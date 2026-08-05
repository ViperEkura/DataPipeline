"""Supervised fine-tuning data processor."""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, ProcessorSchema, encode_with_mask
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("sft")
class SFTProcessor(BaseProcessor):
    """Supervised fine-tuning data processor.

    Supports two input formats:
      1. messages (recommended):
         ``{"messages": [{"role": "user", "content": "..."},
                         {"role": "assistant", "content": "..."}]}``
         Multi-turn and system prompts are supported.
         The tokenizer's ``apply_chat_template`` is used for rendering.
      2. legacy query/response:
         ``{"query": "...", "response": "..."}``
         Falls back to the configured PromptStrategy (ChatML by default).

    Output schema:
        - sequence: int32 tensor - Combined token IDs (prompt + response)
        - loss_mask: bool tensor - True for response tokens (compute loss)
        - position_ids: int32 tensor - Per-sample position IDs starting from 0

    Only the final assistant message is trained (mask_history behavior).
    All earlier turns are context/prompt and masked from loss.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        strategy: Optional[PromptStrategy] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_seq_len = max_seq_len

    @property
    def schema(self) -> ProcessorSchema:
        return ProcessorSchema(
            input_fields={
                "messages": list,
                "query": str,
                "response": str,
            },
            output_fields={
                "sequence": torch.int32,
                "loss_mask": torch.bool,
                "position_ids": torch.int32,
            },
        )

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        if "messages" in input_dict:
            return self._process_messages(input_dict["messages"])
        if "query" in input_dict and "response" in input_dict:
            return self._process_legacy(input_dict)
        raise KeyError(
            "Input must contain 'messages' or 'query'/'response' pair"
        )

    def process_batch(self, input_dicts: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        results: List[Optional[Dict[str, Tensor]]] = [None] * len(input_dicts)
        message_indices = [i for i, item in enumerate(input_dicts) if "messages" in item]
        legacy_indices = [
            i
            for i, item in enumerate(input_dicts)
            if "messages" not in item and "query" in item and "response" in item
        ]
        if len(message_indices) + len(legacy_indices) != len(input_dicts):
            raise KeyError("Input must contain 'messages' or 'query'/'response' pair")

        if message_indices:
            items = [input_dicts[i] for i in message_indices]
            batch_results = self._process_messages_batch(
                [item["messages"] for item in items]
            )
            for index, result in zip(message_indices, batch_results):
                results[index] = result

        if legacy_indices:
            items = [input_dicts[i] for i in legacy_indices]
            batch_results = self._process_legacy_batch(items)
            for index, result in zip(legacy_indices, batch_results):
                results[index] = result

        if any(result is None for result in results):
            raise RuntimeError("Batch processing did not produce all results")
        return results

    def _process_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Tensor]:
        if not messages:
            raise ValueError("Messages list is empty")
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must have role 'assistant'")

        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:last_asst_idx],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        resp_ids = full_ids[len(prompt_ids) :]
        if not resp_ids:
            raise ValueError("Empty assistant response")

        tokens, loss_mask = encode_with_mask(prompt_ids, list(resp_ids))

        if self.max_seq_len and len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
            loss_mask = loss_mask[: self.max_seq_len]

        position_ids = torch.arange(len(tokens), dtype=torch.int32)
        return {
            "sequence": tokens,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    def _process_messages_batch(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Dict[str, Tensor]]:
        for messages in conversations:
            if not messages:
                raise ValueError("Messages list is empty")
            if messages[-1]["role"] != "assistant":
                raise ValueError("Last message must have role 'assistant'")

        assistant_indices = [
            max(i for i, message in enumerate(messages) if message["role"] == "assistant")
            for messages in conversations
        ]
        full_texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in conversations
        ]
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                messages[:assistant_idx], tokenize=False, add_generation_prompt=True
            )
            for messages, assistant_idx in zip(conversations, assistant_indices)
        ]
        full_ids_batch = self.tokenizer.encode(full_texts, add_special_tokens=False)
        prompt_ids_batch = self.tokenizer.encode(prompt_texts, add_special_tokens=False)

        results = []
        for full_ids, prompt_ids in zip(full_ids_batch, prompt_ids_batch):
            resp_ids = full_ids[len(prompt_ids) :]
            if not resp_ids:
                raise ValueError("Empty assistant response")
            tokens, loss_mask = encode_with_mask(prompt_ids, list(resp_ids))
            if self.max_seq_len and len(tokens) > self.max_seq_len:
                tokens = tokens[: self.max_seq_len]
                loss_mask = loss_mask[: self.max_seq_len]
            results.append(
                {
                    "sequence": tokens,
                    "loss_mask": loss_mask,
                    "position_ids": torch.arange(len(tokens), dtype=torch.int32),
                }
            )
        return results

    def _process_legacy(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        strategy = self.strategy or ChatMLStrategy(self.tokenizer)

        query_tokens = self.tokenizer.encode(input_dict["query"])
        response_tokens = self.tokenizer.encode(input_dict["response"])

        prompt = strategy.assemble_prompt(query_tokens)
        response = strategy.assemble_response(response_tokens)

        tokens, loss_mask = encode_with_mask(prompt, response)
        position_ids = torch.arange(len(tokens), dtype=torch.int32)
        return {"sequence": tokens, "loss_mask": loss_mask, "position_ids": position_ids}

    def _process_legacy_batch(
        self, input_dicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Tensor]]:
        strategy = self.strategy or ChatMLStrategy(self.tokenizer)
        query_batch = self.tokenizer.encode([item["query"] for item in input_dicts])
        response_batch = self.tokenizer.encode(
            [item["response"] for item in input_dicts]
        )
        results = []
        for query_tokens, response_tokens in zip(query_batch, response_batch):
            prompt = strategy.assemble_prompt(query_tokens)
            response = strategy.assemble_response(response_tokens)
            tokens, loss_mask = encode_with_mask(prompt, response)
            results.append(
                {
                    "sequence": tokens,
                    "loss_mask": loss_mask,
                    "position_ids": torch.arange(len(tokens), dtype=torch.int32),
                }
            )
        return results

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask", "position_ids"]

"""Supervised fine-tuning data processor."""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, ProcessorSchema
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("sft")
class SFTProcessor(BaseProcessor):
    """Supervised fine-tuning data processor.

    Input formats:
      1. messages (recommended):
         ``{"messages": [{"role": "user", "content": "..."},
                         {"role": "assistant", "content": "..."}]}``
         Multi-turn and system prompts are supported.  Each assistant
         turn gets ``loss_mask = 1``; all other roles get 0.
      2. legacy query/response:
         ``{"query": "...", "response": "..."}``
         Internally converted to messages.

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
            return self._process_messages([
                {"role": "user", "content": input_dict["query"]},
                {"role": "assistant", "content": input_dict["response"]},
            ])
        raise KeyError(
            "Input must contain 'messages' or 'query'/'response' pair"
        )

    def _extract_messages(self, input_dict: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        if "messages" in input_dict:
            return input_dict["messages"]
        if "query" in input_dict and "response" in input_dict:
            return [
                {"role": "user", "content": input_dict["query"]},
                {"role": "assistant", "content": input_dict["response"]},
            ]
        return None

    def _process_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Tensor]:
        if not messages:
            raise ValueError("Messages list is empty")
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must have role 'assistant'")

        strategy = self.strategy or ChatMLStrategy(self.tokenizer)

        prompt, resp = strategy.format_messages(messages)

        sequence = torch.tensor(prompt + resp, dtype=torch.int32)
        loss_mask = torch.zeros(len(sequence), dtype=torch.bool)
        loss_mask[len(prompt) :] = True
        if self.max_seq_len and len(sequence) > self.max_seq_len:
            sequence = sequence[: self.max_seq_len]
            loss_mask = loss_mask[: self.max_seq_len]
        position_ids = torch.arange(len(sequence), dtype=torch.int32)
        return {
            "sequence": sequence,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    def process_batch(self, input_dicts: List[Dict[str, Any]]) -> List[Optional[Dict[str, Tensor]]]:
        strategy = self.strategy or ChatMLStrategy(self.tokenizer)

        prompts_text: List[str] = []
        fulls_text: List[str] = []
        indices: List[int] = []
        results: List[Optional[Dict[str, Tensor]]] = [None] * len(input_dicts)

        for i, d in enumerate(input_dicts):
            try:
                messages = self._extract_messages(d)
                if not messages or messages[-1]["role"] != "assistant":
                    continue
                last_asst = max(j for j, m in enumerate(messages) if m["role"] == "assistant")
                prompt_text = self.tokenizer.apply_chat_template(
                    messages[:last_asst], add_generation_prompt=True, tokenize=False
                )
                full_text = self.tokenizer.apply_chat_template(
                    messages[: last_asst + 1], add_generation_prompt=False, tokenize=False
                )
                prompts_text.append(prompt_text)
                fulls_text.append(full_text)
                indices.append(i)
            except Exception:
                continue

        if not prompts_text:
            return results

        prompt_tokens_list = self.tokenizer.encode(prompts_text)
        full_tokens_list = self.tokenizer.encode(fulls_text)

        for j, idx in enumerate(indices):
            prompt_tokens = prompt_tokens_list[j]
            full_tokens = full_tokens_list[j]
            resp_tokens = full_tokens[len(prompt_tokens):]
            sequence = torch.tensor(prompt_tokens + resp_tokens, dtype=torch.int32)
            loss_mask = torch.zeros(len(sequence), dtype=torch.bool)
            loss_mask[len(prompt_tokens):] = True
            if self.max_seq_len and len(sequence) > self.max_seq_len:
                sequence = sequence[: self.max_seq_len]
                loss_mask = loss_mask[: self.max_seq_len]
            position_ids = torch.arange(len(sequence), dtype=torch.int32)
            results[idx] = {
                "sequence": sequence,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

        return results

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask", "position_ids"]

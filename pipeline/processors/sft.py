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
        - sequence: int32 tensor - Combined token IDs
        - loss_mask: bool tensor - True for assistant response tokens
        - position_ids: int32 tensor - Per-sample position IDs, start from 0
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        strategy: Optional[PromptStrategy] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy

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
        position_ids = torch.arange(len(sequence), dtype=torch.int32)
        return {
            "sequence": sequence,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask", "position_ids"]

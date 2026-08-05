"""DPO preference learning data processor."""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, ProcessorSchema, encode_with_mask
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("dpo")
class DPOProcessor(BaseProcessor):
    """DPO (Direct Preference Optimization) data processor.

    Processes query, chosen, and rejected responses for preference learning.

    Input schema:
        - query: str - User query/prompt
        - chosen: str - Preferred assistant response
        - rejected: str - Dispreferred assistant response

    Output schema:
        - chosen: int32 tensor - Token IDs for preferred response
        - chosen_mask: bool tensor - True for response tokens
        - rejected: int32 tensor - Token IDs for dispreferred response
        - rejected_mask: bool tensor - True for response tokens
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        strategy: Optional[PromptStrategy] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy or ChatMLStrategy(tokenizer)

    @property
    def schema(self) -> ProcessorSchema:
        return ProcessorSchema(
            input_fields={
                "query": str,
                "chosen": str,
                "rejected": str,
            },
            output_fields={
                "chosen": torch.int32,
                "chosen_mask": torch.bool,
                "rejected": torch.int32,
                "rejected_mask": torch.bool,
            },
        )

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        query_tokens = self.tokenizer.encode(input_dict["query"])
        chosen_tokens = self.tokenizer.encode(input_dict["chosen"])
        rejected_tokens = self.tokenizer.encode(input_dict["rejected"])

        prompt = self.strategy.assemble_prompt(query_tokens)

        chosen_t, chosen_m = encode_with_mask(
            prompt, self.strategy.assemble_response(chosen_tokens)
        )
        rejected_t, rejected_m = encode_with_mask(
            prompt, self.strategy.assemble_response(rejected_tokens)
        )

        return {
            "chosen": chosen_t,
            "chosen_mask": chosen_m,
            "rejected": rejected_t,
            "rejected_mask": rejected_m,
        }

    def process_batch(self, input_dicts: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        query_batch = self.tokenizer.encode([item["query"] for item in input_dicts])
        chosen_batch = self.tokenizer.encode([item["chosen"] for item in input_dicts])
        rejected_batch = self.tokenizer.encode(
            [item["rejected"] for item in input_dicts]
        )
        results = []
        for query_tokens, chosen_tokens, rejected_tokens in zip(
            query_batch, chosen_batch, rejected_batch
        ):
            prompt = self.strategy.assemble_prompt(query_tokens)
            chosen_t, chosen_m = encode_with_mask(
                prompt, self.strategy.assemble_response(chosen_tokens)
            )
            rejected_t, rejected_m = encode_with_mask(
                prompt, self.strategy.assemble_response(rejected_tokens)
            )
            results.append(
                {
                    "chosen": chosen_t,
                    "chosen_mask": chosen_m,
                    "rejected": rejected_t,
                    "rejected_mask": rejected_m,
                }
            )
        return results

    @property
    def output_keys(self) -> List[str]:
        return ["chosen", "chosen_mask", "rejected", "rejected_mask"]

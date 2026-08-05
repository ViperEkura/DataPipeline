"""Pre-training data processor."""

from typing import Any, Dict, List

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.processors.base import BaseProcessor, ProcessorSchema
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("pt")
class PreTrainProcessor(BaseProcessor):
    """Pre-training data processor.

    Processes raw text into tokenized sequences with EOS tokens.

    Input schema:
        - text: str - Raw text string to tokenize

    Output schema:
        - sequence: int32 tensor - Token IDs with EOS appended
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        eos_token: str = "<｜end▁of▁sentence｜>",
    ):
        self.tokenizer = tokenizer
        self._eos_token = eos_token

    @property
    def schema(self) -> ProcessorSchema:
        return ProcessorSchema(
            input_fields={"text": str},
            output_fields={"sequence": torch.int32},
        )

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        segment = input_dict["text"]
        tokens = self.tokenizer.encode(f"{segment}{self._eos_token}")
        return {"sequence": torch.tensor(tokens, dtype=torch.int32)}

    def process_batch(self, input_dicts: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        texts = [f"{item['text']}{self._eos_token}" for item in input_dicts]
        encoded = self.tokenizer.encode(texts)
        return [
            {"sequence": torch.tensor(tokens, dtype=torch.int32)}
            for tokens in encoded
        ]

    @property
    def output_keys(self) -> List[str]:
        return ["sequence"]

"""Processor base class and shared utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProcessorSchema:
    """Schema definition for processor input/output contracts.

    Attributes:
        input_fields: Required input field names and their types.
        output_fields: Output field names and their tensor dtypes.
    """

    input_fields: Dict[str, type]
    output_fields: Dict[str, torch.dtype]


class BaseProcessor(ABC):
    """Abstract base class for data processors.

    Processors transform raw data (e.g., text, JSON) into tokenized tensors
    suitable for model training.

    Subclasses must implement:
    - process(): Transform a single input sample
    - output_keys: Declare output tensor names
    - schema: Define input/output contracts (optional but recommended)

    Example::

        class MyProcessor(BaseProcessor):
            @property
            def schema(self) -> ProcessorSchema:
                return ProcessorSchema(
                    input_fields={"text": str},
                    output_fields={"tokens": torch.int32}
                )

            def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
                tokens = self.tokenizer.encode(input_dict["text"])
                return {"tokens": torch.tensor(tokens, dtype=torch.int32)}

            @property
            def output_keys(self) -> List[str]:
                return ["tokens"]
    """

    @property
    def schema(self) -> ProcessorSchema:
        """Return the input/output schema for this processor.

        Override this property to define explicit contracts.
        Default returns None, meaning schema is not defined.
        """
        return None

    @abstractmethod
    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Process a single input sample.

        Args:
            input_dict: Dictionary containing input fields as defined by schema.

        Returns:
            Dictionary mapping output key names to tensors.

        Raises:
            KeyError: If required input fields are missing.
            ValueError: If input data is invalid.
        """
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Return list of output tensor key names."""
        pass

    def process_batch(self, input_dicts: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        """Process a batch of input samples.

        Default implementation calls process() for each sample.
        Subclasses should override for efficient batch processing
        (e.g., using tokenizer.encode_batch).

        Args:
            input_dicts: List of input dictionaries.

        Returns:
            List of output dictionaries mapping output key names to tensors.
        """
        return [self.process(d) for d in input_dicts]

    def validate_input(self, input_dict: Dict[str, Any]) -> None:
        """Validate input against schema before processing.

        Args:
            input_dict: Input dictionary to validate.

        Raises:
            KeyError: If required fields are missing.
            TypeError: If field types don't match schema.
        """
        schema = self.schema
        if schema is None:
            return

        for field_name, expected_type in schema.input_fields.items():
            if field_name not in input_dict:
                raise KeyError(f"Missing required input field: '{field_name}'")
            if not isinstance(input_dict[field_name], expected_type):
                raise TypeError(
                    f"Field '{field_name}' expected type {expected_type.__name__}, "
                    f"got {type(input_dict[field_name]).__name__}"
                )


def encode_with_mask(
    prompt_tokens: List[int],
    response_tokens: List[int],
) -> Tuple[Tensor, Tensor]:
    """Concatenate token lists and build loss mask (prompt=False, response=True).

    Args:
        prompt_tokens: Token IDs for the prompt/question.
        response_tokens: Token IDs for the response/answer.

    Returns:
        Tuple of (combined_tokens, loss_mask) where loss_mask is True for response tokens.
    """
    q_len = len(prompt_tokens)
    combined = torch.tensor(prompt_tokens + response_tokens, dtype=torch.int32)
    mask = torch.zeros(q_len + len(response_tokens), dtype=torch.bool)
    mask[q_len:] = True
    return combined, mask

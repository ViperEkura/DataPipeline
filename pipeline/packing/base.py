from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torch import Tensor


class BasePacker(ABC):
    """Abstract base class for sequence packing algorithms.

    All packers must implement pack() and reset().
    pack() takes a list of 1D tensors and returns a list of packed fixed-size tensors.
    """

    def __init__(
        self,
        pack_size: int,
        pad_value: Union[int, bool] = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        self.pack_size = pack_size
        self.pad_value = pad_value
        self.dtype = dtype

    @abstractmethod
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        """Pack sequences into fixed-size chunks."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset packer state for instance reuse."""
        ...

    def _validate_and_normalize(self, sequences: List[Tensor]) -> List[Tensor]:
        """Validate 1D tensors and unify dtype."""
        if self.dtype is None and sequences:
            self.dtype = sequences[0].dtype

        normalized: List[Tensor] = []
        for i, seq in enumerate(sequences):
            if seq.dim() != 1:
                raise ValueError(
                    f"Expected 1D tensor at index {i}, got {seq.dim()}D tensor with shape {seq.shape}"
                )
            if seq.dtype != self.dtype:
                seq = seq.to(self.dtype)
            normalized.append(seq)
        return normalized

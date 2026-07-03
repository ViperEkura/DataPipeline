from typing import List, Optional, Union

import torch
from torch import Tensor

from pipeline.packing.base import BasePacker
from pipeline.utils import error_handler


def _truncate(tokens: List, max_len: int) -> List:
    return tokens[:max_len]


def _pad_bin(bin_list: List, target_len: int, pad_value: Union[int, bool], dtype: torch.dtype) -> Tensor:
    bin_list.extend([pad_value] * (target_len - len(bin_list)))
    return torch.tensor(bin_list, dtype=dtype)


class GreedyPacker(BasePacker):
    """Greedy first-fit packer (no sorting).

    Sequences are packed in input order into the first bin with enough space.
    Overlong sequences (> pack_size) are truncated to pack_size.
    """

    def __init__(
        self,
        pack_size: int,
        pad_value: Union[int, bool] = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(pack_size, pad_value, dtype)
        self._bins: List[List] = []

    def reset(self) -> None:
        self._bins = []

    @error_handler()
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        if not sequences:
            return []

        normalized = self._validate_and_normalize(sequences)
        self._bins = []
        pack_size = self.pack_size
        pad_value = self.pad_value

        for seq in normalized:
            seq_len = int(seq.shape[0])
            if seq_len > pack_size:
                self._bins.append(_truncate(seq.tolist(), pack_size))
                continue
            placed = False
            for bin_list in self._bins:
                if len(bin_list) + seq_len <= pack_size:
                    bin_list.extend(seq.tolist())
                    placed = True
                    break
            if not placed:
                self._bins.append(list(seq.tolist()))

        packages: List[Tensor] = []
        for bin_list in self._bins:
            packages.append(_pad_bin(bin_list, pack_size, pad_value, self.dtype))

        return packages


class FfDPacker(BasePacker):
    """First-Fit Decreasing (FFD) bin-packing packer.

    Sequences are sorted by descending length, then packed into the first
    bin with enough space. Overlong sequences are truncated to pack_size.
    """

    def __init__(
        self,
        pack_size: int,
        pad_value: Union[int, bool] = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(pack_size, pad_value, dtype)
        self._bins: List[List] = []

    def reset(self) -> None:
        self._bins = []

    @error_handler()
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        if not sequences:
            return []

        normalized = self._validate_and_normalize(sequences)
        self._bins = []
        pack_size = self.pack_size
        pad_value = self.pad_value

        indexed = [(int(s.shape[0]), s) for s in normalized]
        indexed.sort(key=lambda x: x[0], reverse=True)

        for seq_len, seq in indexed:
            if seq_len > pack_size:
                self._bins.append(_truncate(seq.tolist(), pack_size))
                continue
            placed = False
            for bin_list in self._bins:
                if len(bin_list) + seq_len <= pack_size:
                    bin_list.extend(seq.tolist())
                    placed = True
                    break
            if not placed:
                self._bins.append(list(seq.tolist()))

        packages: List[Tensor] = []
        for bin_list in self._bins:
            packages.append(_pad_bin(bin_list, pack_size, pad_value, self.dtype))

        return packages


class BfdPacker(BasePacker):
    """Best-Fit Decreasing (BFD) bin-packing packer.

    Sequences are sorted by descending length, then packed into the bin
    that minimizes remaining space (tightest fit).
    Overlong sequences are truncated to pack_size.
    """

    def __init__(
        self,
        pack_size: int,
        pad_value: Union[int, bool] = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(pack_size, pad_value, dtype)
        self._bins: List[List] = []

    def reset(self) -> None:
        self._bins = []

    @error_handler()
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        if not sequences:
            return []

        normalized = self._validate_and_normalize(sequences)
        self._bins = []
        pack_size = self.pack_size
        pad_value = self.pad_value

        indexed = [(int(s.shape[0]), s) for s in normalized]
        indexed.sort(key=lambda x: x[0], reverse=True)

        for seq_len, seq in indexed:
            if seq_len > pack_size:
                self._bins.append(_truncate(seq.tolist(), pack_size))
                continue
            best_idx = -1
            best_remain = pack_size + 1
            for i, bin_list in enumerate(self._bins):
                remain = pack_size - len(bin_list)
                if seq_len <= remain < best_remain:
                    best_remain = remain
                    best_idx = i
            if best_idx >= 0:
                self._bins[best_idx].extend(seq.tolist())
            else:
                self._bins.append(list(seq.tolist()))

        packages: List[Tensor] = []
        for bin_list in self._bins:
            packages.append(_pad_bin(bin_list, pack_size, pad_value, self.dtype))

        return packages

"""Sequence packing algorithms for LLM training data.

Available packers:
    - BfdPacker:        Best-Fit Decreasing, samples never split (default)
    - FfDPacker:        First-Fit Decreasing, samples never split
    - GreedyPacker:     First-fit in input order, samples never split
"""

from typing import Dict, List, Optional, Union

import torch

from pipeline.packing.base import BasePacker
from pipeline.packing.binpack import GreedyPacker, FfDPacker, BfdPacker


def pack_tensors(
    tensors: Dict[str, List[torch.Tensor]],
    pack_size: int,
    pad_value: Union[int, bool] = 0,
    dtypes: Optional[Dict[str, torch.dtype]] = None,
    pad_values: Optional[Dict[str, Union[int, bool]]] = None,
    algo: Optional[Union[str, BasePacker]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Pack multiple named tensor groups in parallel.

    Each group is packed independently with its own packer instance.

    Args:
        tensors: Dict mapping key names to lists of 1D tensors.
        pack_size: Fixed chunk length.
        pad_value: Default padding value, used for keys not in pad_values.
        dtypes: Optional per-key dtype declarations.
        pad_values: Optional per-key padding values (e.g. pad_token_id for
            'sequence', False for 'loss_mask', 0 for 'position_ids').
        algo: Packing algorithm to use. Can be 'bfd' (default),
              'ffd', 'greedy', or a BasePacker instance.

    Returns:
        Dict mapping key names to lists of packed tensors.
    """
    if dtypes is None:
        dtypes = {}
    if pad_values is None:
        pad_values = {}

    output: Dict[str, List[torch.Tensor]] = {}
    for key, seqs in tensors.items():
        key_pad = pad_values.get(key, pad_value)
        actual_packer = _resolve_algo(algo, pack_size, key_pad)
        dtype = dtypes.get(key)
        if dtype is not None:
            actual_packer.dtype = dtype
        output[key] = actual_packer.pack(seqs)
    return output


def _resolve_algo(
    algo: Optional[Union[str, BasePacker]],
    pack_size: int,
    pad_value: Union[int, bool],
) -> BasePacker:
    if algo is None or algo == "bfd":
        return BfdPacker(pack_size, pad_value)
    if isinstance(algo, BasePacker):
        cls = type(algo)
        return cls(pack_size, pad_value)
    if algo == "ffd":
        return FfDPacker(pack_size, pad_value)
    if algo == "greedy":
        return GreedyPacker(pack_size, pad_value)
    raise ValueError(
        f"Unknown packing algorithm: {algo}. "
        f"Choose from: bfd, ffd, greedy"
    )


__all__ = [
    "BasePacker",
    "BfdPacker",
    "FfDPacker",
    "GreedyPacker",
    "pack_tensors",
]

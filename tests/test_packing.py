"""Tests for pipeline.packing module."""

import pytest
import torch
from pipeline.packing import (
    GreedyPacker,
    FfDPacker,
    BfdPacker,
    pack_tensors,
)


class TestBfdPacker:
    def test_best_fit_tight(self):
        packer = BfdPacker(pack_size=10, pad_value=-1)
        sequences = [
            torch.tensor([5, 6], dtype=torch.int32),
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            torch.tensor([5, 6, 7, 8], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 1
        assert packages[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 5, 6]

    def test_different_dtypes(self):
        for dtype in [torch.int32, torch.int64, torch.float32]:
            packer = BfdPacker(pack_size=10, dtype=dtype)
            val = 1.0 if dtype == torch.float32 else 1
            packages = packer.pack([torch.tensor([val, 2, 3], dtype=dtype)])
            assert packages[0].dtype == dtype

    def test_dtype_conversion_on_mismatch(self):
        packer = BfdPacker(pack_size=10, dtype=torch.int32)
        packages = packer.pack([torch.tensor([1, 2, 3], dtype=torch.int64)])
        assert packages[0].dtype == torch.int32
        assert packages[0][:3].tolist() == [1, 2, 3]

    def test_non_1d_tensor_raises_error(self):
        packer = BfdPacker(pack_size=10)
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack([torch.tensor([[1, 2], [3, 4]])])
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack([torch.tensor(5)])

    def test_empty_input(self):
        packer = BfdPacker(pack_size=10)
        assert packer.pack([]) == []

    def test_reset(self):
        packer = BfdPacker(pack_size=10)
        packer.pack([torch.tensor([1, 2, 3], dtype=torch.int32)])
        assert len(packer._bins) == 1
        packer.reset()
        assert len(packer._bins) == 0

    def test_overlong_sample_truncated(self):
        """Overlong sample is truncated to pack_size."""
        packer = BfdPacker(pack_size=6, pad_value=-1)
        packages = packer.pack(
            [
                torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
                torch.tensor([8, 9], dtype=torch.int32),
            ]
        )
        assert len(packages) == 2
        assert packages[0].tolist() == [1, 2, 3, 4, 5, 6]
        assert packages[1].tolist() == [8, 9, -1, -1, -1, -1]

    def test_uses_two_bins_when_needed(self):
        packer = BfdPacker(pack_size=10, pad_value=0)
        sequences = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6, 7], dtype=torch.int32),
            torch.tensor([8, 9, 10], dtype=torch.int32),
            torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 2
        for pkg in packages:
            assert pkg.shape == (10,)

    def test_minimizes_waste_vs_ffd(self):
        sequences = [
            torch.tensor([6] * i, dtype=torch.int32)
            for i in [3, 5, 5, 7, 2, 4, 1, 4, 6, 2]
        ]
        bfd = BfdPacker(pack_size=10, pad_value=0)
        ffd = FfDPacker(pack_size=10, pad_value=0)
        assert len(bfd.pack(sequences)) <= len(ffd.pack(sequences))


class TestFfDPacker:
    def test_fills_tightly(self):
        packer = FfDPacker(pack_size=10, pad_value=0)
        sequences = [
            torch.tensor([7, 8], dtype=torch.int32),
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32),
            torch.tensor([9, 10], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 1

    def test_overlong_sample_truncated(self):
        packer = FfDPacker(pack_size=5, pad_value=0)
        packages = packer.pack(
            [torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)]
        )
        assert len(packages) == 1
        assert packages[0].tolist() == [1, 2, 3, 4, 5]

    def test_sort_descending_order(self):
        packer = FfDPacker(pack_size=10, pad_value=-1)
        sequences = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int32),
            torch.tensor([9, 10], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 1
        assert packages[0].tolist() == [3, 4, 5, 6, 7, 8, 1, 2, 9, 10]

    def test_reduces_bins_vs_greedy(self):
        sequences = [
            torch.tensor([6] * i, dtype=torch.int32)
            for i in [3, 8, 2, 7, 1, 4, 5, 3, 2, 6]
        ]
        greedy = GreedyPacker(pack_size=10, pad_value=0)
        ffd = FfDPacker(pack_size=10, pad_value=0)
        assert len(ffd.pack(sequences)) <= len(greedy.pack(sequences))

    def test_reset(self):
        packer = FfDPacker(pack_size=10)
        packer.pack([torch.tensor([1, 2, 3], dtype=torch.int32)])
        assert len(packer._bins) == 1
        packer.reset()
        assert len(packer._bins) == 0

    def test_empty_input(self):
        packer = FfDPacker(pack_size=10)
        assert packer.pack([]) == []


class TestGreedyPacker:
    def test_basic_packing(self):
        packer = GreedyPacker(pack_size=10, pad_value=0)
        sequences = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8, 9], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 1
        assert packages[0].shape == (10,)
        assert packages[0][:9].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert packages[0][9] == 0

    def test_overlong_sample_truncated(self):
        """Overlong sample is truncated to pack_size."""
        packer = GreedyPacker(pack_size=5, pad_value=0)
        packages = packer.pack(
            [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)]
        )
        assert len(packages) == 1
        assert packages[0].tolist() == [1, 2, 3, 4, 5]

    def test_multiple_fill(self):
        packer = GreedyPacker(pack_size=6, pad_value=0)
        sequences = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3, 4], dtype=torch.int32),
            torch.tensor([5, 6], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) == 2
        for pkg in packages:
            assert pkg.shape == (6,)

    def test_reset(self):
        packer = GreedyPacker(pack_size=10)
        packer.pack([torch.tensor([1, 2, 3], dtype=torch.int32)])
        assert len(packer._bins) == 1
        packer.reset()
        assert len(packer._bins) == 0

    def test_empty_input(self):
        packer = GreedyPacker(pack_size=10)
        assert packer.pack([]) == []


class TestPackTensors:
    def test_default_is_bfd(self):
        result = pack_tensors(
            tensors={
                "input_ids": [
                    torch.tensor([1, 2], dtype=torch.int32),
                    torch.tensor([3, 4], dtype=torch.int32),
                    torch.tensor([5], dtype=torch.int32),
                ],
            },
            pack_size=5,
            pad_value=0,
        )
        assert result["input_ids"][0].tolist() == [1, 2, 3, 4, 5]

    def test_greedy(self):
        result = pack_tensors(
            tensors={
                "input_ids": [
                    torch.tensor([1, 2, 3], dtype=torch.int32),
                    torch.tensor([4, 5], dtype=torch.int32),
                ],
            },
            pack_size=5,
            pad_value=0,
            algo="greedy",
        )
        assert result["input_ids"][0].tolist() == [1, 2, 3, 4, 5]

    def test_ffd(self):
        result = pack_tensors(
            tensors={
                "input_ids": [
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([2, 3, 4], dtype=torch.int32),
                    torch.tensor([5], dtype=torch.int32),
                ],
            },
            pack_size=5,
            pad_value=0,
            algo="ffd",
        )
        assert result["input_ids"][0].tolist() == [2, 3, 4, 1, 5]

    def test_bfd_explicit(self):
        result = pack_tensors(
            tensors={
                "input_ids": [
                    torch.tensor([1, 2], dtype=torch.int32),
                    torch.tensor([3, 4], dtype=torch.int32),
                    torch.tensor([5], dtype=torch.int32),
                ],
            },
            pack_size=5,
            pad_value=0,
            algo="bfd",
        )
        assert result["input_ids"][0].tolist() == [1, 2, 3, 4, 5]

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            pack_tensors(
                tensors={"input_ids": [torch.tensor([1, 2, 3])]},
                pack_size=10,
                pad_value=0,
                algo="unknown_algo",
            )


class TestPositionIdsPacking:
    """Verify position_ids reset to zero at sample boundaries after packing."""

    def test_position_ids_reset_in_packed_chunk(self):
        """After packing multiple SFT samples, position_ids restart from 0 at each boundary."""
        seqs = [
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),          # len=5
            torch.tensor([0, 1, 2], dtype=torch.int32),                 # len=3
            torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32),    # len=7
        ]
        result = pack_tensors(
            tensors={"position_ids": seqs},
            pack_size=16,
            pad_value=-1,
            algo="greedy",
        )
        packed = result["position_ids"][0].tolist()
        assert packed == [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, -1]

    def test_position_ids_reset_with_bfd(self):
        """BFD may reorder, but each sample's position_ids still start from 0."""
        seqs = [
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        ]
        result = pack_tensors(
            tensors={"position_ids": seqs},
            pack_size=16,
            pad_value=-1,
            algo="bfd",
        )
        packed = result["position_ids"][0].tolist()
        assert packed[0] == 0
        zeros = [i for i, v in enumerate(packed) if v == 0 and (i == 0 or packed[i - 1] != 0)]
        assert len(zeros) == 3

    def test_multiple_keys_share_same_boundaries(self):
        """sequence, loss_mask, position_ids share identical chunk boundaries after packing."""
        seq_a = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        seq_b = torch.tensor([201, 202, 203, 204, 205, 206, 207], dtype=torch.int32)
        seq_c = torch.tensor([301, 302, 303, 304, 305], dtype=torch.int32)

        mask_a = torch.tensor([False, False, True, True], dtype=torch.bool)
        mask_b = torch.tensor([False, False, False, False, True, True, True], dtype=torch.bool)
        mask_c = torch.tensor([False, False, False, True, True], dtype=torch.bool)

        pos_a = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        pos_b = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)
        pos_c = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)

        result = pack_tensors(
            tensors={
                "sequence": [seq_a, seq_b, seq_c],
                "loss_mask": [mask_a, mask_b, mask_c],
                "position_ids": [pos_a, pos_b, pos_c],
            },
            pack_size=16,
            pad_value=-1,
            algo="greedy",
        )

        seq_chunk = result["sequence"][0]
        mask_chunk = result["loss_mask"][0]
        pos_chunk = result["position_ids"][0]

        assert len(seq_chunk) == len(mask_chunk) == len(pos_chunk) == 16

        for i in range(16):
            if seq_chunk[i] == -1:
                assert mask_chunk[i] == -1
                assert pos_chunk[i] == -1

        pos_ids = pos_chunk.tolist()
        zeros = [i for i, v in enumerate(pos_ids) if v == 0]
        assert len(zeros) == 3

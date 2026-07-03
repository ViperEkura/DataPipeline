"""Tests for pipeline.io module."""

import os
import tempfile
import pytest
import torch
import h5py
from pathlib import Path

from pipeline.io import FileScanner, HDF5Handler, cache_jsonl


class TestFileScanner:
    def test_scan_files_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            Path(subdir, "file3.txt").touch()

            files = FileScanner.scan(tmpdir)
            assert len(files) == 3

    def test_scan_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert FileScanner.scan(tmpdir) == []

    def test_scan_with_suffix_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.json").touch()

            txt_files = FileScanner.scan(tmpdir, suffix=".txt")
            assert len(txt_files) == 1
            assert txt_files[0].endswith(".txt")

    def test_scan_folders_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "folder1"))
            os.makedirs(os.path.join(tmpdir, "folder2"))
            os.makedirs(os.path.join(tmpdir, "folder1", "nested"))

            folders = FileScanner.scan_folders(tmpdir)
            assert len(folders) == 3

    def test_scan_folders_with_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "folder1"))
            os.makedirs(os.path.join(tmpdir, "folder2"))

            folders = FileScanner.scan_folders(
                tmpdir, filter_func=lambda x: "folder1" in x
            )
            assert len(folders) == 1

    def test_group_by_extension(self):
        files = ["/path/file1.txt", "/path/file2.txt", "/path/file3.json"]
        groups = FileScanner.group_by_extension(files)
        assert ".txt" in groups
        assert ".json" in groups
        assert len(groups[".txt"]) == 2
        assert len(groups[".json"]) == 1


class TestHDF5Handler:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "sequence": [torch.tensor([1, 2, 3], dtype=torch.int32)],
                "labels": [torch.tensor([4, 5], dtype=torch.int32)],
            }
            h5_path = HDF5Handler.save(tmpdir, "test", tensor_group)

            assert os.path.exists(h5_path)
            loaded = HDF5Handler.load(h5_path, share_memory=False)
            assert "sequence" in loaded
            assert "labels" in loaded
            assert torch.equal(
                loaded["sequence"][0], torch.tensor([1, 2, 3], dtype=torch.int32)
            )
            assert torch.equal(
                loaded["labels"][0], torch.tensor([4, 5], dtype=torch.int32)
            )

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")
            HDF5Handler.save(output_dir, "test", {"data": [torch.tensor([1, 2, 3])]})
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "test.h5"))

    def test_load_directory_with_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, data in enumerate([[1, 2, 3], [4, 5, 6]]):
                h5_path = os.path.join(tmpdir, f"file{i}.h5")
                with h5py.File(h5_path, "w") as f:
                    grp = f.create_group("data")
                    grp.create_dataset("data_0", data=data)

            loaded = HDF5Handler.load(tmpdir, share_memory=False)
            assert len(loaded["data"]) == 2

    def test_load_directory_with_nested_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            h5_path = os.path.join(subdir, "nested.h5")
            with h5py.File(h5_path, "w") as f:
                grp = f.create_group("test")
                grp.create_dataset("data_0", data=[1, 2])

            loaded = HDF5Handler.load(tmpdir, share_memory=False)
            assert "test" in loaded
            assert len(loaded["test"]) == 1

    def test_save_multiple_tensors_per_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "batch": [
                    torch.tensor([1, 2]),
                    torch.tensor([3, 4, 5]),
                    torch.tensor([6]),
                ],
            }
            HDF5Handler.save(tmpdir, "multi", tensor_group)
            loaded = HDF5Handler.load(tmpdir, share_memory=False)
            assert len(loaded["batch"]) == 3

    def test_get_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "data": [torch.tensor([1, 2, 3]) for _ in range(5)],
            }
            HDF5Handler.save(tmpdir, "meta", tensor_group)

            h5_path = os.path.join(tmpdir, "meta.h5")
            metadata = HDF5Handler.get_metadata(h5_path)
            assert metadata["data"] == 5


class DummyTokenizer:
    im_end = "<|im_end|>"

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]

    def token_to_id(self, token: str):
        return ord(token)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True
    ):
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class TestPositionIds:
    def test_example_specific_position_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "data.jsonl")
            with open(jsonl_path, "w") as f:
                f.write('{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bc"}]}\n')
                f.write('{"messages": [{"role": "user", "content": "def"}, {"role": "assistant", "content": "g"}]}\n')

            from pipeline.processors import SFTProcessor

            processor = SFTProcessor(DummyTokenizer())
            out_dir = os.path.join(tmpdir, "cached")
            cache_jsonl([jsonl_path], out_dir, processor, pack_size=-1, group_size=0)

            h5_path = os.path.join(out_dir, "data.h5")
            loaded = HDF5Handler.load(h5_path, share_memory=False)

            assert "position_ids" in loaded
            assert len(loaded["position_ids"]) == 2
            assert len(loaded["position_ids"]) == len(loaded["sequence"])

            for i, pos in enumerate(loaded["position_ids"]):
                seq_len = len(loaded["sequence"][i])
                assert len(pos) == seq_len
                assert pos[0].item() == 0
                assert (pos == torch.arange(seq_len, dtype=torch.int32)).all()

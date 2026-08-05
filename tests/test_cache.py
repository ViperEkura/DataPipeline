"""Tests for pipeline.cache module."""

import json
import os
import tempfile
import torch
from pathlib import Path

from pipeline.io import cache_jsonl
from pipeline.processors import BaseProcessor


class DummyProcessor(BaseProcessor):
    """Dummy processor for testing."""

    def __init__(self):
        self._output_keys = ["sequence", "loss_mask"]

    @property
    def output_keys(self):
        return self._output_keys

    def process(self, item):
        text = item.get("text", "")
        tokens = [ord(c) for c in text[:10]]
        return {
            "sequence": torch.tensor(tokens, dtype=torch.int32),
            "loss_mask": torch.ones(len(tokens), dtype=torch.int32),
        }


class BatchTrackingProcessor(DummyProcessor):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def process_batch(self, items):
        self.batch_sizes.append(len(items))
        return super().process_batch(items)


class TestCacheJsonl:
    def test_basic_cache_functionality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [{"text": "hello"}, {"text": "world"}, {"text": "test"}]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            processor = DummyProcessor()
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,
                pad_value=0,
            )
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])

    def test_packer_state_independence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [{"text": "ab"}, {"text": "abcde"}, {"text": "abc"}]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            processor = DummyProcessor()
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=10,
                pad_value=0,
            )
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])

    def test_no_packing_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [{"text": "hello"}, {"text": "world"}]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            processor = DummyProcessor()
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=0,
                pad_value=-1,
            )
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])

    def test_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(2):
                jsonl_path = os.path.join(tmpdir, f"test{i}.jsonl")
                test_data = [{"text": f"data{i}"}]
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(test_data[0]) + "\n")
                files.append(jsonl_path)

            processor = DummyProcessor()
            output_files = cache_jsonl(
                files=files,
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,
                pad_value=0,
            )
            assert len(output_files) == 2

    def test_empty_file_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "empty.jsonl")
            Path(jsonl_path).touch()

            processor = DummyProcessor()
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,
                pad_value=0,
            )
            assert len(output_files) == 0

    def test_uses_configured_batch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(5):
                    f.write(json.dumps({"text": str(i)}) + "\n")

            processor = BatchTrackingProcessor()
            cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                batch_size=2,
            )
            assert processor.batch_sizes == [2, 2, 1]

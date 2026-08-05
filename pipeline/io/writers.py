"""Storage backends for tensor / text output (Strategy + Factory).

Each backend implements a common ``save()`` interface so callers use
polymorphism instead of ``if fmt == "h5" ... elif fmt == "bin" ...``.

Supports:
    - **H5Writer**: HDF5 format (via HDF5Handler)
    - **BinWriter**: binary format – meta.json + {key}.bin (memmap-compatible)
    - **TextWriter**: raw JSONL text (for dedup output)
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor


class BaseWriter(ABC):
    """Abstract writer – call ``save(dir, name, data)`` without caring
    about the underlying format."""

    @abstractmethod
    def save(self, output_dir: str, file_name: str, data: Dict[str, List[Tensor]]) -> str:
        ...


class H5Writer(BaseWriter):
    def save(self, output_dir: str, file_name: str, data: Dict[str, List[Tensor]]) -> str:
        from pipeline.io.hdf5_handler import HDF5Handler
        return HDF5Handler.save(output_dir, file_name, data)


class BinWriter(BaseWriter):
    def save(self, output_dir: str, file_name: str, data: Dict[str, List[Tensor]]) -> str:
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)
        sub_dir = os.path.join(output_dir, file_name)
        os.makedirs(sub_dir, exist_ok=True)

        meta: Dict[str, Dict] = {}
        for key, tensors in data.items():
            cat = torch.cat(tensors, dim=0)
            meta[key] = {"shape": list(cat.shape), "dtype": str(cat.dtype).split(".")[-1]}
            np.asarray(cat.cpu().numpy()).tofile(os.path.join(sub_dir, f"{key}.bin"))

        with open(os.path.join(sub_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return sub_dir


class TextWriter(BaseWriter):
    """Write raw text records as JSONL (used by dedup output)."""

    def __init__(self, chunk_size: int = 1_000_000):
        self._chunk_size = chunk_size
        self._buffer: List[dict] = []
        self._chunk_idx = 0

    def save(self, output_dir: str, file_name: str, data: Dict[str, List[Tensor]]) -> str:
        raise NotImplementedError("TextWriter.save_one is for tensor data; use write_record()")

    def write_record(self, record: dict, output_dir: Path):
        self._buffer.append(record)
        if len(self._buffer) >= self._chunk_size:
            self._flush(output_dir)

    def flush(self, output_dir: Path):
        if self._buffer:
            self._flush(output_dir)

    def _flush(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        fpath = output_dir / f"chunk_{self._chunk_idx}.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._chunk_idx += 1
        self._buffer = []


_WRITER_REGISTRY: Dict[str, type] = {}


def register_writer(name: str):
    def decorator(cls):
        _WRITER_REGISTRY[name] = cls
        return cls
    return decorator


def create_writer(name: str, **kwargs) -> BaseWriter:
    cls = _WRITER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown writer: {name}. Available: {list(_WRITER_REGISTRY)}")
    return cls(**kwargs)


# Register built-in writers
register_writer("h5")(H5Writer)
register_writer("bin")(BinWriter)
register_writer("jsonl")(TextWriter)

"""I/O module for file operations, HDF5 storage, and dataset export.

This module provides:
- FileScanner: File and directory scanning utilities
- HDF5Handler: Tensor data persistence
- export_dataset: HuggingFace Dataset to JSONL export
- cache_jsonl: JSONL to HDF5/binary tokenization and caching
- dedup_jsonl: MinHash+LSH deduplication for pretraining text
- writers: BaseWriter / H5Writer / BinWriter / TextWriter (Strategy + Factory)
"""

from pipeline.io.file_scanner import FileScanner
from pipeline.io.hdf5_handler import HDF5Handler
from pipeline.io.export import export_dataset, cache_jsonl
from pipeline.io.dedup import dedup_jsonl
from pipeline.io.writers import BaseWriter, H5Writer, BinWriter, TextWriter, create_writer

__all__ = [
    "FileScanner",
    "HDF5Handler",
    "export_dataset",
    "cache_jsonl",
    "dedup_jsonl",
    "BaseWriter",
    "H5Writer",
    "BinWriter",
    "TextWriter",
    "create_writer",
]

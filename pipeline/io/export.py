"""Dataset export and caching utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datasets import Dataset
from tqdm import tqdm

from pipeline.io.file_scanner import FileScanner
from pipeline.io.hdf5_handler import HDF5Handler
from pipeline.processors import BaseProcessor
from pipeline.packing import pack_tensors
from pipeline.utils import error_handler

logger = logging.getLogger(__name__)


@error_handler()
def export_dataset(
    dataset: Dataset,
    output_dir: str,
    output_prefix: str,
    *,
    chunk_size: int = 1_000_000,
    max_chunks: Optional[int] = None,
    process_func: Optional[
        Callable[[Dict[str, Any]], Union[Dict[str, Any], List[Dict[str, Any]]]]
    ] = None,
    column: str = "text",
) -> List[str]:
    """Export HuggingFace Dataset to JSONL files in chunks.

    Args:
        dataset: HuggingFace Dataset object.
        output_dir: Output directory.
        output_prefix: Output file name prefix, e.g., "chinese-c4-pretrain".
        chunk_size: Maximum number of samples per file.
        max_chunks: Maximum number of chunks to process (for debugging).
        process_func: Single sample transformation function (dict) -> dict | list[dict].
        column: Default text column name (only used when process_func is None).

    Returns:
        List of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(dataset)
    num_chunks = (total + chunk_size - 1) // chunk_size
    lim = min(max_chunks, num_chunks) if max_chunks else num_chunks

    output_files: List[str] = []
    for i in range(lim):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = dataset.select(range(start, end))

        path = os.path.join(output_dir, f"{output_prefix}_chunk_{i}.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for example in chunk:
                    processed = (
                        process_func(example)
                        if process_func
                        else {column: example[column]}
                    )
                    items = processed if isinstance(processed, list) else [processed]
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_files.append(path)
            logger.info(f"[{i + 1}/{lim}] Saved {path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to write chunk {i} to {path}: {e}")

    return output_files


@error_handler()
def cache_jsonl(
    files: List[str],
    output_dir: str,
    processor: BaseProcessor,
    *,
    pack_size: int = -1,
    pad_value: int = 0,
    batch_size: int = 256,
) -> List[str]:
    """Tokenize JSONL files and pack them into HDF5 storage.

    Args:
        files: List of JSONL file paths.
        output_dir: H5 output directory.
        processor: Initialized Processor instance.
        pack_size: Packing length, <=0 means no packing.
        pad_value: Padding value.
        batch_size: Number of records passed to the processor at once.

    Returns:
        List of generated H5 file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files: List[str] = []
    output_keys = processor.output_keys

    for file_path in files:
        file_name = Path(file_path).stem

        arrows: Dict[str, List] = {key: [] for key in output_keys}

        def append_batch(batch):
            items = [item for _, item in batch]
            try:
                results = processor.process_batch(items)
                if len(results) != len(items):
                    raise RuntimeError(
                        "Batch processor returned a different number of results"
                    )
            except Exception:
                results = []
                for line_num, item in batch:
                    try:
                        results.append(processor.process(item))
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error processing line {line_num} "
                            f"in {file_path}: {e}. Skipping line."
                        )
                        results.append(None)

            for result in results:
                if result is not None:
                    for key in output_keys:
                        arrows[key].append(result[key])

        batch = []
        batch_size = max(1, batch_size)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(
                tqdm(f, desc=f"Processing {file_name}", leave=False), start=1
            ):
                try:
                    batch.append((line_num, json.loads(line)))
                    if len(batch) >= batch_size:
                        append_batch(batch)
                        batch = []
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON decode error in {file_path} line {line_num}: {e}. Skipping line."
                    )
            if batch:
                append_batch(batch)

        if pack_size > 0:
            dtypes = (
                dict(processor.schema.output_fields)
                if processor.schema is not None
                else None
            )
            output = pack_tensors(arrows, pack_size, pad_value, dtypes)
        else:
            output = arrows

        h5_path = HDF5Handler.save(output_dir, file_name, output)
        output_files.append(h5_path)
        logger.info(f"Saved {h5_path}")

    return output_files

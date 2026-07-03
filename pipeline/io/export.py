"""Dataset export and caching utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from torch import Tensor
from tqdm import tqdm

from pipeline.io.file_scanner import FileScanner
from pipeline.io.hdf5_handler import HDF5Handler
from pipeline.processors import BaseProcessor
from pipeline.packing import pack_tensors, BasePacker
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


def merge_tensors(
    tensors: List[Tensor],
    group_size: int,
) -> List[Tensor]:
    """Merge a list of tensors into fewer larger tensors.

    Concatenates every group_size consecutive tensors into one merged
    tensor. This reduces the number of shm blocks when loading.

    Args:
        tensors: List of 1D tensors.
        group_size: Number of tensors to merge into each group.

    Returns:
        List of merged tensors.
    """
    if not tensors:
        return []

    merged: List[Tensor] = []
    for i in range(0, len(tensors), group_size):
        merged.append(torch.cat(tensors[i : i + group_size]))

    return merged


@error_handler()
def cache_jsonl(
    files: List[str],
    output_dir: str,
    processor: BaseProcessor,
    *,
    pack_size: int = -1,
    pad_value: int = 0,
    group_size: int = 1_000,
    pack_algo: Optional[str] = None,
) -> List[str]:
    """Tokenize JSONL files and pack them into HDF5 storage.

    BFD packs in group_size-bounded batches to avoid O(N²), then all
    packed chunks are merged and saved as one HDF5 file per input file.

    Args:
        files: List of JSONL file paths.
        output_dir: H5 output directory.
        processor: Initialized Processor instance.
        pack_size: Packing length, <=0 means no packing.
        pad_value: Padding value.
        group_size: BFD batch granularity (token count threshold for each
            packing batch) and merge granularity, <=0 means no merging.
        pack_algo: Packing algorithm: 'bfd' (default), 'ffd',
            'greedy'. Only used when pack_size > 0.

    Returns:
        List of generated H5 file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files: List[str] = []
    output_keys = processor.output_keys

    dtypes = (
        dict(processor.schema.output_fields)
        if processor.schema is not None
        else None
    )
    pad_values = {k: (0 if k == "position_ids" else (False if k.endswith("_mask") else pad_value)) for k in output_keys}

    target_tokens = group_size * pack_size if group_size > 0 and pack_size > 0 else 0

    for file_path in files:
        file_name = Path(file_path).stem

        all_packed: Dict[str, List[Tensor]] = {key: [] for key in output_keys}
        arrows_batch: Dict[str, List] = {key: [] for key in output_keys}
        batch_tokens: int = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(
                tqdm(f, desc=f"Processing {file_name}", leave=False), start=1
            ):
                try:
                    result = processor.process(json.loads(line))
                    if result is not None:
                        for key in output_keys:
                            arrows_batch[key].append(result[key])
                        if target_tokens > 0:
                            batch_tokens += int(result[output_keys[0]].shape[0])
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON decode error in {file_path} line {line_num}: {e}. Skipping line."
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"Unexpected error processing line {line_num} in {file_path}: {e}. Skipping line."
                    )
                    continue

                if target_tokens > 0 and batch_tokens >= target_tokens:
                    packed = pack_tensors(arrows_batch, pack_size, pad_value, dtypes, pad_values=pad_values, algo=pack_algo)
                    for key in output_keys:
                        all_packed[key].extend(packed[key])
                        arrows_batch[key] = []
                    batch_tokens = 0

        if arrows_batch[output_keys[0]]:
            if pack_size > 0:
                packed = pack_tensors(arrows_batch, pack_size, pad_value, dtypes, pad_values=pad_values, algo=pack_algo)
                for key in output_keys:
                    all_packed[key].extend(packed[key])
            else:
                for key in output_keys:
                    all_packed[key].extend(arrows_batch[key])

        if not all_packed[output_keys[0]]:
            logger.warning(f"No valid samples in {file_path}, skipping")
            continue

        if pack_size <= 0:
            output = all_packed
        elif group_size > 0 and all_packed[output_keys[0]]:
            output = {
                key: merge_tensors(tensors, group_size)
                for key, tensors in all_packed.items()
            }
        else:
            output = all_packed

        h5_path = HDF5Handler.save(output_dir, file_name, output)
        output_files.append(h5_path)
        logger.info(f"Saved {h5_path}")

    return output_files

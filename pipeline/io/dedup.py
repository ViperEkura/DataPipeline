"""MinHash + LSH deduplication for pretraining text data."""

import json
import logging
import os
from pathlib import Path
from typing import Iterator, List, Set, Tuple

from tqdm import tqdm

from pipeline.io.writers import TextWriter
from pipeline.utils import error_handler

logger = logging.getLogger(__name__)


def _tokenize(text: str, ngram: int = 3) -> Set[str]:
    return {text[i : i + ngram] for i in range(len(text) - ngram + 1)}


def _iter_docs(input_dir: Path) -> Iterator[Tuple[str, dict]]:
    for fpath in sorted(input_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    yield text, record


def _write_h5(records: List[dict], output_dir: str, chunk_idx: int):
    import h5py

    fname = os.path.join(output_dir, f"chunk_{chunk_idx}.h5")
    texts = [rec.get("text", "") for rec in records]
    with h5py.File(fname, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset("text", (len(texts),), dtype=dt)
        for i, t in enumerate(texts):
            ds[i] = t


def _write_bin(records: List[dict], output_dir: Path, chunk_idx: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [rec.get("text", "") + "\n" for rec in records]

    meta = {"chunk": chunk_idx, "count": len(texts), "format": "text", "encoding": "utf-8"}
    meta_path = output_dir / "meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing[str(chunk_idx)] = meta
    meta_path.write_text(json.dumps(existing, indent=2))

    (output_dir / f"text_{chunk_idx}.bin").write_bytes("".join(texts).encode("utf-8"))


_WRITERS = {
    "jsonl": TextWriter,
    "h5": lambda: None,  # handled inline below
    "bin": lambda: None,
}


@error_handler()
def dedup_jsonl(
    input_dir: str,
    output_dir: str,
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram: int = 3,
    output_format: str = "jsonl",
    chunk_size: int = 1_000_000,
) -> Tuple[int, int]:
    """Deduplicate JSONL text files using MinHash + LSH.

    Args:
        input_dir: Directory with source ``*.jsonl`` files.
        output_dir: Directory for deduplicated output.
        threshold: Jaccard similarity threshold (0–1).
        num_perm: Number of MinHash permutations.
        ngram: Character n-gram size.
        output_format: ``"jsonl"``, ``"h5"``, or ``"bin"``.
        chunk_size: Records per output chunk file.

    Returns:
        ``(kept, removed)`` counts.
    """
    from datasketch import MinHash, MinHashLSH

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Deduplicating {input_dir} -> {output_dir} "
        f"(threshold={threshold}, perm={num_perm}, fmt={output_format})"
    )

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    kept = 0
    removed = 0

    dup_doc_ids: Set[int] = set()
    for doc_id, (text, _record) in enumerate(tqdm(_iter_docs(input_path), desc="indexing", unit="docs")):
        shingles = _tokenize(text, ngram=ngram)
        if len(shingles) < ngram * 2:
            dup_doc_ids.add(doc_id)
            continue

        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))

        if lsh.query(m):
            dup_doc_ids.add(doc_id)
        else:
            lsh.insert(doc_id, m)

    logger.info(f"Found {len(dup_doc_ids)} duplicates, writing deduplicated data")

    buffer: List[dict] = []
    chunk_idx = 0
    writer = TextWriter(chunk_size) if output_format == "jsonl" else None

    for doc_id, (_text, record) in enumerate(tqdm(_iter_docs(input_path), desc="writing", unit="docs")):
        if doc_id in dup_doc_ids:
            removed += 1
            continue

        kept += 1
        buffer.append(record)

        if len(buffer) >= chunk_size:
            _flush_chunk(buffer, output_path, chunk_idx, output_format, writer)
            chunk_idx += 1
            buffer = []

    if buffer:
        _flush_chunk(buffer, output_path, chunk_idx, output_format, writer)

    if writer:
        writer.flush(output_path)

    logger.info(f"Done. kept={kept}, removed={removed}")
    return kept, removed


def _flush_chunk(
    records: List[dict],
    output_dir: Path,
    chunk_idx: int,
    output_format: str,
    writer=None,
):
    if output_format == "jsonl":
        for rec in records:
            writer.write_record(rec, output_dir)
    elif output_format == "h5":
        _write_h5(records, str(output_dir), chunk_idx)
    elif output_format == "bin":
        _write_bin(records, output_dir, chunk_idx)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

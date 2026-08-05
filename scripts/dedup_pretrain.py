"""MinHash + LSH deduplication CLI.

Usage:
    python scripts/dedup_pretrain.py --input-dir <data_dir> --output-dir <out_dir> --threshold 0.8 --num-perm 128 --output-format jsonl
"""

import argparse

from pipeline.io import dedup_jsonl


def main():
    parser = argparse.ArgumentParser(description="MinHash + LSH deduplication")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--output-format", default="jsonl", choices=["jsonl", "h5", "bin"])
    args = parser.parse_args()

    kept, removed = dedup_jsonl(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram=args.ngram,
        output_format=args.output_format,
    )

    total = kept + removed
    print(f"kept={kept}, removed={removed} ({removed/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()

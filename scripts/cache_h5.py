"""JSONL to H5 caching script.

Tokenize JSONL files and pack them into HDF5 format.

Usage:
    python scripts/cache_h5.py pt ./dataset/chinese-c4-pretrain
    python scripts/cache_h5.py sft ./dataset/belle-sft --pack-size 4096 --strategy alpaca
    python scripts/cache_h5.py sft ./dataset/Ling-Coder-sft --tokenizer ./my_tokenizer.json
"""

import argparse
import os

from pipeline import (
    AutoTokenizer,
    ProcessorFactory,
    ProcessorConfig,
    FileScanner,
    cache_jsonl,
    setup_logging,
)


def main():
    parser = argparse.ArgumentParser(description="JSONL -> H5 cache")
    parser.add_argument("type", choices=["pt", "sft", "dpo"], help="Processor type")
    parser.add_argument("input_dir", help="Directory containing JSONL files")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="H5 output dir (default: <input_dir>/cached)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="./tokenizer.json",
        help="Tokenizer path (default: ./tokenizer.json)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        default=None,
        help="Prompt strategy: chatml, alpaca (default: chatml)",
    )
    parser.add_argument(
        "-a",
        "--pack-algo",
        default=None,
        choices=[None, "bfd", "ffd", "greedy"],
        help="Packing algorithm: bfd (default), ffd, greedy",
    )
    parser.add_argument(
        "-p",
        "--pack-size",
        type=int,
        default=-1,
        help="Pack size, <=0 to disable (default: -1)",
    )
    parser.add_argument(
        "--pad-value", type=int, default=2, help="Padding token ID (default: 2 = <|pad|>)"
    )
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=1_000,
        help="Merge every N packed chunks into one tensor, <=0 to disable (default: 1000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Initialize logging explicitly (not automatic anymore)
    import logging
    setup_logging(getattr(logging, args.log_level))

    jsonl_files = FileScanner.scan(args.input_dir, suffix=".jsonl")
    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f}")

    if not os.path.exists(args.tokenizer):
        print(f"[ERROR] Tokenizer not found: {args.tokenizer}")
        return
    tokenizer = AutoTokenizer(args.tokenizer)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    # Use unified config interface
    config = ProcessorConfig(
        processor_type=args.type,
        tokenizer=tokenizer,
        strategy_name=args.strategy,
    )
    processor = ProcessorFactory.create_from_config(config)

    print(f"Processor: {args.type} ({processor.__class__.__name__})")
    print(f"Output keys: {processor.output_keys}")

    output_dir = args.output_dir or os.path.join(args.input_dir, "cached")

    print(f"\nStart caching...")
    if args.pack_size > 0:
        algo = args.pack_algo or "bfd"
        print(f"  pack_size={args.pack_size}, pad_value={args.pad_value}, algo={algo}")
    else:
        print(f"  no packing")
    if args.group_size > 0:
        print(f"  group_size={args.group_size} chunks per tensor")
    else:
        print(f"  no grouping")

    cache_jsonl(
        files=jsonl_files,
        output_dir=output_dir,
        processor=processor,
        pack_size=args.pack_size,
        pad_value=args.pad_value,
        group_size=args.group_size,
        pack_algo=args.pack_algo,
    )
    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import tempfile
import shutil

MIN_LEN = 15


def filter_sft(input_path: str) -> tuple[int, int]:
    """Filter SFT JSONL (messages format), remove if any msg content < MIN_LEN chars."""
    kept, total = 0, 0
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(input_path))
    try:
        with open(input_path, encoding="utf-8") as fin, open(tmp_fd, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = obj.get("messages", [])
                short = any(len(m.get("content", "")) < MIN_LEN for m in messages)
                if not short:
                    fout.write(line + "\n")
                    kept += 1
        shutil.move(tmp_path, input_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return kept, total


def filter_pretrain(input_path: str) -> tuple[int, int]:
    """Filter pretrain JSONL (text format), remove if text < MIN_LEN chars."""
    kept, total = 0, 0
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(input_path))
    try:
        with open(input_path, encoding="utf-8") as fin, open(tmp_fd, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if len(text) >= MIN_LEN:
                    fout.write(line + "\n")
                    kept += 1
        shutil.move(tmp_path, input_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return kept, total


def main():
    parser = argparse.ArgumentParser(description="Filter short samples from JSONL datasets")
    parser.add_argument("input_dir", help="Directory containing JSONL files")
    parser.add_argument("--type", choices=["sft", "pt"], required=True, help="Dataset type")
    args = parser.parse_args()

    from pipeline import FileScanner

    jsonl_files = FileScanner.scan(args.input_dir, suffix=".jsonl")
    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    filter_fn = filter_sft if args.type == "sft" else filter_pretrain

    total_kept, total_lines = 0, 0
    for fpath in jsonl_files:
        kept, lines = filter_fn(fpath)
        total_kept += kept
        total_lines += lines
        removed = lines - kept
        print(f"  {os.path.basename(fpath)}: {lines} -> {kept} (removed {removed})")

    print(f"\nTotal: {total_lines} -> {total_kept} (removed {total_lines - total_kept})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from punctuation import LABELS, split_corpus_line, strip_target_punctuation


def split_for_article(article_id: str) -> str:
    bucket = int(hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare People's Daily punctuation restoration data.")
    parser.add_argument("--input-dir", type=Path, default=Path("1998-上半年-带音"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-chars", type=int, default=254,
                        help="Content characters per training example; leave room for [CLS]/[SEP].")
    args = parser.parse_args()
    if args.max_chars < 16:
        parser.error("--max-chars must be at least 16")

    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt corpus files found under {args.input_dir}")

    records: dict[str, list[dict]] = {name: [] for name in ("train", "validation", "test")}
    article_counts = {name: set() for name in records}
    label_counts = {name: Counter() for name in records}
    chars_count = Counter()
    source_line_count = 0
    skipped_lines = 0
    unaligned_punctuation = 0
    replacement_characters_removed = 0

    for file_path in files:
        with file_path.open("r", encoding="gb18030", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                parsed = split_corpus_line(line)
                if parsed is None:
                    skipped_lines += 1
                    continue
                article_id, punctuated = parsed
                replacement_characters_removed += punctuated.count("\ufffd")
                punctuated = punctuated.replace("\ufffd", "")
                text, labels, unaligned = strip_target_punctuation(punctuated)
                unaligned_punctuation += unaligned
                if not text:
                    skipped_lines += 1
                    continue

                split = split_for_article(article_id)
                article_counts[split].add(article_id)
                source_line_count += 1
                chars_count[split] += len(text)
                for label_id in labels:
                    label_counts[split][LABELS[label_id]] += 1

                for start in range(0, len(text), args.max_chars):
                    end = min(start + args.max_chars, len(text))
                    chunk = text[start:end]
                    chunk_labels = labels[start:end]
                    if chunk:
                        records[split].append({
                            "text": chunk,
                            "labels": chunk_labels,
                            "article_id": article_id,
                            "source": file_path.name,
                            "line": line_number,
                        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_records in records.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_records)

    summary = {
        "encoding": "GB18030",
        "input_files": [file.name for file in files],
        "split_policy": "deterministic SHA-256 split by article id: 80/10/10",
        "target_labels": LABELS,
        "max_chars_per_training_example": args.max_chars,
        "source_lines_parsed": source_line_count,
        "skipped_lines": skipped_lines,
        "replacement_characters_removed": replacement_characters_removed,
        "unlabelled_or_collapsed_punctuation_marks": unaligned_punctuation,
        "splits": {
            split: {
                "examples": len(records[split]),
                "articles": len(article_counts[split]),
                "characters": chars_count[split],
                "label_counts": {label: label_counts[split][label] for label in LABELS},
            }
            for split in records
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

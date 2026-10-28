from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

os.environ.setdefault("USE_TF", "0")

from inference import PunctuationPredictor
from punctuation import LABELS, restore_punctuation


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group_articles(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for index, row in enumerate(records):
        grouped[row["article_id"]].append((index, row))
    articles = []
    for article_id, rows in grouped.items():
        rows.sort(key=lambda item: (item[1]["source"], item[1]["line"], item[0]))
        articles.append({
            "article_id": article_id,
            "text": "".join(row["text"] for _, row in rows),
            "labels": [label for _, row in rows for label in row["labels"]],
        })
    return articles


def score(targets: list[int], predictions: list[int]) -> dict:
    class_report = classification_report(
        targets, predictions, labels=list(range(len(LABELS))), target_names=LABELS,
        output_dict=True, zero_division=0)
    punctuation_ids = list(range(1, len(LABELS)))
    boundary_ids = [2, 3, 4]
    boundary_targets = [int(label in boundary_ids) for label in targets]
    boundary_predictions = [int(label in boundary_ids) for label in predictions]
    return {
        "characters": len(targets),
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))) if targets else 0.0,
        "punctuation_macro_f1_excluding_O": float(f1_score(
            targets, predictions, labels=punctuation_ids, average="macro", zero_division=0)),
        "sentence_terminator_f1": float(f1_score(
            boundary_targets, boundary_predictions, zero_division=0)),
        "per_class": {label: class_report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(LABELS)))).tolist(),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(predictor: PunctuationPredictor, articles: list[dict], batch_sizes: list[int],
              repeats: int) -> list[dict]:
    if not articles:
        return []
    predictor.predict_logits(articles[0]["text"][:predictor.window_chars + 1], batch_size=max(batch_sizes))
    results = []
    for batch_size in batch_sizes:
        durations = []
        final_predictions = {}
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(predictor.device)
        for _ in range(repeats):
            synchronize(predictor.device)
            started = time.perf_counter()
            for article in articles:
                final_predictions[article["article_id"]] = predictor.predict(
                    article["text"], batch_size=batch_size)
            synchronize(predictor.device)
            durations.append(time.perf_counter() - started)
        targets = [label for article in articles for label in article["labels"]]
        predictions = [label for article in articles
                       for label in final_predictions[article["article_id"]]]
        elapsed = statistics.median(durations)
        chars = sum(len(article["text"]) for article in articles)
        metrics = score(targets, predictions)
        results.append({
            "batch_size": batch_size,
            "articles": len(articles),
            "characters": chars,
            "median_seconds": elapsed,
            "characters_per_second": chars / elapsed if elapsed else 0.0,
            "punctuation_macro_f1_excluding_O": metrics["punctuation_macro_f1_excluding_O"],
            "sentence_terminator_f1": metrics["sentence_terminator_f1"],
            "peak_allocated_gib": (torch.cuda.max_memory_allocated(predictor.device) / 1024**3
                                    if predictor.device.type == "cuda" else None),
            "peak_reserved_gib": (torch.cuda.max_memory_reserved(predictor.device) / 1024**3
                                  if predictor.device.type == "cuda" else None),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate punctuation quality and batched window throughput.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/bert-punctuation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--window-chars", type=int, default=448)
    parser.add_argument("--stride", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benchmark-articles", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.window_chars < 1:
        parser.error("--window-chars must be at least 1")
    if args.stride < 1 or args.stride > args.window_chars:
        parser.error("--stride must satisfy 1 <= stride <= window-chars")
    if args.benchmark_articles < 1:
        parser.error("--benchmark-articles must be at least 1")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("--batch-sizes values must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    test_records = read_jsonl(args.data_dir / "test.jsonl")
    articles = group_articles(test_records)
    predictor = PunctuationPredictor(args.model_dir, window_chars=args.window_chars,
                                     stride=args.stride, batch_size=args.batch_size)
    all_targets = []
    all_predictions = []
    article_predictions = {}
    started = time.perf_counter()
    for index, article in enumerate(articles, start=1):
        predicted = predictor.predict(article["text"], batch_size=args.batch_size)
        all_targets.extend(article["labels"])
        all_predictions.extend(predicted)
        article_predictions[article["article_id"]] = predicted
        if index % 100 == 0 or index == len(articles):
            print(f"test_articles={index}/{len(articles)}", flush=True)
    evaluation_seconds = time.perf_counter() - started
    test_metrics = score(all_targets, all_predictions)

    benchmark_articles = sorted(articles, key=lambda item: len(item["text"]), reverse=True)
    benchmark_articles = benchmark_articles[:args.benchmark_articles]
    speed_results = benchmark(predictor, benchmark_articles, args.batch_sizes, args.repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for article in articles:
            predicted = article_predictions[article["article_id"]]
            stream.write(json.dumps({
                "article_id": article["article_id"],
                "text": article["text"],
                "reference": restore_punctuation(article["text"], article["labels"]),
                "prediction": restore_punctuation(article["text"], predicted),
            }, ensure_ascii=False) + "\n")

    examples = []
    example_records = [row for row in test_records if any(label != 0 for label in row["labels"])][:8]
    for row in example_records:
        predicted = predictor.predict(row["text"], batch_size=1)
        examples.append({
            "text": row["text"],
            "reference": restore_punctuation(row["text"], row["labels"]),
            "prediction": restore_punctuation(row["text"], predicted),
        })
    results = {
        "model_dir": str(args.model_dir),
        "device": str(predictor.device),
        "gpu_name": torch.cuda.get_device_name(0) if predictor.device.type == "cuda" else None,
        "gpu_memory_gib": (torch.cuda.get_device_properties(0).total_memory / 1024**3
                           if predictor.device.type == "cuda" else None),
        "precision": "FP16 autocast" if predictor.amp else "FP32",
        "window_chars": args.window_chars,
        "stride_chars": args.stride,
        "overlap_chars": args.window_chars - args.stride,
        "merge": "center-weighted mean of overlapping-window logits; edge weight capped at 64",
        "test_articles": len(articles),
        "test_source_examples": len(test_records),
        "test_evaluation_seconds": evaluation_seconds,
        "test_metrics": test_metrics,
        "throughput_method": "same fixed window spans; only inference batch size changes; median over repeats",
        "throughput_repeats": args.repeats,
        "throughput_articles": len(benchmark_articles),
        "throughput_results": speed_results,
        "examples": examples,
    }
    with (args.output_dir / "evaluation_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "test_metrics": {key: value for key, value in test_metrics.items()
                         if key not in ("per_class", "confusion_matrix")},
        "throughput_results": speed_results,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

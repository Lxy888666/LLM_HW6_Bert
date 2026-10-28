from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("USE_TF", "0")
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from punctuation import LABELS


class PunctuationDataset(Dataset):
    def __init__(self, path: Path):
        self.records = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class TokenCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        chars = [list(row["text"]) for row in rows]
        encoded = self.tokenizer(
            chars,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_labels = []
        for batch_index, row in enumerate(rows):
            labels = [-100] * encoded.input_ids.shape[1]
            last_positions = {}
            for token_index, word_id in enumerate(encoded.word_ids(batch_index=batch_index)):
                if word_id is not None:
                    last_positions[word_id] = token_index
            if len(last_positions) != len(row["labels"]):
                raise ValueError(
                    "Tokenizer truncated a training example before all characters received "
                    "token positions; increase --max-length or reduce the prepared chunk size"
                )
            for word_id, token_index in last_positions.items():
                if word_id < len(row["labels"]):
                    labels[token_index] = row["labels"][word_id]
            batch_labels.append(labels)
        encoded["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return encoded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, amp: bool, class_weights: torch.Tensor) -> tuple[float, dict, float]:
    model.eval()
    losses = []
    targets = []
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, len(LABELS)), labels.reshape(-1),
                    weight=class_weights, ignore_index=-100)
            losses.append(float(loss.item()))
            valid = labels != -100
            predictions.extend(output.logits.argmax(dim=-1)[valid].cpu().tolist())
            targets.extend(labels[valid].cpu().tolist())
    report = classification_report(targets, predictions, labels=list(range(len(LABELS))),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    return float(np.mean(losses)), report, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Chinese BERT for punctuation restoration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-name", default="google-bert/bert-base-chinese")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert-punctuation"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.grad_accumulation < 1:
        parser.error("--grad-accumulation must be at least 1")
    if args.max_length < 3:
        parser.error("--max-length must be at least 3")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
    ).to(device)

    train_data = PunctuationDataset(args.data_dir / "train.jsonl")
    validation_data = PunctuationDataset(args.data_dir / "validation.jsonl")
    collator = TokenCollator(tokenizer, args.max_length)
    loader_options = {"num_workers": args.num_workers, "pin_memory": amp,
                      "collate_fn": collator}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              **loader_options)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   **loader_options)

    counts = Counter(label for row in train_data.records for label in row["labels"])
    total = sum(counts.values())
    class_weights = torch.tensor([
        math.sqrt(total / max(len(LABELS) * counts[index], 1)) for index in range(len(LABELS))
    ], dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(x in n for x in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            group_start = ((step - 1) // args.grad_accumulation) * args.grad_accumulation + 1
            accumulation_count = min(args.grad_accumulation, len(train_loader) - group_start + 1)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(**batch)
                raw_loss = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, len(LABELS)), labels.reshape(-1),
                    weight=class_weights, ignore_index=-100)
                loss = raw_loss / accumulation_count
            scaler.scale(loss).backward()
            running_loss += float(raw_loss.item())
            if step % args.grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = not amp or scaler.get_scale() >= previous_scale
                optimizer.zero_grad(set_to_none=True)
                if updated:
                    scheduler.step()
            if step % 100 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} "
                      f"train_loss={running_loss / step:.5f}", flush=True)

        val_loss, val_report, val_seconds = evaluate(model, validation_loader, device, amp, class_weights)
        val_macro_f1 = float(np.mean([val_report[label]["f1-score"] for label in LABELS[1:]]))
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": val_loss,
            "validation_macro_f1_excluding_O": val_macro_f1,
            "validation_seconds": val_seconds,
            "validation_classification_report": val_report,
        }
        history.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "validation_classification_report"},
                         ensure_ascii=False), flush=True)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    results = {
        "model_name": args.model_name,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if amp else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3) if amp else None,
        "precision": "FP16 autocast" if amp else "FP32",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_excluding_O": best_macro_f1,
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    with (args.output_dir / "training_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in results.items() if key != "history"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
from transformers import AutoModelForTokenClassification, AutoTokenizer

from punctuation import LABELS, restore_punctuation, strip_target_punctuation


class PunctuationPredictor:
    """Batched overlapping-window inference with center-weighted logit merging."""

    def __init__(self, model_dir: Path, device: str | None = None,
                 window_chars: int = 448, stride: int = 320, batch_size: int = 8):
        if window_chars <= 0 or stride <= 0 or stride > window_chars:
            raise ValueError("Require 0 < stride <= window_chars")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = self.device.type == "cuda"
        self.window_chars = window_chars
        self.stride = stride
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        if window_chars > model_limit - 2:
            raise ValueError(
                f"window_chars={window_chars} exceeds the safe character limit "
                f"{model_limit - 2} for this model"
            )

    def window_spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans = []
        start = 0
        while start < len(text):
            end = min(start + self.window_chars, len(text))
            spans.append((start, end))
            if end == len(text):
                break
            start += self.stride
        return spans

    def _forward_windows(self, text: str, spans: list[tuple[int, int]]) -> list[np.ndarray]:
        if not spans:
            return []
        char_batches = [list(text[start:end]) for start, end in spans]
        encoded = self.tokenizer(
            char_batches,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        word_positions = []
        for batch_index, chars in enumerate(char_batches):
            last_positions = {}
            for token_index, word_id in enumerate(encoded.word_ids(batch_index=batch_index)):
                if word_id is not None:
                    last_positions[word_id] = token_index
            positions = [last_positions.get(i) for i in range(len(chars))]
            if any(position is None for position in positions):
                raise ValueError(
                    "Tokenizer truncated a window before all characters received token positions; "
                    "reduce window_chars or use a tokenizer-aware window builder"
                )
            word_positions.append(positions)

        model_inputs = {key: value.to(self.device, non_blocking=True)
                        for key, value in encoded.items() if key in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(**model_inputs).logits.float().cpu().numpy()

        outputs = []
        for batch_index, positions in enumerate(word_positions):
            row = np.zeros((len(positions), len(LABELS)), dtype=np.float32)
            for char_index, token_index in enumerate(positions):
                if token_index is not None:
                    row[char_index] = logits[batch_index, token_index]
            outputs.append(row)
        return outputs

    def predict_logits(self, text: str, batch_size: int | None = None) -> np.ndarray:
        spans = self.window_spans(text)
        if not spans:
            return np.zeros((0, len(LABELS)), dtype=np.float32)
        size = self.batch_size if batch_size is None else batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive")
        aggregate = np.zeros((len(text), len(LABELS)), dtype=np.float32)
        weights = np.zeros(len(text), dtype=np.float32)
        for offset in range(0, len(spans), size):
            span_batch = spans[offset:offset + size]
            for (start, end), window_logits in zip(span_batch, self._forward_windows(text, span_batch)):
                local = np.arange(end - start)
                edge_distance = np.minimum(local + 1, end - start - local).astype(np.float32)
                # Cap the ramp so that a very long overlap does not dominate the merge.
                window_weights = np.minimum(edge_distance, 64.0)
                aggregate[start:end] += window_logits * window_weights[:, None]
                weights[start:end] += window_weights
        return aggregate / np.maximum(weights[:, None], 1e-6)

    def predict(self, text: str, batch_size: int | None = None) -> list[int]:
        logits = self.predict_logits(text, batch_size=batch_size)
        return logits.argmax(axis=-1).tolist()

    def restore(self, text: str, batch_size: int | None = None) -> str:
        clean_text, _, _ = strip_target_punctuation(text)
        return restore_punctuation(clean_text, self.predict(clean_text, batch_size=batch_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore Chinese punctuation with a fine-tuned BERT model.")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/bert-punctuation"))
    parser.add_argument("--input", type=Path, help="UTF-8 text file; one input passage per line.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.txt"))
    parser.add_argument("--window-chars", type=int, default=448)
    parser.add_argument("--stride", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required")

    predictor = PunctuationPredictor(args.model_dir, window_chars=args.window_chars,
                                     stride=args.stride, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            target.write(predictor.restore(line.rstrip("\r\n")) + "\n")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

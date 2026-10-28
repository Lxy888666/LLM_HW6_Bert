"""Utilities for converting the People's Daily corpus to punctuation labels."""

from __future__ import annotations

import re
from collections.abc import Sequence


LABELS = ["O", "，", "。", "？", "！", "；", "：", "、"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

PUNCTUATION_MAP = {
    ",": "，",
    "，": "，",
    ".": "。",
    "。": "。",
    "．": "。",
    "?": "？",
    "？": "？",
    "!": "！",
    "！": "！",
    ";": "；",
    "；": "；",
    ":": "：",
    "：": "：",
    "、": "、",
}

PINYIN_RE = re.compile(r"\{[^{}\s]*\}")
LINE_RE = re.compile(r"^(?P<article>\d{8}-\d{2}-\d{3})-\d{3}\s+(?P<text>.*)$")


def split_corpus_line(line: str) -> tuple[str, str] | None:
    """Return the article identifier and punctuation-bearing text for a corpus line."""
    match = LINE_RE.match(line.rstrip("\r\n"))
    if not match:
        return None
    text = PINYIN_RE.sub("", match.group("text"))
    text = "".join(text.split())
    return match.group("article"), text


def strip_target_punctuation(text: str) -> tuple[str, list[int], int]:
    """Remove target punctuation and label the preceding visible character.

    Multiple target marks at one character boundary are reduced to one label.
    The first mark is retained, matching the left-to-right surface convention.
    """
    chars: list[str] = []
    labels: list[int] = []
    unaligned = 0
    for char in text:
        label = PUNCTUATION_MAP.get(char)
        if label is None:
            chars.append(char)
            labels.append(0)
            continue
        if not labels:
            unaligned += 1
        elif labels[-1] == 0:
            labels[-1] = LABEL_TO_ID[label]
        else:
            unaligned += 1
    return "".join(chars), labels, unaligned


def restore_punctuation(text: str, predictions: Sequence[int]) -> str:
    """Insert one predicted punctuation mark after each corresponding character."""
    if len(text) != len(predictions):
        raise ValueError(
            f"Prediction length {len(predictions)} does not match text length {len(text)}"
        )
    output: list[str] = []
    for char, label_id in zip(text, predictions):
        output.append(char)
        if int(label_id) != 0:
            output.append(ID_TO_LABEL[int(label_id)])
    return "".join(output)


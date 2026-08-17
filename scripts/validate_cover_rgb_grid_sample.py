"""Validate a tiny handcrafted cover-vector smoke test with no third-party packages."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VECTOR_PATH = ROOT / "outputs" / "visual_smoke" / "rgb_grid_8x8_sample.csv"
OUTPUT_PATH = ROOT / "outputs" / "visual_smoke" / "rgb_grid_8x8_smoke_report.json"


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def normalized(values: list[float]) -> list[float]:
    # Center each image so similarity is less dominated by overall brightness.
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    norm = math.sqrt(sum(value * value for value in centered))
    return [value / norm for value in centered] if norm else centered


def main() -> None:
    with VECTOR_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if len(reader.fieldnames or []) != 193:
            raise ValueError("Expected item plus 192 RGB features")
        vectors = {
            int(row["item"]): normalized([float(row[name]) for name in reader.fieldnames[1:]])
            for row in reader
        }
    if len(vectors) != 32:
        raise ValueError(f"Expected 32 sample vectors, found {len(vectors)}")

    known_pairs = [
        {"label": "same Naruto frame", "items": [1186, 16683]},
        {"label": "same comic template / different episode number", "items": [313, 8425]},
        {"label": "dHash false positive: dog vs football", "items": [1852, 13443]},
        {"label": "dHash false positive: game vs film", "items": [1864, 9319]},
    ]
    for pair in known_pairs:
        left, right = pair["items"]
        pair["cosine_similarity"] = cosine(vectors[left], vectors[right])

    all_pairs = []
    items = sorted(vectors)
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            all_pairs.append((cosine(vectors[left], vectors[right]), left, right))
    all_pairs.sort(reverse=True)
    report = {
        "feature": "mean-centered 8x8 RGB grid",
        "purpose": "local pipeline smoke test, not the final Visual-only model",
        "rows": len(vectors),
        "dimensions": 192,
        "vector_file_sha256": hashlib.sha256(VECTOR_PATH.read_bytes()).hexdigest(),
        "known_pairs": known_pairs,
        "top_sample_pairs": [
            {"items": [left, right], "cosine_similarity": score}
            for score, left, right in all_pairs[:10]
        ],
        "checks": {
            "all_vectors_have_unit_norm": all(
                abs(math.sqrt(sum(value * value for value in vector)) - 1) < 1e-9
                for vector in vectors.values()
            ),
            "same_frame_more_similar_than_false_positive_pairs": (
                known_pairs[0]["cosine_similarity"]
                > max(known_pairs[2]["cosine_similarity"], known_pairs[3]["cosine_similarity"])
            ),
        },
        "limitations": [
            "The 8x8 grid captures coarse color and layout only.",
            "It is not invariant to crop, text overlays, or large composition changes.",
            "It is not a semantic pretrained visual representation.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

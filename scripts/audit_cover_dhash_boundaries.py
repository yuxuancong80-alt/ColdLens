"""Find perceptual-hash neighbors crossing train/cold split boundaries."""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HASH_PATH = ROOT / "data" / "processed" / "cover_dhash.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
OUTPUT_DIR = ROOT / "outputs" / "m0"


def read_hashes() -> dict[int, int]:
    with HASH_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["item"]): int(row["dhash"], 16) for row in csv.DictReader(handle)}


def split_items(filename: str) -> set[int]:
    with gzip.open(SPLIT_DIR / filename, "rt", encoding="utf-8", newline="") as handle:
        return {int(row["item"]) for row in csv.DictReader(handle)}


def boundary_audit(
    hashes: dict[int, int], train_items: set[int], cold_items: set[int]
) -> dict[str, object]:
    train = [(item, hashes[item]) for item in sorted(train_items)]
    distance_counts: Counter[int] = Counter()
    nearest_examples: list[dict[str, int]] = []
    for cold_item in sorted(cold_items):
        cold_hash = hashes[cold_item]
        nearest_item = -1
        nearest_distance = 65
        for train_item, train_hash in train:
            distance = (cold_hash ^ train_hash).bit_count()
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_item = train_item
                if distance == 0:
                    break
        distance_counts[nearest_distance] += 1
        if nearest_distance <= 5:
            nearest_examples.append(
                {
                    "cold_item": cold_item,
                    "train_item": nearest_item,
                    "hamming_distance": nearest_distance,
                }
            )
    nearest_examples.sort(key=lambda row: (row["hamming_distance"], row["cold_item"]))
    return {
        "train_items": len(train_items),
        "cold_items": len(cold_items),
        "cold_items_by_minimum_hamming_distance": {
            str(distance): count for distance, count in sorted(distance_counts.items())
        },
        "cold_items_with_min_distance_0": sum(count for distance, count in distance_counts.items() if distance == 0),
        "cold_items_with_min_distance_lte_2": sum(count for distance, count in distance_counts.items() if distance <= 2),
        "cold_items_with_min_distance_lte_5": sum(count for distance, count in distance_counts.items() if distance <= 5),
        "nearest_examples_lte_5": nearest_examples[:100],
    }


def main() -> None:
    started = time.perf_counter()
    hashes = read_hashes()
    if len(hashes) != 19_220:
        raise ValueError(f"Expected 19,220 hashes, found {len(hashes)}")
    report = {
        "method": {
            "name": "64-bit difference hash (dHash)",
            "resize": "9x8",
            "distance": "Hamming distance",
            "warning": "Low distance is a screening signal, not proof of duplicate content.",
        },
        "hash_rows": len(hashes),
        "validation_boundary": boundary_audit(
            hashes,
            split_items("dev_train.csv.gz"),
            split_items("validation_cold_candidates.csv.gz"),
        ),
        "test_boundary": boundary_audit(
            hashes,
            split_items("final_train.csv.gz"),
            split_items("test_cold_candidates.csv.gz"),
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "cover_dhash_boundary_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    concise = {
        "method": report["method"],
        "hash_rows": report["hash_rows"],
        "validation_boundary": {
            key: value
            for key, value in report["validation_boundary"].items()
            if key != "nearest_examples_lte_5"
        },
        "test_boundary": {
            key: value
            for key, value in report["test_boundary"].items()
            if key != "nearest_examples_lte_5"
        },
        "runtime_seconds": report["runtime_seconds"],
    }
    print(json.dumps(concise, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

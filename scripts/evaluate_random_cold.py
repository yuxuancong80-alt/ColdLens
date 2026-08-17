"""Validate the cold-candidate evaluator with a deterministic random baseline."""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "split_v1.json").read_text(encoding="utf-8"))
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
OUTPUT_DIR = ROOT / "outputs" / "m0"


def read_items(path: Path) -> list[int]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [int(row["item"]) for row in csv.DictReader(handle)]


def read_positives(path: Path) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            positives[int(row["user"])].add(int(row["item"]))
    return positives


def evaluate(candidates: list[int], positives: dict[int, set[int]], seed: int) -> dict[str, float | int]:
    k_values = [int(value) for value in CONFIG["evaluation"]["k_values"]]
    totals = {f"Recall@{k}": 0.0 for k in k_values}
    totals.update({f"NDCG@{k}": 0.0 for k in k_values})
    totals.update({f"HitRate@{k}": 0.0 for k in k_values})
    candidate_set = set(candidates)
    if any(not targets <= candidate_set for targets in positives.values()):
        raise ValueError("A positive item is absent from the candidate pool")

    for user, targets in sorted(positives.items()):
        ranking = candidates.copy()
        random.Random(seed + user).shuffle(ranking)
        for k in k_values:
            top_k = ranking[:k]
            hits = [1 if item in targets else 0 for item in top_k]
            hit_count = sum(hits)
            totals[f"Recall@{k}"] += hit_count / len(targets)
            totals[f"HitRate@{k}"] += float(hit_count > 0)
            dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
            ideal_hits = min(len(targets), k)
            idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
            totals[f"NDCG@{k}"] += dcg / idcg

    user_count = len(positives)
    result: dict[str, float | int] = {
        "users": user_count,
        "candidate_items": len(candidates),
        "positive_user_item_pairs": sum(len(items) for items in positives.values()),
        "seed": seed,
    }
    result.update({metric: value / user_count for metric, value in totals.items()})
    return result


def main() -> None:
    seed = int(CONFIG["evaluation"]["random_seed"])
    report = {
        "protocol": CONFIG["version"],
        "baseline": "deterministic per-user random ranking",
        "validation_cold": evaluate(
            read_items(SPLIT_DIR / "validation_cold_candidates.csv.gz"),
            read_positives(SPLIT_DIR / "validation_cold.csv.gz"),
            seed,
        ),
        "test_cold": evaluate(
            read_items(SPLIT_DIR / "test_cold_candidates.csv.gz"),
            read_positives(SPLIT_DIR / "test_cold.csv.gz"),
            seed,
        ),
        "interpretation": "Pipeline smoke test only; a single seeded random run is not a model result.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "random_cold_split_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

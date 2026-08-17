"""Evaluate a frozen MobileNetV3-Small Visual-only baseline on Validation Cold.

The final-train and test files are deliberately not read by this script.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
FEATURE_DIR = ROOT / "artifacts" / "features"
HASH_PATH = ROOT / "data" / "processed" / "cover_dhash.csv"
CONFIG = json.loads((ROOT / "configs" / "split_v1.json").read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "outputs" / "experiments"
BATCH_SIZE = 256


def read_interactions(path: Path) -> list[tuple[int, int, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [
            (int(row["user"]), int(row["item"]), int(row["timestamp"]))
            for row in csv.DictReader(handle)
        ]


def read_candidate_items(path: Path) -> list[int]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [int(row["item"]) for row in csv.DictReader(handle)]


def read_hashes() -> dict[int, int]:
    with HASH_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["item"]): int(row["dhash"], 16) for row in csv.DictReader(handle)}


def history_bucket(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def metric_contributions(top_items: list[int], targets: set[int], k: int) -> dict[str, float]:
    hits = [1 if item in targets else 0 for item in top_items[:k]]
    hit_count = sum(hits)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(targets), k)
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"Recall@{k}": hit_count / len(targets),
        f"NDCG@{k}": dcg / idcg,
        f"HitRate@{k}": float(hit_count > 0),
    }


def build_profiles(
    users: list[int],
    histories: dict[int, list[int]],
    feature_rows: dict[int, int],
    item_vectors: np.ndarray,
) -> tuple[np.ndarray, int]:
    profiles = np.empty((len(users), item_vectors.shape[1]), dtype=np.float32)
    empty_profiles = 0
    for position, user in enumerate(users):
        rows = [feature_rows[item] for item in histories[user]]
        profile = item_vectors[rows].sum(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(profile))
        if norm == 0:
            empty_profiles += 1
            profiles[position] = 0
        else:
            profiles[position] = profile / norm
    return profiles, empty_profiles


def evaluate(
    candidate_items: list[int],
    positives: dict[int, set[int]],
    histories: dict[int, list[int]],
    feature_rows: dict[int, int],
    item_vectors: np.ndarray,
) -> dict[str, object]:
    users = sorted(positives)
    profiles, empty_profiles = build_profiles(users, histories, feature_rows, item_vectors)
    candidate_rows = [feature_rows[item] for item in candidate_items]
    candidate_vectors = item_vectors[candidate_rows]
    k_values = [int(value) for value in CONFIG["evaluation"]["k_values"]]
    max_k = max(k_values)
    seed = int(CONFIG["evaluation"]["random_seed"])

    metric_names = [
        f"{metric}@{k}"
        for k in k_values
        for metric in ("Recall", "NDCG", "HitRate")
    ]
    totals = {name: 0.0 for name in metric_names}
    bucket_totals: dict[str, dict[str, float]] = {
        bucket: {name: 0.0 for name in metric_names}
        for bucket in ("1-3", "4-6", "7-10", "11+")
    }
    bucket_users = {bucket: 0 for bucket in bucket_totals}
    users_with_all_zero_scores = 0

    for batch_start in range(0, len(users), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(users))
        scores = profiles[batch_start:batch_end] @ candidate_vectors.T
        for offset, user in enumerate(users[batch_start:batch_end]):
            user_scores = scores[offset].tolist()
            users_with_all_zero_scores += not any(user_scores)
            shuffled_indices = list(range(len(candidate_items)))
            random.Random(seed + user).shuffle(shuffled_indices)
            top_indices = heapq.nlargest(
                max_k, shuffled_indices, key=user_scores.__getitem__
            )
            top_items = [candidate_items[index] for index in top_indices]
            bucket = history_bucket(len(histories[user]))
            bucket_users[bucket] += 1
            for k in k_values:
                contributions = metric_contributions(top_items, positives[user], k)
                for name, value in contributions.items():
                    totals[name] += value
                    bucket_totals[bucket][name] += value

    user_count = len(users)
    metrics = {name: total / user_count for name, total in totals.items()}
    bucket_metrics = {
        bucket: {
            "users": bucket_users[bucket],
            **{
                name: total / bucket_users[bucket]
                for name, total in bucket_totals[bucket].items()
            },
        }
        for bucket in bucket_totals
        if bucket_users[bucket]
    }
    return {
        "users": user_count,
        "candidate_items": len(candidate_items),
        "positive_user_item_pairs": sum(len(items) for items in positives.values()),
        "users_with_empty_profile": empty_profiles,
        "users_with_all_zero_scores": users_with_all_zero_scores,
        **metrics,
        "history_length_buckets": bucket_metrics,
    }


def main() -> None:
    started = time.perf_counter()
    feature_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    item_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    if item_vectors.shape != (19_220, 576):
        raise ValueError(f"Unexpected feature shape: {item_vectors.shape}")
    feature_rows = {int(item): row for row, item in enumerate(feature_ids)}
    if len(feature_rows) != 19_220:
        raise ValueError("Feature item IDs are missing or duplicated")

    train_rows = read_interactions(SPLIT_DIR / "dev_train.csv.gz")
    validation_rows = read_interactions(SPLIT_DIR / "validation_cold.csv.gz")
    candidate_items = read_candidate_items(
        SPLIT_DIR / "validation_cold_candidates.csv.gz"
    )
    train_items = {item for _, item, _ in train_rows}
    if train_items.intersection(candidate_items):
        raise ValueError("Cold candidates overlap dev-train items")
    required_items = train_items.union(candidate_items)
    missing_features = required_items.difference(feature_rows)
    if missing_features:
        raise ValueError(f"Missing features for {len(missing_features)} required items")

    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)
    if any(user not in histories for user in positives):
        raise ValueError("Validation user without dev-train history")

    main_metrics = evaluate(
        candidate_items, positives, histories, feature_rows, item_vectors
    )

    hashes = read_hashes()
    train_hashes = {hashes[item] for item in train_items}
    excluded_items = {
        item for item in candidate_items if hashes[item] in train_hashes
    }
    clean_candidates = [item for item in candidate_items if item not in excluded_items]
    clean_positives = {
        user: targets.difference(excluded_items)
        for user, targets in positives.items()
        if targets.difference(excluded_items)
    }
    clean_metrics = evaluate(
        clean_candidates, clean_positives, histories, feature_rows, item_vectors
    )

    report = {
        "experiment": "visual_mobilenet_v3_small_validation_v1",
        "protocol": CONFIG["version"],
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "model": {
            "item_feature": "frozen torchvision MobileNetV3-Small IMAGENET1K_V1, 576-dimensional global-average-pooled backbone output",
            "item_normalization": "L2",
            "user_profile": "L2-normalized equal-weight sum of historical item cover vectors",
            "score": "cosine similarity",
            "training": "none",
            "tie_breaking": "deterministic per-user random order",
        },
        "invariants": {
            "feature_rows": len(feature_rows),
            "dev_train_items": len(train_items),
            "cold_candidates_overlap_train": 0,
            "missing_required_features": 0,
        },
        "main": main_metrics,
        "visual_clean_dhash0": {
            "rule": "exclude validation cold candidates whose dHash exactly matches any dev-train item",
            "excluded_candidate_items": len(excluded_items),
            "excluded_positive_user_item_pairs": sum(
                len(targets.intersection(excluded_items)) for targets in positives.values()
            ),
            "users_removed_after_positive_filter": len(positives) - len(clean_positives),
            "metrics": clean_metrics,
        },
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Public comments are positive signals, not exposure, click, or watch labels.",
            "Equal history weighting ignores recency and varying feedback strength.",
            "ImageNet pretraining may underrepresent anime, games, and Chinese text overlays.",
            "dHash=0 is a sensitivity filter, not proof that every excluded cover is duplicated.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "visual_mobilenet_v3_small_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

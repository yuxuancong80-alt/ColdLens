"""Analyze Text/Visual complementarity and select a rank-fusion weight on validation.

Only dev-train and Validation Cold data are read. The final test split is not read.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_text_tfidf_validation import (
    MIN_DOCUMENT_FREQUENCY,
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
FEATURE_DIR = ROOT / "artifacts" / "features"
HASH_PATH = ROOT / "data" / "processed" / "cover_dhash.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments"
CONFIG = json.loads((ROOT / "configs" / "split_v1.json").read_text(encoding="utf-8"))
TEXT_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
PRIMARY_METRIC = "NDCG@10"
SECONDARY_METRIC = "Recall@10"
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


def deterministic_order(scores: np.ndarray, user: int) -> np.ndarray:
    """Sort descending, using the frozen per-user shuffle for exact score ties."""
    shuffled = list(range(len(scores)))
    random.Random(int(CONFIG["evaluation"]["random_seed"]) + user).shuffle(shuffled)
    shuffled_array = np.asarray(shuffled, dtype=np.int32)
    stable_positions = np.argsort(-scores[shuffled_array], kind="stable")
    return shuffled_array[stable_positions]


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(order), dtype=np.uint16)
    ranks[order] = np.arange(len(order), dtype=np.uint16)
    return ranks


def empty_totals() -> dict[str, float]:
    return {
        f"{metric}@{k}": 0.0
        for k in CONFIG["evaluation"]["k_values"]
        for metric in ("Recall", "NDCG", "HitRate")
    }


def add_user_metrics(
    totals: dict[str, float],
    ranked_items: list[int],
    targets: set[int],
) -> None:
    for k in CONFIG["evaluation"]["k_values"]:
        hits = [1 if item in targets else 0 for item in ranked_items[: int(k)]]
        hit_count = sum(hits)
        totals[f"Recall@{k}"] += hit_count / len(targets)
        totals[f"HitRate@{k}"] += float(hit_count > 0)
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal_hits = min(len(targets), int(k))
        idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
        totals[f"NDCG@{k}"] += dcg / idcg


def finalized_metrics(
    totals: dict[str, float],
    users: int,
    candidates: int,
    positive_pairs: int,
) -> dict[str, float | int]:
    return {
        "users": users,
        "candidate_items": candidates,
        "positive_user_item_pairs": positive_pairs,
        **{name: value / users for name, value in totals.items()},
    }


def assert_endpoint(
    actual: dict[str, float | int], expected: dict[str, object], label: str
) -> None:
    for metric in empty_totals():
        difference = abs(float(actual[metric]) - float(expected[metric]))
        if difference > 1e-12:
            raise ValueError(
                f"{label} endpoint mismatch for {metric}: "
                f"actual={actual[metric]} expected={expected[metric]}"
            )


def main() -> None:
    started = time.perf_counter()
    train_rows = read_interactions(SPLIT_DIR / "dev_train.csv.gz")
    validation_rows = read_interactions(SPLIT_DIR / "validation_cold.csv.gz")
    candidate_items = read_candidate_items(
        SPLIT_DIR / "validation_cold_candidates.csv.gz"
    )
    train_items = {item for _, item, _ in train_rows}
    if train_items.intersection(candidate_items):
        raise ValueError("Validation cold candidates overlap dev-train items")

    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)
    users = sorted(positives)

    # Text features exactly reuse the frozen TF-IDF baseline implementation.
    titles = load_titles()
    idf = fit_idf(train_items, titles)
    text_item_vectors = {
        item: vectorize(titles[item], idf) for item in train_items
    }
    text_candidate_vectors = {
        item: vectorize(titles[item], idf) for item in candidate_items
    }
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for candidate_index, item in enumerate(candidate_items):
        for term, weight in text_candidate_vectors[item].items():
            postings[term].append((candidate_index, weight))

    # Visual profiles and scores exactly mirror the frozen Visual-only baseline.
    feature_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    visual_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    feature_rows = {int(item): row for row, item in enumerate(feature_ids)}
    required_items = train_items.union(candidate_items)
    if required_items.difference(feature_rows):
        raise ValueError("Missing one or more required visual features")
    visual_profiles = np.empty((len(users), visual_vectors.shape[1]), dtype=np.float32)
    for user_position, user in enumerate(users):
        rows = [feature_rows[item] for item in histories[user]]
        profile = visual_vectors[rows].sum(axis=0, dtype=np.float32)
        visual_profiles[user_position] = profile / np.linalg.norm(profile)
    visual_candidates = visual_vectors[[feature_rows[item] for item in candidate_items]]
    visual_scores = np.empty((len(users), len(candidate_items)), dtype=np.float32)
    for start in range(0, len(users), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(users))
        visual_scores[start:end] = visual_profiles[start:end] @ visual_candidates.T

    text_ranks = np.empty((len(users), len(candidate_items)), dtype=np.uint16)
    visual_ranks = np.empty_like(text_ranks)
    text_totals = empty_totals()
    visual_totals = empty_totals()
    overlap_size_total = 0
    jaccard_total = 0.0
    hit_categories = {"both": 0, "text_only": 0, "visual_only": 0, "neither": 0}

    max_k = max(int(value) for value in CONFIG["evaluation"]["k_values"])
    for user_position, user in enumerate(users):
        profile = build_user_profile(histories[user], text_item_vectors)
        text_scores = np.zeros(len(candidate_items), dtype=np.float64)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                text_scores[candidate_index] += profile_weight * candidate_weight

        text_order = deterministic_order(text_scores, user)
        visual_order = deterministic_order(visual_scores[user_position], user)
        text_ranks[user_position] = ranks_from_order(text_order)
        visual_ranks[user_position] = ranks_from_order(visual_order)
        text_top = [candidate_items[index] for index in text_order[:max_k]]
        visual_top = [candidate_items[index] for index in visual_order[:max_k]]
        add_user_metrics(text_totals, text_top, positives[user])
        add_user_metrics(visual_totals, visual_top, positives[user])

        text_top10 = set(text_top[:10])
        visual_top10 = set(visual_top[:10])
        intersection = len(text_top10.intersection(visual_top10))
        overlap_size_total += intersection
        jaccard_total += intersection / len(text_top10.union(visual_top10))
        text_hit = bool(text_top10.intersection(positives[user]))
        visual_hit = bool(visual_top10.intersection(positives[user]))
        category = (
            "both"
            if text_hit and visual_hit
            else "text_only"
            if text_hit
            else "visual_only"
            if visual_hit
            else "neither"
        )
        hit_categories[category] += 1

    positive_pairs = sum(len(items) for items in positives.values())
    text_metrics = finalized_metrics(
        text_totals, len(users), len(candidate_items), positive_pairs
    )
    visual_metrics = finalized_metrics(
        visual_totals, len(users), len(candidate_items), positive_pairs
    )
    expected_text = json.loads(
        (OUTPUT_DIR / "text_tfidf_validation_v1.json").read_text(encoding="utf-8")
    )["metrics"]
    expected_visual = json.loads(
        (OUTPUT_DIR / "visual_mobilenet_v3_small_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )["main"]
    assert_endpoint(text_metrics, expected_text, "text")
    assert_endpoint(visual_metrics, expected_visual, "visual")

    grid_results: dict[str, dict[str, float | int]] = {}
    for text_weight in TEXT_WEIGHTS:
        totals = empty_totals()
        for user_position, user in enumerate(users):
            fusion_cost = (
                text_weight * text_ranks[user_position].astype(np.float32)
                + (1.0 - text_weight)
                * visual_ranks[user_position].astype(np.float32)
            )
            fusion_order = deterministic_order(-fusion_cost, user)
            ranked_items = [candidate_items[index] for index in fusion_order[:max_k]]
            add_user_metrics(totals, ranked_items, positives[user])
        grid_results[str(text_weight)] = finalized_metrics(
            totals, len(users), len(candidate_items), positive_pairs
        )

    selected_weight = max(
        TEXT_WEIGHTS,
        key=lambda weight: (
            float(grid_results[str(weight)][PRIMARY_METRIC]),
            float(grid_results[str(weight)][SECONDARY_METRIC]),
        ),
    )

    # Apply the selected main-validation weight to the frozen dHash=0 clean set.
    hashes = read_hashes()
    train_hashes = {hashes[item] for item in train_items}
    excluded_items = {
        item for item in candidate_items if hashes[item] in train_hashes
    }
    allowed_mask = np.asarray(
        [item not in excluded_items for item in candidate_items], dtype=bool
    )
    clean_candidates = [
        item for item in candidate_items if item not in excluded_items
    ]
    clean_positives = {
        user: targets.difference(excluded_items)
        for user, targets in positives.items()
        if targets.difference(excluded_items)
    }
    clean_users = sorted(clean_positives)
    clean_visual_profiles = np.empty(
        (len(clean_users), visual_vectors.shape[1]), dtype=np.float32
    )
    for clean_position, user in enumerate(clean_users):
        rows = [feature_rows[item] for item in histories[user]]
        profile = visual_vectors[rows].sum(axis=0, dtype=np.float32)
        clean_visual_profiles[clean_position] = profile / np.linalg.norm(profile)
    clean_visual_candidates = visual_candidates[allowed_mask]
    clean_visual_score_matrix = np.empty(
        (len(clean_users), len(clean_candidates)), dtype=np.float32
    )
    for start in range(0, len(clean_users), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(clean_users))
        clean_visual_score_matrix[start:end] = (
            clean_visual_profiles[start:end] @ clean_visual_candidates.T
        )
    clean_postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for candidate_index, item in enumerate(clean_candidates):
        for term, weight in text_candidate_vectors[item].items():
            clean_postings[term].append((candidate_index, weight))
    clean_totals = {"text": empty_totals(), "visual": empty_totals(), "fusion": empty_totals()}
    for clean_position, user in enumerate(clean_users):
        text_profile = build_user_profile(histories[user], text_item_vectors)
        clean_text_scores = np.zeros(len(clean_candidates), dtype=np.float64)
        for term, profile_weight in text_profile.items():
            for candidate_index, candidate_weight in clean_postings.get(term, []):
                clean_text_scores[candidate_index] += profile_weight * candidate_weight
        clean_visual_scores = clean_visual_score_matrix[clean_position]
        clean_text_order = deterministic_order(clean_text_scores, user)
        clean_visual_order = deterministic_order(clean_visual_scores, user)
        clean_text_ranks = ranks_from_order(clean_text_order)
        clean_visual_ranks = ranks_from_order(clean_visual_order)
        fusion_cost = (
            selected_weight * clean_text_ranks.astype(np.float32)
            + (1.0 - selected_weight) * clean_visual_ranks.astype(np.float32)
        )
        clean_fusion_order = deterministic_order(-fusion_cost, user)
        add_user_metrics(
            clean_totals["text"],
            [clean_candidates[index] for index in clean_text_order[:max_k]],
            clean_positives[user],
        )
        add_user_metrics(
            clean_totals["visual"],
            [clean_candidates[index] for index in clean_visual_order[:max_k]],
            clean_positives[user],
        )
        add_user_metrics(
            clean_totals["fusion"],
            [clean_candidates[index] for index in clean_fusion_order[:max_k]],
            clean_positives[user],
        )

    clean_pairs = sum(len(items) for items in clean_positives.values())
    clean_metrics = {
        name: finalized_metrics(
            totals, len(clean_positives), len(clean_candidates), clean_pairs
        )
        for name, totals in clean_totals.items()
    }
    assert_endpoint(
        clean_metrics["visual"],
        expected_visual := json.loads(
            (OUTPUT_DIR / "visual_mobilenet_v3_small_validation_v1.json").read_text(
                encoding="utf-8"
            )
        )["visual_clean_dhash0"]["metrics"],
        "clean visual",
    )

    report = {
        "experiment": "multimodal_rank_fusion_validation_v1",
        "protocol": CONFIG["version"],
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "selection_rule_frozen_before_grid": {
            "fusion": "weighted sum of per-user zero-based candidate ranks",
            "text_weights": TEXT_WEIGHTS,
            "visual_weight": "1 - text_weight",
            "primary_metric": PRIMARY_METRIC,
            "secondary_tie_break_metric": SECONDARY_METRIC,
            "clean_weight_retuned": False,
        },
        "endpoint_reproduction": {
            "text_matches_existing_baseline": True,
            "visual_matches_existing_baseline": True,
            "clean_visual_matches_existing_baseline": True,
        },
        "top10_complementarity": {
            "mean_recommendation_intersection_size": overlap_size_total / len(users),
            "mean_recommendation_jaccard": jaccard_total / len(users),
            "positive_hit_users": hit_categories,
            "users_hit_by_either": len(users) - hit_categories["neither"],
        },
        "validation_grid": grid_results,
        "selected_text_weight": selected_weight,
        "selected_visual_weight": 1.0 - selected_weight,
        "selected_validation_metrics": grid_results[str(selected_weight)],
        "visual_clean_dhash0": {
            "excluded_candidate_items": len(excluded_items),
            "users_removed_after_positive_filter": len(users) - len(clean_positives),
            "selected_weight_retuned": False,
            "metrics": clean_metrics,
        },
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The fusion weight is selected and reported on the same validation split.",
            "A five-point grid can miss a better intermediate weight and still incurs selection bias.",
            "User-level hit complementarity does not guarantee that a simple fusion improves ranking.",
            "Comment positives are not exposure, click, or watch labels.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "multimodal_rank_fusion_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

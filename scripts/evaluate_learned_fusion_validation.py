"""Evaluate the single frozen learned-fusion model on outer Validation Cold."""

from __future__ import annotations

import csv
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_multimodal_rank_fusion_validation import (
    BATCH_SIZE,
    CONFIG,
    FEATURE_DIR,
    HASH_PATH,
    OUTPUT_DIR,
    SPLIT_DIR,
    deterministic_order,
    read_candidate_items,
    read_hashes,
    read_interactions,
)
from run_text_tfidf_validation import (
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "learned_fusion" / "learned_fusion_model_v1.json"


def empty_totals() -> dict[str, float]:
    return {
        f"{metric}@{k}": 0.0
        for k in CONFIG["evaluation"]["k_values"]
        for metric in ("Recall", "NDCG", "HitRate")
    }


def add_metrics(
    totals: dict[str, float], ranked_items: list[int], targets: set[int]
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


def finalize(
    totals: dict[str, float], users: int, candidates: int, positives: int
) -> dict[str, float | int]:
    return {
        "users": users,
        "candidate_items": candidates,
        "positive_user_item_pairs": positives,
        **{name: value / users for name, value in totals.items()},
    }


def build_postings(
    candidate_items: list[int], text_vectors: dict[int, dict[str, float]]
) -> defaultdict[str, list[tuple[int, float]]]:
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for candidate_index, item in enumerate(candidate_items):
        for term, value in text_vectors[item].items():
            postings[term].append((candidate_index, value))
    return postings


def evaluate_pool(
    candidate_items: list[int],
    positives: dict[int, set[int]],
    histories: dict[int, list[int]],
    text_item_vectors: dict[int, dict[str, float]],
    text_candidate_vectors: dict[int, dict[str, float]],
    visual_vectors: np.ndarray,
    visual_rows: dict[int, int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    model_weights: np.ndarray,
) -> dict[str, object]:
    users = sorted(positives)
    postings = build_postings(candidate_items, text_candidate_vectors)
    visual_profiles = np.empty((len(users), visual_vectors.shape[1]), dtype=np.float32)
    empty_text_profiles = 0
    text_profiles: list[dict[str, float]] = []
    for position, user in enumerate(users):
        history_items = histories[user]
        text_profile = build_user_profile(history_items, text_item_vectors)
        text_profiles.append(text_profile)
        empty_text_profiles += not text_profile
        visual_profile = visual_vectors[[visual_rows[item] for item in history_items]].sum(
            axis=0, dtype=np.float32
        )
        visual_profiles[position] = visual_profile / np.linalg.norm(visual_profile)
    visual_candidates = visual_vectors[[visual_rows[item] for item in candidate_items]]
    visual_scores = np.empty((len(users), len(candidate_items)), dtype=np.float32)
    for start in range(0, len(users), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(users))
        visual_scores[start:end] = visual_profiles[start:end] @ visual_candidates.T

    text_totals = empty_totals()
    learned_totals = empty_totals()
    max_k = max(int(k) for k in CONFIG["evaluation"]["k_values"])
    learned_text_top10_intersection = 0
    learned_text_jaccard = 0.0
    users_changed_at_top10 = 0
    learned_feature_min = np.full(3, np.inf, dtype=np.float64)
    learned_feature_max = np.full(3, -np.inf, dtype=np.float64)

    for position, user in enumerate(users):
        text_scores = np.zeros(len(candidate_items), dtype=np.float64)
        for term, profile_weight in text_profiles[position].items():
            for candidate_index, candidate_weight in postings.get(term, []):
                text_scores[candidate_index] += profile_weight * candidate_weight
        text_float32 = text_scores.astype(np.float32)
        visual_float32 = visual_scores[position]
        features = np.column_stack(
            [text_float32, visual_float32, text_float32 * visual_float32]
        ).astype(np.float32, copy=False)
        learned_feature_min = np.minimum(learned_feature_min, features.min(axis=0))
        learned_feature_max = np.maximum(learned_feature_max, features.max(axis=0))
        standardized = (features - feature_mean) / feature_std
        learned_scores = standardized @ model_weights

        text_order = deterministic_order(text_scores, user)
        learned_order = deterministic_order(learned_scores, user)
        text_ranked = [candidate_items[index] for index in text_order[:max_k]]
        learned_ranked = [candidate_items[index] for index in learned_order[:max_k]]
        add_metrics(text_totals, text_ranked, positives[user])
        add_metrics(learned_totals, learned_ranked, positives[user])

        text_top10 = set(text_ranked[:10])
        learned_top10 = set(learned_ranked[:10])
        intersection = len(text_top10.intersection(learned_top10))
        learned_text_top10_intersection += intersection
        learned_text_jaccard += intersection / len(text_top10.union(learned_top10))
        users_changed_at_top10 += text_top10 != learned_top10

    positive_pairs = sum(len(items) for items in positives.values())
    return {
        "text": finalize(
            text_totals, len(users), len(candidate_items), positive_pairs
        ),
        "learned_fusion": finalize(
            learned_totals, len(users), len(candidate_items), positive_pairs
        ),
        "diagnostics": {
            "empty_text_profiles": empty_text_profiles,
            "feature_min": learned_feature_min.tolist(),
            "feature_max": learned_feature_max.tolist(),
            "mean_text_learned_top10_intersection": learned_text_top10_intersection
            / len(users),
            "mean_text_learned_top10_jaccard": learned_text_jaccard / len(users),
            "users_with_changed_top10_set": users_changed_at_top10,
        },
    }


def assert_text_endpoint(
    actual: dict[str, object], expected: dict[str, object], label: str
) -> None:
    for name in empty_totals():
        if abs(float(actual[name]) - float(expected[name])) > 1e-12:
            raise ValueError(
                f"{label} text endpoint mismatch for {name}: "
                f"actual={actual[name]} expected={expected[name]}"
            )


def main() -> None:
    started = time.perf_counter()
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("test_split_read") is not False:
        raise ValueError("Learned model metadata does not confirm test isolation")
    feature_mean = np.asarray(model["standardization"]["mean"], dtype=np.float32)
    feature_std = np.asarray(model["standardization"]["std"], dtype=np.float32)
    model_weights = np.asarray(
        model["final_retrain"]["softmax_weights_on_standardized_features"],
        dtype=np.float32,
    )
    if not np.isclose(model_weights.sum(), 1.0, atol=1e-6) or np.any(
        model_weights < 0
    ):
        raise ValueError("Learned fusion weights are not a non-negative simplex")

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

    titles = load_titles()
    idf = fit_idf(train_items, titles)
    required_text_items = train_items.union(candidate_items)
    text_vectors = {
        item: vectorize(titles[item], idf) for item in required_text_items
    }
    visual_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    visual_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    visual_rows = {int(item): row for row, item in enumerate(visual_ids)}
    if required_text_items.difference(visual_rows):
        raise ValueError("Missing required visual features")

    main_result = evaluate_pool(
        candidate_items,
        positives,
        histories,
        text_vectors,
        text_vectors,
        visual_vectors,
        visual_rows,
        feature_mean,
        feature_std,
        model_weights,
    )
    existing_text = json.loads(
        (OUTPUT_DIR / "text_tfidf_validation_v1.json").read_text(encoding="utf-8")
    )["metrics"]
    assert_text_endpoint(main_result["text"], existing_text, "main")

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
    clean_result = evaluate_pool(
        clean_candidates,
        clean_positives,
        histories,
        text_vectors,
        text_vectors,
        visual_vectors,
        visual_rows,
        feature_mean,
        feature_std,
        model_weights,
    )
    expected_clean_text = json.loads(
        (OUTPUT_DIR / "multimodal_rank_fusion_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )["visual_clean_dhash0"]["metrics"]["text"]
    assert_text_endpoint(clean_result["text"], expected_clean_text, "clean")

    learned = main_result["learned_fusion"]
    accepted = (
        float(learned["NDCG@10"]) > float(existing_text["NDCG@10"])
        and float(learned["Recall@10"]) > float(existing_text["Recall@10"])
    )
    report = {
        "experiment": "learned_fusion_validation_v1",
        "protocol": CONFIG["version"],
        "model": "three-weight non-negative pairwise learned late fusion",
        "model_path": str(MODEL_PATH.resolve()),
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "outer_validation_tuning": False,
        "weights_on_standardized_features": model_weights.tolist(),
        "selected_internal_epoch": model["selection"]["selected_epoch"],
        "main": main_result,
        "visual_clean_dhash0": {
            "excluded_candidate_items": len(excluded_items),
            "users_removed_after_positive_filter": len(positives)
            - len(clean_positives),
            "model_retrained_or_retuned": False,
            "results": clean_result,
        },
        "acceptance_rule": "main NDCG@10 and Recall@10 must both strictly exceed Text-only",
        "accepted": accepted,
        "retained_final_candidate": "learned_fusion" if accepted else "text_only",
        "main_delta_vs_text": {
            name: float(learned[name]) - float(existing_text[name])
            for name in empty_totals()
        },
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Training negatives are uniformly sampled unobserved items, while evaluation ranks the full cold pool.",
            "First observed comment time is an availability proxy, not upload time.",
            "The learned model combines frozen similarity features and cannot repair weak encoders.",
            "Comment positives are not exposure, click, or watch labels.",
        ],
    }
    output = OUTPUT_DIR / "learned_fusion_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

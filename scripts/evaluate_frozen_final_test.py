"""Evaluate the frozen Text-only candidate and Visual-only baseline on Test Cold."""

from __future__ import annotations

import argparse
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
FINAL_OUTPUT_DIR = ROOT / "outputs" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Reproduce frozen development endpoints without reading final/test files.",
    )
    return parser.parse_args()


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


def evaluate_pool(
    candidate_items: list[int],
    positives: dict[int, set[int]],
    histories: dict[int, list[int]],
    train_text_vectors: dict[int, dict[str, float]],
    candidate_text_vectors: dict[int, dict[str, float]],
    visual_vectors: np.ndarray,
    visual_rows: dict[int, int],
) -> dict[str, object]:
    users = sorted(positives)
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for candidate_index, item in enumerate(candidate_items):
        for term, value in candidate_text_vectors[item].items():
            postings[term].append((candidate_index, value))

    text_profiles: list[dict[str, float]] = []
    visual_profiles = np.empty((len(users), visual_vectors.shape[1]), dtype=np.float32)
    empty_text_profiles = 0
    for position, user in enumerate(users):
        history_items = histories[user]
        text_profile = build_user_profile(history_items, train_text_vectors)
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
    visual_totals = empty_totals()
    text_all_zero_users = 0
    visual_all_zero_users = 0
    max_k = max(int(k) for k in CONFIG["evaluation"]["k_values"])
    for position, user in enumerate(users):
        text_scores = np.zeros(len(candidate_items), dtype=np.float64)
        for term, profile_weight in text_profiles[position].items():
            for candidate_index, candidate_weight in postings.get(term, []):
                text_scores[candidate_index] += profile_weight * candidate_weight
        text_all_zero_users += not np.any(text_scores)
        visual_all_zero_users += not np.any(visual_scores[position])
        text_order = deterministic_order(text_scores, user)
        visual_order = deterministic_order(visual_scores[position], user)
        text_ranked = [candidate_items[index] for index in text_order[:max_k]]
        visual_ranked = [candidate_items[index] for index in visual_order[:max_k]]
        add_metrics(text_totals, text_ranked, positives[user])
        add_metrics(visual_totals, visual_ranked, positives[user])

    positive_pairs = sum(len(items) for items in positives.values())
    return {
        "text": finalize(
            text_totals, len(users), len(candidate_items), positive_pairs
        ),
        "visual": finalize(
            visual_totals, len(users), len(candidate_items), positive_pairs
        ),
        "diagnostics": {
            "text_empty_profile_users": empty_text_profiles,
            "text_all_zero_score_users": text_all_zero_users,
            "visual_all_zero_score_users": visual_all_zero_users,
            "zero_vector_candidate_titles": sum(
                not candidate_text_vectors[item] for item in candidate_items
            ),
        },
    }


def assert_metrics(
    actual: dict[str, object], expected: dict[str, object], label: str
) -> None:
    for name in empty_totals():
        difference = abs(float(actual[name]) - float(expected[name]))
        if difference > 1e-12:
            raise ValueError(
                f"{label} mismatch for {name}: actual={actual[name]} "
                f"expected={expected[name]}"
            )


def run_stage(
    train_filename: str,
    evaluation_filename: str,
    candidate_filename: str,
) -> dict[str, object]:
    train_rows = read_interactions(SPLIT_DIR / train_filename)
    evaluation_rows = read_interactions(SPLIT_DIR / evaluation_filename)
    candidate_items = read_candidate_items(SPLIT_DIR / candidate_filename)
    train_items = {item for _, item, _ in train_rows}
    if train_items.intersection(candidate_items):
        raise ValueError("Cold candidate overlaps corresponding train items")

    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in evaluation_rows:
        positives[user].add(item)
    if any(user not in histories for user in positives):
        raise ValueError("Evaluation user lacks corresponding training history")

    titles = load_titles()
    idf = fit_idf(train_items, titles)
    train_text_vectors = {
        item: vectorize(titles[item], idf) for item in train_items
    }
    candidate_text_vectors = {
        item: vectorize(titles[item], idf) for item in candidate_items
    }
    visual_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    visual_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    visual_rows = {int(item): row for row, item in enumerate(visual_ids)}
    if train_items.union(candidate_items).difference(visual_rows):
        raise ValueError("Missing required visual feature")

    main = evaluate_pool(
        candidate_items,
        positives,
        histories,
        train_text_vectors,
        candidate_text_vectors,
        visual_vectors,
        visual_rows,
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
    clean_candidate_text_vectors = {
        item: candidate_text_vectors[item] for item in clean_candidates
    }
    clean = evaluate_pool(
        clean_candidates,
        clean_positives,
        histories,
        train_text_vectors,
        clean_candidate_text_vectors,
        visual_vectors,
        visual_rows,
    )
    return {
        "train": {
            "rows": len(train_rows),
            "users": len(histories),
            "items": len(train_items),
            "idf_vocabulary_size": len(idf),
            "zero_vector_train_titles": sum(
                not vector for vector in train_text_vectors.values()
            ),
        },
        "main": main,
        "visual_clean_dhash0": {
            "excluded_candidate_items": len(excluded_items),
            "excluded_positive_user_item_pairs": sum(
                len(targets.intersection(excluded_items)) for targets in positives.values()
            ),
            "users_removed_after_positive_filter": len(positives)
            - len(clean_positives),
            "results": clean,
        },
    }


def run_preflight() -> None:
    result = run_stage(
        "dev_train.csv.gz",
        "validation_cold.csv.gz",
        "validation_cold_candidates.csv.gz",
    )
    expected_text = json.loads(
        (OUTPUT_DIR / "text_tfidf_validation_v1.json").read_text(encoding="utf-8")
    )["metrics"]
    expected_visual_report = json.loads(
        (OUTPUT_DIR / "visual_mobilenet_v3_small_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected_clean_text = json.loads(
        (OUTPUT_DIR / "multimodal_rank_fusion_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )["visual_clean_dhash0"]["metrics"]["text"]
    assert_metrics(result["main"]["text"], expected_text, "main text")
    assert_metrics(
        result["main"]["visual"], expected_visual_report["main"], "main visual"
    )
    assert_metrics(
        result["visual_clean_dhash0"]["results"]["text"],
        expected_clean_text,
        "clean text",
    )
    assert_metrics(
        result["visual_clean_dhash0"]["results"]["visual"],
        expected_visual_report["visual_clean_dhash0"]["metrics"],
        "clean visual",
    )
    print(
        json.dumps(
            {
                "preflight": "passed",
                "test_split_read": False,
                "reproduced_endpoints": [
                    "main_text",
                    "main_visual",
                    "visual_clean_dhash0_text",
                    "visual_clean_dhash0_visual",
                ],
            },
            indent=2,
        )
    )


def run_final() -> None:
    started = time.perf_counter()
    result = run_stage(
        "final_train.csv.gz",
        "test_cold.csv.gz",
        "test_cold_candidates.csv.gz",
    )
    actual_train_counts = {
        name: result["train"][name] for name in ("rows", "users", "items")
    }
    expected_train_counts = {"rows": 323_736, "users": 49_887, "items": 18_212}
    if actual_train_counts != expected_train_counts:
        raise ValueError(f"Unexpected final-train counts: {result['train']}")
    main = result["main"]
    if (
        main["text"]["users"] != 13_810
        or main["text"]["candidate_items"] != 1_008
        or main["text"]["positive_user_item_pairs"] != 18_812
    ):
        raise ValueError(f"Unexpected Test Cold counts: {main['text']}")
    if result["visual_clean_dhash0"]["excluded_candidate_items"] != 13:
        raise ValueError(
            "Test dHash0 exclusion count differs from the frozen audit record"
        )
    report = {
        "experiment": "frozen_final_test_v1",
        "protocol": CONFIG["version"],
        "freeze_record": "docs/MODEL_FREEZE_V1.md",
        "test_split_read": True,
        "test_run_after_model_freeze": True,
        "models_evaluated": ["text_tfidf_v1", "visual_mobilenet_v3_small"],
        "rejected_validation_models_not_evaluated": [
            "global_rank_fusion",
            "conditional_modality_gate",
            "learned_fusion_v1",
        ],
        **result,
        "runtime_seconds": time.perf_counter() - started,
        "post_test_rule": "No model, feature, threshold, or hyperparameter changes followed by another test run.",
        "interpretation_boundary": "Public-comment preference matching; not CTR, watch time, retention, exposure, or online causal lift.",
    }
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = FINAL_OUTPUT_DIR / "frozen_final_test_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        run_preflight()
        return
    run_final()


if __name__ == "__main__":
    main()

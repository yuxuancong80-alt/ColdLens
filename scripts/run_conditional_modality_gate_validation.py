"""Test a predeclared low-text-confidence gate on held-out validation users.

The gate is tuned on one deterministic half of Validation Cold users and
audited once on the other half. Final-train and test files are not read.
"""

from __future__ import annotations

import hashlib
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
    read_interactions,
)
from run_text_tfidf_validation import (
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
VISUAL_SHARE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PRIMARY_METRIC = "NDCG@10"
SECONDARY_METRIC = "Recall@10"
SPLIT_SALT = "conditional-modality-gate-v1"


def gate_partition(user: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{user}".encode("ascii")).digest()
    return "gate_dev" if digest[0] % 2 == 0 else "gate_audit"


def empty_totals() -> dict[str, float]:
    return {
        f"{metric}@{k}": 0.0
        for k in CONFIG["evaluation"]["k_values"]
        for metric in ("Recall", "NDCG", "HitRate")
    }


def add_metrics(
    totals: dict[str, float], ranked_items: np.ndarray, targets: set[int]
) -> None:
    for k in CONFIG["evaluation"]["k_values"]:
        hits = [1 if int(item) in targets else 0 for item in ranked_items[: int(k)]]
        hit_count = sum(hits)
        totals[f"Recall@{k}"] += hit_count / len(targets)
        totals[f"HitRate@{k}"] += float(hit_count > 0)
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal_hits = min(len(targets), int(k))
        idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
        totals[f"NDCG@{k}"] += dcg / idcg


def evaluate(
    positions: list[int],
    users: list[int],
    positives: dict[int, set[int]],
    text_top_items: np.ndarray,
    visual_top_items: np.ndarray,
    text_max_scores: np.ndarray,
    threshold: float | None,
) -> dict[str, object]:
    totals = empty_totals()
    visual_users = 0
    hit_categories = {
        "both": 0,
        "text_only": 0,
        "visual_only": 0,
        "neither": 0,
    }
    gated_categories = {key: 0 for key in hit_categories}
    for position in positions:
        user = users[position]
        targets = positives[user]
        text_hit = bool(set(map(int, text_top_items[position, :10])).intersection(targets))
        visual_hit = bool(
            set(map(int, visual_top_items[position, :10])).intersection(targets)
        )
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
        use_visual = threshold is not None and text_max_scores[position] <= threshold
        if use_visual:
            visual_users += 1
            gated_categories[category] += 1
            ranked = visual_top_items[position]
        else:
            ranked = text_top_items[position]
        add_metrics(totals, ranked, targets)
    count = len(positions)
    return {
        "users": count,
        "positive_user_item_pairs": sum(len(positives[users[p]]) for p in positions),
        "visual_users": visual_users,
        "actual_visual_share": visual_users / count,
        **{name: total / count for name, total in totals.items()},
        "underlying_hit_categories": hit_categories,
        "gated_user_hit_categories": gated_categories,
    }


def assert_main_endpoint(
    actual: dict[str, object], expected: dict[str, object], label: str
) -> None:
    for name in empty_totals():
        if abs(float(actual[name]) - float(expected[name])) > 1e-12:
            raise ValueError(
                f"{label} mismatch for {name}: actual={actual[name]} expected={expected[name]}"
            )


def main() -> None:
    started = time.perf_counter()
    train_rows = read_interactions(SPLIT_DIR / "dev_train.csv.gz")
    validation_rows = read_interactions(SPLIT_DIR / "validation_cold.csv.gz")
    candidate_items = read_candidate_items(
        SPLIT_DIR / "validation_cold_candidates.csv.gz"
    )
    train_items = {item for _, item, _ in train_rows}
    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)
    users = sorted(positives)

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

    feature_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    visual_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    feature_rows = {int(item): row for row, item in enumerate(feature_ids)}
    visual_profiles = np.empty((len(users), visual_vectors.shape[1]), dtype=np.float32)
    for position, user in enumerate(users):
        rows = [feature_rows[item] for item in histories[user]]
        profile = visual_vectors[rows].sum(axis=0, dtype=np.float32)
        visual_profiles[position] = profile / np.linalg.norm(profile)
    visual_candidates = visual_vectors[[feature_rows[item] for item in candidate_items]]
    visual_scores = np.empty((len(users), len(candidate_items)), dtype=np.float32)
    for start in range(0, len(users), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(users))
        visual_scores[start:end] = visual_profiles[start:end] @ visual_candidates.T

    max_k = max(int(k) for k in CONFIG["evaluation"]["k_values"])
    text_top_items = np.empty((len(users), max_k), dtype=np.int32)
    visual_top_items = np.empty_like(text_top_items)
    text_max_scores = np.empty(len(users), dtype=np.float64)
    candidate_array = np.asarray(candidate_items, dtype=np.int32)
    for position, user in enumerate(users):
        profile = build_user_profile(histories[user], text_item_vectors)
        text_scores = np.zeros(len(candidate_items), dtype=np.float64)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                text_scores[candidate_index] += profile_weight * candidate_weight
        text_order = deterministic_order(text_scores, user)
        visual_order = deterministic_order(visual_scores[position], user)
        text_top_items[position] = candidate_array[text_order[:max_k]]
        visual_top_items[position] = candidate_array[visual_order[:max_k]]
        text_max_scores[position] = float(text_scores.max())

    all_positions = list(range(len(users)))
    dev_positions = [p for p, user in enumerate(users) if gate_partition(user) == "gate_dev"]
    audit_positions = [p for p, user in enumerate(users) if gate_partition(user) == "gate_audit"]
    if set(dev_positions).intersection(audit_positions) or len(dev_positions) + len(
        audit_positions
    ) != len(users):
        raise ValueError("Gate user partition is not exhaustive and disjoint")

    all_text = evaluate(
        all_positions,
        users,
        positives,
        text_top_items,
        visual_top_items,
        text_max_scores,
        None,
    )
    all_visual = evaluate(
        all_positions,
        users,
        positives,
        text_top_items,
        visual_top_items,
        text_max_scores,
        float("inf"),
    )
    existing_text = json.loads(
        (OUTPUT_DIR / "text_tfidf_validation_v1.json").read_text(encoding="utf-8")
    )["metrics"]
    existing_visual = json.loads(
        (OUTPUT_DIR / "visual_mobilenet_v3_small_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )["main"]
    assert_main_endpoint(all_text, existing_text, "text endpoint")
    assert_main_endpoint(all_visual, existing_visual, "visual endpoint")

    dev_scores = text_max_scores[dev_positions]
    dev_grid: dict[str, dict[str, object]] = {}
    thresholds: dict[str, float | None] = {}
    for visual_share in VISUAL_SHARE_GRID:
        threshold = (
            None
            if visual_share == 0
            else float(np.quantile(dev_scores, visual_share, method="lower"))
        )
        key = str(visual_share)
        thresholds[key] = threshold
        dev_grid[key] = evaluate(
            dev_positions,
            users,
            positives,
            text_top_items,
            visual_top_items,
            text_max_scores,
            threshold,
        )
        dev_grid[key]["threshold"] = threshold

    selected_share = max(
        VISUAL_SHARE_GRID,
        key=lambda share: (
            float(dev_grid[str(share)][PRIMARY_METRIC]),
            float(dev_grid[str(share)][SECONDARY_METRIC]),
        ),
    )
    selected_threshold = thresholds[str(selected_share)]
    dev_text = dev_grid["0.0"]
    audit_text = evaluate(
        audit_positions,
        users,
        positives,
        text_top_items,
        visual_top_items,
        text_max_scores,
        None,
    )
    audit_visual = evaluate(
        audit_positions,
        users,
        positives,
        text_top_items,
        visual_top_items,
        text_max_scores,
        float("inf"),
    )
    audit_gate = evaluate(
        audit_positions,
        users,
        positives,
        text_top_items,
        visual_top_items,
        text_max_scores,
        selected_threshold,
    )
    accepted = (
        float(audit_gate[PRIMARY_METRIC]) > float(audit_text[PRIMARY_METRIC])
        and float(audit_gate[SECONDARY_METRIC])
        > float(audit_text[SECONDARY_METRIC])
    )

    report = {
        "experiment": "conditional_modality_gate_validation_v1",
        "protocol": CONFIG["version"],
        "evaluated_split": "validation_cold_user_partition",
        "test_split_read": False,
        "rule_frozen_before_run": {
            "partition": f"SHA-256 parity of '{SPLIT_SALT}:<user_id>'",
            "gate_feature": "maximum TF-IDF cosine score across all validation cold candidates",
            "direction": "use Visual ranking when maximum Text score is at or below threshold",
            "gate_dev_visual_share_grid": VISUAL_SHARE_GRID,
            "primary_metric": PRIMARY_METRIC,
            "secondary_tie_break_metric": SECONDARY_METRIC,
            "acceptance": "gate-audit NDCG@10 and Recall@10 must both strictly exceed Text-only",
        },
        "endpoint_reproduction": {
            "text_matches_existing_baseline": True,
            "visual_matches_existing_baseline": True,
        },
        "partition_counts": {
            "gate_dev_users": len(dev_positions),
            "gate_audit_users": len(audit_positions),
        },
        "gate_dev_grid": dev_grid,
        "selected_gate_dev_visual_share": selected_share,
        "selected_max_text_score_threshold": selected_threshold,
        "gate_dev_text_metrics": dev_text,
        "gate_audit_text_metrics": audit_text,
        "gate_audit_visual_metrics": audit_visual,
        "gate_audit_frozen_gate_metrics": audit_gate,
        "accepted_by_predeclared_rule": accepted,
        "retained_final_candidate": "conditional_gate" if accepted else "text_only",
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The user partition is an internal validation audit, not the final temporal test.",
            "Only one confidence feature and six switching shares are tested.",
            "The threshold is specific to the frozen candidate pool and TF-IDF representation.",
            "Comment positives are not exposure, click, or watch labels.",
        ],
    }
    output = OUTPUT_DIR / "conditional_modality_gate_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

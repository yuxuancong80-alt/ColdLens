"""Create frozen descriptive and Validation-only product diagnostics.

This script deliberately does not read final-train, Test Cold, or final reports.
It does not select or tune a model.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_text_tfidf_validation import (
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "microlens50k"
SPLIT = ROOT / "data" / "processed" / "split_v1"
FEATURES = ROOT / "artifacts" / "features"
OUTPUT = ROOT / "outputs" / "analysis" / "product_analysis_v1.json"
ASSETS = ROOT / "docs" / "assets"
SEED = 20260813
TOP_K = 10
HISTORY_BUCKETS = ("1-3", "4-6", "7-10", "11+")


def read_gzip_rows(path: Path) -> list[tuple[int, int, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [
            (int(row["user"]), int(row["item"]), int(row["timestamp"]))
            for row in csv.DictReader(handle)
        ]


def read_candidates(path: Path) -> list[int]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [int(row["item"]) for row in csv.DictReader(handle)]


def percentile_nearest_rank(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def gini(values: list[int]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered) or ordered.sum() == 0:
        return 0.0
    positions = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float(
        (2 * np.dot(positions, ordered) / (len(ordered) * ordered.sum()))
        - (len(ordered) + 1) / len(ordered)
    )


def distribution(counts: list[int]) -> dict[str, object]:
    ordered_desc = sorted(counts, reverse=True)
    total = sum(counts)
    concentration = {}
    for share in (0.01, 0.05, 0.10, 0.20):
        entities = math.ceil(share * len(counts))
        concentration[f"top_{int(share * 100)}pct"] = {
            "entities": entities,
            "interaction_share": sum(ordered_desc[:entities]) / total,
        }
    return {
        "entities": len(counts),
        "interactions": total,
        "quantiles": {
            "min": min(counts),
            "p25": percentile_nearest_rank(counts, 0.25),
            "p50": percentile_nearest_rank(counts, 0.50),
            "p75": percentile_nearest_rank(counts, 0.75),
            "p90": percentile_nearest_rank(counts, 0.90),
            "p95": percentile_nearest_rank(counts, 0.95),
            "p99": percentile_nearest_rank(counts, 0.99),
            "max": max(counts),
        },
        "gini": gini(counts),
        "concentration": concentration,
    }


def history_bucket(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def deterministic_top(scores: np.ndarray, user: int, k: int) -> np.ndarray:
    shuffled = list(range(len(scores)))
    random.Random(SEED + user).shuffle(shuffled)
    shuffled_array = np.asarray(shuffled, dtype=np.int32)
    positions = np.argsort(-scores[shuffled_array], kind="stable")
    return shuffled_array[positions[:k]]


def recall_at_10(top_items: list[int], targets: set[int]) -> float:
    return sum(item in targets for item in top_items) / len(targets)


def exposure_summary(exposure: np.ndarray, candidate_count: int) -> dict[str, float | int]:
    ordered = sorted((int(value) for value in exposure), reverse=True)
    slots = sum(ordered)
    top_one_count = math.ceil(0.01 * candidate_count)
    top_ten_count = math.ceil(0.10 * candidate_count)
    return {
        "top_k": TOP_K,
        "total_slots": slots,
        "unique_recommended_items": sum(value > 0 for value in ordered),
        "catalog_coverage": sum(value > 0 for value in ordered) / candidate_count,
        "top_1pct_candidate_count": top_one_count,
        "top_1pct_exposure_share": sum(ordered[:top_one_count]) / slots,
        "top_10pct_candidate_count": top_ten_count,
        "top_10pct_exposure_share": sum(ordered[:top_ten_count]) / slots,
        "exposure_gini_including_zero": gini(ordered),
        "max_item_exposures": ordered[0],
    }


def svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_concentration_svg(user_stats: dict, item_stats: dict) -> None:
    shares = ("top_1pct", "top_5pct", "top_10pct", "top_20pct")
    labels = ("Top 1%", "Top 5%", "Top 10%", "Top 20%")
    rows = []
    for i, (key, label) in enumerate(zip(shares, labels)):
        y = 135 + i * 64
        user = user_stats["concentration"][key]["interaction_share"]
        item = item_stats["concentration"][key]["interaction_share"]
        rows.append(
            f'<text x="48" y="{y + 17}" class="label">{label}</text>'
            f'<rect x="150" y="{y}" width="{user * 650:.1f}" height="20" rx="4" fill="#38bdf8"/>'
            f'<text x="{160 + user * 650:.1f}" y="{y + 16}" class="value">{user:.1%}</text>'
            f'<rect x="150" y="{y + 26}" width="{item * 650:.1f}" height="14" rx="4" fill="#a78bfa"/>'
            f'<text x="{160 + item * 650:.1f}" y="{y + 39}" class="small">{item:.1%}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" viewBox="0 0 920 440">
<style>.title{{font:700 25px Segoe UI,Arial;fill:#f8fafc}}.sub{{font:15px Segoe UI,Arial;fill:#94a3b8}}.label{{font:15px Segoe UI,Arial;fill:#e2e8f0}}.value{{font:700 13px Segoe UI,Arial;fill:#e0f2fe}}.small{{font:13px Segoe UI,Arial;fill:#ddd6fe}}</style>
<rect width="920" height="440" rx="18" fill="#0f172a"/><text x="48" y="52" class="title">互动分布具有明显长尾</text>
<text x="48" y="82" class="sub">各头部实体贡献的全量评论互动占比</text>{''.join(rows)}
<rect x="48" y="405" width="14" height="14" rx="3" fill="#38bdf8"/><text x="70" y="417" class="sub">用户</text>
<rect x="145" y="405" width="14" height="14" rx="3" fill="#a78bfa"/><text x="167" y="417" class="sub">视频</text></svg>'''
    (ASSETS / "interaction_concentration.svg").write_text(svg, encoding="utf-8")


def write_history_svg(segments: dict[str, dict[str, float | int]]) -> None:
    rows = []
    for i, bucket in enumerate(HISTORY_BUCKETS):
        y = 135 + i * 64
        text_value = float(segments[bucket]["text_recall_at_10"])
        visual_value = float(segments[bucket]["visual_recall_at_10"])
        rows.append(
            f'<text x="48" y="{y + 17}" class="label">{svg_escape(bucket)} 条</text>'
            f'<rect x="150" y="{y}" width="{text_value / 0.24 * 650:.1f}" height="20" rx="4" fill="#38bdf8"/>'
            f'<text x="{160 + text_value / 0.24 * 650:.1f}" y="{y + 16}" class="value">{text_value:.3f}</text>'
            f'<rect x="150" y="{y + 26}" width="{visual_value / 0.24 * 650:.1f}" height="14" rx="4" fill="#a78bfa"/>'
            f'<text x="{160 + visual_value / 0.24 * 650:.1f}" y="{y + 39}" class="small">{visual_value:.3f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" viewBox="0 0 920 440">
<style>.title{{font:700 25px Segoe UI,Arial;fill:#f8fafc}}.sub{{font:15px Segoe UI,Arial;fill:#94a3b8}}.label{{font:15px Segoe UI,Arial;fill:#e2e8f0}}.value{{font:700 13px Segoe UI,Arial;fill:#e0f2fe}}.small{{font:13px Segoe UI,Arial;fill:#ddd6fe}}</style>
<rect width="920" height="440" rx="18" fill="#0f172a"/><text x="48" y="52" class="title">历史越丰富，冷启动匹配越稳定</text>
<text x="48" y="82" class="sub">Validation Cold Recall@10；相关关系，不作因果解释</text>{''.join(rows)}
<rect x="48" y="405" width="14" height="14" rx="3" fill="#38bdf8"/><text x="70" y="417" class="sub">Text-only</text>
<rect x="175" y="405" width="14" height="14" rx="3" fill="#a78bfa"/><text x="197" y="417" class="sub">Visual-only</text></svg>'''
    (ASSETS / "history_segment_recall.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    user_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    with (RAW / "MicroLens-50k_pairs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            user_counts[int(row["user"])] += 1
            item_counts[int(row["item"])] += 1

    train_rows = read_gzip_rows(SPLIT / "dev_train.csv.gz")
    validation_rows = read_gzip_rows(SPLIT / "validation_cold.csv.gz")
    candidates = read_candidates(SPLIT / "validation_cold_candidates.csv.gz")
    train_items = {item for _, item, _ in train_rows}
    if train_items.intersection(candidates):
        raise ValueError("Cold candidates overlap dev-train items")

    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)
    users = sorted(positives)

    titles = load_titles()
    idf = fit_idf(train_items, titles)
    train_text = {item: vectorize(titles[item], idf) for item in train_items}
    candidate_text = {item: vectorize(titles[item], idf) for item in candidates}
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, item in enumerate(candidates):
        for term, weight in candidate_text[item].items():
            postings[term].append((index, weight))

    feature_ids = np.load(FEATURES / "mobilenet_v3_small_item_ids.npy", allow_pickle=False)
    vectors = np.load(FEATURES / "mobilenet_v3_small_vectors.npy", allow_pickle=False)
    feature_rows = {int(item): index for index, item in enumerate(feature_ids)}
    if (train_items | set(candidates)).difference(feature_rows):
        raise ValueError("Missing visual features")
    visual_candidates = vectors[[feature_rows[item] for item in candidates]]

    text_exposure = np.zeros(len(candidates), dtype=np.int64)
    visual_exposure = np.zeros(len(candidates), dtype=np.int64)
    segments = {
        bucket: {"users": 0, "text_recall_sum": 0.0, "visual_recall_sum": 0.0}
        for bucket in HISTORY_BUCKETS
    }
    intersection_sum = 0
    text_empty_profiles = 0
    text_all_zero_scores = 0

    for user in users:
        profile = build_user_profile(histories[user], train_text)
        text_empty_profiles += not profile
        text_scores = np.zeros(len(candidates), dtype=np.float64)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                text_scores[candidate_index] += profile_weight * candidate_weight
        text_all_zero_scores += not np.any(text_scores)
        text_top = deterministic_top(text_scores, user, TOP_K)

        rows = [feature_rows[item] for item in histories[user]]
        visual_profile = vectors[rows].sum(axis=0, dtype=np.float32)
        visual_profile /= np.linalg.norm(visual_profile)
        visual_scores = visual_profile @ visual_candidates.T
        visual_top = deterministic_top(visual_scores, user, TOP_K)

        text_exposure[text_top] += 1
        visual_exposure[visual_top] += 1
        intersection_sum += len(set(text_top).intersection(visual_top))
        text_items = [candidates[index] for index in text_top]
        visual_items = [candidates[index] for index in visual_top]
        bucket = history_bucket(len(histories[user]))
        segments[bucket]["users"] += 1
        segments[bucket]["text_recall_sum"] += recall_at_10(text_items, positives[user])
        segments[bucket]["visual_recall_sum"] += recall_at_10(visual_items, positives[user])

    segment_output = {}
    for bucket in HISTORY_BUCKETS:
        values = segments[bucket]
        count = int(values["users"])
        segment_output[bucket] = {
            "users": count,
            "text_recall_at_10": values["text_recall_sum"] / count,
            "visual_recall_at_10": values["visual_recall_sum"] / count,
        }

    normalized_titles = [
        " ".join(re.findall(r"[a-z0-9]+", titles[item].lower())) for item in item_counts
    ]
    token_counts = [len(value.split()) for value in normalized_titles]
    user_stats = distribution(list(user_counts.values()))
    item_stats = distribution(list(item_counts.values()))
    report = {
        "analysis": "product_analysis_v1",
        "protocol": "docs/PRODUCT_ANALYSIS_PROTOCOL.md",
        "test_split_read": False,
        "model_selection_performed": False,
        "dataset": {
            "interactions": sum(user_counts.values()),
            "users": len(user_counts),
            "items": len(item_counts),
            "user_activity": user_stats,
            "item_activity": item_stats,
            "titles": {
                "items": len(normalized_titles),
                "unique_normalized_titles": len(set(normalized_titles)),
                "empty_after_tokenization": sum(not title for title in normalized_titles),
                "token_count_quantiles": distribution(token_counts)["quantiles"],
            },
        },
        "validation_cold": {
            "dev_train_interactions": len(train_rows),
            "candidate_items": len(candidates),
            "eligible_users": len(users),
            "positive_user_item_pairs": sum(len(value) for value in positives.values()),
            "history_segments": segment_output,
            "text": {
                **exposure_summary(text_exposure, len(candidates)),
                "empty_profile_users": text_empty_profiles,
                "all_zero_score_users": text_all_zero_scores,
            },
            "visual": exposure_summary(visual_exposure, len(candidates)),
            "mean_text_visual_top10_intersection": intersection_sum / len(users),
        },
        "limitations": [
            "Interactions are public comments, not exposure, click, watch, or retention labels.",
            "First observed comment is not the upload timestamp.",
            "Global like/view totals are not used because they are not time-safe model features.",
            "History-segment differences are correlational and may mix activity, preference stability, and target count.",
            "Recommendation exposure diagnostics are Validation-only post-hoc descriptions, not online exposure effects.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_concentration_svg(user_stats, item_stats)
    write_history_svg(segment_output)
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {OUTPUT}")


if __name__ == "__main__":
    main()

"""Explain TF-IDF validation successes and failures without reading the test split."""

from __future__ import annotations

import heapq
import json
import math
import random
import time
from collections import defaultdict

from run_text_tfidf_validation import (
    CONFIG,
    OUTPUT_DIR,
    SPLIT_DIR,
    build_user_profile,
    fit_idf,
    load_titles,
    normalized_title,
    read_candidate_items,
    read_interactions,
    vectorize,
    tokenize,
)


def history_bucket(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def top_contributions(
    profile: dict[str, float], candidate: dict[str, float], limit: int = 6
) -> list[dict[str, float | str]]:
    contributions = [
        (term, profile_weight * candidate[term])
        for term, profile_weight in profile.items()
        if term in candidate
    ]


def token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
    contributions.sort(key=lambda pair: (-pair[1], pair[0]))
    return [
        {"term": term, "contribution": round(value, 6)}
        for term, value in contributions[:limit]
    ]


def case_record(
    user: int,
    history: list[int],
    target: int,
    target_rank: int,
    target_score: float,
    top_item: int,
    top_score: float,
    profile: dict[str, float],
    item_vectors: dict[int, dict[str, float]],
    candidate_vectors: dict[int, dict[str, float]],
    titles: dict[int, str],
) -> dict[str, object]:
    return {
        "user": user,
        "history_length": len(history),
        "recent_history": [
            {"item": item, "title": titles[item]} for item in history[-3:]
        ],
        "best_positive": {
            "item": target,
            "rank": target_rank,
            "score": round(target_score, 6),
            "title": titles[target],
            "shared_terms": top_contributions(profile, candidate_vectors[target]),
            "exact_title_in_history": normalized_title(titles[target])
            in {normalized_title(titles[item]) for item in history},
        },
        "top_recommendation": {
            "item": top_item,
            "score": round(top_score, 6),
            "title": titles[top_item],
            "shared_terms": top_contributions(profile, candidate_vectors[top_item]),
        },
    }


def main() -> None:
    started = time.perf_counter()
    titles = load_titles()
    train_rows = read_interactions(SPLIT_DIR / "dev_train.csv.gz")
    validation_rows = read_interactions(SPLIT_DIR / "validation_cold.csv.gz")
    candidate_items = read_candidate_items(SPLIT_DIR / "validation_cold_candidates.csv.gz")
    train_items = {item for _, item, _ in train_rows}

    histories: defaultdict[int, list[int]] = defaultdict(list)
    for user, item, _ in train_rows:
        histories[user].append(item)
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)

    idf = fit_idf(train_items, titles)
    item_vectors = {item: vectorize(titles[item], idf) for item in train_items}
    candidate_vectors = {item: vectorize(titles[item], idf) for item in candidate_items}
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, item in enumerate(candidate_items):
        for term, weight in candidate_vectors[item].items():
            postings[term].append((index, weight))

    seed = int(CONFIG["evaluation"]["random_seed"])
    buckets: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"users": 0, "recall_10_sum": 0.0, "ndcg_10_sum": 0.0, "hit_rate_10_sum": 0.0}
    )
    strong_hit_pool: list[tuple[float, int, dict[str, object]]] = []
    confident_miss_pool: list[tuple[float, int, dict[str, object]]] = []
    zero_signal_cases: list[dict[str, object]] = []
    positive_pairs_with_zero_score = 0
    best_positive_rank_sum = 0
    near_duplicate_pair_counts = {"jaccard_gte_0_5": 0, "jaccard_gte_0_8": 0}
    filtered_totals = {"users": 0, "positive_pairs": 0, "recall_10_sum": 0.0, "ndcg_10_sum": 0.0, "hit_rate_10_sum": 0.0}
    title_token_sets = {item: set(tokenize(title)) for item, title in titles.items()}

    for user, targets in sorted(positives.items()):
        history = histories[user]
        profile = build_user_profile(history, item_vectors)
        scores = [0.0] * len(candidate_items)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                scores[candidate_index] += profile_weight * candidate_weight

        shuffled_indices = list(range(len(candidate_items)))
        random.Random(seed + user).shuffle(shuffled_indices)
        ranking = sorted(shuffled_indices, key=scores.__getitem__, reverse=True)
        rank_by_item = {candidate_items[index]: rank for rank, index in enumerate(ranking, start=1)}
        best_target = min(targets, key=lambda item: rank_by_item[item])
        best_rank = rank_by_item[best_target]
        best_positive_rank_sum += best_rank
        top_item = candidate_items[ranking[0]]
        positive_pairs_with_zero_score += sum(
            scores[candidate_items.index(target)] == 0 for target in targets
        )

        top_10 = {candidate_items[index] for index in ranking[:10]}
        hit_count = len(targets & top_10)
        hits = [1 if candidate_items[index] in targets else 0 for index in ranking[:10]]
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        idcg = sum(1 / math.log2(index + 2) for index in range(min(len(targets), 10)))
        bucket = buckets[history_bucket(len(history))]
        bucket["users"] += 1
        bucket["recall_10_sum"] += hit_count / len(targets)
        bucket["ndcg_10_sum"] += dcg / idcg
        bucket["hit_rate_10_sum"] += float(hit_count > 0)

        target_near_duplicate = {}
        history_token_sets = [title_token_sets[item] for item in history]
        for target in targets:
            max_similarity = max(
                (token_jaccard(title_token_sets[target], history_tokens) for history_tokens in history_token_sets),
                default=0.0,
            )
            target_near_duplicate[target] = max_similarity
            near_duplicate_pair_counts["jaccard_gte_0_5"] += max_similarity >= 0.5
            near_duplicate_pair_counts["jaccard_gte_0_8"] += max_similarity >= 0.8

        non_near_targets = {
            target for target, similarity in target_near_duplicate.items() if similarity < 0.8
        }
        if non_near_targets:
            filtered_hits = [
                1 if candidate_items[index] in non_near_targets else 0 for index in ranking[:10]
            ]
            filtered_hit_count = sum(filtered_hits)
            filtered_dcg = sum(
                hit / math.log2(index + 2) for index, hit in enumerate(filtered_hits)
            )
            filtered_idcg = sum(
                1 / math.log2(index + 2) for index in range(min(len(non_near_targets), 10))
            )
            filtered_totals["users"] += 1
            filtered_totals["positive_pairs"] += len(non_near_targets)
            filtered_totals["recall_10_sum"] += filtered_hit_count / len(non_near_targets)
            filtered_totals["ndcg_10_sum"] += filtered_dcg / filtered_idcg
            filtered_totals["hit_rate_10_sum"] += float(filtered_hit_count > 0)

        record = case_record(
            user,
            history,
            best_target,
            best_rank,
            scores[candidate_items.index(best_target)],
            top_item,
            scores[ranking[0]],
            profile,
            item_vectors,
            candidate_vectors,
            titles,
        )
        exact_match = record["best_positive"]["exact_title_in_history"]  # type: ignore[index]
        if best_rank <= 3 and not exact_match:
            heapq.heappush(strong_hit_pool, (scores[candidate_items.index(best_target)], user, record))
        if best_rank > 50 and scores[ranking[0]] > 0:
            heapq.heappush(confident_miss_pool, (scores[ranking[0]], user, record))
        if not any(scores) and len(zero_signal_cases) < 5:
            zero_signal_cases.append(record)

    stratified = {}
    for name in ("1-3", "4-6", "7-10", "11+"):
        bucket = buckets[name]
        users = int(bucket["users"])
        stratified[name] = {
            "users": users,
            "Recall@10": bucket["recall_10_sum"] / users,
            "NDCG@10": bucket["ndcg_10_sum"] / users,
            "HitRate@10": bucket["hit_rate_10_sum"] / users,
        }

    strong_hits = [record for _, _, record in heapq.nlargest(5, strong_hit_pool, key=lambda pair: pair[0])]
    confident_misses = [
        record for _, _, record in heapq.nlargest(5, confident_miss_pool, key=lambda pair: pair[0])
    ]
    filtered_users = int(filtered_totals["users"])
    report = {
        "analysis": "text_tfidf_validation_error_analysis_v1",
        "protocol": CONFIG["version"],
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "summary": {
            "users": len(positives),
            "positive_pairs": sum(len(items) for items in positives.values()),
            "positive_pairs_with_zero_text_similarity": positive_pairs_with_zero_score,
            "mean_best_positive_rank": best_positive_rank_sum / len(positives),
        },
        "near_duplicate_audit": {
            "definition": "maximum word-token Jaccard similarity against the same user's training-history titles",
            **near_duplicate_pair_counts,
            "metrics_after_excluding_positive_pairs_with_jaccard_gte_0_8": {
                "users": filtered_users,
                "positive_pairs": int(filtered_totals["positive_pairs"]),
                "Recall@10": filtered_totals["recall_10_sum"] / filtered_users,
                "NDCG@10": filtered_totals["ndcg_10_sum"] / filtered_users,
                "HitRate@10": filtered_totals["hit_rate_10_sum"] / filtered_users,
            },
        },
        "by_training_history_length": stratified,
        "representative_cases": {
            "strong_hits_without_exact_title_copy": strong_hits,
            "confident_misses": confident_misses,
            "all_candidate_scores_zero": zero_signal_cases,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "text_tfidf_validation_error_analysis_v1.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "representative_cases"}, indent=2))
    print("\nRepresentative case counts:")
    for name, cases in report["representative_cases"].items():
        print(f"  {name}: {len(cases)}")
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

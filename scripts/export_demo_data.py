"""Export deterministic, anonymized Validation-only samples for the local demo."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import random
from collections import defaultdict
from pathlib import Path

from run_text_tfidf_validation import (
    CONFIG,
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "processed" / "split_v1"
OUTPUT = ROOT / "demo" / "public" / "demo-data.json"
TOP_K = 10
MAX_HISTORY_TITLES = 6
MAX_POSITIVE_TITLES = 4
MAX_TITLE_CHARS = 180
HISTORY_ITEMS_PER_TOPIC = 60
BUCKETS = ("1-3", "4-6", "7-10", "11+")
SANDBOX_PRESETS = (
    {"id": "anime-game", "label": "动漫游戏", "keywords": "anime game naruto"},
    {"id": "food", "label": "美食", "keywords": "food spicy"},
    {"id": "pets", "label": "萌宠", "keywords": "cat dog pet"},
    {"id": "sports", "label": "运动", "keywords": "basketball football fitness"},
    {"id": "music-dance", "label": "音乐舞蹈", "keywords": "music dance song"},
    {"id": "film-drama", "label": "影视剧情", "keywords": "movie drama story"},
)
INTEREST_TERM_GROUPS = (
    {
        "id": "anime-game",
        "label": "动漫游戏",
        "terms": ("anime", "game", "naruto", "dragon", "hero", "battle", "magic", "cartoon", "animation", "war"),
    },
    {
        "id": "food",
        "label": "美食",
        "terms": ("food", "spicy", "eat", "chicken", "cake", "noodles", "meat", "rice", "delicious"),
    },
    {
        "id": "pets",
        "label": "萌宠",
        "terms": ("cat", "dog", "pet", "cute", "animal", "bird", "fish", "baby", "funny", "rabbit"),
    },
    {
        "id": "sports",
        "label": "运动",
        "terms": ("basketball", "football", "fitness", "sports", "run", "running", "training", "player", "game"),
    },
    {
        "id": "music-dance",
        "label": "音乐舞蹈",
        "terms": ("music", "dance", "song", "singing", "singer", "piano", "band", "performance", "stage"),
    },
    {
        "id": "film-drama",
        "label": "影视剧情",
        "terms": ("movie", "drama", "story", "film", "actor", "scene", "comedy", "episode", "love", "family", "life"),
    },
)


def read_rows(path: Path) -> list[tuple[int, int, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [
            (int(row["user"]), int(row["item"]), int(row["timestamp"]))
            for row in csv.DictReader(handle)
        ]


def read_candidates(path: Path) -> list[int]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [int(row["item"]) for row in csv.DictReader(handle)]


def bucket_for(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 10:
        return "7-10"
    return "11+"


def shorten(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) <= MAX_TITLE_CHARS:
        return clean
    return clean[: MAX_TITLE_CHARS - 1].rstrip() + "…"


def selection_hash(user: int) -> str:
    return hashlib.sha256(f"coldlens_demo_v1:{user}".encode()).hexdigest()


def history_selection_hash(topic_id: str, item: int) -> str:
    return hashlib.sha256(
        f"coldlens_history_catalog_v1:{topic_id}:{item}".encode()
    ).hexdigest()


def top_order(scores: list[float], user: int) -> list[int]:
    shuffled = list(range(len(scores)))
    random.Random(int(CONFIG["evaluation"]["random_seed"]) + user).shuffle(shuffled)
    return heapq.nlargest(TOP_K, shuffled, key=scores.__getitem__)


def explain_terms(
    profile: dict[str, float], candidate: dict[str, float]
) -> list[dict[str, float | str]]:
    contributions = [
        (term, profile_weight * candidate[term])
        for term, profile_weight in profile.items()
        if term in candidate
    ]
    contributions.sort(key=lambda pair: (-pair[1], pair[0]))
    return [
        {"term": term, "contribution": round(value, 6)}
        for term, value in contributions[:3]
    ]


def main() -> None:
    train_rows = read_rows(SPLIT / "dev_train.csv.gz")
    validation_rows = read_rows(SPLIT / "validation_cold.csv.gz")
    candidates = read_candidates(SPLIT / "validation_cold_candidates.csv.gz")
    train_items = {item for _, item, _ in train_rows}
    if train_items.intersection(candidates):
        raise ValueError("Validation Cold candidates overlap dev-train")

    histories: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for user, item, timestamp in train_rows:
        histories[user].append((timestamp, item))
    positives: defaultdict[int, set[int]] = defaultdict(set)
    for user, item, _ in validation_rows:
        positives[user].add(item)

    titles = load_titles()
    idf = fit_idf(train_items, titles)
    train_vectors = {item: vectorize(titles[item], idf) for item in train_items}
    candidate_vectors = {item: vectorize(titles[item], idf) for item in candidates}
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, item in enumerate(candidates):
        for term, weight in candidate_vectors[item].items():
            postings[term].append((index, weight))

    evaluated: dict[int, dict[str, object]] = {}
    selection_groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    total_recall = 0.0
    total_ndcg_hits = 0
    for user in sorted(positives):
        ordered_history = sorted(histories[user])
        history_items = [item for _, item in ordered_history]
        profile = build_user_profile(history_items, train_vectors)
        scores = [0.0] * len(candidates)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                scores[candidate_index] += profile_weight * candidate_weight
        top_indices = top_order(scores, user)
        top_items = [candidates[index] for index in top_indices]
        hits = [item for item in top_items if item in positives[user]]
        outcome = "hit" if hits else "miss"
        bucket = bucket_for(len(history_items))
        selection_groups[(bucket, outcome)].append(user)
        total_recall += len(hits) / len(positives[user])
        total_ndcg_hits += int(bool(hits))
        evaluated[user] = {
            "history_items": history_items,
            "profile": profile,
            "scores": scores,
            "top_indices": top_indices,
            "bucket": bucket,
            "outcome": outcome,
        }

    chosen: list[int] = []
    for bucket in BUCKETS:
        for outcome in ("hit", "miss"):
            group = selection_groups[(bucket, outcome)]
            if not group:
                raise ValueError(f"No eligible demo user for {bucket}/{outcome}")
            chosen.append(min(group, key=selection_hash))

    samples = []
    for position, user in enumerate(chosen):
        details = evaluated[user]
        history_items = details["history_items"]
        profile = details["profile"]
        top_indices = details["top_indices"]
        scores = details["scores"]
        recommendations = []
        for rank, candidate_index in enumerate(top_indices, start=1):
            item = candidates[candidate_index]
            recommendations.append(
                {
                    "rank": rank,
                    "title": shorten(titles[item]),
                    "score": round(float(scores[candidate_index]), 6),
                    "is_validation_positive": item in positives[user],
                    "matched_terms": explain_terms(profile, candidate_vectors[item]),
                }
            )
        samples.append(
            {
                "id": f"U{position + 1}",
                "history_bucket": details["bucket"],
                "history_count": len(history_items),
                "sample_outcome": details["outcome"],
                "history_titles": [
                    shorten(titles[item]) for item in history_items[-MAX_HISTORY_TITLES:]
                ],
                "validation_positive_titles": [
                    shorten(titles[item])
                    for item in sorted(positives[user])[:MAX_POSITIVE_TITLES]
                ],
                "recommendations": recommendations,
            }
        )

    candidate_terms = sorted(
        {
            term
            for vector in candidate_vectors.values()
            for term in vector
        }
    )
    missing_preset_terms = sorted(
        {
            term
            for preset in SANDBOX_PRESETS
            for term in preset["keywords"].split()
            if term not in candidate_terms
        }
    )
    if missing_preset_terms:
        raise ValueError(f"Sandbox preset terms missing from candidates: {missing_preset_terms}")
    missing_interest_terms = sorted(
        {
            term
            for group in INTEREST_TERM_GROUPS
            for term in group["terms"]
            if term not in candidate_terms
        }
    )
    if missing_interest_terms:
        raise ValueError(
            f"Interest-library terms missing from candidates: {missing_interest_terms}"
        )

    reference_queries = []
    for preset in SANDBOX_PRESETS:
        query_vector = vectorize(preset["keywords"], idf)
        scores = [
            sum(
                query_weight * candidate_vectors[item].get(term, 0.0)
                for term, query_weight in query_vector.items()
            )
            for item in candidates
        ]
        expected_top = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )[:TOP_K]
        reference_queries.append(
            {
                "id": preset["id"],
                "keywords": preset["keywords"],
                "expected_top_candidate_indices": expected_top,
            }
        )

    history_catalog = []
    used_history_items: set[int] = set()
    for preset in SANDBOX_PRESETS:
        topic_terms = set(preset["keywords"].split())
        eligible = [
            item
            for item in train_items
            if item not in used_history_items
            and train_vectors[item]
            and topic_terms.intersection(train_vectors[item])
        ]
        eligible.sort(key=lambda item: history_selection_hash(preset["id"], item))
        chosen_items = eligible[:HISTORY_ITEMS_PER_TOPIC]
        if len(chosen_items) < HISTORY_ITEMS_PER_TOPIC:
            raise ValueError(
                f"Not enough history catalog items for {preset['id']}: "
                f"{len(chosen_items)}"
            )
        for item in chosen_items:
            used_history_items.add(item)
            history_catalog.append(
                {
                    "id": f"H{len(history_catalog) + 1:03d}",
                    "topic_id": preset["id"],
                    "topic_label": preset["label"],
                    "title": shorten(titles[item]),
                    "vector": [
                        [term, weight]
                        for term, weight in sorted(train_vectors[item].items())
                    ],
                }
            )

    reference_history_ids = [
        next(item["id"] for item in history_catalog if item["topic_id"] == topic_id)
        for topic_id in ("anime-game", "sports", "film-drama")
    ]
    reference_history_items = [
        next(
            item
            for item in history_catalog
            if item["id"] == history_id
        )
        for history_id in reference_history_ids
    ]
    reference_profile: defaultdict[str, float] = defaultdict(float)
    for history_item in reference_history_items:
        for term, weight in history_item["vector"]:
            reference_profile[term] += weight
    reference_norm = sum(weight * weight for weight in reference_profile.values()) ** 0.5
    normalized_reference_profile = {
        term: weight / reference_norm for term, weight in reference_profile.items()
    }
    history_scores = [
        sum(
            profile_weight * candidate_vectors[item].get(term, 0.0)
            for term, profile_weight in normalized_reference_profile.items()
        )
        for item in candidates
    ]
    history_expected_top = sorted(
        (index for index, score in enumerate(history_scores) if score > 0),
        key=lambda index: (-history_scores[index], index),
    )[:TOP_K]
    history_expected_evidence_ids = []
    for candidate_index in history_expected_top:
        candidate_vector = candidate_vectors[candidates[candidate_index]]
        evidence_scores = [
            sum(
                weight * candidate_vector.get(term, 0.0)
                for term, weight in history_item["vector"]
            )
            for history_item in reference_history_items
        ]
        evidence_index = max(
            range(len(reference_history_items)),
            key=lambda index: (evidence_scores[index], -index),
        )
        history_expected_evidence_ids.append(
            reference_history_items[evidence_index]["id"]
        )

    report = {
        "demo": "coldlens_demo_v1",
        "protocol": "docs/DEMO_PROTOCOL.md",
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "model": "Text-only TF-IDF",
        "overall_validation": {
            "users": len(positives),
            "candidate_items": len(candidates),
            "positive_pairs": sum(len(items) for items in positives.values()),
            "recall_at_10": round(total_recall / len(positives), 6),
            "hit_rate_at_10": round(total_ndcg_hits / len(positives), 6),
        },
        "selection_note": (
            "Two deterministic samples per history bucket: one Top-10 hit and one miss. "
            "The eight samples are balanced for explanation and are not prevalence estimates."
        ),
        "interpretation_boundary": (
            "Public-comment preference matching; not exposure, CTR, watch time, retention, "
            "or online causal lift. Scores are cosine similarities, not probabilities."
        ),
        "samples": samples,
        "sandbox": {
            "mode": "interest_text_or_selected_history_profile",
            "candidate_items": len(candidates),
            "tokenization": "lowercase ASCII letters and digits",
            "input_language_boundary": (
                "The frozen tokenizer and translated title field support English/digit tokens; "
                "this is not Chinese semantic understanding."
            ),
            "tie_breaking": "descending score, then frozen candidate order",
            "presets": list(SANDBOX_PRESETS),
            "interest_term_groups": [
                {**group, "terms": list(group["terms"])}
                for group in INTEREST_TERM_GROUPS
            ],
            "reference_queries": reference_queries,
            "history_catalog_note": (
                "Deterministic, topic-balanced training-item sample for mechanism demonstration; "
                "not a representative catalog or a new evaluation set."
            ),
            "history_catalog": history_catalog,
            "reference_history_scenario": {
                "selected_history_ids": reference_history_ids,
                "expected_top_candidate_indices": history_expected_top,
                "expected_evidence_history_ids": history_expected_evidence_ids,
            },
            "idf": {term: idf[term] for term in candidate_terms},
            "candidates": [
                {
                    "title": shorten(titles[item]),
                    "vector": [
                        [term, weight]
                        for term, weight in sorted(candidate_vectors[item].items())
                    ],
                }
                for item in candidates
            ],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "samples": len(samples),
                "buckets": [sample["history_bucket"] for sample in samples],
                "outcomes": [sample["sample_outcome"] for sample in samples],
                "test_split_read": False,
                "sandbox_candidates": len(candidates),
                "sandbox_presets": len(SANDBOX_PRESETS),
                "sandbox_interest_terms": sum(
                    len(group["terms"]) for group in INTEREST_TERM_GROUPS
                ),
                "sandbox_history_items": len(history_catalog),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Run a zero-dependency TF-IDF title baseline on Validation Cold only.

The final test split is deliberately not read by this script.
"""

from __future__ import annotations

import csv
import gzip
import heapq
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
TITLE_PATH = ROOT / "data" / "raw" / "microlens50k" / "MicroLens-50k_titles.csv"
CONFIG = json.loads((ROOT / "configs" / "split_v1.json").read_text(encoding="utf-8"))
OUTPUT_DIR = ROOT / "outputs" / "experiments"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MIN_DOCUMENT_FREQUENCY = 2


def tokenize(title: str) -> list[str]:
    """Use lowercase alphanumeric word tokens; punctuation and hashtag markers are ignored."""
    return TOKEN_PATTERN.findall(title.lower())


def normalized_title(title: str) -> str:
    return " ".join(tokenize(title))


def load_titles() -> dict[int, str]:
    with TITLE_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["item"]): row["title"] for row in csv.DictReader(handle)}


def read_interactions(path: Path) -> list[tuple[int, int, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [
            (int(row["user"]), int(row["item"]), int(row["timestamp"]))
            for row in csv.DictReader(handle)
        ]


def read_candidate_items(path: Path) -> list[int]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [int(row["item"]) for row in csv.DictReader(handle)]


def fit_idf(train_items: set[int], titles: dict[int, str]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for item in train_items:
        document_frequency.update(set(tokenize(titles[item])))
    document_count = len(train_items)
    return {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
        if frequency >= MIN_DOCUMENT_FREQUENCY
    }


def vectorize(title: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokenize(title) if token in idf)
    weights = {term: (1 + math.log(count)) * idf[term] for term, count in counts.items()}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm:
        return {term: weight / norm for term, weight in weights.items()}
    return {}


def build_user_profile(history_items: list[int], item_vectors: dict[int, dict[str, float]]) -> dict[str, float]:
    profile: defaultdict[str, float] = defaultdict(float)
    used_items = 0
    for item in history_items:
        vector = item_vectors[item]
        if not vector:
            continue
        used_items += 1
        for term, weight in vector.items():
            profile[term] += weight
    if not used_items:
        return {}
    norm = math.sqrt(sum(weight * weight for weight in profile.values()))
    return {term: weight / norm for term, weight in profile.items()}


def evaluate(
    candidate_items: list[int],
    candidate_vectors: dict[int, dict[str, float]],
    histories: dict[int, list[int]],
    item_vectors: dict[int, dict[str, float]],
    positives: dict[int, set[int]],
) -> dict[str, float | int]:
    postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, item in enumerate(candidate_items):
        for term, weight in candidate_vectors[item].items():
            postings[term].append((index, weight))

    k_values = [int(value) for value in CONFIG["evaluation"]["k_values"]]
    max_k = max(k_values)
    totals = {f"Recall@{k}": 0.0 for k in k_values}
    totals.update({f"NDCG@{k}": 0.0 for k in k_values})
    totals.update({f"HitRate@{k}": 0.0 for k in k_values})
    users_with_empty_profile = 0
    users_with_all_zero_scores = 0
    seed = int(CONFIG["evaluation"]["random_seed"])

    for user, targets in sorted(positives.items()):
        profile = build_user_profile(histories[user], item_vectors)
        users_with_empty_profile += not profile
        scores = [0.0] * len(candidate_items)
        for term, profile_weight in profile.items():
            for candidate_index, candidate_weight in postings.get(term, []):
                scores[candidate_index] += profile_weight * candidate_weight
        users_with_all_zero_scores += not any(scores)

        # Randomized but reproducible ordering removes item-ID bias when scores tie.
        shuffled_indices = list(range(len(candidate_items)))
        random.Random(seed + user).shuffle(shuffled_indices)
        top_indices = heapq.nlargest(max_k, shuffled_indices, key=scores.__getitem__)
        top_items = [candidate_items[index] for index in top_indices]

        for k in k_values:
            hits = [1 if item in targets else 0 for item in top_items[:k]]
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
        "candidate_items": len(candidate_items),
        "positive_user_item_pairs": sum(len(items) for items in positives.values()),
        "users_with_empty_profile": users_with_empty_profile,
        "users_with_all_zero_scores": users_with_all_zero_scores,
    }
    result.update({metric: total / user_count for metric, total in totals.items()})
    return result


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
    train_normalized_titles = {normalized_title(titles[item]) for item in train_items}
    candidate_normalized_counts = Counter(normalized_title(titles[item]) for item in candidate_items)
    exact_history_match_pairs = 0
    for user, targets in positives.items():
        history_titles = {normalized_title(titles[item]) for item in histories[user]}
        exact_history_match_pairs += sum(
            normalized_title(titles[target]) in history_titles for target in targets
        )
    metrics = evaluate(
        candidate_items,
        candidate_vectors,
        histories,
        item_vectors,
        positives,
    )

    report = {
        "experiment": "text_tfidf_validation_v1",
        "protocol": CONFIG["version"],
        "evaluated_split": "validation_cold",
        "test_split_read": False,
        "model": {
            "text": "English-translated video title",
            "tokenization": "lowercase alphanumeric word unigrams",
            "minimum_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "idf_fit_items": "dev_train items only",
            "user_profile": "L2-normalized sum of L2-normalized historical title vectors",
            "score": "cosine similarity",
            "tie_breaking": "deterministic per-user random order",
            "vocabulary_size": len(idf),
            "zero_vector_train_items": sum(not vector for vector in item_vectors.values()),
            "zero_vector_candidate_items": sum(not vector for vector in candidate_vectors.values()),
        },
        "title_overlap_audit": {
            "candidate_items_with_exact_title_in_dev_train": sum(
                normalized_title(titles[item]) in train_normalized_titles for item in candidate_items
            ),
            "candidate_rows_beyond_first_with_duplicate_title_inside_candidate_pool": sum(
                count - 1 for count in candidate_normalized_counts.values()
            ),
            "positive_user_item_pairs_with_exact_title_in_user_history": exact_history_match_pairs,
        },
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Titles are English translations/transliterations rather than original-language raw text.",
            "TF-IDF ignores word order, semantics, negation, and visual content.",
            "Comment positives are not exposure, click, or watch labels.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "text_tfidf_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

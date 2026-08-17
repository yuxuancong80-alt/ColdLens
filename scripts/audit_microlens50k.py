"""Audit MicroLens-50K metadata before any modeling or media download."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "microlens50k"
REPORT_DIR = ROOT / "outputs" / "m0"


def iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def quantile(sorted_values: list[int], fraction: float) -> int:
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def summarize_counts(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "p90": quantile(ordered, 0.90),
        "max": ordered[-1],
    }


def load_interactions() -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int]]]]:
    interactions: list[tuple[int, int, int]] = []
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with (RAW / "MicroLens-50k_pairs.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user", "item", "timestamp"]:
            raise ValueError(f"Unexpected interaction header: {reader.fieldnames}")
        for row in reader:
            user, item, timestamp = int(row["user"]), int(row["item"]), int(row["timestamp"])
            interactions.append((user, item, timestamp))
            by_user[user].append((timestamp, item))
    return interactions, by_user


def load_tsv_sequences() -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    with (RAW / "MicroLens-50k_pairs.tsv").open("r", encoding="utf-8") as handle:
        for line in handle:
            user_text, items_text = line.rstrip("\n").split("\t", 1)
            user = int(user_text)
            if user in sequences:
                raise ValueError(f"Duplicate user in TSV: {user}")
            sequences[user] = [int(value) for value in items_text.split()]
    return sequences


def load_titles() -> tuple[dict[int, str], int]:
    titles: dict[int, str] = {}
    duplicate_ids = 0
    with (RAW / "MicroLens-50k_titles.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["item", "title"]:
            raise ValueError(f"Unexpected title header: {reader.fieldnames}")
        for row in reader:
            item = int(row["item"])
            duplicate_ids += item in titles
            titles[item] = row["title"].strip()
    return titles, duplicate_ids


def load_engagement() -> tuple[dict[int, tuple[int, int]], int]:
    engagement: dict[int, tuple[int, int]] = {}
    duplicate_ids = 0
    with (RAW / "MicroLens-50k_likes_and_views.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            item_text, likes_text, views_text = line.rstrip("\n").split("\t")
            item = int(item_text)
            duplicate_ids += item in engagement
            engagement[item] = (int(likes_text), int(views_text))
    return engagement, duplicate_ids


def main() -> None:
    interactions, by_user = load_interactions()
    tsv_sequences = load_tsv_sequences()
    titles, duplicate_title_ids = load_titles()
    engagement, duplicate_engagement_ids = load_engagement()

    users = {user for user, _, _ in interactions}
    items = {item for _, item, _ in interactions}
    timestamps = sorted(timestamp for _, _, timestamp in interactions)
    user_counts = Counter(user for user, _, _ in interactions)
    item_counts = Counter(item for _, item, _ in interactions)
    pair_counts = Counter((user, item) for user, item, _ in interactions)

    csv_sequences = {
        user: [item for _, item in sorted(events)] for user, events in by_user.items()
    }
    mismatched_users = [
        user for user in users | set(tsv_sequences) if csv_sequences.get(user) != tsv_sequences.get(user)
    ]
    non_monotonic_users = sum(
        any(events[index][0] > events[index + 1][0] for index in range(len(events) - 1))
        for events in by_user.values()
    )

    train_cutoff = quantile(timestamps, 0.80)
    validation_cutoff = quantile(timestamps, 0.90)
    train_items = {item for _, item, timestamp in interactions if timestamp < train_cutoff}
    train_users = {user for user, _, timestamp in interactions if timestamp < train_cutoff}
    validation_cold = [
        (user, item, timestamp)
        for user, item, timestamp in interactions
        if train_cutoff <= timestamp < validation_cutoff and item not in train_items
    ]
    pre_test_items = {item for _, item, timestamp in interactions if timestamp < validation_cutoff}
    test_cold = [
        (user, item, timestamp)
        for user, item, timestamp in interactions
        if timestamp >= validation_cutoff and item not in pre_test_items
    ]

    report = {
        "source": {
            "dataset": "MicroLens-50K",
            "files": sorted(path.name for path in RAW.iterdir() if path.is_file()),
        },
        "interactions": {
            "rows": len(interactions),
            "unique_users": len(users),
            "unique_items": len(items),
            "duplicate_user_item_rows_beyond_first": sum(count - 1 for count in pair_counts.values()),
            "users_with_non_monotonic_csv_row_order": non_monotonic_users,
            "timestamp_min_ms": timestamps[0],
            "timestamp_min_utc": iso_utc(timestamps[0]),
            "timestamp_max_ms": timestamps[-1],
            "timestamp_max_utc": iso_utc(timestamps[-1]),
            "user_interactions": summarize_counts(list(user_counts.values())),
            "item_interactions": summarize_counts(list(item_counts.values())),
        },
        "csv_tsv_consistency": {
            "tsv_users": len(tsv_sequences),
            "mismatched_user_sequences": len(mismatched_users),
            "mismatch_examples": mismatched_users[:10],
        },
        "content_coverage": {
            "title_rows": len(titles),
            "duplicate_title_ids": duplicate_title_ids,
            "empty_titles": sum(not title for title in titles.values()),
            "interaction_items_missing_title": len(items - set(titles)),
            "title_items_without_interaction": len(set(titles) - items),
            "engagement_rows": len(engagement),
            "duplicate_engagement_ids": duplicate_engagement_ids,
            "interaction_items_missing_engagement": len(items - set(engagement)),
            "engagement_items_without_interaction": len(set(engagement) - items),
        },
        "temporal_cold_feasibility": {
            "split_rule": "global interaction timestamp quantiles; boundaries use < then >=",
            "train_cutoff_ms": train_cutoff,
            "train_cutoff_utc": iso_utc(train_cutoff),
            "validation_cutoff_ms": validation_cutoff,
            "validation_cutoff_utc": iso_utc(validation_cutoff),
            "train_rows": sum(timestamp < train_cutoff for _, _, timestamp in interactions),
            "train_users": len(train_users),
            "train_items": len(train_items),
            "validation_cold_rows": len(validation_cold),
            "validation_cold_users": len({user for user, _, _ in validation_cold}),
            "validation_cold_users_with_train_history": len(
                {user for user, _, _ in validation_cold if user in train_users}
            ),
            "validation_cold_items": len({item for _, item, _ in validation_cold}),
            "test_cold_rows": len(test_cold),
            "test_cold_users": len({user for user, _, _ in test_cold}),
            "test_cold_users_with_train_history": len(
                {user for user, _, _ in test_cold if user in train_users}
            ),
            "test_cold_items": len({item for _, item, _ in test_cold}),
            "test_cold_items_missing_title": len({item for _, item, _ in test_cold} - set(titles)),
        },
        "known_limitations": [
            "First observed interaction is not the same as the video's upload time.",
            "The source paper says seed videos were popularity-filtered, so this is model-cold rather than a representative sample of newly uploaded videos.",
            "Global item engagement statistics must not be used as training features because they summarize information outside the training window.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "microlens50k_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

"""Build deterministic temporal leave-one-out pairs for learned late fusion."""

from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_text_tfidf_validation import (
    build_user_profile,
    fit_idf,
    load_titles,
    vectorize,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data" / "processed" / "split_v1" / "dev_train.csv.gz"
FEATURE_DIR = ROOT / "artifacts" / "features"
OUTPUT_DIR = ROOT / "artifacts" / "learned_fusion"
PAIR_PATH = OUTPUT_DIR / "temporal_leave_one_out_pairs_v1.npz"
METADATA_PATH = OUTPUT_DIR / "temporal_leave_one_out_pairs_v1.json"
NEGATIVES_PER_POSITIVE = 5
SEED = 20260814
PARTITION_SALT = "learned-fusion-v1"


def read_interactions() -> list[tuple[int, int, int]]:
    with gzip.open(SPLIT_PATH, "rt", encoding="utf-8", newline="") as handle:
        return [
            (int(row["user"]), int(row["item"]), int(row["timestamp"]))
            for row in csv.DictReader(handle)
        ]


def is_calibration_user(user: int) -> bool:
    digest = hashlib.sha256(f"{PARTITION_SALT}:{user}".encode("ascii")).digest()
    return digest[0] < 26


def sparse_dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def candidate_features(
    item: int,
    text_profile: dict[str, float],
    text_vectors: dict[int, dict[str, float]],
    visual_profile: np.ndarray,
    visual_vectors: np.ndarray,
    visual_rows: dict[int, int],
) -> np.ndarray:
    text_score = sparse_dot(text_profile, text_vectors[item])
    visual_score = float(np.dot(visual_profile, visual_vectors[visual_rows[item]]))
    return np.asarray(
        [text_score, visual_score, text_score * visual_score], dtype=np.float32
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    rows = read_interactions()
    histories: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    first_seen: dict[int, int] = {}
    for user, item, timestamp in rows:
        histories[user].append((timestamp, item))
        first_seen[item] = min(timestamp, first_seen.get(item, timestamp))
    for sequence in histories.values():
        sequence.sort()

    train_items = set(first_seen)
    titles = load_titles()
    idf = fit_idf(train_items, titles)
    text_vectors = {item: vectorize(titles[item], idf) for item in train_items}
    visual_ids = np.load(
        FEATURE_DIR / "mobilenet_v3_small_item_ids.npy", allow_pickle=False
    )
    visual_vectors = np.load(
        FEATURE_DIR / "mobilenet_v3_small_vectors.npy", allow_pickle=False
    )
    visual_rows = {int(item): row for row, item in enumerate(visual_ids)}
    if train_items.difference(visual_rows):
        raise ValueError("Missing visual features for one or more dev-train items")

    available_items = sorted((timestamp, item) for item, timestamp in first_seen.items())
    available_times = [timestamp for timestamp, _ in available_items]
    eligible_users = sorted(user for user, sequence in histories.items() if len(sequence) >= 2)
    pair_count = len(eligible_users) * NEGATIVES_PER_POSITIVE
    positive_features = np.empty((pair_count, 3), dtype=np.float32)
    negative_features = np.empty_like(positive_features)
    pair_users = np.empty(pair_count, dtype=np.int32)
    calibration_mask = np.empty(pair_count, dtype=bool)

    empty_text_profiles = 0
    minimum_negative_pool = len(train_items)
    cursor = 0
    for user_index, user in enumerate(eligible_users, start=1):
        sequence = histories[user]
        target_timestamp, target_item = sequence[-1]
        history_items = [item for _, item in sequence[:-1]]
        all_user_items = {item for _, item in sequence}
        if target_item in history_items:
            raise ValueError(f"Target self-inclusion for user {user}")

        text_profile = build_user_profile(history_items, text_vectors)
        empty_text_profiles += not text_profile
        visual_profile = visual_vectors[[visual_rows[item] for item in history_items]].sum(
            axis=0, dtype=np.float32
        )
        visual_norm = float(np.linalg.norm(visual_profile))
        if visual_norm == 0:
            raise ValueError(f"Empty visual profile for user {user}")
        visual_profile /= visual_norm
        positive = candidate_features(
            target_item,
            text_profile,
            text_vectors,
            visual_profile,
            visual_vectors,
            visual_rows,
        )

        available_count = bisect.bisect_right(available_times, target_timestamp)
        negative_pool_size = available_count - len(all_user_items)
        minimum_negative_pool = min(minimum_negative_pool, negative_pool_size)
        if negative_pool_size < NEGATIVES_PER_POSITIVE:
            raise ValueError(f"Insufficient negative pool for user {user}")
        generator = random.Random(SEED + user)
        negatives: list[int] = []
        negative_set: set[int] = set()
        while len(negatives) < NEGATIVES_PER_POSITIVE:
            _, candidate = available_items[generator.randrange(available_count)]
            if candidate in all_user_items or candidate in negative_set:
                continue
            negative_set.add(candidate)
            negatives.append(candidate)

        is_calibration = is_calibration_user(user)
        for negative_item in negatives:
            if first_seen[negative_item] > target_timestamp:
                raise ValueError("Negative item appears after target timestamp")
            positive_features[cursor] = positive
            negative_features[cursor] = candidate_features(
                negative_item,
                text_profile,
                text_vectors,
                visual_profile,
                visual_vectors,
                visual_rows,
            )
            pair_users[cursor] = user
            calibration_mask[cursor] = is_calibration
            cursor += 1
        if user_index % 10_000 == 0:
            print(f"users_processed={user_index}/{len(eligible_users)}", flush=True)

    if cursor != pair_count:
        raise ValueError(f"Pair count mismatch: {cursor} != {pair_count}")
    if not np.isfinite(positive_features).all() or not np.isfinite(
        negative_features
    ).all():
        raise ValueError("Non-finite pair feature detected")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PAIR_PATH,
        positive_features=positive_features,
        negative_features=negative_features,
        user_ids=pair_users,
        calibration_mask=calibration_mask,
    )
    train_mask = ~calibration_mask
    combined_train = np.concatenate(
        [positive_features[train_mask], negative_features[train_mask]], axis=0
    )
    metadata = {
        "protocol": "temporal_leave_one_out_pairs_v1",
        "source": str(SPLIT_PATH.resolve()),
        "eligible_users": len(eligible_users),
        "excluded_single_interaction_users": sum(
            len(sequence) == 1 for sequence in histories.values()
        ),
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "pairs": pair_count,
        "train_users": len(set(pair_users[train_mask].tolist())),
        "calibration_users": len(set(pair_users[calibration_mask].tolist())),
        "train_pairs": int(train_mask.sum()),
        "calibration_pairs": int(calibration_mask.sum()),
        "minimum_negative_pool": minimum_negative_pool,
        "empty_text_profiles": empty_text_profiles,
        "feature_names": ["text_cosine", "visual_cosine", "product"],
        "feature_range": {
            "positive_min": positive_features.min(axis=0).tolist(),
            "positive_max": positive_features.max(axis=0).tolist(),
            "negative_min": negative_features.min(axis=0).tolist(),
            "negative_max": negative_features.max(axis=0).tolist(),
        },
        "train_candidate_feature_mean": combined_train.mean(axis=0).tolist(),
        "train_candidate_feature_std": combined_train.std(axis=0).tolist(),
        "pair_file": {
            "path": str(PAIR_PATH.resolve()),
            "bytes": PAIR_PATH.stat().st_size,
            "sha256": sha256(PAIR_PATH),
        },
        "runtime_seconds": time.perf_counter() - started,
        "known_biases": [
            "First observed comment time is a proxy for item availability, not upload time.",
            "Uniform unobserved negatives can contain unexposed or actually preferred items.",
            "One last positive per user balances users but discards earlier target opportunities.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

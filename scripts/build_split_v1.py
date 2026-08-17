"""Build the frozen ColdLens split_v1 and enforce leakage invariants."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "split_v1.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "split_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_interactions(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user", "item", "timestamp"]:
            raise ValueError(f"Unexpected interaction header: {reader.fieldnames}")
        for row in reader:
            rows.append((int(row["user"]), int(row["item"]), int(row["timestamp"])))
    return rows


def read_title_items(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["item", "title"]:
            raise ValueError(f"Unexpected title header: {reader.fieldnames}")
        return {int(row["item"]) for row in reader}


def write_gzip_csv(path: Path, header: list[str], rows: Iterable[tuple[int, ...]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(header)
                for row in rows:
                    writer.writerow(row)
                    count += 1
    return count


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    interaction_path = ROOT / config["interaction_file"]
    title_path = ROOT / config["title_file"]
    if sha256(interaction_path) != config["interaction_sha256"]:
        raise ValueError("Interaction source hash differs from frozen split_v1 config")
    if sha256(title_path) != config["title_sha256"]:
        raise ValueError("Title source hash differs from frozen split_v1 config")

    rows = read_interactions(interaction_path)
    title_items = read_title_items(title_path)
    dev_end = int(config["dev_train_end_ms_exclusive"])
    final_end = int(config["final_train_end_ms_exclusive"])
    if not dev_end < final_end:
        raise ValueError("Development cutoff must precede final cutoff")

    dev_train = [row for row in rows if row[2] < dev_end]
    final_train = [row for row in rows if row[2] < final_end]
    validation_window = [row for row in rows if dev_end <= row[2] < final_end]
    test_window = [row for row in rows if row[2] >= final_end]

    dev_users = Counter(user for user, _, _ in dev_train)
    final_users = Counter(user for user, _, _ in final_train)
    dev_items = {item for _, item, _ in dev_train}
    final_items = {item for _, item, _ in final_train}

    validation_cold_candidates = {item for _, item, _ in validation_window if item not in dev_items}
    test_cold_candidates = {item for _, item, _ in test_window if item not in final_items}
    validation_cold = [
        row
        for row in validation_window
        if row[1] in validation_cold_candidates and dev_users[row[0]] >= 1
    ]
    test_cold = [
        row
        for row in test_window
        if row[1] in test_cold_candidates and final_users[row[0]] >= 1
    ]
    validation_warm = [
        row for row in validation_window if row[1] in dev_items and dev_users[row[0]] >= 1
    ]
    test_warm = [
        row for row in test_window if row[1] in final_items and final_users[row[0]] >= 1
    ]

    # Leakage, eligibility, coverage, and boundary invariants.
    assert not (validation_cold_candidates & dev_items)
    assert not (test_cold_candidates & final_items)
    assert validation_cold_candidates <= title_items
    assert test_cold_candidates <= title_items
    assert all(timestamp < dev_end for _, _, timestamp in dev_train)
    assert all(timestamp < final_end for _, _, timestamp in final_train)
    assert all(dev_end <= timestamp < final_end for _, _, timestamp in validation_cold)
    assert all(timestamp >= final_end for _, _, timestamp in test_cold)
    assert all(user in dev_users for user, _, _ in validation_cold)
    assert all(user in final_users for user, _, _ in test_cold)
    assert {item for _, item, _ in validation_cold} <= validation_cold_candidates
    assert {item for _, item, _ in test_cold} <= test_cold_candidates
    assert set(dev_train) <= set(final_train)

    files: dict[str, dict[str, int | str]] = {}
    outputs: list[tuple[str, list[str], Iterable[tuple[int, ...]]]] = [
        ("dev_train.csv.gz", ["user", "item", "timestamp"], dev_train),
        ("validation_cold.csv.gz", ["user", "item", "timestamp"], validation_cold),
        ("validation_warm.csv.gz", ["user", "item", "timestamp"], validation_warm),
        ("final_train.csv.gz", ["user", "item", "timestamp"], final_train),
        ("test_cold.csv.gz", ["user", "item", "timestamp"], test_cold),
        ("test_warm.csv.gz", ["user", "item", "timestamp"], test_warm),
        ("validation_cold_candidates.csv.gz", ["item"], ((item,) for item in sorted(validation_cold_candidates))),
        ("test_cold_candidates.csv.gz", ["item"], ((item,) for item in sorted(test_cold_candidates))),
        ("dev_warm_candidates.csv.gz", ["item"], ((item,) for item in sorted(dev_items))),
        ("final_warm_candidates.csv.gz", ["item"], ((item,) for item in sorted(final_items))),
    ]
    for name, header, output_rows in outputs:
        path = OUTPUT_DIR / name
        count = write_gzip_csv(path, header, output_rows)
        files[name] = {"rows": count, "bytes": path.stat().st_size, "sha256": sha256(path)}

    manifest = {
        "version": config["version"],
        "config_sha256": sha256(CONFIG_PATH),
        "source_interaction_sha256": config["interaction_sha256"],
        "source_title_sha256": config["title_sha256"],
        "counts": {
            "source_rows": len(rows),
            "dev_train_users": len(dev_users),
            "dev_train_items": len(dev_items),
            "final_train_users": len(final_users),
            "final_train_items": len(final_items),
            "validation_cold_candidate_items": len(validation_cold_candidates),
            "validation_cold_eval_users": len({user for user, _, _ in validation_cold}),
            "test_cold_candidate_items": len(test_cold_candidates),
            "test_cold_eval_users": len({user for user, _, _ in test_cold}),
        },
        "files": files,
        "invariants": {
            "validation_cold_items_seen_in_dev_train": 0,
            "test_cold_items_seen_in_final_train": 0,
            "cold_items_missing_title": 0,
            "eval_users_without_corresponding_train_history": 0,
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote split to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

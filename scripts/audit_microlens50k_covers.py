"""Audit the MicroLens-50K cover archive in place without extracting it."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import struct
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "raw" / "microlens50k" / "MicroLens-50k_covers.zip"
TITLE_PATH = ROOT / "data" / "raw" / "microlens50k" / "MicroLens-50k_titles.csv"
OUTPUT_DIR = ROOT / "outputs" / "m0"
SPLIT_DIR = ROOT / "data" / "processed" / "split_v1"
EXPECTED_ZIP_SHA256 = "135255149cc74d47f1fb04985b2d16b862ee3d00c963ff47cb756b2136ca5892"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_items() -> set[int]:
    with TITLE_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["item"]) for row in csv.DictReader(handle)}


def split_items(filename: str) -> set[int]:
    with gzip.open(SPLIT_DIR / filename, "rt", encoding="utf-8", newline="") as handle:
        return {int(row["item"]) for row in csv.DictReader(handle)}


def jpeg_dimensions(handle) -> tuple[int, int] | None:
    if handle.read(2) != b"\xff\xd8":
        return None
    while True:
        byte = handle.read(1)
        while byte and byte != b"\xff":
            byte = handle.read(1)
        if not byte:
            return None
        marker_byte = handle.read(1)
        while marker_byte == b"\xff":
            marker_byte = handle.read(1)
        if not marker_byte:
            return None
        marker = marker_byte[0]
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            return None
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            return None
        if marker in JPEG_SOF_MARKERS:
            payload = handle.read(5)
            if len(payload) != 5:
                return None
            height, width = struct.unpack(">HH", payload[1:])
            return width, height
        handle.seek(segment_length - 2, 1)


def main() -> None:
    started = time.perf_counter()
    archive_hash = sha256_file(ZIP_PATH)
    if archive_hash != EXPECTED_ZIP_SHA256:
        raise ValueError("Cover ZIP hash differs from the downloaded file frozen for this audit")

    expected = expected_items()
    ids: list[int] = []
    invalid_names: list[str] = []
    unsafe_paths: list[str] = []
    extensions: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    unreadable_headers: list[str] = []
    tiny_files: list[dict[str, int | str]] = []
    crc_size_groups: defaultdict[tuple[int, int], list[zipfile.ZipInfo]] = defaultdict(list)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        bad_crc_entry = archive.testzip()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                unsafe_paths.append(entry.filename)
            extensions[path.suffix.lower()] += 1
            try:
                ids.append(int(path.stem))
            except ValueError:
                invalid_names.append(entry.filename)
            if entry.file_size < 1024:
                tiny_files.append({"name": entry.filename, "bytes": entry.file_size})
            crc_size_groups[(entry.CRC, entry.file_size)].append(entry)
            with archive.open(entry, "r") as image:
                size = jpeg_dimensions(image)
            if size is None:
                unreadable_headers.append(entry.filename)
            else:
                width, height = size
                dimensions[f"{width}x{height}"] += 1
                if width == height:
                    orientations["square"] += 1
                elif width > height:
                    orientations["landscape"] += 1
                else:
                    orientations["portrait"] += 1

        # CRC+size identifies duplicate candidates; SHA-256 verifies only those groups.
        exact_duplicate_groups: list[dict[str, object]] = []
        for group in crc_size_groups.values():
            if len(group) < 2:
                continue
            by_hash: defaultdict[str, list[str]] = defaultdict(list)
            for entry in group:
                digest = hashlib.sha256()
                with archive.open(entry, "r") as image:
                    for chunk in iter(lambda: image.read(1024 * 1024), b""):
                        digest.update(chunk)
                by_hash[digest.hexdigest()].append(entry.filename)
            for digest, names in by_hash.items():
                if len(names) > 1:
                    exact_duplicate_groups.append(
                        {"sha256": digest, "count": len(names), "entries": sorted(names)}
                    )

        uncompressed_bytes = sum(entry.file_size for entry in entries)
        compressed_bytes = sum(entry.compress_size for entry in entries)

    id_counts = Counter(ids)
    actual_ids = set(ids)
    duplicate_id_rows = sum(count - 1 for count in id_counts.values())
    exact_duplicate_groups.sort(key=lambda group: (-int(group["count"]), group["sha256"]))
    dev_train_items = split_items("dev_train.csv.gz")
    validation_cold_items = split_items("validation_cold_candidates.csv.gz")
    final_train_items = split_items("final_train.csv.gz")
    test_cold_items = split_items("test_cold_candidates.csv.gz")

    def ids_from_group(group: dict[str, object]) -> set[int]:
        return {int(PurePosixPath(name).stem) for name in group["entries"]}  # type: ignore[union-attr]

    validation_cross_boundary = []
    test_cross_boundary = []
    for group in exact_duplicate_groups:
        group_ids = ids_from_group(group)
        if group_ids & dev_train_items and group_ids & validation_cold_items:
            validation_cross_boundary.append(group)
        if group_ids & final_train_items and group_ids & test_cold_items:
            test_cross_boundary.append(group)
    report = {
        "archive": {
            "path": str(ZIP_PATH),
            "bytes": ZIP_PATH.stat().st_size,
            "sha256": archive_hash,
            "file_entries": len(entries),
            "uncompressed_bytes": uncompressed_bytes,
            "compressed_payload_bytes": compressed_bytes,
            "compression_ratio": uncompressed_bytes / compressed_bytes,
        },
        "integrity": {
            "bad_crc_entry": bad_crc_entry,
            "unsafe_paths": unsafe_paths,
            "invalid_item_filenames": invalid_names,
            "duplicate_item_id_rows": duplicate_id_rows,
            "expected_item_ids": len(expected),
            "actual_item_ids": len(actual_ids),
            "missing_expected_item_ids": sorted(expected - actual_ids),
            "unexpected_item_ids": sorted(actual_ids - expected),
        },
        "images": {
            "extensions": dict(sorted(extensions.items())),
            "unreadable_image_headers": unreadable_headers,
            "files_smaller_than_1KiB": tiny_files,
            "orientations": dict(sorted(orientations.items())),
            "unique_dimensions": len(dimensions),
            "top_dimensions": [
                {"size": size, "count": count} for size, count in dimensions.most_common(20)
            ],
        },
        "exact_duplicates": {
            "groups": len(exact_duplicate_groups),
            "rows_beyond_first": sum(int(group["count"]) - 1 for group in exact_duplicate_groups),
            "largest_groups": exact_duplicate_groups[:20],
            "validation_groups_crossing_train_cold_boundary": validation_cross_boundary,
            "test_groups_crossing_train_cold_boundary": test_cross_boundary,
            "scope_note": "Exact byte duplicates only; perceptual near-duplicates require decoded-image analysis later.",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "microlens50k_cover_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "exact_duplicates"}, ensure_ascii=False, indent=2))
    print("\nExact duplicates:")
    print(json.dumps({key: value for key, value in report["exact_duplicates"].items() if key != "largest_groups"}, indent=2))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()

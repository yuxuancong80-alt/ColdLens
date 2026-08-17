"""Run a fixed local MobileNetV3-Small cover-embedding smoke test.

This script reads images directly from the MicroLens-50K cover archive. It does
not extract the archive, train a model, or evaluate recommendation labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "data" / "raw" / "microlens50k" / "MicroLens-50k_covers.zip"
OUTPUT_DIR = ROOT / "outputs" / "visual_smoke"
VECTOR_PATH = OUTPUT_DIR / "mobilenet_v3_small_sample.csv"
REPORT_PATH = OUTPUT_DIR / "mobilenet_v3_small_smoke_report.json"

# Known audit pairs plus a deterministic spread across the item-ID range.
SAMPLE_IDS = [
    46,
    313,
    792,
    902,
    1186,
    1792,
    1852,
    1864,
    2491,
    3478,
    3658,
    3683,
    4141,
    4237,
    5036,
    5628,
    6061,
    6443,
    7157,
    7979,
    8425,
    8557,
    8714,
    9319,
    10204,
    11800,
    13064,
    13443,
    15000,
    16683,
    17620,
    18187,
]

KNOWN_PAIRS = [
    {"label": "same Naruto frame", "items": [1186, 16683]},
    {"label": "same comic template / different episode number", "items": [313, 8425]},
    {"label": "dHash false positive: dog vs football", "items": [1852, 13443]},
    {"label": "dHash false positive: game vs film", "items": [1864, 9319]},
]


def load_tensors(transform: object) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    with ZipFile(ARCHIVE_PATH) as archive:
        for item_id in SAMPLE_IDS:
            member = f"MicroLens-50k_covers/{item_id}.jpg"
            with archive.open(member) as source:
                with Image.open(BytesIO(source.read())) as image:
                    tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors)


def embed(model: torch.nn.Module, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            features = model.features(batch)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            batches.append(torch.nn.functional.normalize(features, p=2, dim=1))
    return torch.cat(batches)


def cosine(vectors: np.ndarray, index: dict[int, int], left: int, right: int) -> float:
    return float(np.dot(vectors[index[left]], vectors[index[right]]))


def write_vectors(vectors: np.ndarray) -> None:
    with VECTOR_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["item", *[f"f{column:03d}" for column in range(vectors.shape[1])]])
        for item_id, vector in zip(SAMPLE_IDS, vectors, strict=True):
            writer.writerow([item_id, *[format(float(value), ".9g") for value in vector]])


def main() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Missing cover archive: {ARCHIVE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    torch.manual_seed(20260814)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model_started = time.perf_counter()
    model = mobilenet_v3_small(weights=weights).eval()
    model_seconds = time.perf_counter() - model_started

    preprocess_started = time.perf_counter()
    images = load_tensors(weights.transforms())
    preprocess_seconds = time.perf_counter() - preprocess_started

    inference_started = time.perf_counter()
    first = embed(model, images)
    first_inference_seconds = time.perf_counter() - inference_started

    repeat_started = time.perf_counter()
    second = embed(model, images)
    repeat_inference_seconds = time.perf_counter() - repeat_started

    max_repeat_difference = float(torch.max(torch.abs(first - second)))
    vectors = first.cpu().numpy()
    write_vectors(vectors)

    item_index = {item_id: position for position, item_id in enumerate(SAMPLE_IDS)}
    known_pairs = []
    for pair in KNOWN_PAIRS:
        left, right = pair["items"]
        known_pairs.append(
            {
                **pair,
                "cosine_similarity": cosine(vectors, item_index, left, right),
            }
        )

    all_pairs: list[tuple[float, int, int]] = []
    for position, left in enumerate(SAMPLE_IDS):
        for right in SAMPLE_IDS[position + 1 :]:
            all_pairs.append((cosine(vectors, item_index, left, right), left, right))
    all_pairs.sort(reverse=True)

    norms = np.linalg.norm(vectors, axis=1)
    end_to_end_seconds = preprocess_seconds + first_inference_seconds
    report = {
        "purpose": "local pretrained visual-embedding smoke test; no recommendation-label evaluation",
        "archive": str(ARCHIVE_PATH),
        "model": "torchvision MobileNetV3-Small DEFAULT weights",
        "weights": str(weights),
        "feature": "576-dimensional global-average-pooled backbone output, L2-normalized",
        "device": "cpu",
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "rows": len(SAMPLE_IDS),
        "dimensions": int(vectors.shape[1]),
        "timing_seconds": {
            "model_setup_including_cached_or_first_weight_load": model_seconds,
            "zip_decode_and_preprocess": preprocess_seconds,
            "first_inference": first_inference_seconds,
            "repeat_inference": repeat_inference_seconds,
            "first_end_to_end_excluding_model_setup": end_to_end_seconds,
            "projected_19220_images_seconds_from_smoke": end_to_end_seconds
            * 19220
            / len(SAMPLE_IDS),
        },
        "determinism": {"repeat_max_abs_difference": max_repeat_difference},
        "norm_range": {"minimum": float(norms.min()), "maximum": float(norms.max())},
        "vector_file_sha256": hashlib.sha256(VECTOR_PATH.read_bytes()).hexdigest(),
        "known_pairs": known_pairs,
        "top_sample_pairs": [
            {"items": [left, right], "cosine_similarity": score}
            for score, left, right in all_pairs[:10]
        ],
        "checks": {
            "expected_shape": list(vectors.shape) == [32, 576],
            "all_vectors_are_unit_norm": bool(np.allclose(norms, 1.0, atol=1e-6)),
            "repeat_is_deterministic": max_repeat_difference <= 1e-7,
            "same_frame_more_similar_than_false_positive_pairs": known_pairs[0][
                "cosine_similarity"
            ]
            > max(
                known_pairs[2]["cosine_similarity"],
                known_pairs[3]["cosine_similarity"],
            ),
        },
        "limitations": [
            "A 32-image audit sample cannot establish recommendation quality.",
            "The time projection is linear and should be confirmed on a larger local batch.",
            "ImageNet pretraining may underrepresent anime, games, and Chinese text overlays.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

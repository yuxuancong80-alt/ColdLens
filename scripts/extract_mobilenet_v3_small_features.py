"""Extract MobileNetV3-Small cover features directly from the cover ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT / "data" / "raw" / "microlens50k" / "MicroLens-50k_covers.zip"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "features"
MEMBER_PATTERN = re.compile(r"^MicroLens-50k_covers/(\d+)\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N numeric item IDs for a local benchmark.",
    )
    return parser.parse_args()


def archive_items(archive: ZipFile) -> list[tuple[int, str]]:
    items = []
    for member in archive.namelist():
        match = MEMBER_PATTERN.fullmatch(member)
        if match:
            items.append((int(match.group(1)), member))
    items.sort()
    return items


def load_batch(
    archive: ZipFile, members: list[tuple[int, str]], transform: object
) -> torch.Tensor:
    tensors = []
    for _, member in members:
        with archive.open(member) as source:
            with Image.open(BytesIO(source.read())) as image:
                tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors)


def embed(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        features = model.features(images)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)
        return torch.nn.functional.normalize(features, p=2, dim=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)

    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    torch.manual_seed(20260814)
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model_started = time.perf_counter()
    model = mobilenet_v3_small(weights=weights).eval()
    model_seconds = time.perf_counter() - model_started
    transform = weights.transforms()

    with ZipFile(args.archive) as archive:
        members = archive_items(archive)
        if args.limit is not None:
            members = members[: args.limit]
        if not members:
            raise ValueError("No cover JPG files found in archive")

        item_ids = np.empty(len(members), dtype=np.int32)
        vectors = np.empty((len(members), 576), dtype=np.float32)
        extraction_started = time.perf_counter()
        for start in range(0, len(members), args.batch_size):
            batch_members = members[start : start + args.batch_size]
            images = load_batch(archive, batch_members, transform)
            batch_vectors = embed(model, images).cpu().numpy()
            end = start + len(batch_members)
            item_ids[start:end] = [item_id for item_id, _ in batch_members]
            vectors[start:end] = batch_vectors
            if end == len(members) or end % 512 == 0:
                print(f"processed={end}/{len(members)}", flush=True)
        extraction_seconds = time.perf_counter() - extraction_started

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite feature value detected")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError("One or more feature vectors are not unit normalized")
    if len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("Duplicate item ID detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{len(members)}" if args.limit is not None else ""
    ids_path = args.output_dir / f"mobilenet_v3_small_item_ids{suffix}.npy"
    vectors_path = args.output_dir / f"mobilenet_v3_small_vectors{suffix}.npy"
    metadata_path = args.output_dir / f"mobilenet_v3_small_metadata{suffix}.json"
    np.save(ids_path, item_ids, allow_pickle=False)
    np.save(vectors_path, vectors, allow_pickle=False)

    metadata = {
        "purpose": "frozen cover features for the Visual-only and multimodal baselines",
        "archive": str(args.archive.resolve()),
        "model": "torchvision MobileNetV3-Small",
        "weights": str(weights),
        "feature": "576-dimensional global-average-pooled backbone output, L2-normalized",
        "rows": len(members),
        "dimensions": 576,
        "batch_size": args.batch_size,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "timing_seconds": {
            "model_setup": model_seconds,
            "zip_decode_preprocess_and_inference": extraction_seconds,
            "images_per_second": len(members) / extraction_seconds,
            "projected_19220_images": extraction_seconds * 19220 / len(members),
        },
        "norm_range": {"minimum": float(norms.min()), "maximum": float(norms.max())},
        "outputs": {
            "item_ids": {"path": str(ids_path.resolve()), "sha256": sha256(ids_path)},
            "vectors": {
                "path": str(vectors_path.resolve()),
                "sha256": sha256(vectors_path),
            },
        },
        "is_partial_benchmark": args.limit is not None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

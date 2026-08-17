"""Train the frozen three-weight pairwise late-fusion ranker on local CPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "artifacts" / "learned_fusion"
PAIR_PATH = PAIR_DIR / "temporal_leave_one_out_pairs_v1.npz"
OUTPUT_PATH = PAIR_DIR / "learned_fusion_model_v1.json"
SEED = 20260814
LEARNING_RATE = 0.01
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 5
MIN_IMPROVEMENT = 1e-6


class FusionRanker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weights()


def standardized_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(PAIR_PATH, allow_pickle=False) as data:
        positive = data["positive_features"].astype(np.float32, copy=True)
        negative = data["negative_features"].astype(np.float32, copy=True)
        calibration = data["calibration_mask"].astype(bool, copy=True)
    train = ~calibration
    train_candidates = np.concatenate([positive[train], negative[train]], axis=0)
    mean = train_candidates.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_candidates.std(axis=0, dtype=np.float64).astype(np.float32)
    if np.any(std <= 0):
        raise ValueError(f"Non-positive feature standard deviation: {std}")
    positive = (positive - mean) / std
    negative = (negative - mean) / std
    return positive, negative, calibration, mean, std


def pairwise_loss(
    model: FusionRanker, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    margin = model(positive) - model(negative)
    return torch.nn.functional.softplus(-margin).mean()


def evaluate(
    model: FusionRanker, positive: torch.Tensor, negative: torch.Tensor
) -> tuple[float, float, float]:
    model.eval()
    with torch.inference_mode():
        margin = model(positive) - model(negative)
        loss = torch.nn.functional.softplus(-margin).mean().item()
        strict_accuracy = (margin > 0).float().mean().item()
        tie_rate = (margin == 0).float().mean().item()
    return loss, strict_accuracy, tie_rate


def train_for_epochs(
    positive: torch.Tensor,
    negative: torch.Tensor,
    epochs: int,
    *,
    calibration: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[FusionRanker, list[dict[str, object]], int]:
    torch.manual_seed(SEED)
    model = FusionRanker()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_calibration_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(positive), generator=generator)
        loss_sum = 0.0
        examples = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            loss = pairwise_loss(model, positive[indices], negative[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(indices)
            loss_sum += loss.item() * count
            examples += count

        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": loss_sum / examples,
            "weights": model.weights().detach().tolist(),
        }
        if calibration is not None:
            calibration_loss, calibration_accuracy, calibration_tie_rate = evaluate(
                model, *calibration
            )
            row.update(
                {
                    "calibration_loss": calibration_loss,
                    "calibration_strict_pair_accuracy": calibration_accuracy,
                    "calibration_tie_rate": calibration_tie_rate,
                }
            )
            if calibration_loss < best_calibration_loss - MIN_IMPROVEMENT:
                best_calibration_loss = calibration_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history.append(row)
            if epochs_without_improvement >= PATIENCE:
                break
        else:
            history.append(row)

    if calibration is None:
        best_epoch = epochs
    return model, history, best_epoch


def baseline_pair_accuracy(
    positive: np.ndarray, negative: np.ndarray, column: int
) -> dict[str, float]:
    margin = positive[:, column] - negative[:, column]
    return {
        "strict_accuracy": float(np.mean(margin > 0)),
        "tie_rate": float(np.mean(margin == 0)),
    }


def main() -> None:
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    positive, negative, calibration_mask, mean, std = standardized_arrays()
    train_mask = ~calibration_mask
    train_positive = torch.from_numpy(positive[train_mask])
    train_negative = torch.from_numpy(negative[train_mask])
    calibration_positive = torch.from_numpy(positive[calibration_mask])
    calibration_negative = torch.from_numpy(negative[calibration_mask])

    _, selection_history, best_epoch = train_for_epochs(
        train_positive,
        train_negative,
        MAX_EPOCHS,
        calibration=(calibration_positive, calibration_negative),
    )
    if best_epoch < 1:
        raise ValueError("No calibration epoch selected")

    all_positive = torch.from_numpy(positive)
    all_negative = torch.from_numpy(negative)
    final_model, final_history, _ = train_for_epochs(
        all_positive, all_negative, best_epoch
    )
    final_loss, final_accuracy, final_tie_rate = evaluate(
        final_model, all_positive, all_negative
    )
    softmax_weights = np.asarray(
        final_model.weights().detach().numpy(), dtype=np.float64
    )
    original_scale_coefficients = softmax_weights / std.astype(np.float64)
    normalized_original_scale = (
        original_scale_coefficients / original_scale_coefficients.sum()
    )

    selected_row = next(
        row for row in selection_history if int(row["epoch"]) == best_epoch
    )
    report = {
        "model": "learned_fusion_v1",
        "pair_protocol": "temporal_leave_one_out_pairs_v1",
        "feature_names": ["text_cosine", "visual_cosine", "product"],
        "standardization": {
            "fit": "internal train candidate features only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "training_budget": {
            "seed": SEED,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "minimum_improvement": MIN_IMPROVEMENT,
            "loss": "pairwise softplus negative margin",
            "device": "cpu",
            "torch_version": torch.__version__,
        },
        "selection": {
            "selected_epoch": best_epoch,
            "epochs_run_before_early_stop": len(selection_history),
            "selected_calibration_loss": selected_row["calibration_loss"],
            "selected_calibration_strict_pair_accuracy": selected_row[
                "calibration_strict_pair_accuracy"
            ],
            "selected_calibration_tie_rate": selected_row["calibration_tie_rate"],
            "history": selection_history,
        },
        "final_retrain": {
            "users": "all eligible train and calibration users",
            "epochs": best_epoch,
            "softmax_weights_on_standardized_features": softmax_weights.tolist(),
            "normalized_effective_coefficients_on_original_features": normalized_original_scale.tolist(),
            "pairwise_loss": final_loss,
            "strict_pair_accuracy": final_accuracy,
            "tie_rate": final_tie_rate,
            "history": final_history,
        },
        "calibration_baselines": {
            "text_cosine": baseline_pair_accuracy(
                positive[calibration_mask], negative[calibration_mask], 0
            ),
            "visual_cosine": baseline_pair_accuracy(
                positive[calibration_mask], negative[calibration_mask], 1
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "test_split_read": False,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

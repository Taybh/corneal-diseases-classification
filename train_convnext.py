from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from oct_patch_dataset import (
    CORNEA_CLASS,
    LIMBUS_CLASS,
    OCTPatchDataset,
    SourceCase,
    build_source_cases,
    collect_region_json_files,
    split_cases,
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


# ============================================================
# Data / patch settings agreed for the first experiment
# ============================================================

TRAIN_FRACTION = 0.80      # exactly 40/10 when there are 50 sources
PATCH_WIDTH = 256          # original OCT pixels
PATCH_STEP = 128           # 50% overlap during training
MARGIN_ABOVE_ANTERIOR = 15
MASK_VALUE = 0             # below posterior / outside anatomical band
PADDING_VALUE = 0          # same value as mask
MODEL_INPUT_SIZE = 224     # aspect-ratio-preserving fit + padding


# ============================================================
# Training settings
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0
TOTAL_EPOCHS = 50
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
MIN_VAL_LOSS_DELTA = 1e-5

# Main augmentation: small rotation only.
# No blur, no brightness/contrast, no vertical jitter, no flips.
ROTATION_DEGREES = 8


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN where available.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Logging
# ============================================================

def log_message(message: str, log_path: Path) -> None:
    print(message)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


# ============================================================
# Transform
# ============================================================

def create_transforms():
    """
    The dataset itself already:
      1. extracts a 256-pixel-wide anatomical patch,
      2. keeps 15 px above anterior through posterior,
      3. masks invalid pixels with 0,
      4. preserves aspect ratio,
      5. pads to 224x224 with 0,
      6. converts grayscale to RGB.

    Therefore transforms here only handle augmentation + tensor/normalization.
    """

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=ROTATION_DEGREES,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, validation_transform


# ============================================================
# Dataset creation
# ============================================================

def create_datasets(input_paths: list[Path]):
    json_files = collect_region_json_files(input_paths)
    cases = build_source_cases(json_files)

    if len(cases) < 2:
        raise RuntimeError(
            f"Too few complete annotated OCT sources were found: {len(cases)}"
        )

    splits = split_cases(
        cases=cases,
        train_fraction=TRAIN_FRACTION,
        seed=SEED,
    )

    train_transform, validation_transform = create_transforms()

    train_dataset = OCTPatchDataset(
        cases=splits["train"],
        transform=train_transform,
        patch_width=PATCH_WIDTH,
        patch_step=PATCH_STEP,
        margin_above_anterior=MARGIN_ABOVE_ANTERIOR,
        output_size=MODEL_INPUT_SIZE,
        mask_value=MASK_VALUE,
        padding_value=PADDING_VALUE,
        return_metadata=False,
    )

    validation_dataset = OCTPatchDataset(
        cases=splits["validation"],
        transform=validation_transform,
        patch_width=PATCH_WIDTH,
        patch_step=PATCH_STEP,
        margin_above_anterior=MARGIN_ABOVE_ANTERIOR,
        output_size=MODEL_INPUT_SIZE,
        mask_value=MASK_VALUE,
        padding_value=PADDING_VALUE,
        return_metadata=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No training patches were generated.")
    if len(validation_dataset) == 0:
        raise RuntimeError("No validation patches were generated.")

    return train_dataset, validation_dataset, splits, cases


# ============================================================
# DataLoaders
# ============================================================

def create_loaders(train_dataset, validation_dataset):
    # Generator makes train shuffling reproducible.
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


# ============================================================
# Model
# ============================================================

def build_model() -> nn.Module:
    """
    ImageNet-pretrained ConvNeXt-Tiny.

    We replace only the final classifier and fine-tune ALL model weights
    from epoch 1. There is no frozen-head stage.
    """

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)

    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


# ============================================================
# Class-weighted cross entropy
# ============================================================

def create_criterion(
    train_dataset: OCTPatchDataset,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    counts = train_dataset.class_counts()
    limbus_count = counts["limbus"]
    cornea_count = counts["cornea"]

    if limbus_count == 0 or cornea_count == 0:
        raise RuntimeError(
            "Both limbus and cornea patches must exist in the training set. "
            f"Counts: {counts}"
        )

    total = limbus_count + cornea_count
    weights = torch.tensor(
        [
            total / (2.0 * limbus_count),
            total / (2.0 * cornea_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    return nn.CrossEntropyLoss(weight=weights), weights.detach().cpu().tolist()


# ============================================================
# Metrics
# ============================================================

def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int | list]:
    # Rows = true label, columns = predicted label.
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_class = []
    recalls = []
    f1s = []

    for class_index, class_name in (
        (LIMBUS_CLASS, "limbus"),
        (CORNEA_CLASS, "cornea"),
    ):
        tp = int(confusion[class_index, class_index])
        fn = int(confusion[class_index, :].sum() - tp)
        fp = int(confusion[:, class_index].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        recalls.append(recall)
        f1s.append(f1)
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[class_index, :].sum()),
            }
        )

    balanced_accuracy = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================
# One epoch
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    confusion = np.zeros((2, 2), dtype=np.int64)

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            predictions = torch.argmax(logits, dim=1)

            true_np = targets.detach().cpu().numpy()
            pred_np = predictions.detach().cpu().numpy()

            for true_label, pred_label in zip(true_np, pred_np):
                confusion[int(true_label), int(pred_label)] += 1

    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples if total_examples else float("nan")
    metrics["n_examples"] = total_examples
    return metrics


# ============================================================
# Save split metadata
# ============================================================

def case_to_dict(case: SourceCase) -> dict:
    return {
        "source_id": case.source_id,
        "region_json_path": str(case.region_json_path),
        "source_json_path": (
            str(case.source_json_path)
            if case.source_json_path is not None
            else None
        ),
        "image_path": (
            str(case.image_path)
            if case.image_path is not None
            else None
        ),
    }


def save_splits(splits: dict[str, list[SourceCase]], output_path: Path) -> None:
    payload = {
        split_name: [case_to_dict(case) for case in split_cases_list]
        for split_name, split_cases_list in splits.items()
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


# ============================================================
# Checkpoint helpers
# ============================================================

def make_config() -> dict:
    return {
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "patch_width": PATCH_WIDTH,
        "patch_step": PATCH_STEP,
        "margin_above_anterior": MARGIN_ABOVE_ANTERIOR,
        "mask_value": MASK_VALUE,
        "padding_value": PADDING_VALUE,
        "model_input_size": MODEL_INPUT_SIZE,
        "model": "convnext_tiny",
        "pretrained_weights": "ConvNeXt_Tiny_Weights.DEFAULT",
        "fine_tune_all_weights_from_epoch_1": True,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "total_epochs": TOTAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "min_val_loss_delta": MIN_VAL_LOSS_DELTA,
        "early_stopping": False,
        "rotation_degrees": ROTATION_DEGREES,
        "blur": False,
        "brightness_contrast_augmentation": False,
        "vertical_jitter": False,
        "horizontal_flip": False,
        "vertical_flip": False,
        "imagenet_normalization": True,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
    history: list[dict],
    class_weights: list[float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "class_weights": class_weights,
            "config": make_config(),
        },
        path,
    )


# ============================================================
# Main training
# ============================================================

def train(
    input_paths: list[Path],
    output_dir: Path,
    resume_path: Path | None = None,
) -> None:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.txt"
    best_model_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    splits_path = output_dir / "splits.json"
    history_path = output_dir / "history.json"
    config_path = output_dir / "config.json"

    # Fresh run gets a fresh log. Resume appends to existing log.
    if resume_path is None:
        log_path.write_text("", encoding="utf-8")

    device = get_device()

    train_dataset, validation_dataset, splits, all_cases = create_datasets(input_paths)
    train_loader, validation_loader = create_loaders(
        train_dataset,
        validation_dataset,
    )

    save_splits(splits, splits_path)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(make_config(), file, indent=2)

    criterion, class_weights = create_criterion(train_dataset, device)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))

    log_message("=" * 72, log_path)
    log_message("AS-OCT cornea/limbus ConvNeXt training", log_path)
    log_message("=" * 72, log_path)
    log_message(f"Device: {device}", log_path)
    log_message(f"Complete annotated source images found: {len(all_cases)}", log_path)
    log_message(f"Train source images: {len(splits['train'])}", log_path)
    log_message(f"Validation source images: {len(splits['validation'])}", log_path)
    log_message("Internal test split: none (external test set kept separate)", log_path)
    log_message(f"Training patches: {len(train_dataset)} {train_dataset.class_counts()}", log_path)
    log_message(
        f"Validation patches: {len(validation_dataset)} "
        f"{validation_dataset.class_counts()}",
        log_path,
    )
    log_message(f"Class weights [limbus, cornea]: {class_weights}", log_path)
    log_message("", log_path)
    log_message(f"Patch width: {PATCH_WIDTH} original OCT pixels", log_path)
    log_message(f"Patch step: {PATCH_STEP} (50% horizontal overlap)", log_path)
    log_message(
        f"Vertical anatomy: {MARGIN_ABOVE_ANTERIOR}px above anterior "
        "through posterior boundary",
        log_path,
    )
    log_message("Posterior bottom margin: 0", log_path)
    log_message(f"Mask value: {MASK_VALUE}", log_path)
    log_message(f"Padding value: {PADDING_VALUE}", log_path)
    log_message(
        f"Resize: preserve aspect ratio, then pad to "
        f"{MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}",
        log_path,
    )
    log_message("", log_path)
    log_message("Model: ConvNeXt-Tiny with ImageNet-pretrained weights", log_path)
    log_message("Training: fine-tune ALL weights from epoch 1", log_path)
    log_message(f"Learning rate: {LEARNING_RATE}", log_path)
    log_message(f"Weight decay: {WEIGHT_DECAY}", log_path)
    log_message(f"Total epochs: {TOTAL_EPOCHS}", log_path)
    log_message("Early stopping: disabled", log_path)
    log_message(f"Best-checkpoint min delta: {MIN_VAL_LOSS_DELTA}", log_path)
    log_message(f"Training augmentation: random rotation +/-{ROTATION_DEGREES} deg", log_path)
    log_message("Blur / brightness-contrast / vertical jitter / flips: disabled", log_path)
    log_message("ImageNet normalization: enabled", log_path)
    if resume_path is not None:
        log_message(f"Resuming from: {resume_path}", log_path)
        log_message(f"Next epoch: {start_epoch}", log_path)
    log_message("=" * 72, log_path)

    if start_epoch > TOTAL_EPOCHS:
        log_message(
            f"Checkpoint is already at epoch {start_epoch - 1}, which is >= "
            f"TOTAL_EPOCHS={TOTAL_EPOCHS}. Nothing to train.",
            log_path,
        )
        return

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "learning_rate": lr,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        val_per_class = {
            item["class"]: item
            for item in validation_metrics["per_class"]
        }

        log_message(
            (
                f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | lr={lr:.2e} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={validation_metrics['loss']:.6f} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_acc={validation_metrics['accuracy']:.4f} | "
                f"val_bal_acc={validation_metrics['balanced_accuracy']:.4f} | "
                f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
            ),
            log_path,
        )

        log_message(
            (
                "  Val limbus: "
                f"P={val_per_class['limbus']['precision']:.4f} "
                f"R={val_per_class['limbus']['recall']:.4f} "
                f"F1={val_per_class['limbus']['f1']:.4f} | "
                "Val cornea: "
                f"P={val_per_class['cornea']['precision']:.4f} "
                f"R={val_per_class['cornea']['recall']:.4f} "
                f"F1={val_per_class['cornea']['f1']:.4f} | "
                f"CM={validation_metrics['confusion_matrix']}"
            ),
            log_path,
        )

        current_val_loss = float(validation_metrics["loss"])

        if current_val_loss < best_val_loss - MIN_VAL_LOSS_DELTA:
            previous_best = best_val_loss
            absolute_improvement = (
                previous_best - current_val_loss
                if np.isfinite(previous_best)
                else float("nan")
            )
            relative_improvement = (
                absolute_improvement / previous_best
                if np.isfinite(previous_best) and previous_best != 0.0
                else float("nan")
            )

            best_val_loss = current_val_loss
            best_epoch = epoch

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                history=history,
                class_weights=class_weights,
            )

            if np.isfinite(previous_best):
                log_message(
                    (
                        "  >>> NEW BEST MODEL | "
                        f"epoch={epoch} | previous_val_loss={previous_best:.6f} | "
                        f"new_val_loss={current_val_loss:.6f} | "
                        f"absolute_improvement={absolute_improvement:.6f} | "
                        f"relative_improvement={100.0 * relative_improvement:.4f}%"
                    ),
                    log_path,
                )
            else:
                log_message(
                    f"  >>> NEW BEST MODEL | epoch={epoch} | "
                    f"val_loss={current_val_loss:.6f}",
                    log_path,
                )

        # Always overwrite the resume checkpoint after every completed epoch.
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            history=history,
            class_weights=class_weights,
        )

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    log_message("=" * 72, log_path)
    log_message("Training complete.", log_path)
    log_message(f"Best validation-loss epoch: {best_epoch}", log_path)
    log_message(f"Best validation loss: {best_val_loss:.6f}", log_path)
    log_message(f"Best model: {best_model_path}", log_path)
    log_message(
        f"Last checkpoint (epoch {TOTAL_EPOCHS}, resumable): "
        f"{last_checkpoint_path}",
        log_path,
    )
    log_message("No early stopping was used.", log_path)
    log_message("External test set has not been evaluated by this script.", log_path)
    log_message("=" * 72, log_path)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train ImageNet-pretrained ConvNeXt-Tiny for binary "
            "cornea/limbus classification on AS-OCT sliding patches."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Annotation JSON file(s) and/or folders searched recursively.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_convnext_output"),
        help="Folder for logs, checkpoints, split metadata, and history.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional last_checkpoint.pt to resume from. "
            "Increase TOTAL_EPOCHS in this file first if continuing beyond 50."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        resume_path=args.resume,
    )

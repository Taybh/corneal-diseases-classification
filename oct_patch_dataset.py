from __future__ import annotations

import base64
import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Annotation labels stored in the canonical region JSON
# ============================================================

LEFT_LIMBUS = 0
CORNEA = 1
RIGHT_LIMBUS = 2

# Binary training labels
LIMBUS_CLASS = 0
CORNEA_CLASS = 1

CLASS_NAMES = {
    LIMBUS_CLASS: "limbus",
    CORNEA_CLASS: "cornea",
}


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class SourceCase:
    """One original annotated OCT source image."""

    region_json_path: Path
    source_json_path: Path | None
    image_path: Path | None
    source_id: str


@dataclass(frozen=True)
class PatchInfo:
    """Lightweight description of one sliding-window patch."""

    source_index: int
    center_x: int
    left: int
    right: int
    label: int
    cornea_fraction: float
    limbus_fraction: float
    limbus_side: str


# ============================================================
# JSON helpers
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return data


# ============================================================
# Resolve source images
# ============================================================

def resolve_source_json(
    region_json_path: Path,
    region_data: dict[str, Any],
) -> Path:
    source_value = region_data.get("source_json")

    if not source_value:
        raise ValueError(
            f"'source_json' is missing from {region_json_path}"
        )

    source_path = Path(str(source_value))

    candidates = [
        source_path,
        region_json_path.parent / source_path,
        region_json_path.parent / source_path.name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the original LabelMe JSON.\n"
        f"Region JSON: {region_json_path}\n"
        f"Stored source_json: {source_value}"
    )


def resolve_image_only_path(
    region_json_path: Path,
    region_data: dict[str, Any],
) -> Path:
    """Resolve an image for annotations whose source_json is null."""

    image_value = region_data.get("imagePath")
    if not image_value:
        raise ValueError(
            f"'imagePath' is missing from image-only annotation: "
            f"{region_json_path}"
        )

    normalized = str(image_value).replace("\\", "/")
    stored_path = Path(normalized)
    filename = stored_path.name

    candidates: list[Path] = []
    if stored_path.is_absolute():
        candidates.append(stored_path)

    candidates.extend(
        [
            region_json_path.parent / stored_path,
            region_json_path.parent / filename,
            region_json_path.parent.parent / stored_path,
            region_json_path.parent.parent / "MCOA_Normal_images" / filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Last-resort exact filename search within the local project subtree.
    project_root = region_json_path.parent.parent
    matches = list(
        dict.fromkeys(
            path.resolve()
            for path in project_root.rglob(filename)
            if path.is_file()
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileNotFoundError(
            "Found multiple possible OCT images for "
            f"{region_json_path}.\nMatches:\n"
            + "\n".join(f"  - {match}" for match in matches)
        )

    raise FileNotFoundError(
        "Could not locate the original OCT image for "
        f"{region_json_path}.\nStored imagePath: {image_value}"
    )


def load_image_from_labelme(source_json_path: Path) -> np.ndarray:
    data = load_json(source_json_path)

    image_data = data.get("imageData")
    if image_data:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        return np.asarray(image)

    image_value = data.get("imagePath")
    if not image_value:
        raise ValueError(
            f"{source_json_path} contains neither imageData nor imagePath."
        )

    normalized = str(image_value).replace("\\", "/")
    image_path = Path(normalized)

    if not image_path.is_absolute():
        image_path = source_json_path.parent / image_path

    if not image_path.exists():
        fallback = source_json_path.parent / Path(normalized).name
        if fallback.exists():
            image_path = fallback
        else:
            raise FileNotFoundError(
                f"Could not locate OCT image for {source_json_path}.\n"
                f"Tried: {image_path}\nTried: {fallback}"
            )

    return np.asarray(Image.open(image_path).convert("L"))


def load_image_from_path(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"OCT image does not exist: {image_path}")
    return np.asarray(Image.open(image_path).convert("L"))


def load_image_for_case(case: SourceCase) -> np.ndarray:
    if case.source_json_path is not None:
        return load_image_from_labelme(case.source_json_path)
    if case.image_path is not None:
        return load_image_from_path(case.image_path)
    raise RuntimeError("SourceCase has neither source_json_path nor image_path.")


# ============================================================
# Find/build source cases
# ============================================================

def collect_region_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []

    for path in paths:
        if path.is_file():
            if path.suffix.lower() == ".json":
                files.append(path.resolve())
        elif path.is_dir():
            files.extend(
                file.resolve()
                for file in path.rglob("*.json")
            )
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    return list(dict.fromkeys(files))


def _source_id_from_path(path: Path) -> str:
    resolved = str(path.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def build_source_cases(json_files: list[Path]) -> list[SourceCase]:
    """
    Build one case per original OCT image.

    Only canonical annotations containing anterior boundary, posterior
    boundary, and anterior region labels are accepted.
    """

    cases: list[SourceCase] = []
    seen_sources: set[str] = set()

    for region_json_path in json_files:
        data = load_json(region_json_path)

        required = (
            "anterior_boundary_points",
            "posterior_boundary_points",
            "anterior_boundary_region_labels",
        )
        if not all(key in data for key in required):
            continue

        source_value = data.get("source_json")

        if source_value:
            source_json_path = resolve_source_json(region_json_path, data)
            image_path = None
            source_key = str(source_json_path.resolve())
            source_id = _source_id_from_path(source_json_path)
        else:
            source_json_path = None
            image_path = resolve_image_only_path(region_json_path, data)
            source_key = str(image_path.resolve())
            source_id = _source_id_from_path(image_path)

        if source_key in seen_sources:
            continue

        cases.append(
            SourceCase(
                region_json_path=region_json_path,
                source_json_path=source_json_path,
                image_path=image_path,
                source_id=source_id,
            )
        )
        seen_sources.add(source_key)

    return cases


# ============================================================
# Source-level train/validation split
# ============================================================

def split_cases(
    cases: list[SourceCase],
    train_fraction: float = 0.80,
    seed: int = 42,
) -> dict[str, list[SourceCase]]:
    """
    Split ORIGINAL OCT images before any patches are generated.

    With exactly 50 cases and train_fraction=0.80 this gives 40 train
    and 10 validation cases. There is intentionally no internal test
    split; the external test set should remain separate and untouched.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    shuffled = list(cases)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total < 2:
        raise ValueError("At least two source images are needed.")

    n_train = int(round(n_total * train_fraction))
    n_train = max(1, min(n_train, n_total - 1))

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:],
    }


# ============================================================
# Read canonical boundary annotation
# ============================================================

def read_boundary_annotation(
    region_json_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        anterior_boundary: (N, 2), sorted left->right
        posterior_boundary: (M, 2), sorted left->right
        anterior_region_labels: (N,)
    """

    data = load_json(region_json_path)

    anterior_points = data.get("anterior_boundary_points")
    posterior_points = data.get("posterior_boundary_points")
    labels = data.get("anterior_boundary_region_labels")

    if anterior_points is None or posterior_points is None or labels is None:
        raise ValueError(
            f"Missing anterior/posterior boundary annotation in "
            f"{region_json_path}"
        )

    anterior = np.asarray(anterior_points, dtype=np.float64)
    posterior = np.asarray(posterior_points, dtype=np.float64)
    region_labels = np.asarray(labels, dtype=np.int32)

    if anterior.ndim != 2 or anterior.shape[1] != 2:
        raise ValueError("anterior_boundary_points must have shape (N, 2).")
    if posterior.ndim != 2 or posterior.shape[1] != 2:
        raise ValueError("posterior_boundary_points must have shape (M, 2).")
    if region_labels.ndim != 1 or len(region_labels) != len(anterior):
        raise ValueError("Anterior boundary points and labels do not match.")
    if len(anterior) < 2 or len(posterior) < 2:
        raise ValueError("Anterior and posterior boundaries need >=2 points.")

    valid_labels = {LEFT_LIMBUS, CORNEA, RIGHT_LIMBUS}
    found = set(np.unique(region_labels).tolist())
    if not found.issubset(valid_labels):
        raise ValueError(f"Unexpected anterior labels found: {found}")

    anterior_order = np.argsort(anterior[:, 0])
    anterior = anterior[anterior_order]
    region_labels = region_labels[anterior_order]

    posterior_order = np.argsort(posterior[:, 0])
    posterior = posterior[posterior_order]

    return anterior, posterior, region_labels


def _unique_xy_for_interpolation(boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate x coordinates by averaging their y values."""

    x = boundary[:, 0]
    y = boundary[:, 1]

    unique_x, inverse = np.unique(x, return_inverse=True)
    y_sum = np.zeros(len(unique_x), dtype=np.float64)
    counts = np.zeros(len(unique_x), dtype=np.float64)

    np.add.at(y_sum, inverse, y)
    np.add.at(counts, inverse, 1.0)

    unique_y = y_sum / counts
    return unique_x, unique_y


# ============================================================
# Patch label
# ============================================================

def determine_patch_label(
    anterior_boundary: np.ndarray,
    region_labels: np.ndarray,
    patch_left: int,
    patch_right: int,
    center_x: int,
) -> tuple[int, float, float, str] | None:
    """
    Binary target from the annotated ANTERIOR boundary.

    Majority rule:
        original 0/2 -> limbus
        original 1   -> cornea

    Exact 50/50 tie:
        use the anterior-boundary label nearest the patch center.
    """

    inside = (
        (anterior_boundary[:, 0] >= patch_left)
        & (anterior_boundary[:, 0] < patch_right)
    )

    labels_inside = region_labels[inside]
    if len(labels_inside) == 0:
        return None

    cornea_fraction = float(np.mean(labels_inside == CORNEA))
    limbus_mask = (labels_inside == LEFT_LIMBUS) | (labels_inside == RIGHT_LIMBUS)
    limbus_fraction = float(np.mean(limbus_mask))

    if cornea_fraction > limbus_fraction:
        training_label = CORNEA_CLASS
    elif limbus_fraction > cornea_fraction:
        training_label = LIMBUS_CLASS
    else:
        center_index = int(
            np.argmin(np.abs(anterior_boundary[:, 0] - center_x))
        )
        center_region = int(region_labels[center_index])
        training_label = (
            CORNEA_CLASS if center_region == CORNEA else LIMBUS_CLASS
        )

    left_fraction = float(np.mean(labels_inside == LEFT_LIMBUS))
    right_fraction = float(np.mean(labels_inside == RIGHT_LIMBUS))

    if training_label == CORNEA_CLASS:
        limbus_side = "none"
    elif left_fraction > right_fraction:
        limbus_side = "left"
    elif right_fraction > left_fraction:
        limbus_side = "right"
    else:
        limbus_side = "unknown"

    return (
        training_label,
        cornea_fraction,
        limbus_fraction,
        limbus_side,
    )


# ============================================================
# Anatomical patch extraction
# ============================================================

def extract_anatomical_patch(
    image: np.ndarray,
    anterior_boundary: np.ndarray,
    posterior_boundary: np.ndarray,
    patch_left: int,
    patch_width: int,
    margin_above_anterior: int = 15,
    mask_value: int = 0,
) -> np.ndarray | None:
    """
    Extract a variable-height anatomical patch.

    Horizontal extent:
        fixed patch_width in ORIGINAL OCT pixels.

    Vertical retained anatomy, column by column:
        anterior_y(x) - margin_above_anterior
        through
        posterior_y(x)

    Everything outside that band is set to mask_value. There is no
    bottom margin below the posterior boundary.

    The caller must ensure the full horizontal patch lies inside the
    x-overlap of the annotated anterior and posterior boundaries.
    """

    if image.ndim != 2:
        raise ValueError("Expected grayscale OCT image with shape (H, W).")

    image_height, image_width = image.shape
    patch_right = patch_left + patch_width

    if patch_left < 0 or patch_right > image_width:
        return None

    anterior_x, anterior_y_values = _unique_xy_for_interpolation(anterior_boundary)
    posterior_x, posterior_y_values = _unique_xy_for_interpolation(posterior_boundary)

    global_x = np.arange(patch_left, patch_right, dtype=np.float64)

    # No boundary extrapolation is allowed.
    if (
        global_x[0] < anterior_x.min()
        or global_x[-1] > anterior_x.max()
        or global_x[0] < posterior_x.min()
        or global_x[-1] > posterior_x.max()
    ):
        return None

    anterior_y = np.interp(global_x, anterior_x, anterior_y_values)
    posterior_y = np.interp(global_x, posterior_x, posterior_y_values)

    # Anatomically invalid annotation/crossing.
    if np.any(posterior_y < anterior_y):
        return None

    upper_y = anterior_y - float(margin_above_anterior)
    lower_y = posterior_y

    # Bounding rectangle that contains the entire valid band.
    crop_top = max(0, int(np.floor(np.min(upper_y))))
    crop_bottom = min(image_height, int(np.ceil(np.max(lower_y))) + 1)

    if crop_bottom <= crop_top:
        return None

    patch = image[crop_top:crop_bottom, patch_left:patch_right].copy()

    # Column-wise anatomical mask.
    global_rows = np.arange(crop_top, crop_bottom, dtype=np.float64)[:, None]
    keep = (
        (global_rows >= upper_y[None, :])
        & (global_rows <= lower_y[None, :])
    )

    patch[~keep] = np.asarray(mask_value, dtype=patch.dtype)
    return patch


# ============================================================
# Aspect-ratio-preserving resize + zero padding
# ============================================================

def resize_and_pad_to_square(
    patch: np.ndarray,
    output_size: int = 224,
    padding_value: int = 0,
) -> Image.Image:
    """
    Preserve aspect ratio, fit the whole patch inside output_size^2,
    and pad the rest. Nothing is cropped.

    Output is RGB because ImageNet-pretrained ConvNeXt expects 3 channels.
    The original OCT grayscale content is simply repeated across R/G/B.
    """

    if patch.ndim != 2:
        raise ValueError("Expected a 2-D grayscale patch.")

    height, width = patch.shape
    if height <= 0 or width <= 0:
        raise ValueError("Patch has invalid dimensions.")

    scale = min(output_size / width, output_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    pil_patch = Image.fromarray(patch, mode="L")
    resized = pil_patch.resize(
        (new_width, new_height),
        resample=Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "L",
        (output_size, output_size),
        color=int(padding_value),
    )

    left = (output_size - new_width) // 2
    top = (output_size - new_height) // 2
    canvas.paste(resized, (left, top))

    return canvas.convert("RGB")


# ============================================================
# Dataset
# ============================================================

class OCTPatchDataset(Dataset):
    """
    Sliding-window OCT dataset.

    Patches are generated ON THE FLY and are never written to disk.
    """

    def __init__(
        self,
        cases: list[SourceCase],
        transform: Callable | None = None,
        patch_width: int = 256,
        patch_step: int = 128,
        margin_above_anterior: int = 15,
        output_size: int = 224,
        mask_value: int = 0,
        padding_value: int = 0,
        return_metadata: bool = False,
    ) -> None:
        self.cases = cases
        self.transform = transform
        self.patch_width = patch_width
        self.patch_step = patch_step
        self.margin_above_anterior = margin_above_anterior
        self.output_size = output_size
        self.mask_value = mask_value
        self.padding_value = padding_value
        self.return_metadata = return_metadata

        self.samples: list[PatchInfo] = []
        self.image_cache: dict[int, np.ndarray] = {}
        self.annotation_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self._build_patch_index()

    def _get_image(self, source_index: int) -> np.ndarray:
        if source_index not in self.image_cache:
            self.image_cache[source_index] = load_image_for_case(
                self.cases[source_index]
            )
        return self.image_cache[source_index]

    def _get_annotation(
        self,
        source_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if source_index not in self.annotation_cache:
            self.annotation_cache[source_index] = read_boundary_annotation(
                self.cases[source_index].region_json_path
            )
        return self.annotation_cache[source_index]

    def _build_patch_index(self) -> None:
        half_width = self.patch_width // 2

        for source_index, case in enumerate(self.cases):
            anterior, posterior, region_labels = self._get_annotation(source_index)
            image = self._get_image(source_index)
            _, image_width = image.shape

            # Use ONLY the common x-range where both boundaries are annotated.
            common_start_x = int(
                np.ceil(max(anterior[:, 0].min(), posterior[:, 0].min()))
            )
            common_end_x = int(
                np.floor(min(anterior[:, 0].max(), posterior[:, 0].max()))
            )

            # Also respect the image itself.
            common_start_x = max(common_start_x, 0)
            common_end_x = min(common_end_x, image_width - 1)

            start_center_x = common_start_x + half_width
            # patch_right is exclusive, so the last included x is right-1.
            end_center_x = common_end_x - (self.patch_width - half_width - 1)

            if end_center_x < start_center_x:
                continue

            for center_x in range(
                start_center_x,
                end_center_x + 1,
                self.patch_step,
            ):
                patch_left = center_x - half_width
                patch_right = patch_left + self.patch_width

                # Full patch must stay in image and in common annotation range.
                if patch_left < common_start_x:
                    continue
                if patch_right - 1 > common_end_x:
                    continue
                if patch_left < 0 or patch_right > image_width:
                    continue

                result = determine_patch_label(
                    anterior_boundary=anterior,
                    region_labels=region_labels,
                    patch_left=patch_left,
                    patch_right=patch_right,
                    center_x=center_x,
                )
                if result is None:
                    continue

                # Validate that this patch can actually be anatomically extracted.
                patch = extract_anatomical_patch(
                    image=image,
                    anterior_boundary=anterior,
                    posterior_boundary=posterior,
                    patch_left=patch_left,
                    patch_width=self.patch_width,
                    margin_above_anterior=self.margin_above_anterior,
                    mask_value=self.mask_value,
                )
                if patch is None:
                    continue

                label, cornea_fraction, limbus_fraction, limbus_side = result

                self.samples.append(
                    PatchInfo(
                        source_index=source_index,
                        center_x=center_x,
                        left=patch_left,
                        right=patch_right,
                        label=label,
                        cornea_fraction=cornea_fraction,
                        limbus_fraction=limbus_fraction,
                        limbus_side=limbus_side,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._get_image(sample.source_index)
        anterior, posterior, _ = self._get_annotation(sample.source_index)

        patch = extract_anatomical_patch(
            image=image,
            anterior_boundary=anterior,
            posterior_boundary=posterior,
            patch_left=sample.left,
            patch_width=self.patch_width,
            margin_above_anterior=self.margin_above_anterior,
            mask_value=self.mask_value,
        )

        if patch is None:
            raise RuntimeError(
                f"Patch became invalid at dataset index {index}. "
                "The source files may have changed after indexing."
            )

        image_for_model = resize_and_pad_to_square(
            patch=patch,
            output_size=self.output_size,
            padding_value=self.padding_value,
        )

        if self.transform is not None:
            image_for_model = self.transform(image_for_model)

        target = torch.tensor(sample.label, dtype=torch.long)

        if not self.return_metadata:
            return image_for_model, target

        case = self.cases[sample.source_index]
        metadata = {
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
            "center_x": sample.center_x,
            "patch_left": sample.left,
            "patch_right": sample.right,
            "label": sample.label,
            "class_name": CLASS_NAMES[sample.label],
            "cornea_fraction": sample.cornea_fraction,
            "limbus_fraction": sample.limbus_fraction,
            "limbus_side": sample.limbus_side,
            "raw_patch_shape": tuple(int(v) for v in patch.shape),
        }

        return image_for_model, target, metadata

    def class_counts(self) -> dict[str, int]:
        counts = {
            "limbus": 0,
            "cornea": 0,
        }

        for sample in self.samples:
            counts[CLASS_NAMES[sample.label]] += 1

        return counts

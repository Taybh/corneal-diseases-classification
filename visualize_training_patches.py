from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oct_patch_dataset import (
    OCTPatchDataset,
    build_source_cases,
    collect_region_json_files,
    split_cases,
)


# ============================================================
# Settings
# ============================================================

SEED = 42

PATCH_WIDTH = 256
PATCH_STEP = 64

# Keep this much real OCT above the anterior boundary.
MARGIN_ABOVE_ANTERIOR = 30

# IMPORTANT:
# There is deliberately NO margin below the posterior boundary.
# Everything immediately below posterior is masked.
MARGIN_BELOW_POSTERIOR = 0

MASK_VALUE = 0

# Small safety padding used only for the rectangular display crop.
# This does NOT mean these pixels survive the mask.
DISPLAY_BOTTOM_PADDING = 10


# ============================================================
# Read anterior + posterior annotation
# ============================================================

def read_anatomical_boundaries(
    region_json_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read:

        anterior_boundary_points
        anterior_boundary_region_labels
        posterior_boundary_points

    Both boundaries are returned in left-to-right order.
    """

    with region_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    anterior_points = data.get(
        "anterior_boundary_points"
    )

    posterior_points = data.get(
        "posterior_boundary_points"
    )

    region_labels = data.get(
        "anterior_boundary_region_labels"
    )

    if anterior_points is None:
        raise ValueError(
            f"Missing anterior_boundary_points in "
            f"{region_json_path}"
        )

    if posterior_points is None:
        raise ValueError(
            f"Missing posterior_boundary_points in "
            f"{region_json_path}"
        )

    if region_labels is None:
        raise ValueError(
            f"Missing anterior_boundary_region_labels in "
            f"{region_json_path}"
        )

    anterior = np.asarray(
        anterior_points,
        dtype=np.float64,
    )

    posterior = np.asarray(
        posterior_points,
        dtype=np.float64,
    )

    labels = np.asarray(
        region_labels,
        dtype=np.int32,
    )

    if (
        anterior.ndim != 2
        or anterior.shape[1] != 2
    ):
        raise ValueError(
            "anterior_boundary_points must have shape (N, 2)."
        )

    if (
        posterior.ndim != 2
        or posterior.shape[1] != 2
    ):
        raise ValueError(
            "posterior_boundary_points must have shape (N, 2)."
        )

    if len(anterior) != len(labels):
        raise ValueError(
            "Anterior boundary and region labels do not match."
        )

    if len(anterior) < 2:
        raise ValueError(
            "Too few anterior boundary points."
        )

    if len(posterior) < 2:
        raise ValueError(
            "Too few posterior boundary points."
        )

    # --------------------------------------------------------
    # Make both boundaries left -> right.
    # --------------------------------------------------------

    anterior_order = np.argsort(
        anterior[:, 0]
    )

    anterior = anterior[
        anterior_order
    ]

    labels = labels[
        anterior_order
    ]

    posterior_order = np.argsort(
        posterior[:, 0]
    )

    posterior = posterior[
        posterior_order
    ]

    return (
        anterior,
        posterior,
        labels,
    )


# ============================================================
# Build dataset
# ============================================================

def create_dataset(
    input_paths: list[Path],
) -> OCTPatchDataset:

    json_files = collect_region_json_files(
        input_paths
    )

    cases = build_source_cases(
        json_files
    )

    # --------------------------------------------------------
    # Only retain annotations that actually contain a
    # posterior boundary.
    # --------------------------------------------------------

    cases_with_posterior = []

    for case in cases:

        try:

            with case.region_json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            posterior = data.get(
                "posterior_boundary_points"
            )

            if (
                posterior is not None
                and len(posterior) >= 2
            ):
                cases_with_posterior.append(
                    case
                )

        except Exception as error:

            print(
                f"Skipping {case.region_json_path.name}: "
                f"{error}"
            )

    cases = cases_with_posterior

    if len(cases) == 0:
        raise RuntimeError(
            "No OCT annotations containing a posterior "
            "boundary were found."
        )

    print(
        f"Found {len(cases)} OCT images with "
        f"anterior + posterior annotations."
    )

    # --------------------------------------------------------
    # Same source-level split strategy as training.
    # --------------------------------------------------------

    splits = split_cases(
        cases=cases,
        train_fraction=0.70,
        validation_fraction=0.15,
        seed=SEED,
    )

    print(
        f"Training images: {len(splits['train'])}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We still use the existing OCTPatchDataset only to
    # generate:
    #
    #   - horizontal patch locations
    #   - labels
    #   - cornea/limbus fractions
    #
    # We DO NOT use its fixed 192-pixel crop for the
    # visualization below.
    #
    # A large temporary patch height is NOT needed here
    # because we construct the anatomical crop ourselves.
    # --------------------------------------------------------

    dataset = OCTPatchDataset(
        cases=splits["train"],
        transform=None,
        patch_width=PATCH_WIDTH,

        # Keep existing value only so OCTPatchDataset can
        # generate its sample index.
        patch_height=192,

        patch_step=PATCH_STEP,
        depth_above_boundary=MARGIN_ABOVE_ANTERIOR,
        vertical_jitter=0,
        return_metadata=True,
    )

    print(
        f"Indexed patches: {len(dataset)}"
    )

    print(
        "Class counts:",
        dataset.class_counts(),
    )

    return dataset


# ============================================================
# Collapse duplicate x coordinates
# ============================================================

def collapse_duplicate_x(
    boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    x = boundary[:, 0]
    y = boundary[:, 1]

    order = np.argsort(
        x
    )

    x = x[
        order
    ]

    y = y[
        order
    ]

    unique_x = np.unique(
        x
    )

    unique_y = np.zeros(
        len(unique_x),
        dtype=np.float64,
    )

    for i, x_value in enumerate(
        unique_x
    ):

        same_x = (
            x == x_value
        )

        unique_y[i] = np.mean(
            y[
                same_x
            ]
        )

    return (
        unique_x,
        unique_y,
    )


# ============================================================
# Interpolate one boundary over requested x coordinates
# ============================================================

def interpolate_boundary(
    boundary: np.ndarray,
    global_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a boundary only where it is actually annotated.

    IMPORTANT:
    We do NOT extrapolate beyond the annotated x-range.
    """

    boundary_x, boundary_y = (
        collapse_duplicate_x(
            boundary
        )
    )

    valid = (
        (global_x >= boundary_x.min())
        &
        (global_x <= boundary_x.max())
    )

    interpolated_y = np.full(
        global_x.shape,
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid):

        interpolated_y[
            valid
        ] = np.interp(
            global_x[
                valid
            ],
            boundary_x,
            boundary_y,
        )

    return (
        interpolated_y,
        valid,
    )


# ============================================================
# Create one anatomical patch
# ============================================================

def create_anatomical_patch(
    dataset: OCTPatchDataset,
    sample_index: int,
) -> dict:

    sample = dataset.samples[
        sample_index
    ]

    case = dataset.cases[
        sample.source_index
    ]

    image = dataset._get_image(
        sample.source_index
    )

    (
        anterior,
        posterior,
        _,
    ) = read_anatomical_boundaries(
        case.region_json_path
    )

    image_height, image_width = (
        image.shape
    )

    patch_left = int(
        sample.left
    )

    patch_right = (
        patch_left
        + PATCH_WIDTH
    )

    if (
        patch_left < 0
        or patch_right > image_width
    ):
        raise RuntimeError(
            "Horizontal patch lies outside image."
        )

    # --------------------------------------------------------
    # Every x-column belonging to this patch in ORIGINAL
    # image coordinates.
    # --------------------------------------------------------

    global_x = np.arange(
        patch_left,
        patch_right,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Interpolate both anatomical boundaries.
    # --------------------------------------------------------

    (
        anterior_y,
        anterior_valid,
    ) = interpolate_boundary(
        anterior,
        global_x,
    )

    (
        posterior_y,
        posterior_valid,
    ) = interpolate_boundary(
        posterior,
        global_x,
    )

    # --------------------------------------------------------
    # For the final anatomical mask we need BOTH boundaries.
    #
    # If posterior was not annotated for a column, we do not
    # invent/extrapolate it.
    # --------------------------------------------------------

    both_valid = (
        anterior_valid
        &
        posterior_valid
        &
        np.isfinite(anterior_y)
        &
        np.isfinite(posterior_y)
    )

    if np.sum(both_valid) < 2:
        raise RuntimeError(
            "This patch has insufficient overlap between "
            "the anterior and posterior annotations."
        )

    # --------------------------------------------------------
    # Sanity check:
    #
    # In normal image coordinates, y increases downward.
    # Posterior should therefore be below anterior:
    #
    # posterior_y >= anterior_y
    # --------------------------------------------------------

    anatomically_valid = (
        both_valid
        &
        (posterior_y >= anterior_y)
    )

    if np.sum(anatomically_valid) < 2:
        raise RuntimeError(
            "Posterior boundary is not below anterior "
            "boundary in this patch."
        )

    # --------------------------------------------------------
    # Determine the rectangular DISPLAY crop.
    #
    # TOP:
    # 30 px above the highest anterior point.
    #
    # BOTTOM:
    # just beyond the deepest posterior point, plus a tiny
    # display-only padding so that we can visually see that
    # pixels underneath have been masked.
    #
    # This padding is NOT retained by the anatomical mask.
    # --------------------------------------------------------

    valid_anterior_values = (
        anterior_y[
            anatomically_valid
        ]
    )

    valid_posterior_values = (
        posterior_y[
            anatomically_valid
        ]
    )

    crop_top = int(
        np.floor(
            np.min(
                valid_anterior_values
            )
            - MARGIN_ABOVE_ANTERIOR
        )
    )

    crop_bottom = int(
        np.ceil(
            np.max(
                valid_posterior_values
            )
            + DISPLAY_BOTTOM_PADDING
        )
    )

    crop_top = max(
        0,
        crop_top,
    )

    crop_bottom = min(
        image_height,
        crop_bottom,
    )

    if crop_bottom <= crop_top:
        raise RuntimeError(
            "Invalid anatomical crop height."
        )

    raw_patch = image[
        crop_top:crop_bottom,
        patch_left:patch_right,
    ].copy()

    patch_height = (
        raw_patch.shape[0]
    )

    # --------------------------------------------------------
    # Convert boundaries into PATCH-LOCAL coordinates.
    # --------------------------------------------------------

    local_x = np.arange(
        PATCH_WIDTH,
        dtype=np.float64,
    )

    anterior_local_y = (
        anterior_y
        - crop_top
    )

    posterior_local_y = (
        posterior_y
        - crop_top
    )

    # --------------------------------------------------------
    # Build anatomical mask.
    #
    # KEEP:
    #
    # anterior - 30 px
    #        ↓
    # tissue
    #        ↓
    # posterior
    #
    # MASK:
    #
    # everything above anterior - 30
    # everything immediately below posterior
    #
    # NO bottom margin.
    # --------------------------------------------------------

    mask = np.zeros(
        (
            patch_height,
            PATCH_WIDTH,
        ),
        dtype=bool,
    )

    for x_index in range(
        PATCH_WIDTH
    ):

        if not anatomically_valid[
            x_index
        ]:
            continue

        anterior_value = (
            anterior_local_y[
                x_index
            ]
        )

        posterior_value = (
            posterior_local_y[
                x_index
            ]
        )

        keep_top = int(
            np.floor(
                anterior_value
                - MARGIN_ABOVE_ANTERIOR
            )
        )

        # ----------------------------------------------------
        # +1 because Python slice endpoint is exclusive.
        #
        # We want the posterior boundary pixel itself to stay.
        # Everything AFTER it is masked.
        # ----------------------------------------------------

        keep_bottom = int(
            np.floor(
                posterior_value
            )
        ) + 1

        keep_top = int(
            np.clip(
                keep_top,
                0,
                patch_height,
            )
        )

        keep_bottom = int(
            np.clip(
                keep_bottom,
                0,
                patch_height,
            )
        )

        if keep_bottom > keep_top:

            mask[
                keep_top:keep_bottom,
                x_index
            ] = True

    # --------------------------------------------------------
    # Apply mask.
    # --------------------------------------------------------

    masked_patch = (
        raw_patch.copy()
    )

    masked_patch[
        ~mask
    ] = MASK_VALUE

    return {
        "raw_patch":
            raw_patch,

        "masked_patch":
            masked_patch,

        "local_x":
            local_x,

        "anterior_y":
            anterior_local_y,

        "posterior_y":
            posterior_local_y,

        "valid":
            anatomically_valid,

        "crop_top":
            crop_top,

        "crop_bottom":
            crop_bottom,
    }


# ============================================================
# Check whether a patch has usable posterior coverage
# ============================================================

def patch_has_posterior_coverage(
    dataset: OCTPatchDataset,
    sample_index: int,
    minimum_fraction: float = 0.95,
) -> bool:
    """
    Require posterior annotation across almost the whole patch.

    This avoids selecting a visualization patch where most
    columns have no posterior boundary.
    """

    try:

        sample = dataset.samples[
            sample_index
        ]

        case = dataset.cases[
            sample.source_index
        ]

        (
            anterior,
            posterior,
            _,
        ) = read_anatomical_boundaries(
            case.region_json_path
        )

        global_x = np.arange(
            sample.left,
            sample.left + PATCH_WIDTH,
            dtype=np.float64,
        )

        (
            anterior_y,
            anterior_valid,
        ) = interpolate_boundary(
            anterior,
            global_x,
        )

        (
            posterior_y,
            posterior_valid,
        ) = interpolate_boundary(
            posterior,
            global_x,
        )

        valid = (
            anterior_valid
            &
            posterior_valid
            &
            np.isfinite(
                anterior_y
            )
            &
            np.isfinite(
                posterior_y
            )
            &
            (
                posterior_y
                >= anterior_y
            )
        )

        coverage = float(
            np.mean(
                valid
            )
        )

        return (
            coverage
            >= minimum_fraction
        )

    except Exception:

        return False


# ============================================================
# Find one source containing usable left/cornea/right patches
# ============================================================

def find_source_with_all_regions(
    dataset: OCTPatchDataset,
) -> int:

    source_indices = sorted(
        {
            sample.source_index
            for sample in dataset.samples
        }
    )

    for source_index in source_indices:

        indexed_samples = [
            (
                index,
                sample,
            )
            for index, sample in enumerate(
                dataset.samples
            )
            if sample.source_index
            == source_index
        ]

        has_left = any(
            (
                sample.limbus_side == "left"
                and
                sample.limbus_fraction >= 0.80
                and
                patch_has_posterior_coverage(
                    dataset,
                    index,
                )
            )
            for index, sample
            in indexed_samples
        )

        has_cornea = any(
            (
                sample.cornea_fraction >= 0.80
                and
                patch_has_posterior_coverage(
                    dataset,
                    index,
                )
            )
            for index, sample
            in indexed_samples
        )

        has_right = any(
            (
                sample.limbus_side == "right"
                and
                sample.limbus_fraction >= 0.80
                and
                patch_has_posterior_coverage(
                    dataset,
                    index,
                )
            )
            for index, sample
            in indexed_samples
        )

        if (
            has_left
            and has_cornea
            and has_right
        ):
            return source_index

    raise RuntimeError(
        "Could not find one OCT containing clear "
        "left-limbus, cornea, and right-limbus patches "
        "with sufficient posterior-boundary coverage."
    )


# ============================================================
# Select representative patches
# ============================================================

def select_three_patches(
    dataset: OCTPatchDataset,
    source_index: int,
) -> dict[str, int]:

    indexed_samples = [
        (
            index,
            sample,
        )
        for index, sample in enumerate(
            dataset.samples
        )
        if sample.source_index
        == source_index
    ]

    # ========================================================
    # LEFT LIMBUS
    # ========================================================

    left_candidates = [
        (
            index,
            sample,
        )
        for index, sample
        in indexed_samples
        if (
            sample.limbus_side == "left"
            and
            sample.limbus_fraction >= 0.80
            and
            patch_has_posterior_coverage(
                dataset,
                index,
            )
        )
    ]

    if len(left_candidates) == 0:
        raise RuntimeError(
            "No clear usable left-limbus patch found."
        )

    left_index, _ = min(
        left_candidates,
        key=lambda item:
            item[1].center_x,
    )

    # ========================================================
    # RIGHT LIMBUS
    # ========================================================

    right_candidates = [
        (
            index,
            sample,
        )
        for index, sample
        in indexed_samples
        if (
            sample.limbus_side == "right"
            and
            sample.limbus_fraction >= 0.80
            and
            patch_has_posterior_coverage(
                dataset,
                index,
            )
        )
    ]

    if len(right_candidates) == 0:
        raise RuntimeError(
            "No clear usable right-limbus patch found."
        )

    right_index, _ = max(
        right_candidates,
        key=lambda item:
            item[1].center_x,
    )

    # ========================================================
    # CORNEA
    # ========================================================

    cornea_candidates = [
        (
            index,
            sample,
        )
        for index, sample
        in indexed_samples
        if (
            sample.cornea_fraction >= 0.80
            and
            patch_has_posterior_coverage(
                dataset,
                index,
            )
        )
    ]

    if len(cornea_candidates) == 0:
        raise RuntimeError(
            "No clear usable corneal patch found."
        )

    cornea_x_values = [
        sample.center_x
        for _, sample
        in cornea_candidates
    ]

    cornea_middle_x = (
        min(
            cornea_x_values
        )
        +
        max(
            cornea_x_values
        )
    ) / 2.0

    cornea_index, _ = min(
        cornea_candidates,
        key=lambda item: abs(
            item[1].center_x
            - cornea_middle_x
        ),
    )

    return {
        "left":
            left_index,

        "cornea":
            cornea_index,

        "right":
            right_index,
    }


# ============================================================
# Visualization
# ============================================================

def visualize_three_patches(
    dataset: OCTPatchDataset,
    selected: dict[str, int],
    save_path: Path | None = None,
) -> None:

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
    )

    positions = [
        (
            "left",
            "LEFT LIMBUS",
        ),
        (
            "cornea",
            "MIDDLE CORNEA",
        ),
        (
            "right",
            "RIGHT LIMBUS",
        ),
    ]

    for column, (
        key,
        title,
    ) in enumerate(
        positions
    ):

        sample_index = selected[
            key
        ]

        sample = dataset.samples[
            sample_index
        ]

        result = create_anatomical_patch(
            dataset=dataset,
            sample_index=sample_index,
        )

        raw_patch = result[
            "raw_patch"
        ]

        masked_patch = result[
            "masked_patch"
        ]

        local_x = result[
            "local_x"
        ]

        anterior_y = result[
            "anterior_y"
        ]

        posterior_y = result[
            "posterior_y"
        ]

        valid = result[
            "valid"
        ]

        # ====================================================
        # RAW
        # ====================================================

        raw_axis = axes[
            0,
            column,
        ]

        raw_axis.imshow(
            raw_patch,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        # ----------------------------------------------------
        # Anterior boundary
        # ----------------------------------------------------

        raw_axis.plot(
            local_x[
                valid
            ],
            anterior_y[
                valid
            ],
            linewidth=1.2,
            label="Anterior",
        )

        # ----------------------------------------------------
        # 30 px above anterior
        # ----------------------------------------------------

        raw_axis.plot(
            local_x[
                valid
            ],
            (
                anterior_y[
                    valid
                ]
                - MARGIN_ABOVE_ANTERIOR
            ),
            linestyle="--",
            linewidth=1.0,
            label="Upper mask",
        )

        # ----------------------------------------------------
        # Posterior boundary = EXACT lower cutoff.
        # ----------------------------------------------------

        raw_axis.plot(
            local_x[
                valid
            ],
            posterior_y[
                valid
            ],
            linewidth=1.2,
            label="Posterior / lower mask",
        )

        raw_axis.set_title(
            f"{title}\n"
            f"C={sample.cornea_fraction:.2f} | "
            f"L={sample.limbus_fraction:.2f}",
            fontsize=11,
        )

        raw_axis.set_xticks([])
        raw_axis.set_yticks([])

        # ====================================================
        # MASKED
        # ====================================================

        masked_axis = axes[
            1,
            column,
        ]

        masked_axis.imshow(
            masked_patch,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        masked_axis.set_title(
            "Anatomically masked input",
            fontsize=11,
        )

        masked_axis.set_xticks([])
        masked_axis.set_yticks([])

    # ========================================================
    # Row labels
    # ========================================================

    axes[
        0,
        0,
    ].set_ylabel(
        "RAW",
        fontsize=14,
        fontweight="bold",
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "MASKED",
        fontsize=14,
        fontweight="bold",
    )

    # ========================================================
    # Source
    # ========================================================

    source_index = dataset.samples[
        selected[
            "cornea"
        ]
    ].source_index

    case = dataset.cases[
        source_index
    ]

    figure.suptitle(
        (
            "OCT anatomical patch sanity check\n"
            f"Width = {PATCH_WIDTH}px   |   "
            f"Step = {PATCH_STEP}px   |   "
            f"Above anterior = "
            f"{MARGIN_ABOVE_ANTERIOR}px   |   "
            "Below posterior = 0px\n"
            f"Source: "
            f"{case.region_json_path.name}"
        ),
        fontsize=14,
    )

    figure.text(
        0.5,
        0.015,
        (
            "The model region keeps 30 px above the "
            "anterior boundary through the posterior "
            "boundary. Everything immediately below "
            "posterior is masked."
        ),
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[
            0.02,
            0.06,
            1.0,
            0.88,
        ]
    )

    if save_path is not None:

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print()
        print(
            f"Saved visualization to: "
            f"{save_path}"
        )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize OCT patches using anterior and "
            "posterior anatomical boundaries."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Folder(s) containing CLJ annotation JSON files."
        ),
    )

    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=(
            "Optional path for saving the visualization."
        ),
    )

    args = parser.parse_args()

    dataset = create_dataset(
        args.inputs
    )

    source_index = (
        find_source_with_all_regions(
            dataset
        )
    )

    selected = select_three_patches(
        dataset=dataset,
        source_index=source_index,
    )

    print()
    print(
        f"Selected source index: "
        f"{source_index}"
    )

    print()
    print(
        "Selected patches:"
    )

    for name, index in selected.items():

        sample = dataset.samples[
            index
        ]

        print(
            f"  {name:8s} "
            f"| patch index = {index:4d} "
            f"| x = {sample.center_x:4d} "
            f"| C = {sample.cornea_fraction:.2f} "
            f"| L = {sample.limbus_fraction:.2f} "
            f"| side = {sample.limbus_side}"
        )

    visualize_three_patches(
        dataset=dataset,
        selected=selected,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
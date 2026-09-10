from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


# ============================================================
# Labels
# ============================================================

LEFT_LABEL = "left_CLJ"
RIGHT_LABEL = "right_CLJ"


# ============================================================
# Region labels
#
# 0 = left limbus
# 1 = cornea
# 2 = right limbus
# ============================================================

LEFT_LIMBUS = 0
CORNEA = 1
RIGHT_LIMBUS = 2


CLASS_MAP = {
    "0": "left_limbus",
    "1": "cornea",
    "2": "right_limbus",
}


# ============================================================
# Visualization settings
# ============================================================

CONTROL_POINT_COLOR = "tab:blue"

FITTED_BOUNDARY_COLOR = "yellow"

LEFT_LIMBUS_COLOR = "tab:orange"
CORNEA_COLOR = "tab:green"
RIGHT_LIMBUS_COLOR = "tab:orange"

CONTROL_POINT_SIZE = 35
CLJ_POINT_SIZE = 140

FITTED_BOUNDARY_LINE_WIDTH = 1.5
REGION_LINE_WIDTH = 2.5


# ============================================================
# Boundary fitting settings
# ============================================================

# We want approximately one boundary point per x pixel,
# consistent with the JSON-based annotator.

BOUNDARY_X_STEP = 1.0


# The spline is mainly used to interpolate between the
# manually clicked control points.
#
# We do NOT strongly smooth here because the final dense
# boundary will subsequently receive the same Savitzky-Golay
# smoothing used in the JSON-based annotator.

SPLINE_SMOOTHING = 0.0


# Minimum number of points required before fitting.
#
# In practice you can use many more:
# 20, 30, 40, etc.

MIN_BOUNDARY_POINTS = 4


# ============================================================
# Supported image formats
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Load image
# ============================================================

def load_image(
    image_path: Path,
) -> np.ndarray:

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    return np.asarray(
        image
    )


# ============================================================
# Find images
# ============================================================

def collect_images(
    paths: list[Path],
) -> list[Path]:

    image_paths: list[Path] = []

    for path in paths:

        # ----------------------------------------------------
        # Individual image
        # ----------------------------------------------------

        if path.is_file():

            if (
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                image_paths.append(
                    path.resolve()
                )

        # ----------------------------------------------------
        # Folder
        # ----------------------------------------------------

        elif path.is_dir():

            for file in path.rglob("*"):

                if (
                    file.is_file()
                    and
                    file.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):

                    image_paths.append(
                        file.resolve()
                    )

        else:

            raise FileNotFoundError(
                f"Path does not exist: {path}"
            )

    # Remove duplicates and sort.
    image_paths = sorted(
        dict.fromkeys(
            image_paths
        )
    )

    if not image_paths:

        raise RuntimeError(
            "No OCT images were found."
        )

    return image_paths


# ============================================================
# Savitzky-Golay smoothing
# ============================================================

def smooth_dense_boundary(
    y: np.ndarray,
) -> np.ndarray:
    """
    Apply the SAME Savitzky-Golay smoothing logic used by
    the JSON-based annotation code.

    This is applied after the manually selected control
    points have been converted into a dense boundary.
    """

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if len(y) >= 7:

        window = min(
            101,
            len(y)
            if len(y) % 2
            else len(y) - 1,
        )

        window = max(
            7,
            window,
        )

        if window >= len(y):

            window = (
                len(y) - 1
                if len(y) % 2 == 0
                else len(y)
            )

        if window % 2 == 0:

            window -= 1

        y = savgol_filter(
            y,
            window_length=window,
            polyorder=min(
                3,
                window - 1,
            ),
        )

    return y


# ============================================================
# Fit anterior boundary
# ============================================================

def fit_boundary(
    control_points: list[
        tuple[float, float]
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert manually selected points into a dense,
    smoothed anterior boundary.

    Pipeline:

        manually clicked points
                ↓
        sort left -> right
                ↓
        remove duplicate x positions
                ↓
        spline interpolation
                ↓
        dense sampling (~1 point/x pixel)
                ↓
        Savitzky-Golay smoothing
                ↓
        final anterior boundary

    Returns:

        boundary
        sorted_control_points
    """

    if (
        len(control_points)
        <
        MIN_BOUNDARY_POINTS
    ):

        raise ValueError(
            f"Please select at least "
            f"{MIN_BOUNDARY_POINTS} boundary points."
        )

    points = np.asarray(
        control_points,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Remove invalid points
    # --------------------------------------------------------

    valid = np.isfinite(
        points
    ).all(axis=1)

    points = points[
        valid
    ]

    if (
        len(points)
        <
        MIN_BOUNDARY_POINTS
    ):

        raise ValueError(
            "Not enough valid boundary points."
        )

    # --------------------------------------------------------
    # Sort from left to right
    # --------------------------------------------------------

    order = np.argsort(
        points[:, 0]
    )

    points = points[
        order
    ]

    x = points[:, 0]
    y = points[:, 1]

    # --------------------------------------------------------
    # Collapse duplicate / almost duplicate x positions.
    #
    # This is important because spline fitting requires
    # increasing x coordinates.
    # --------------------------------------------------------

    rounded_x = np.round(
        x
    ).astype(int)

    unique_x_values = np.unique(
        rounded_x
    )

    collapsed_x = []
    collapsed_y = []

    for x_value in unique_x_values:

        mask = (
            rounded_x == x_value
        )

        collapsed_x.append(
            float(
                np.mean(
                    x[mask]
                )
            )
        )

        collapsed_y.append(
            float(
                np.median(
                    y[mask]
                )
            )
        )

    x = np.asarray(
        collapsed_x,
        dtype=np.float64,
    )

    y = np.asarray(
        collapsed_y,
        dtype=np.float64,
    )

    if (
        len(x)
        <
        MIN_BOUNDARY_POINTS
    ):

        raise ValueError(
            "Too few unique x coordinates remain "
            "after removing duplicate x positions."
        )

    # --------------------------------------------------------
    # Make sure x is strictly increasing
    # --------------------------------------------------------

    if not np.all(
        np.diff(x) > 0
    ):

        raise ValueError(
            "Boundary x coordinates are not "
            "strictly increasing."
        )

    # --------------------------------------------------------
    # Cubic spline
    #
    # With enough points:
    #     k = 3
    #
    # With very few points the degree is automatically
    # reduced.
    # --------------------------------------------------------

    spline_degree = min(
        3,
        len(x) - 1,
    )

    spline = UnivariateSpline(
        x,
        y,
        s=SPLINE_SMOOTHING,
        k=spline_degree,
    )

    # --------------------------------------------------------
    # Dense x coordinates
    #
    # Approximately one boundary point for every x pixel.
    #
    # This matches the sampling strategy of the existing
    # JSON-based annotator.
    # --------------------------------------------------------

    x_min = float(
        np.ceil(
            x.min()
        )
    )

    x_max = float(
        np.floor(
            x.max()
        )
    )

    if x_max <= x_min:

        raise ValueError(
            "Boundary has insufficient horizontal extent."
        )

    dense_x = np.arange(
        x_min,
        x_max + BOUNDARY_X_STEP,
        BOUNDARY_X_STEP,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Spline interpolation
    # --------------------------------------------------------

    dense_y = spline(
        dense_x
    )

    # --------------------------------------------------------
    # SAME final smoothing as the JSON-based annotator
    # --------------------------------------------------------

    dense_y = smooth_dense_boundary(
        dense_y
    )

    # --------------------------------------------------------
    # Final boundary
    # --------------------------------------------------------

    boundary = np.column_stack(
        (
            dense_x,
            dense_y,
        )
    )

    # --------------------------------------------------------
    # These are only used for visualization.
    # They are NOT saved in the training JSON.
    # --------------------------------------------------------

    sorted_control_points = np.column_stack(
        (
            x,
            y,
        )
    )

    return (
        boundary,
        sorted_control_points,
    )


# ============================================================
# Find closest boundary point
# ============================================================

def closest_boundary_index(
    boundary: np.ndarray,
    point: tuple[float, float],
) -> int:
    """
    Snap a mouse click to the nearest point on the final
    smoothed anterior boundary.
    """

    x_click, y_click = point

    distance_squared = (
        (boundary[:, 0] - x_click) ** 2
        +
        (boundary[:, 1] - y_click) ** 2
    )

    return int(
        np.argmin(
            distance_squared
        )
    )


# ============================================================
# Create region labels
# ============================================================

def create_region_labels(
    boundary: np.ndarray,
    left_clj_index: int,
    right_clj_index: int,
) -> np.ndarray:
    """
    Create one label for every boundary point.

        0 = left limbus
        1 = cornea
        2 = right limbus
    """

    if (
        left_clj_index
        >
        right_clj_index
    ):

        (
            left_clj_index,
            right_clj_index,
        ) = (
            right_clj_index,
            left_clj_index,
        )

    # --------------------------------------------------------
    # Start with -1, exactly like your JSON-based annotator.
    # --------------------------------------------------------

    labels = np.full(
        len(boundary),
        -1,
        dtype=np.int32,
    )

    # --------------------------------------------------------
    # Left limbus
    # --------------------------------------------------------

    labels[
        :left_clj_index
    ] = LEFT_LIMBUS

    # --------------------------------------------------------
    # Cornea
    #
    # The CLJ points themselves belong to the corneal region.
    # --------------------------------------------------------

    labels[
        left_clj_index:
        right_clj_index + 1
    ] = CORNEA

    # --------------------------------------------------------
    # Right limbus
    # --------------------------------------------------------

    labels[
        right_clj_index + 1:
    ] = RIGHT_LIMBUS

    return labels


# ============================================================
# Main annotator
# ============================================================

class ImageOnlyCLJAnnotator:

    def __init__(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> None:

        self.image_paths = image_paths

        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index = 0

        # ----------------------------------------------------
        # Current image
        # ----------------------------------------------------

        self.image: np.ndarray | None = None

        # ----------------------------------------------------
        # Manual control points
        # ----------------------------------------------------

        self.control_points: list[
            tuple[float, float]
        ] = []

        self.sorted_control_points: (
            np.ndarray | None
        ) = None

        # ----------------------------------------------------
        # Final dense smoothed boundary
        # ----------------------------------------------------

        self.boundary: (
            np.ndarray | None
        ) = None

        # ----------------------------------------------------
        # CLJ indices
        # ----------------------------------------------------

        self.left_clj_index: (
            int | None
        ) = None

        self.right_clj_index: (
            int | None
        ) = None

        # ----------------------------------------------------
        # Modes
        #
        # boundary_points
        # left_clj
        # right_clj
        # complete
        # ----------------------------------------------------

        self.mode = "boundary_points"

        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------

        self.fig, self.ax_img = plt.subplots(
            figsize=(16, 8)
        )

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_click,
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )

        self.load_current()

        self.draw()


    # ========================================================
    # Paths
    # ========================================================

    @property
    def current_image_path(
        self,
    ) -> Path:

        return self.image_paths[
            self.index
        ]


    @property
    def output_path(
        self,
    ) -> Path:

        return (
            self.output_dir
            /
            f"{self.current_image_path.stem}_CLJ.json"
        )


    # ========================================================
    # Load current image
    # ========================================================

    def load_current(
        self,
    ) -> None:

        self.image = load_image(
            self.current_image_path
        )

        # ----------------------------------------------------
        # Reset annotation state
        # ----------------------------------------------------

        self.control_points = []

        self.sorted_control_points = None

        self.boundary = None

        self.left_clj_index = None
        self.right_clj_index = None

        self.mode = "boundary_points"

        # ----------------------------------------------------
        # Load annotation if this image has already
        # been annotated.
        # ----------------------------------------------------

        if self.output_path.exists():

            self.load_existing_annotation()


    # ========================================================
    # Load existing annotation
    # ========================================================

    def load_existing_annotation(
        self,
    ) -> None:

        try:

            saved = json.loads(
                self.output_path.read_text(
                    encoding="utf-8"
                )
            )

            boundary_points = saved.get(
                "anterior_boundary_points"
            )

            left_index = saved.get(
                "left_CLJ_boundary_index"
            )

            right_index = saved.get(
                "right_CLJ_boundary_index"
            )

            if boundary_points is not None:

                self.boundary = np.asarray(
                    boundary_points,
                    dtype=np.float64,
                )

            if left_index is not None:

                self.left_clj_index = int(
                    left_index
                )

            if right_index is not None:

                self.right_clj_index = int(
                    right_index
                )

            if (
                self.boundary is not None
                and
                self.left_clj_index is not None
                and
                self.right_clj_index is not None
            ):

                self.mode = "complete"

            elif self.boundary is not None:

                self.mode = "left_clj"

            print(
                f"Loaded existing annotation: "
                f"{self.output_path.name}"
            )

        except Exception as error:

            print()
            print(
                "Could not load existing annotation:"
            )

            print(
                error
            )


    # ========================================================
    # Mouse click
    # ========================================================

    def on_click(
        self,
        event,
    ) -> None:

        if (
            event.inaxes
            !=
            self.ax_img
        ):

            return

        if (
            event.button
            !=
            MouseButton.LEFT
        ):

            return

        if (
            event.xdata is None
            or
            event.ydata is None
        ):

            return

        x = float(
            event.xdata
        )

        y = float(
            event.ydata
        )

        # ----------------------------------------------------
        # MODE 1:
        # Select boundary control points
        # ----------------------------------------------------

        if self.mode == "boundary_points":

            self.control_points.append(
                (
                    x,
                    y,
                )
            )

            print(
                f"Boundary point "
                f"{len(self.control_points)}: "
                f"({x:.1f}, {y:.1f})"
            )

            self.draw()

            return

        # ----------------------------------------------------
        # MODE 2:
        # Select LEFT CLJ
        # ----------------------------------------------------

        if (
            self.mode == "left_clj"
            and
            self.boundary is not None
        ):

            self.left_clj_index = (
                closest_boundary_index(
                    self.boundary,
                    (
                        x,
                        y,
                    ),
                )
            )

            self.mode = "right_clj"

            self.draw()

            return

        # ----------------------------------------------------
        # MODE 3:
        # Select RIGHT CLJ
        # ----------------------------------------------------

        if (
            self.mode == "right_clj"
            and
            self.boundary is not None
        ):

            self.right_clj_index = (
                closest_boundary_index(
                    self.boundary,
                    (
                        x,
                        y,
                    ),
                )
            )

            # ------------------------------------------------
            # Ensure left/right ordering
            # ------------------------------------------------

            if (
                self.left_clj_index is not None
                and
                self.right_clj_index
                <
                self.left_clj_index
            ):

                (
                    self.left_clj_index,
                    self.right_clj_index,
                ) = (
                    self.right_clj_index,
                    self.left_clj_index,
                )

            self.mode = "complete"

            self.draw()

            return

        # ----------------------------------------------------
        # MODE 4:
        #
        # Both CLJs already exist.
        #
        # Behave like your JSON annotator:
        # clicking near one CLJ moves whichever selected
        # CLJ is closer to the new click.
        # ----------------------------------------------------

        if (
            self.mode == "complete"
            and
            self.boundary is not None
            and
            self.left_clj_index is not None
            and
            self.right_clj_index is not None
        ):

            new_index = (
                closest_boundary_index(
                    self.boundary,
                    (
                        x,
                        y,
                    ),
                )
            )

            left_point = self.boundary[
                self.left_clj_index
            ]

            right_point = self.boundary[
                self.right_clj_index
            ]

            new_point = self.boundary[
                new_index
            ]

            left_distance = np.sum(
                (
                    left_point
                    -
                    new_point
                ) ** 2
            )

            right_distance = np.sum(
                (
                    right_point
                    -
                    new_point
                ) ** 2
            )

            if (
                left_distance
                <=
                right_distance
            ):

                self.left_clj_index = (
                    new_index
                )

            else:

                self.right_clj_index = (
                    new_index
                )

            # Ensure correct ordering again.
            if (
                self.left_clj_index
                >
                self.right_clj_index
            ):

                (
                    self.left_clj_index,
                    self.right_clj_index,
                ) = (
                    self.right_clj_index,
                    self.left_clj_index,
                )

            self.draw()


    # ========================================================
    # Fit boundary
    # ========================================================

    def fit_current_boundary(
        self,
    ) -> None:

        if (
            self.mode
            !=
            "boundary_points"
        ):

            return

        try:

            (
                self.boundary,
                self.sorted_control_points,
            ) = fit_boundary(
                self.control_points
            )

            self.left_clj_index = None
            self.right_clj_index = None

            self.mode = "left_clj"

            print()
            print(
                "===================================="
            )

            print(
                "Anterior boundary fitted."
            )

            print(
                f"Manual points: "
                f"{len(self.control_points)}"
            )

            print(
                f"Dense boundary points: "
                f"{len(self.boundary)}"
            )

            print()
            print(
                "Inspect the YELLOW curve carefully."
            )

            print(
                "If it follows the anterior boundary "
                "correctly, click the LEFT CLJ."
            )

            print(
                "If it is incorrect, press R and "
                "annotate the boundary again."
            )

            print(
                "===================================="
            )

        except Exception as error:

            print()
            print(
                "Could not fit anterior boundary:"
            )

            print(
                error
            )

        self.draw()


    # ========================================================
    # Remove last control point
    # ========================================================

    def remove_last_control_point(
        self,
    ) -> None:

        if (
            self.mode
            !=
            "boundary_points"
        ):

            print(
                "Boundary has already been fitted. "
                "Press R if you want to redo it."
            )

            return

        if not self.control_points:

            print(
                "No boundary points to remove."
            )

            return

        removed = (
            self.control_points.pop()
        )

        print(
            f"Removed boundary point: "
            f"({removed[0]:.1f}, "
            f"{removed[1]:.1f})"
        )

        self.draw()


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.control_points = []

        self.sorted_control_points = None

        self.boundary = None

        self.left_clj_index = None
        self.right_clj_index = None

        self.mode = "boundary_points"

        print()
        print(
            "Annotation reset."
        )

        print(
            "Start clicking points along "
            "the anterior boundary again."
        )

        self.draw()


    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
    ) -> bool:

        if self.boundary is None:

            print(
                "Fit the anterior boundary before saving."
            )

            return False

        if (
            self.left_clj_index is None
            or
            self.right_clj_index is None
        ):

            print(
                "Select both left and right CLJ "
                "points before saving."
            )

            return False

        # ----------------------------------------------------
        # Ensure ordering
        # ----------------------------------------------------

        left_index = min(
            self.left_clj_index,
            self.right_clj_index,
        )

        right_index = max(
            self.left_clj_index,
            self.right_clj_index,
        )

        # ----------------------------------------------------
        # Same safety checks as JSON annotator
        # ----------------------------------------------------

        if left_index <= 0:

            print(
                "Left CLJ is too close to the "
                "start of the boundary."
            )

            return False

        if (
            right_index
            >=
            len(self.boundary) - 1
        ):

            print(
                "Right CLJ is too close to the "
                "end of the boundary."
            )

            return False

        # ----------------------------------------------------
        # CLJ coordinates
        # ----------------------------------------------------

        left_point = self.boundary[
            left_index
        ]

        right_point = self.boundary[
            right_index
        ]

        # ----------------------------------------------------
        # Region labels
        # ----------------------------------------------------

        region_labels = (
            create_region_labels(
                boundary=self.boundary,
                left_clj_index=left_index,
                right_clj_index=right_index,
            )
        )

        # ----------------------------------------------------
        # Same representation as JSON annotator
        # ----------------------------------------------------

        boundary_points = np.asarray(
            self.boundary,
            dtype=np.float64,
        )

        # ====================================================
        # OUTPUT
        #
        # IMPORTANT:
        #
        # This structure intentionally matches the output
        # produced by your JSON-based annotator.
        # ====================================================

        output = {

            # ------------------------------------------------
            # No original LabelMe JSON exists for these cases.
            # Keep the field so the schema remains consistent.
            # ------------------------------------------------

            "source_json": None,

            # ------------------------------------------------
            # Image reference
            #
            # We save only the filename rather than an
            # absolute computer-specific path.
            # ------------------------------------------------

            "imagePath":
                self.current_image_path.name,

            # ------------------------------------------------
            # Same annotation description
            # ------------------------------------------------

            "annotation_note": (
                "The anterior boundary between left_CLJ "
                "and right_CLJ is defined as cornea. "
                "Boundary points outside those junctions "
                "are defined as limbus. The selected "
                "points are snapped to the smoothed "
                "anterior boundary and may be revised."
            ),

            # ------------------------------------------------
            # Same class map
            # ------------------------------------------------

            "class_map": {
                "0": "left_limbus",
                "1": "cornea",
                "2": "right_limbus",
            },

            # ------------------------------------------------
            # CLJs
            # ------------------------------------------------

            LEFT_LABEL:
                left_point.tolist(),

            RIGHT_LABEL:
                right_point.tolist(),

            "left_CLJ_boundary_index":
                int(
                    left_index
                ),

            "right_CLJ_boundary_index":
                int(
                    right_index
                ),

            # ------------------------------------------------
            # Final dense + smoothed anterior boundary
            # ------------------------------------------------

            "anterior_boundary_points":
                boundary_points.tolist(),

            # ------------------------------------------------
            # One anatomical label for every boundary point
            # ------------------------------------------------

            "anterior_boundary_region_labels":
                region_labels.tolist(),

            # ------------------------------------------------
            # Same region structure as your JSON annotator
            # ------------------------------------------------

            "regions": {

                "left_limbus": {

                    "start_index":
                        0,

                    "end_index":
                        left_index - 1,

                    "points":
                        boundary_points[
                            :left_index
                        ].tolist(),
                },

                "cornea": {

                    "start_index":
                        left_index,

                    "end_index":
                        right_index,

                    "points":
                        boundary_points[
                            left_index:
                            right_index + 1
                        ].tolist(),
                },

                "right_limbus": {

                    "start_index":
                        right_index + 1,

                    "end_index":
                        len(
                            boundary_points
                        ) - 1,

                    "points":
                        boundary_points[
                            right_index + 1:
                        ].tolist(),
                },
            },
        }

        # ----------------------------------------------------
        # Write JSON only.
        #
        # No annotated image is saved.
        # ----------------------------------------------------

        self.output_path.write_text(
            json.dumps(
                output,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            f"Saved: {self.output_path}"
        )

        return True


    # ========================================================
    # Move between images
    # ========================================================

    def move(
        self,
        step: int,
    ) -> None:

        new_index = (
            self.index
            +
            step
        )

        if not (
            0
            <=
            new_index
            <
            len(self.image_paths)
        ):

            print(
                "No more cases in that direction."
            )

            return

        self.index = new_index

        self.load_current()

        self.draw()


    # ========================================================
    # Save + next
    # ========================================================

    def save_and_next(
        self,
    ) -> None:

        # ----------------------------------------------------
        # Same general behavior as the JSON annotator:
        # save when the annotation is complete.
        # ----------------------------------------------------

        if (
            self.left_clj_index is not None
            and
            self.right_clj_index is not None
        ):

            if not self.save():
                return

        else:

            print(
                "Complete the annotation before "
                "moving to the next image."
            )

            return

        self.move(
            1
        )


    # ========================================================
    # Keyboard
    # ========================================================

    def on_key(
        self,
        event,
    ) -> None:

        key = (
            event.key.lower()
            if event.key
            else ""
        )

        # ----------------------------------------------------
        # ENTER:
        # finish selecting boundary points and fit boundary
        # ----------------------------------------------------

        if key in (
            "enter",
            "return",
        ):

            self.fit_current_boundary()

        # ----------------------------------------------------
        # BACKSPACE / DELETE:
        # remove most recently selected boundary point
        # ----------------------------------------------------

        elif key in (
            "backspace",
            "delete",
        ):

            self.remove_last_control_point()

        # ----------------------------------------------------
        # R:
        # reset entire annotation
        # ----------------------------------------------------

        elif key == "r":

            self.reset()

        # ----------------------------------------------------
        # S:
        # save JSON
        # ----------------------------------------------------

        elif key == "s":

            self.save()

        # ----------------------------------------------------
        # N / RIGHT:
        # save and next
        # ----------------------------------------------------

        elif key in (
            "n",
            "right",
        ):

            self.save_and_next()

        # ----------------------------------------------------
        # P / LEFT:
        # previous image
        # ----------------------------------------------------

        elif key in (
            "p",
            "left",
        ):

            self.move(
                -1
            )

        # ----------------------------------------------------
        # Q / ESC:
        # quit
        # ----------------------------------------------------

        elif key in (
            "q",
            "escape",
        ):

            plt.close(
                self.fig
            )


    # ========================================================
    # Draw
    # ========================================================

    def draw(
        self,
    ) -> None:

        self.ax_img.clear()

        if self.image is None:

            return

        # ----------------------------------------------------
        # 1. Original OCT image
        # ----------------------------------------------------

        self.ax_img.imshow(
            self.image
        )

        # ----------------------------------------------------
        # 2. Manually selected boundary points
        #
        # Only show these while creating/checking the
        # boundary.
        # ----------------------------------------------------

        if (
            self.control_points
            and
            self.mode
            ==
            "boundary_points"
        ):

            points = np.asarray(
                self.control_points,
                dtype=np.float64,
            )

            self.ax_img.scatter(
                points[:, 0],
                points[:, 1],
                s=CONTROL_POINT_SIZE,
                color=CONTROL_POINT_COLOR,
                edgecolors="white",
                linewidths=0.5,
                zorder=10,
                label=(
                    "Manual anterior-boundary points"
                ),
            )

            # --------------------------------------------
            # Light connection between points.
            #
            # This is ONLY a visual guide.
            # It is not the fitted boundary.
            # --------------------------------------------

            if len(points) > 1:

                self.ax_img.plot(
                    points[:, 0],
                    points[:, 1],
                    color=CONTROL_POINT_COLOR,
                    linewidth=1,
                    alpha=0.4,
                )

        # ----------------------------------------------------
        # 3. Final fitted + smoothed boundary
        # ----------------------------------------------------

        if self.boundary is not None:

            # ------------------------------------------------
            # If BOTH CLJs exist, show anatomical regions.
            # ------------------------------------------------

            if (
                self.left_clj_index is not None
                and
                self.right_clj_index is not None
            ):

                left_index = min(
                    self.left_clj_index,
                    self.right_clj_index,
                )

                right_index = max(
                    self.left_clj_index,
                    self.right_clj_index,
                )

                # --------------------------------------------
                # LEFT LIMBUS
                # --------------------------------------------

                if left_index > 0:

                    self.ax_img.plot(
                        self.boundary[
                            :left_index,
                            0
                        ],
                        self.boundary[
                            :left_index,
                            1
                        ],
                        color=LEFT_LIMBUS_COLOR,
                        linewidth=REGION_LINE_WIDTH,
                        label="Left limbus",
                    )

                # --------------------------------------------
                # CORNEA
                # --------------------------------------------

                self.ax_img.plot(
                    self.boundary[
                        left_index:
                        right_index + 1,
                        0
                    ],
                    self.boundary[
                        left_index:
                        right_index + 1,
                        1
                    ],
                    color=CORNEA_COLOR,
                    linewidth=REGION_LINE_WIDTH,
                    label="Cornea",
                )

                # --------------------------------------------
                # RIGHT LIMBUS
                # --------------------------------------------

                if (
                    right_index
                    <
                    len(self.boundary) - 1
                ):

                    self.ax_img.plot(
                        self.boundary[
                            right_index + 1:,
                            0
                        ],
                        self.boundary[
                            right_index + 1:,
                            1
                        ],
                        color=RIGHT_LIMBUS_COLOR,
                        linewidth=REGION_LINE_WIDTH,
                        label="Right limbus",
                    )

            # ------------------------------------------------
            # Before both CLJs are selected:
            #
            # show complete fitted boundary in yellow.
            #
            # THIS is the curve you should visually inspect.
            # ------------------------------------------------

            else:

                self.ax_img.plot(
                    self.boundary[:, 0],
                    self.boundary[:, 1],
                    color=FITTED_BOUNDARY_COLOR,
                    linewidth=FITTED_BOUNDARY_LINE_WIDTH,
                    alpha = 0.7,
                    label=(
                        "Fitted + smoothed "
                        "anterior boundary"
                    ),
                    zorder=8,
                )

        # ----------------------------------------------------
        # 4. LEFT CLJ marker
        # ----------------------------------------------------

        if (
            self.boundary is not None
            and
            self.left_clj_index is not None
        ):

            left_point = self.boundary[
                self.left_clj_index
            ]

            self.ax_img.scatter(
                left_point[0],
                left_point[1],
                s=CLJ_POINT_SIZE,
                marker="o",
                color=LEFT_LIMBUS_COLOR,
                edgecolors="black",
                linewidths=1.5,
                zorder=20,
                label="Left CLJ",
            )

            self.ax_img.annotate(
                "Left CLJ",
                (
                    left_point[0],
                    left_point[1],
                ),
                xytext=(
                    8,
                    -18,
                ),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.75,
                },
            )

        # ----------------------------------------------------
        # 5. RIGHT CLJ marker
        # ----------------------------------------------------

        if (
            self.boundary is not None
            and
            self.right_clj_index is not None
        ):

            right_point = self.boundary[
                self.right_clj_index
            ]

            self.ax_img.scatter(
                right_point[0],
                right_point[1],
                s=CLJ_POINT_SIZE,
                marker="s",
                color=RIGHT_LIMBUS_COLOR,
                edgecolors="black",
                linewidths=1.5,
                zorder=20,
                label="Right CLJ",
            )

            self.ax_img.annotate(
                "Right CLJ",
                (
                    right_point[0],
                    right_point[1],
                ),
                xytext=(
                    8,
                    -18,
                ),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.75,
                },
            )

        # ----------------------------------------------------
        # 6. Instructions
        # ----------------------------------------------------

        if self.mode == "boundary_points":

            instruction = (
                f"Click points along the anterior boundary "
                f"from LEFT to RIGHT "
                f"({len(self.control_points)} selected). "
                f"Press ENTER when finished."
            )

        elif self.mode == "left_clj":

            instruction = (
                "Inspect the YELLOW fitted + smoothed "
                "boundary. If correct, click LEFT CLJ. "
                "Press R to redo the boundary."
            )

        elif self.mode == "right_clj":

            instruction = (
                "Click RIGHT CLJ."
            )

        else:

            instruction = (
                "Both CLJs selected. "
                "Click near either CLJ to adjust it."
            )

        # ----------------------------------------------------
        # 7. Title
        # ----------------------------------------------------

        self.ax_img.set_title(
            f"{self.index + 1}/"
            f"{len(self.image_paths)} "
            f"— {self.current_image_path.name}\n"
            f"{instruction}\n"
            f"ENTER=fit boundary | "
            f"Backspace=remove last point | "
            f"s=save | r=reset | "
            f"n/right=next | "
            f"p/left=previous | q=quit"
        )

        # ----------------------------------------------------
        # 8. Image coordinates
        # ----------------------------------------------------

        self.ax_img.set_xlim(
            0,
            self.image.shape[1],
        )

        self.ax_img.set_ylim(
            self.image.shape[0],
            0,
        )

        self.ax_img.set_xlabel(
            "x [pixels]"
        )

        self.ax_img.set_ylabel(
            "y [pixels]"
        )

        self.ax_img.set_aspect(
            "equal"
        )

        # ----------------------------------------------------
        # 9. Legend
        # ----------------------------------------------------

        handles, labels = (
            self.ax_img.get_legend_handles_labels()
        )

        if handles:

            self.ax_img.legend(
                loc="best",
                fontsize=8,
            )

        # ----------------------------------------------------
        # 10. Refresh
        # ----------------------------------------------------

        plt.tight_layout()

        self.fig.canvas.draw_idle()


    # ========================================================
    # Run
    # ========================================================

    def run(
        self,
    ) -> None:

        print()
        print(
            "=========================================="
        )

        print(
            "Image-only anterior boundary + CLJ annotator"
        )

        print(
            "=========================================="
        )

        print()

        print(
            "STEP 1"
        )

        print(
            "Click points along the complete anterior "
            "boundary from LEFT to RIGHT."
        )

        print()

        print(
            "The number of points is NOT fixed."
        )

        print(
            "Use as many points as necessary."
        )

        print(
            "Around 20-30 is reasonable, but add more "
            "where the curvature changes strongly."
        )

        print()

        print(
            "Backspace = remove last point"
        )

        print(
            "Enter     = fit the final boundary"
        )

        print()

        print(
            "STEP 2"
        )

        print(
            "After pressing Enter, inspect the YELLOW "
            "fitted + smoothed anterior boundary."
        )

        print()

        print(
            "If it is wrong:"
        )

        print(
            "R = reset and annotate again"
        )

        print()

        print(
            "STEP 3"
        )

        print(
            "If the boundary is correct:"
        )

        print(
            "click LEFT CLJ"
        )

        print(
            "then click RIGHT CLJ"
        )

        print()

        print(
            "STEP 4"
        )

        print(
            "After both CLJs are selected:"
        )

        print(
            "orange = left limbus"
        )

        print(
            "green  = cornea"
        )

        print(
            "orange = right limbus"
        )

        print()

        print(
            "S       = save JSON"
        )

        print(
            "N/right = save JSON + next image"
        )

        print(
            "P/left  = previous image"
        )

        print(
            "R       = reset"
        )

        print(
            "Q/Esc   = quit"
        )

        print()

        print(
            "Only the JSON annotation is saved."
        )

        print(
            "The original OCT image is not modified."
        )

        print()

        plt.show()


# ============================================================
# Command line
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Annotate the anterior boundary and left/right "
            "corneo-limbal junctions in OCT images without "
            "existing LabelMe JSON annotations."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "OCT image file(s) or folder(s) "
            "containing OCT images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "clj_annotations_image_only"
        ),
        help=(
            "Directory where annotation JSON files "
            "will be saved."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    image_paths = collect_images(
        args.inputs
    )

    print()
    print(
        f"Found {len(image_paths)} OCT images."
    )

    print(
        "Annotations will be saved to:"
    )

    print(
        args.output_dir.resolve()
    )

    annotator = ImageOnlyCLJAnnotator(
        image_paths=image_paths,
        output_dir=args.output_dir,
    )

    annotator.run()


if __name__ == "__main__":
    main()
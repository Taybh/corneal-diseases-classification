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


LEFT_LABEL = "left_CLJ"
RIGHT_LABEL = "right_CLJ"

LEFT_LIMBUS = 0
CORNEA = 1
RIGHT_LIMBUS = 2

CLASS_MAP = {
    "0": "left_limbus",
    "1": "cornea",
    "2": "right_limbus",
}

ANTERIOR_POINT_COLOR = "tab:blue"
POSTERIOR_POINT_COLOR = "tab:purple"
ANTERIOR_BOUNDARY_COLOR = "yellow"
POSTERIOR_BOUNDARY_COLOR = "cyan"
LEFT_LIMBUS_COLOR = "tab:orange"
CORNEA_COLOR = "tab:green"
RIGHT_LIMBUS_COLOR = "tab:orange"
POLYGON_FILL_COLOR = "tab:blue"

CONTROL_POINT_SIZE = 25
CLJ_POINT_SIZE = 70

# Thin final fitted lines make CLJ placement more precise.
BOUNDARY_LINE_WIDTH = 0.8
REGION_LINE_WIDTH = 1.0

# While clicking control points, use a more visible temporary connecting line.
CONTROL_CONNECT_LINE_WIDTH = 2.0
CONTROL_CONNECT_ALPHA = 0.8

# The filled tissue band is only a display overlay; it is never written into the OCT.
POLYGON_ALPHA = 0.30

# The posterior boundary should cover nearly the same lateral extent as the anterior
# boundary. This prevents us from losing most of the limbal curve merely because
# deeper angle/iris structures begin immediately below the posterior surface.
POSTERIOR_ENDPOINT_TOLERANCE_FRACTION = 0.05
POSTERIOR_ENDPOINT_TOLERANCE_MIN_PX = 75.0

BOUNDARY_X_STEP = 1.0
SPLINE_SMOOTHING = 0.0
MIN_BOUNDARY_POINTS = 4

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def load_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"))


def collect_images(paths: list[Path]) -> list[Path]:
    image_paths: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path.resolve())
        elif path.is_dir():
            for file in path.rglob("*"):
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(file.resolve())
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    image_paths = sorted(dict.fromkeys(image_paths))
    if not image_paths:
        raise RuntimeError("No OCT images were found.")
    return image_paths


def smooth_dense_boundary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if len(y) >= 7:
        window = min(101, len(y) if len(y) % 2 else len(y) - 1)
        window = max(7, window)
        if window >= len(y):
            window = len(y) - 1 if len(y) % 2 == 0 else len(y)
        if window % 2 == 0:
            window -= 1
        y = savgol_filter(
            y,
            window_length=window,
            polyorder=min(3, window - 1),
        )
    return y


def fit_boundary(
    control_points: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if len(control_points) < MIN_BOUNDARY_POINTS:
        raise ValueError(
            f"Please select at least {MIN_BOUNDARY_POINTS} boundary points."
        )

    points = np.asarray(control_points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < MIN_BOUNDARY_POINTS:
        raise ValueError("Not enough valid boundary points.")

    points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    y = points[:, 1]

    rounded_x = np.round(x).astype(int)
    collapsed_x = []
    collapsed_y = []
    for x_value in np.unique(rounded_x):
        mask = rounded_x == x_value
        collapsed_x.append(float(np.mean(x[mask])))
        collapsed_y.append(float(np.median(y[mask])))

    x = np.asarray(collapsed_x, dtype=np.float64)
    y = np.asarray(collapsed_y, dtype=np.float64)

    if len(x) < MIN_BOUNDARY_POINTS:
        raise ValueError(
            "Too few unique x coordinates remain after removing duplicate x positions."
        )
    if not np.all(np.diff(x) > 0):
        raise ValueError("Boundary x coordinates are not strictly increasing.")

    spline = UnivariateSpline(
        x,
        y,
        s=SPLINE_SMOOTHING,
        k=min(3, len(x) - 1),
    )

    x_min = float(np.ceil(x.min()))
    x_max = float(np.floor(x.max()))
    if x_max <= x_min:
        raise ValueError("Boundary has insufficient horizontal extent.")

    dense_x = np.arange(
        x_min,
        x_max + BOUNDARY_X_STEP,
        BOUNDARY_X_STEP,
        dtype=np.float64,
    )
    dense_y = smooth_dense_boundary(spline(dense_x))
    boundary = np.column_stack((dense_x, dense_y))
    sorted_control_points = np.column_stack((x, y))
    return boundary, sorted_control_points


def closest_boundary_index(
    boundary: np.ndarray,
    point: tuple[float, float],
) -> int:
    x_click, y_click = point
    distance_squared = (
        (boundary[:, 0] - x_click) ** 2
        + (boundary[:, 1] - y_click) ** 2
    )
    return int(np.argmin(distance_squared))


def create_region_labels(
    boundary: np.ndarray,
    left_clj_index: int,
    right_clj_index: int,
) -> np.ndarray:
    if left_clj_index > right_clj_index:
        left_clj_index, right_clj_index = right_clj_index, left_clj_index

    labels = np.full(len(boundary), -1, dtype=np.int32)
    labels[:left_clj_index] = LEFT_LIMBUS
    labels[left_clj_index:right_clj_index + 1] = CORNEA
    labels[right_clj_index + 1:] = RIGHT_LIMBUS
    return labels


def build_tissue_polygon_points(
    anterior_boundary: np.ndarray,
    posterior_boundary: np.ndarray,
) -> np.ndarray:
    anterior = np.asarray(anterior_boundary, dtype=np.float64)
    posterior = np.asarray(posterior_boundary, dtype=np.float64)
    return np.vstack((anterior, posterior[::-1]))


def validate_boundary_pair(
    anterior_boundary: np.ndarray,
    posterior_boundary: np.ndarray,
) -> None:
    """
    Validate the two fitted curves.

    For this project the posterior boundary has TWO roles:
    1. anatomical posterior tissue boundary;
    2. hard lower masking boundary (zero pixels are kept below it).

    Therefore it should extend laterally almost as far as the anterior boundary,
    rather than stopping merely because iris/angle structures begin underneath.
    """
    anterior_x_min = float(anterior_boundary[:, 0].min())
    anterior_x_max = float(anterior_boundary[:, 0].max())
    posterior_x_min = float(posterior_boundary[:, 0].min())
    posterior_x_max = float(posterior_boundary[:, 0].max())

    x_min = max(anterior_x_min, posterior_x_min)
    x_max = min(anterior_x_max, posterior_x_max)

    if x_max <= x_min:
        raise ValueError(
            "Anterior and posterior boundaries do not overlap horizontally."
        )

    anterior_span = anterior_x_max - anterior_x_min
    tolerance = max(
        POSTERIOR_ENDPOINT_TOLERANCE_MIN_PX,
        POSTERIOR_ENDPOINT_TOLERANCE_FRACTION * anterior_span,
    )

    left_gap = posterior_x_min - anterior_x_min
    right_gap = anterior_x_max - posterior_x_max

    if left_gap > tolerance or right_gap > tolerance:
        raise ValueError(
            "Posterior boundary is too short laterally.\n"
            f"Anterior x-range: {anterior_x_min:.0f} .. {anterior_x_max:.0f}\n"
            f"Posterior x-range: {posterior_x_min:.0f} .. {posterior_x_max:.0f}\n"
            f"Allowed endpoint gap: about {tolerance:.0f} px.\n"
            "Extend the posterior boundary farther into BOTH limbal sides. "
            "Do not stop just because another structure begins immediately below it; "
            "everything below the posterior line will be masked later."
        )

    xs = np.arange(np.ceil(x_min), np.floor(x_max) + 1, dtype=np.float64)
    if len(xs) == 0:
        return

    anterior_y = np.interp(
        xs, anterior_boundary[:, 0], anterior_boundary[:, 1]
    )
    posterior_y = np.interp(
        xs, posterior_boundary[:, 0], posterior_boundary[:, 1]
    )
    correct_fraction = float(np.mean(posterior_y >= anterior_y))

    if correct_fraction < 0.98:
        raise ValueError(
            "Posterior boundary crosses above the anterior boundary in too many "
            f"columns ({correct_fraction:.1%} correctly ordered). Please correct it."
        )


class ImageOnlyPolygonCLJAnnotator:
    def __init__(self, image_paths: list[Path], output_dir: Path) -> None:
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0

        self.image: np.ndarray | None = None

        self.anterior_control_points: list[tuple[float, float]] = []
        self.anterior_sorted_control_points: np.ndarray | None = None
        self.anterior_boundary: np.ndarray | None = None

        self.posterior_control_points: list[tuple[float, float]] = []
        self.posterior_sorted_control_points: np.ndarray | None = None
        self.posterior_boundary: np.ndarray | None = None

        self.left_clj_index: int | None = None
        self.right_clj_index: int | None = None

        # anterior_points -> posterior_points -> left_clj -> right_clj -> complete
        self.mode = "anterior_points"

        self.fig, self.ax_img = plt.subplots(figsize=(16, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()
        self.draw()

    @property
    def current_image_path(self) -> Path:
        return self.image_paths[self.index]

    @property
    def output_path(self) -> Path:
        return self.output_dir / f"{self.current_image_path.stem}_CLJ.json"

    def load_current(self) -> None:
        self.image = load_image(self.current_image_path)

        self.anterior_control_points = []
        self.anterior_sorted_control_points = None
        self.anterior_boundary = None

        self.posterior_control_points = []
        self.posterior_sorted_control_points = None
        self.posterior_boundary = None

        self.left_clj_index = None
        self.right_clj_index = None
        self.mode = "anterior_points"

        if self.output_path.exists():
            self.load_existing_annotation()

    def load_existing_annotation(self) -> None:
        """
        Supports both the new polygon format and the previous
        anterior-only format.

        If an old annotation is found, the existing anterior
        boundary and CLJs are preserved, and only the posterior
        boundary needs to be added.
        """
        try:
            saved = json.loads(self.output_path.read_text(encoding="utf-8"))

            anterior_points = saved.get("anterior_boundary_points")
            posterior_points = saved.get("posterior_boundary_points")
            left_index = saved.get("left_CLJ_boundary_index")
            right_index = saved.get("right_CLJ_boundary_index")

            if anterior_points is not None:
                self.anterior_boundary = np.asarray(
                    anterior_points, dtype=np.float64
                )
            if posterior_points is not None:
                self.posterior_boundary = np.asarray(
                    posterior_points, dtype=np.float64
                )
            if left_index is not None:
                self.left_clj_index = int(left_index)
            if right_index is not None:
                self.right_clj_index = int(right_index)

            if (
                self.anterior_boundary is not None
                and self.posterior_boundary is not None
                and self.left_clj_index is not None
                and self.right_clj_index is not None
            ):
                self.mode = "complete"

            elif (
                self.anterior_boundary is not None
                and self.posterior_boundary is not None
            ):
                self.mode = (
                    "left_clj"
                    if self.left_clj_index is None
                    else "right_clj"
                )

            elif self.anterior_boundary is not None:
                self.mode = "posterior_points"
                print("\nExisting anterior-only annotation found.")
                print("Anterior boundary and existing CLJs were preserved.")
                print("Please add the POSTERIOR boundary from RIGHT to LEFT.")
                print("Extend it laterally almost as far as the anterior boundary.")
                print("Everything immediately below it will be masked later.")

            print(f"Loaded existing annotation: {self.output_path.name}")

        except Exception as error:
            print("\nCould not load existing annotation:")
            print(error)

    def on_click(self, event) -> None:
        if event.inaxes != self.ax_img:
            return
        if event.button != MouseButton.LEFT:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if self.mode == "anterior_points":
            self.anterior_control_points.append((x, y))
            print(
                f"Anterior point {len(self.anterior_control_points)}: "
                f"({x:.1f}, {y:.1f})"
            )
            self.draw()
            return

        if self.mode == "posterior_points":
            self.posterior_control_points.append((x, y))
            print(
                f"Posterior point {len(self.posterior_control_points)}: "
                f"({x:.1f}, {y:.1f})"
            )
            self.draw()
            return

        if self.mode == "left_clj" and self.anterior_boundary is not None:
            self.left_clj_index = closest_boundary_index(
                self.anterior_boundary, (x, y)
            )
            self.mode = "right_clj"
            self.draw()
            return

        if self.mode == "right_clj" and self.anterior_boundary is not None:
            self.right_clj_index = closest_boundary_index(
                self.anterior_boundary, (x, y)
            )
            if (
                self.left_clj_index is not None
                and self.right_clj_index < self.left_clj_index
            ):
                self.left_clj_index, self.right_clj_index = (
                    self.right_clj_index,
                    self.left_clj_index,
                )
            self.mode = "complete"
            self.draw()
            return

        if (
            self.mode == "complete"
            and self.anterior_boundary is not None
            and self.left_clj_index is not None
            and self.right_clj_index is not None
        ):
            new_index = closest_boundary_index(
                self.anterior_boundary, (x, y)
            )
            left_point = self.anterior_boundary[self.left_clj_index]
            right_point = self.anterior_boundary[self.right_clj_index]
            new_point = self.anterior_boundary[new_index]

            left_distance = np.sum((left_point - new_point) ** 2)
            right_distance = np.sum((right_point - new_point) ** 2)

            if left_distance <= right_distance:
                self.left_clj_index = new_index
            else:
                self.right_clj_index = new_index

            if self.left_clj_index > self.right_clj_index:
                self.left_clj_index, self.right_clj_index = (
                    self.right_clj_index,
                    self.left_clj_index,
                )
            self.draw()

    def fit_current_stage(self) -> None:
        if self.mode == "anterior_points":
            try:
                (
                    self.anterior_boundary,
                    self.anterior_sorted_control_points,
                ) = fit_boundary(self.anterior_control_points)

                self.left_clj_index = None
                self.right_clj_index = None
                self.mode = "posterior_points"

                print("\n====================================")
                print("Anterior boundary fitted.")
                print(f"Manual points: {len(self.anterior_control_points)}")
                print(f"Dense anterior points: {len(self.anterior_boundary)}")
                print("\nNow click the POSTERIOR boundary from RIGHT to LEFT.")
                print("Extend it laterally almost as far as the anterior boundary.")
                print("Do NOT stop because iris/angle structures begin below it.")
                print("The posterior curve is the HARD LOWER MASK boundary:")
                print("everything immediately below it will be removed later.")
                print("Press ENTER when finished.")
                print("====================================")

            except Exception as error:
                print("\nCould not fit anterior boundary:")
                print(error)

            self.draw()
            return

        if self.mode == "posterior_points":
            try:
                (
                    self.posterior_boundary,
                    self.posterior_sorted_control_points,
                ) = fit_boundary(self.posterior_control_points)

                if self.anterior_boundary is None:
                    raise RuntimeError("Anterior boundary is missing.")

                validate_boundary_pair(
                    self.anterior_boundary,
                    self.posterior_boundary,
                )

                if (
                    self.left_clj_index is not None
                    and self.right_clj_index is not None
                ):
                    self.mode = "complete"
                elif self.left_clj_index is not None:
                    self.mode = "right_clj"
                else:
                    self.mode = "left_clj"

                print("\n====================================")
                print("Posterior boundary fitted.")
                print(f"Manual points: {len(self.posterior_control_points)}")
                print(f"Dense posterior points: {len(self.posterior_boundary)}")
                print("\nInspect both fitted curves carefully:")
                print("YELLOW = anterior")
                print("CYAN   = posterior")

                if self.mode == "left_clj":
                    print("If correct, click LEFT CLJ on the anterior boundary.")
                elif self.mode == "right_clj":
                    print("If correct, click RIGHT CLJ on the anterior boundary.")
                else:
                    print("Existing CLJs were preserved.")

                print("====================================")

            except Exception as error:
                print("\nCould not fit posterior boundary:")
                print(error)

            self.draw()

    def remove_last_control_point(self) -> None:
        if self.mode == "anterior_points":
            points = self.anterior_control_points
            name = "anterior"
        elif self.mode == "posterior_points":
            points = self.posterior_control_points
            name = "posterior"
        else:
            print(
                "No active boundary point selection. "
                "Press R if you want to redo the annotation."
            )
            return

        if not points:
            print(f"No {name} control points to remove.")
            return

        removed = points.pop()
        print(
            f"Removed {name} point: "
            f"({removed[0]:.1f}, {removed[1]:.1f})"
        )
        self.draw()

    def reset(self) -> None:
        self.anterior_control_points = []
        self.anterior_sorted_control_points = None
        self.anterior_boundary = None

        self.posterior_control_points = []
        self.posterior_sorted_control_points = None
        self.posterior_boundary = None

        self.left_clj_index = None
        self.right_clj_index = None
        self.mode = "anterior_points"

        print("\nAnnotation reset.")
        print("Start again with the ANTERIOR boundary from LEFT to RIGHT.")
        self.draw()

    def save(self) -> bool:
        if self.anterior_boundary is None:
            print("Fit the anterior boundary before saving.")
            return False
        if self.posterior_boundary is None:
            print("Fit the posterior boundary before saving.")
            return False
        if self.left_clj_index is None or self.right_clj_index is None:
            print("Select both left and right CLJ points before saving.")
            return False

        left_index = min(self.left_clj_index, self.right_clj_index)
        right_index = max(self.left_clj_index, self.right_clj_index)

        if left_index <= 0:
            print("Left CLJ is too close to the start of the anterior boundary.")
            return False
        if right_index >= len(self.anterior_boundary) - 1:
            print("Right CLJ is too close to the end of the anterior boundary.")
            return False

        left_point = self.anterior_boundary[left_index]
        right_point = self.anterior_boundary[right_index]

        region_labels = create_region_labels(
            self.anterior_boundary,
            left_index,
            right_index,
        )

        anterior_points = np.asarray(self.anterior_boundary, dtype=np.float64)
        posterior_points = np.asarray(self.posterior_boundary, dtype=np.float64)
        tissue_polygon_points = build_tissue_polygon_points(
            anterior_points,
            posterior_points,
        )

        output = {
            "source_json": None,
            "imagePath": self.current_image_path.name,
            "source_polygon_label": None,
            "annotation_note": (
                "The tissue polygon defines the visible corneal/limbal tissue band. "
                "The anterior boundary between left_CLJ and right_CLJ is defined as "
                "cornea; anterior-boundary points outside those junctions are defined "
                "as limbus. The CLJ points themselves belong to the corneal region. "
                "For preprocessing, the posterior boundary is also the hard lower mask "
                "boundary: pixels immediately below it are excluded with zero lower margin."
            ),
            "class_map": CLASS_MAP,
            "tissue_polygon_points": tissue_polygon_points.tolist(),
            "anterior_boundary_points": anterior_points.tolist(),
            "posterior_boundary_points": posterior_points.tolist(),
            LEFT_LABEL: left_point.tolist(),
            RIGHT_LABEL: right_point.tolist(),
            "left_CLJ_boundary_index": int(left_index),
            "right_CLJ_boundary_index": int(right_index),
            "anterior_boundary_region_labels": region_labels.tolist(),
            "regions": {
                "left_limbus": {
                    "start_index": 0,
                    "end_index": int(left_index - 1),
                },
                "cornea": {
                    "start_index": int(left_index),
                    "end_index": int(right_index),
                },
                "right_limbus": {
                    "start_index": int(right_index + 1),
                    "end_index": int(len(anterior_points) - 1),
                },
            },
        }

        self.output_path.write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved: {self.output_path}")
        return True

    def move(self, step: int) -> None:
        new_index = self.index + step
        if not (0 <= new_index < len(self.image_paths)):
            print("No more cases in that direction.")
            return

        self.index = new_index
        self.load_current()
        self.draw()

    def redo_posterior_only(self) -> None:
        if self.anterior_boundary is None:
            print("Fit/load the anterior boundary first.")
            return

        self.posterior_control_points = []
        self.posterior_sorted_control_points = None
        self.posterior_boundary = None
        self.mode = "posterior_points"

        print("\nPosterior boundary cleared; anterior boundary and CLJs were preserved.")
        print("Redraw POSTERIOR boundary RIGHT -> LEFT.")
        print("Extend it laterally almost as far as the anterior boundary.")
        print("Everything immediately below it will be masked later.")
        self.draw()

    def save_and_next(self) -> None:
        if not self.save():
            return
        self.move(1)

    def on_key(self, event) -> None:
        key = event.key.lower() if event.key else ""

        if key in ("enter", "return"):
            self.fit_current_stage()
        elif key in ("backspace", "delete"):
            self.remove_last_control_point()
        elif key == "r":
            self.reset()
        elif key == "b":
            self.redo_posterior_only()
        elif key == "s":
            self.save()
        elif key in ("n", "right"):
            self.save_and_next()
        elif key in ("p", "left"):
            self.move(-1)
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def draw(self) -> None:
        self.ax_img.clear()
        if self.image is None:
            return

        self.ax_img.imshow(self.image)

        if self.anterior_control_points and self.mode == "anterior_points":
            points = np.asarray(self.anterior_control_points, dtype=np.float64)
            self.ax_img.scatter(
                points[:, 0], points[:, 1],
                s=CONTROL_POINT_SIZE,
                color=ANTERIOR_POINT_COLOR,
                edgecolors="white",
                linewidths=0.5,
                zorder=10,
                label="Manual anterior points",
            )
            if len(points) > 1:
                self.ax_img.plot(
                    points[:, 0], points[:, 1],
                    color=ANTERIOR_POINT_COLOR,
                    linewidth=CONTROL_CONNECT_LINE_WIDTH,
                    alpha=CONTROL_CONNECT_ALPHA,
                )

        if self.posterior_control_points and self.mode == "posterior_points":
            points = np.asarray(self.posterior_control_points, dtype=np.float64)
            self.ax_img.scatter(
                points[:, 0], points[:, 1],
                s=CONTROL_POINT_SIZE,
                color=POSTERIOR_POINT_COLOR,
                edgecolors="white",
                linewidths=0.5,
                zorder=10,
                label="Manual posterior points",
            )
            if len(points) > 1:
                self.ax_img.plot(
                    points[:, 0], points[:, 1],
                    color=POSTERIOR_POINT_COLOR,
                    linewidth=CONTROL_CONNECT_LINE_WIDTH,
                    alpha=CONTROL_CONNECT_ALPHA,
                )

        if self.anterior_boundary is not None and self.posterior_boundary is not None:
            polygon = build_tissue_polygon_points(
                self.anterior_boundary,
                self.posterior_boundary,
            )
            self.ax_img.fill(
                polygon[:, 0], polygon[:, 1],
                color=POLYGON_FILL_COLOR,
                alpha=POLYGON_ALPHA,
                label="Tissue polygon",
                zorder=3,
            )

        if self.anterior_boundary is not None:
            if self.left_clj_index is not None and self.right_clj_index is not None:
                left_index = min(self.left_clj_index, self.right_clj_index)
                right_index = max(self.left_clj_index, self.right_clj_index)

                if left_index > 0:
                    self.ax_img.plot(
                        self.anterior_boundary[:left_index, 0],
                        self.anterior_boundary[:left_index, 1],
                        color=LEFT_LIMBUS_COLOR,
                        linewidth=REGION_LINE_WIDTH,
                        label="Left limbus",
                        zorder=8,
                    )

                self.ax_img.plot(
                    self.anterior_boundary[left_index:right_index + 1, 0],
                    self.anterior_boundary[left_index:right_index + 1, 1],
                    color=CORNEA_COLOR,
                    linewidth=REGION_LINE_WIDTH,
                    label="Cornea",
                    zorder=8,
                )

                if right_index < len(self.anterior_boundary) - 1:
                    self.ax_img.plot(
                        self.anterior_boundary[right_index + 1:, 0],
                        self.anterior_boundary[right_index + 1:, 1],
                        color=RIGHT_LIMBUS_COLOR,
                        linewidth=REGION_LINE_WIDTH,
                        label="Right limbus",
                        zorder=8,
                    )
            else:
                self.ax_img.plot(
                    self.anterior_boundary[:, 0],
                    self.anterior_boundary[:, 1],
                    color=ANTERIOR_BOUNDARY_COLOR,
                    linewidth=BOUNDARY_LINE_WIDTH,
                    alpha=0.85,
                    label="Fitted anterior boundary",
                    zorder=8,
                )

        if self.posterior_boundary is not None:
            self.ax_img.plot(
                self.posterior_boundary[:, 0],
                self.posterior_boundary[:, 1],
                color=POSTERIOR_BOUNDARY_COLOR,
                linewidth=BOUNDARY_LINE_WIDTH,
                alpha=0.85,
                label="Fitted posterior boundary",
                zorder=8,
            )

        if self.anterior_boundary is not None and self.left_clj_index is not None:
            left_point = self.anterior_boundary[self.left_clj_index]
            self.ax_img.scatter(
                left_point[0], left_point[1],
                s=CLJ_POINT_SIZE,
                marker="o",
                color=LEFT_LIMBUS_COLOR,
                edgecolors="black",
                linewidths=1.5,
                zorder=20,
                label="Left CLJ",
            )
            """
            self.ax_img.annotate(
                "Left CLJ",
                (left_point[0], left_point[1]),
                xytext=(8, -18),
                textcoords="offset points",
                bbox={"boxstyle": "round", "alpha": 0.75},
            )
            """

        if self.anterior_boundary is not None and self.right_clj_index is not None:
            right_point = self.anterior_boundary[self.right_clj_index]
            self.ax_img.scatter(
                right_point[0], right_point[1],
                s=CLJ_POINT_SIZE,
                marker="s",
                color=RIGHT_LIMBUS_COLOR,
                edgecolors="black",
                linewidths=1.5,
                zorder=20,
                label="Right CLJ",
            )
            """
            self.ax_img.annotate(
                "Right CLJ",
                (right_point[0], right_point[1]),
                xytext=(8, -18),
                textcoords="offset points",
                bbox={"boxstyle": "round", "alpha": 0.75},
            )
            """
        if self.mode == "anterior_points":
            instruction = (
                f"STEP 1: click ANTERIOR boundary LEFT→RIGHT "
                f"({len(self.anterior_control_points)} points). Press ENTER when finished."
            )
        elif self.mode == "posterior_points":
            instruction = (
                f"STEP 2: click POSTERIOR RIGHT→LEFT, nearly full anterior x-range "
                f"({len(self.posterior_control_points)} points). ENTER=fit."
            )
        elif self.mode == "left_clj":
            instruction = (
                "Inspect YELLOW anterior + CYAN posterior. "
                "If correct, click LEFT CLJ on the anterior boundary."
            )
        elif self.mode == "right_clj":
            instruction = "Click RIGHT CLJ on the anterior boundary."
        else:
            instruction = "Annotation complete. Click near either CLJ to adjust it."

        self.ax_img.set_title(
            f"{self.index + 1}/{len(self.image_paths)} — {self.current_image_path.name}\n"
            f"{instruction}\n"
            f"ENTER=fit current boundary | Backspace=remove last point | "
            f"s=save | b=redo posterior | r=reset all | "
            f"n/right=save+next | p/left=previous | q=quit"
        )

        self.ax_img.set_xlim(0, self.image.shape[1])
        self.ax_img.set_ylim(self.image.shape[0], 0)
        self.ax_img.set_xlabel("x [pixels]")
        self.ax_img.set_ylabel("y [pixels]")
        self.ax_img.set_aspect("equal")

        handles, labels = self.ax_img.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            self.ax_img.legend(
                unique.values(),
                unique.keys(),
                loc="best",
                fontsize=8,
            )

        plt.tight_layout()
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        print("\n================================================")
        print("Image-only tissue polygon + CLJ annotator")
        print("================================================\n")
        print("STEP 1: click ANTERIOR boundary LEFT -> RIGHT, then Enter.")
        print("STEP 2: click POSTERIOR boundary RIGHT -> LEFT, then Enter.")
        print("        Extend it almost to the same lateral x-range as the anterior.")
        print("        Do NOT stop because deeper structures start underneath.")
        print("        Everything immediately below posterior will be masked later.")
        print("STEP 3: inspect YELLOW anterior + CYAN posterior.")
        print("STEP 4: click LEFT CLJ, then RIGHT CLJ on the anterior boundary.\n")
        print("Backspace = remove last control point")
        print("Enter     = fit current boundary")
        print("S         = save JSON")
        print("N/right   = save JSON + next image")
        print("P/left    = previous image")
        print("B         = redo posterior only (preserves anterior + CLJs)")
        print("R         = reset entire annotation")
        print("Q/Esc     = quit\n")
        print("Saved JSON contains:")
        print("- tissue_polygon_points")
        print("- anterior_boundary_points")
        print("- posterior_boundary_points")
        print("- left/right CLJs")
        print("- anterior boundary class labels")
        print("- region start/end indices only\n")
        print("Old anterior-only annotations are supported:")
        print("existing anterior boundary and CLJs are preserved;")
        print("you only need to add the posterior boundary.\n")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a complete corneal/limbal tissue band using anterior and "
            "posterior boundaries, then select left/right corneo-limbal junctions."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="OCT image file(s) or folder(s) containing OCT images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clj_annotations_MCOA_Normal_images"),
        help="Directory where canonical annotation JSON files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.inputs)

    print(f"\nFound {len(image_paths)} OCT images.")
    print("Annotations will be saved to:")
    print(args.output_dir.resolve())

    annotator = ImageOnlyPolygonCLJAnnotator(
        image_paths=image_paths,
        output_dir=args.output_dir,
    )
    annotator.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter


LEFT_LABEL = "left_CLJ"
RIGHT_LABEL = "right_CLJ"


def load_case(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("imageData"):
        image_bytes = base64.b64decode(data["imageData"])
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    else:
        image_ref = str(data.get("imagePath", "")).replace("\\", "/")
        image_path = Path(image_ref)
        if not image_path.is_absolute():
            image_path = json_path.parent / image_path
        if not image_path.exists():
            fallback = json_path.parent / Path(image_ref).name
            if fallback.exists():
                image_path = fallback
            else:
                raise FileNotFoundError(f"Image not found for {json_path.name}")
        image = np.asarray(Image.open(image_path).convert("RGB"))

    polygons = [
        np.asarray(shape["points"], dtype=float)
        for shape in data.get("shapes", [])
        if str(shape.get("label", "")).strip().lower() == "cornea"
        and len(shape.get("points", [])) >= 3
    ]
    if not polygons:
        raise ValueError(f"No polygon labeled 'cornea' in {json_path.name}")

    polygon = max(polygons, key=len)
    return data, image, polygon


def extract_anterior_boundary(polygon: np.ndarray):
    closed = np.vstack([polygon, polygon[0]])
    xs = np.arange(
        int(np.ceil(polygon[:, 0].min())),
        int(np.floor(polygon[:, 0].max())) + 1,
        dtype=float,
    )
    ys = np.full(xs.shape, np.nan, dtype=float)

    for i, x in enumerate(xs):
        intersections = []
        for (x1, y1), (x2, y2) in zip(closed[:-1], closed[1:]):
            if np.isclose(x1, x2):
                if np.isclose(x, x1):
                    intersections.extend([y1, y2])
                continue
            if min(x1, x2) <= x <= max(x1, x2):
                t = (x - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    intersections.append(y1 + t * (y2 - y1))
        if intersections:
            ys[i] = min(intersections)

    valid = np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]

    if len(ys) >= 7:
        window = min(101, len(ys) if len(ys) % 2 else len(ys) - 1)
        window = max(7, window)
        if window >= len(ys):
            window = len(ys) - 1 if len(ys) % 2 == 0 else len(ys)
        if window % 2 == 0:
            window -= 1
        ys = savgol_filter(ys, window_length=window, polyorder=min(3, window - 1))

    return xs, ys


def curvature(x: np.ndarray, y: np.ndarray):
    slope = np.gradient(y, x)
    slope_change = np.gradient(slope, x)
    return np.abs(slope_change) / np.power(1.0 + slope**2, 1.5)


class Annotator:
    def __init__(self, json_files: list[Path], output_dir: Path):
        self.files = json_files
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0

        self.fig, (self.ax_img, self.ax_curv) = plt.subplots(
            2, 1, figsize=(15, 10),
            gridspec_kw={"height_ratios": [4, 1]},
        )
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()
        self.draw()

    @property
    def current_json(self):
        return self.files[self.index]

    @property
    def output_path(self):
        return self.output_dir / f"{self.current_json.stem}_CLJ.json"

    def load_current(self):
        self.data, self.image, self.polygon = load_case(self.current_json)
        self.x, self.y = extract_anterior_boundary(self.polygon)
        self.kappa = curvature(self.x, self.y)
        self.points = {}

        if self.output_path.exists():
            saved = json.loads(self.output_path.read_text(encoding="utf-8"))
            for label in (LEFT_LABEL, RIGHT_LABEL):
                if label in saved:
                    self.points[label] = np.asarray(saved[label], dtype=float)

    def nearest_boundary_point(self, x_click, y_click):
        d = (self.x - x_click) ** 2 + (self.y - y_click) ** 2
        i = int(np.argmin(d))
        return np.array([self.x[i], self.y[i]])

    def on_click(self, event):
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return

        point = self.nearest_boundary_point(event.xdata, event.ydata)

        if LEFT_LABEL not in self.points:
            self.points[LEFT_LABEL] = point
        elif RIGHT_LABEL not in self.points:
            self.points[RIGHT_LABEL] = point
        else:
            left_d = np.sum((self.points[LEFT_LABEL] - point) ** 2)
            right_d = np.sum((self.points[RIGHT_LABEL] - point) ** 2)
            label = LEFT_LABEL if left_d <= right_d else RIGHT_LABEL
            self.points[label] = point

        if LEFT_LABEL in self.points and RIGHT_LABEL in self.points:
            if self.points[LEFT_LABEL][0] > self.points[RIGHT_LABEL][0]:
                self.points[LEFT_LABEL], self.points[RIGHT_LABEL] = (
                    self.points[RIGHT_LABEL],
                    self.points[LEFT_LABEL],
                )

        self.draw()

    def on_key(self, event):
        if event.key in ("r", "delete", "backspace"):
            self.points = {}
            self.draw()
        elif event.key == "s":
            self.save()
        elif event.key in ("n", "right"):
            if len(self.points) == 2:
                self.save()
            self.move(1)
        elif event.key in ("p", "left"):
            self.move(-1)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def move(self, step):
        new_index = self.index + step
        if not 0 <= new_index < len(self.files):
            print("No more cases in that direction.")
            return
        self.index = new_index
        self.load_current()
        self.draw()

    def save(self):
        if LEFT_LABEL not in self.points or RIGHT_LABEL not in self.points:
            print("Select both left and right CLJ points before saving.")
            return

        left_point = self.points[LEFT_LABEL]
        right_point = self.points[RIGHT_LABEL]

        left_index = int(
            np.argmin(np.abs(self.x - left_point[0]))
        )
        right_index = int(
            np.argmin(np.abs(self.x - right_point[0]))
        )

        if left_index > right_index:
            left_index, right_index = right_index, left_index

        # Safety check
        if left_index <= 0:
            print("Left CLJ is too close to the start of the boundary.")
            return

        if right_index >= len(self.x) - 1:
            print("Right CLJ is too close to the end of the boundary.")
            return
        
        # Boundary-point labels:
        # 0 = left limbus
        # 1 = cornea
        # 2 = right limbus
        region_labels = np.full(
            len(self.x),
            -1,
            dtype=np.int32,
        )
        region_labels[:left_index] = 0
        region_labels[left_index:right_index + 1] = 1
        region_labels[right_index + 1:] = 2

        boundary_points = np.column_stack(
            [self.x, self.y]
        )

        output = {
            "source_json": str(self.current_json),
            "imagePath": self.data.get("imagePath"),
            "annotation_note": (
                "The anterior boundary between left_CLJ and right_CLJ is "
                "defined as cornea. Boundary points outside those junctions "
                "are defined as limbus. The selected points are snapped to "
                "the smoothed anterior boundary and may be revised."
            ),
            "class_map": {
                "0": "left_limbus",
                "1": "cornea",
                "2": "right_limbus",
            },
            LEFT_LABEL: left_point.tolist(),
            RIGHT_LABEL: right_point.tolist(),
            "left_CLJ_boundary_index": left_index,
            "right_CLJ_boundary_index": right_index,
            "anterior_boundary_points": boundary_points.tolist(),
            "anterior_boundary_region_labels": region_labels.tolist(),
            "regions": {
                "left_limbus": {
                    "start_index": 0,
                    "end_index": left_index - 1,
                    "points": boundary_points[:left_index].tolist(),
                },
                "cornea": {
                    "start_index": left_index,
                    "end_index": right_index,
                    "points": boundary_points[left_index:right_index + 1].tolist(),
                },
                "right_limbus": {
                    "start_index": right_index + 1,
                    "end_index": len(boundary_points) - 1,
                    "points": boundary_points[right_index + 1:].tolist(),
                },
            },
        }

        self.output_path.write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {self.output_path}")

    def draw(self):
        self.ax_img.clear()
        self.ax_curv.clear()

        # --------------------------------------------------
        # 1. Show the original AS-OCT image
        # --------------------------------------------------
        self.ax_img.imshow(self.image)

        # --------------------------------------------------
        # 2. Show the original polygon annotation
        # --------------------------------------------------
        closed = np.vstack(
            [self.polygon, self.polygon[0]]
        )

        self.ax_img.fill(
            self.polygon[:, 0],
            self.polygon[:, 1],
            alpha=0.10,
            label="Existing cornea + limbus polygon",
        )

        self.ax_img.plot(
            closed[:, 0],
            closed[:, 1],
            linewidth=1.2,
            alpha=0.5,
            label="Original polygon border",
        )

        # --------------------------------------------------
        # 3. Show the complete extracted anterior boundary
        # --------------------------------------------------
        self.ax_img.plot(
            self.x,
            self.y,
            linewidth=2,
            alpha=0.6,
            label="Smoothed anterior boundary",
        )

        # --------------------------------------------------
        # 4. If both CLJs have been selected,
        #    divide the boundary into:
        #
        #    left limbus | cornea | right limbus
        # --------------------------------------------------
        if (
            LEFT_LABEL in self.points
            and RIGHT_LABEL in self.points
        ):
            left_x = self.points[LEFT_LABEL][0]
            right_x = self.points[RIGHT_LABEL][0]

            # Find where the selected CLJ points occur
            # in the boundary array.
            left_index = int(
                np.argmin(
                    np.abs(self.x - left_x)
                )
            )

            right_index = int(
                np.argmin(
                    np.abs(self.x - right_x)
                )
            )

            # Safety check in case the points
            # somehow became reversed.
            if left_index > right_index:
                left_index, right_index = (
                    right_index,
                    left_index,
                )

            # ------------------------------
            # LEFT LIMBUS
            # ------------------------------
            if left_index > 0:
                self.ax_img.plot(
                    self.x[:left_index],
                    self.y[:left_index],
                    color="tab:orange",
                    linewidth=6,
                    label="Left limbus",
                )

            # ------------------------------
            # CORNEA
            # ------------------------------
            self.ax_img.plot(
                self.x[
                    left_index:right_index + 1
                ],
                self.y[
                    left_index:right_index + 1
                ],
                color="tab:green",
                linewidth=6,
                label="Cornea",
            )

            # ------------------------------
            # RIGHT LIMBUS
            # ------------------------------
            if right_index < len(self.x) - 1:
                self.ax_img.plot(
                    self.x[right_index + 1:],
                    self.y[right_index + 1:],
                    color="tab:orange",
                    linewidth=6,
                    label="Right limbus",
                )

        # --------------------------------------------------
        # 5. Draw the manually selected CLJ points
        # --------------------------------------------------
        for label, name, marker in [
            (
                LEFT_LABEL,
                "Left CLJ",
                "o",
            ),
            (
                RIGHT_LABEL,
                "Right CLJ",
                "s",
            ),
        ]:
            if label not in self.points:
                continue

            x0, y0 = self.points[label]

            self.ax_img.scatter(
                x0,
                y0,
                s=140,
                marker=marker,
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                label=name,
            )

            self.ax_img.annotate(
                name,
                (x0, y0),
                xytext=(8, -18),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.75,
                },
            )

            # Find the corresponding boundary index.
            i = int(
                np.argmin(
                    np.abs(self.x - x0)
                )
            )

            # Show that location on the curvature plot.
            self.ax_curv.axvline(
                self.x[i],
                linestyle="--",
            )

            self.ax_curv.scatter(
                self.x[i],
                self.kappa[i],
                s=65,
                zorder=5,
            )

        # --------------------------------------------------
        # 6. Instruction shown at the top
        # --------------------------------------------------
        if LEFT_LABEL not in self.points:
            instruction = (
                "Click the LEFT corneo-limbal junction."
            )

        elif RIGHT_LABEL not in self.points:
            instruction = (
                "Click the RIGHT corneo-limbal junction."
            )

        else:
            instruction = (
                "Both CLJs selected. "
                "Click near either side to move "
                "the nearest selected point."
            )

        # --------------------------------------------------
        # 7. Image plot formatting
        # --------------------------------------------------
        self.ax_img.set_title(
            f"{self.index + 1}/{len(self.files)} "
            f"— {self.current_json.name}\n"
            f"{instruction}\n"
            "s=save | r=reset | "
            "n/right=next | "
            "p/left=previous | q=quit"
        )

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

        self.ax_img.legend(
            loc="best",
            fontsize=8,
        )

        # --------------------------------------------------
        # 8. Curvature plot
        # --------------------------------------------------
        self.ax_curv.plot(
            self.x,
            self.kappa,
        )

        self.ax_curv.set_title(
            "Curvature along the anterior boundary "
            "(visual aid only)"
        )

        self.ax_curv.set_xlabel(
            "x [pixels]"
        )

        self.ax_curv.set_ylabel(
            "curvature"
        )

        self.ax_curv.grid(
            alpha=0.25
        )

        # --------------------------------------------------
        # 9. Refresh figure
        # --------------------------------------------------
        plt.tight_layout()

        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


def collect_json_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    raise FileNotFoundError(input_path)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate left and right corneo-limbal junction points."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A LabelMe JSON file or a folder containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clj_annotations_MCOA"),
    )
    args = parser.parse_args()

    files = collect_json_files(args.input)
    if not files:
        raise FileNotFoundError("No JSON files found.")

    Annotator(files, args.output_dir).run()


if __name__ == "__main__":
    main()

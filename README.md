# AS-OCT Cornea–Limbus Classification

This project develops a deep-learning pipeline for localizing and classifying **corneal** and **limbal** regions in anterior-segment optical coherence tomography (**AS-OCT**) images.

The workflow has two main parts:

1. **Annotation** – manually annotate the anterior and posterior tissue boundaries and mark the left and right corneo-limbal junctions (CLJs).
2. **Training** – generate anatomy-aware sliding-window patches from the annotated scans and fine-tune an ImageNet-pretrained **ConvNeXt-Tiny** model for binary classification:
   - `0 = limbus`
   - `1 = cornea`

The patches are created directly from the original OCT images during training and are not saved as separate image files.

---

## Project Structure

```text
project/
├── annotate_tissue_polygon_clj_v2.py
├── oct_patch_dataset.py
├── train.py
├── requirements.txt
├── README.md
└── data/
```

## Environment Setup

It is recommended to use a Python virtual environment.

### 1. Create the virtual environment

```bash
python3 -m venv myenv
```

### 2. Activate the environment

macOS/Linux:

```bash
source myenv/bin/activate
```

Windows:

```bash
myenv\Scripts\activate
```

### 3. Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

or:

```bash
pip3 install --upgrade pip
```

### 4. Install the required packages

```bash
pip3 install -r requirements.txt
```

Check the Python version with:

```bash
python3 --version
```

---

## Annotation

Run the annotation tool with:

```bash
python3 annotate_tissue_polygon_clj.py MCOA_Normal_images
```

The annotation workflow is:

1. Select points along the **anterior boundary** from left to right.
2. Confirm the anterior boundary.
3. Select points along the **posterior boundary**.
4. Confirm the posterior boundary.
5. Mark the **left CLJ**.
6. Mark the **right CLJ**.
7. Save the annotation.

The saved JSON contains the main anatomical information required for training, including:

```text
anterior_boundary_points
posterior_boundary_points
left_CLJ
right_CLJ
anterior_boundary_region_labels
```

The anterior boundary is divided into:

```text
left limbus | cornea | right limbus
```

with:

```text
0 = left limbus
1 = cornea
2 = right limbus
```

For model training, the two limbal regions are combined into one binary class:

```text
0 = limbus
1 = cornea
```

---

## Patch Generation

Training patches are generated automatically by `oct_patch_dataset.py`.

Current preprocessing:

```text
patch width = 256 pixels
training step = 128 pixels
margin above anterior boundary = 15 pixels
lower limit = posterior boundary
```

For each patch:

- the full width must lie inside the region where both anterior and posterior boundaries are annotated;
- the patch starts 15 pixels above the anterior boundary;
- tissue is retained down to the posterior boundary;
- pixels below the posterior boundary are masked with zero;
- the original geometry and curvature are preserved;
- the patch is resized while preserving aspect ratio;
- zero padding is added to obtain a final size of `224 × 224`;
- the grayscale OCT patch is converted to three channels;
- ImageNet normalization is applied before the patch is passed to ConvNeXt.

Patch labels are determined from the annotated anterior boundary. The class occupying the majority of the patch width determines the binary patch label. In an exact 50/50 case, the label at the center of the patch is used.

---

## Training

The training script uses an **ImageNet-pretrained ConvNeXt-Tiny** model. The final classification layer is replaced with a two-class classifier and the **entire network is fine-tuned from the first epoch**.

Current setup:

```text
model = ConvNeXt-Tiny
pretraining = ImageNet
classes = 2
epochs = 50
learning rate = 1e-4
weight decay = 1e-4
early stopping = disabled
```

The annotated OCT images are split at the **source-image level before patch generation** to avoid leakage between training and validation patches.

For 50 annotated images:

```text
40 images = training
10 images = validation
```

The external test set is kept separate and is not used during training or model selection.

### Run training

```bash
python3 train.py /path/to/annotations --output-dir training_output
```

Example:

```bash
python3 train_convnext.py clj_annotations_MCOA_Normal_images
```

The output directory contains files such as:

```text
training_output/
├── best_model.pt
├── last_checkpoint.pt
├── log.txt
├── history.json
├── config.json
└── splits.json
```

`best_model.pt` stores the model with the lowest validation loss observed during training.

`last_checkpoint.pt` stores the latest training state, including the model and optimizer states, and can be used to resume training.

There is no early stopping in the current setup. Training runs for the full 50 epochs while the best validation-loss checkpoint is saved separately.

A new best model is saved when:

```python
val_loss < best_val_loss - 1e-5
```

---

## Training Log

`log.txt` records the training progress for every epoch, including:

```text
training loss
validation loss
training accuracy
validation accuracy
validation balanced accuracy
validation precision
validation recall
validation F1-score
learning rate
```

When a new best model is found, the log also records the epoch and validation-loss improvement.

---

## Reproducibility

A fixed random seed is used for:

```text
Python random
NumPy
PyTorch
train/validation split
DataLoader shuffling
```

This helps keep the source-level split and training setup reproducible between runs.

---

## Notes

- Training patches are generated on the fly and are not stored as PNG files.
- Validation data are not augmented.
- Small rotational augmentation is used during training.
- Neighboring-patch smoothing is not part of training and can be added later during inference.
- The independent external test set should remain untouched until final evaluation.

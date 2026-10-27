# Cell Secretor Classifier

This project prepares multi-layer nanowell TIFF images for binary classifiers:

- insulin `high` / `low`
- glucagon `high` / `low`

Insulin and glucagon are trained as independent models. Use the same pipeline
twice with different `--target` values.

The model input is restricted to morphology-derived channels only:

- Layer 0: brightfield (BF)
- Layer 1: nuclear fluorescence
- Channel 2: a nucleus-guided BF cell mask (derived, not a TIFF layer)
- Channel 3: Layer-0 BF pixels at the primary-nucleus coordinates detected in Layer 1

The nucleus channel identifies the primary nucleus. Its centroid seeds a BF
watershed segmentation, so distant BF structures cannot become the target cell.
The selected UI ROI size is a maximum search/crop size; the final square crop is
automatically tightened from nucleus size and the segmented-cell bounding box.
Background outside the candidate cell is smoothly dimmed instead of hard-zeroed.
Bead pixels are excluded from the nucleus-guided cell mask and replaced again in
the final BF tensor, so a bead attached to a cell cannot be treated as cell body.

Autofluorescence and bead fluorescence channels are used only to find ignored
regions in the model input:

- Layer 2: bead/nanowell autofluorescence
- Layer 7: INS capture beads
- Layer 5: GCG capture beads

Layers 3, 4, and 6 are ignored.

## Recommended Data Layout

Put manually curated TIFF images into this structure:

```text
data/
  raw/
    insulin/
      high/
        sample_001.tif
      low/
        sample_002.tif
    glucagon/
      high/
        sample_101.tif
      low/
        sample_102.tif
```

Then create a manifest:

```bash
python scripts/make_manifest.py \
  --data-root data/raw \
  --out data/manifest.csv
```

The manifest split is made by original image group, not by individual augmented
files. Augmented variants such as `_orig`, `_rot90`, `_rot180`, `_rot270`,
`_flip_lr`, and `_shift_up_5px` / `_shift_down_5px` / `_shift_left_5px` /
`_shift_right_5px` share one source ID and split, including combined suffixes.
Shift suffixes accept any nonnegative integer pixel count (for example,
`_shift_up_10px`), so 5px and 10px variants stay with the same original source.
Numeric source suffixes remain distinct unless provenance confirms their identity.
By default, about one sixth of each high/low group is held out for validation.

Preview preprocessing without training:

```bash
python scripts/preview_preprocess.py \
  --manifest data/manifest.csv \
  --out-dir previews \
  --max-images 50
```

Train later on the workstation/GPU machine:

```bash
python train.py \
  --manifest data/manifest.csv \
  --target insulin \
  --output-dir runs/insulin_resnet18_bf_nucleus \
  --preview-images 50

python train.py \
  --manifest data/manifest.csv \
  --target glucagon \
  --output-dir runs/glucagon_resnet18_bf_nucleus \
  --preview-images 50
```

Training outputs include model metrics, plots, and visual QC examples:

```text
runs/insulin_resnet18_bf_nucleus/
  best_model.pt
  history.csv
  test_metrics.json
  training_curves.png
  test_confusion_matrix.png
  test_roc_curve.png
  test_precision_recall_curve.png
  validation_threshold_curve.png
  validation_thresholds.csv
  preprocess_attention_examples/
    examples.csv
    000_sample_attention.png
    001_sample_attention.png
```

Each model-focus example shows the preprocessed BF channel, preprocessed nucleus
channel, nucleus-guided cell mask, Grad-CAM model-focus map, and overlay for the
model's predicted class.

The high/low decision threshold is selected from validation predictions and
stored in `best_model.pt`. Test metrics and `predict.py` use this stored
threshold instead of always using a fixed 0.5 cutoff. Use
`--threshold-mode high_recall --min-high-recall 0.9` when missing high cells is
more costly than false positives.

## Local UI

Run the Streamlit UI from the project folder:

```bash
python -m streamlit run app.py
```

The UI provides pages for data folder checks, manifest creation, training,
evaluation plots, Grad-CAM preview images, and prediction CSV generation.

## Independent Preprocessing and Training

The optional v2 workflow freezes preprocessing into compressed NPZ tensors so
model training never opens or reprocesses the source TIFF files.

First create the usual source manifest, then preprocess it once:

On Windows, launch the folder-browsing UI with:

```bash
python preprocess_ui.py
```

The UI can select either an input TIFF folder (searched recursively) or an
existing source manifest CSV, plus the output folder. Folder mode writes the
discovered file list to `source_manifest.csv` in the output directory. It also
exposes the object-size and morphology settings. Each click launches an
independent background process, so the UI can be closed and different
input/output folder pairs can run concurrently. A representative
multi-layer TIFF can also be selected to compare a raw BF/nucleus/bead
composite with the final four-channel model input before processing the full
dataset. The equivalent command-line workflow is:

```bash
python preprocess_dataset.py \
  --manifest data/manifest.csv \
  --output-dir data/preprocessed_v2 \
  --roi-size 128 \
  --bead-min-diameter 5 \
  --bead-max-diameter 20 \
  --bead-dilation-px 2 \
  --nucleus-min-diameter 10 \
  --nucleus-max-diameter 100 \
  --cell-opening-px 4
```

Add `--background` to return immediately while preprocessing continues in a
detached process:

```bash
python preprocess_dataset.py \
  --input-dir data/raw_tiffs_a \
  --output-dir data/preprocessed_a \
  --background

python preprocess_dataset.py \
  --input-dir data/raw_tiffs_b \
  --output-dir data/preprocessed_b \
  --background
```

Each output directory receives `preprocess_progress.json`, `preprocess.log`, and
a temporary `preprocessing.lock`. Monitor one or several jobs from a terminal:

```bash
python monitor_preprocessing.py \
  --output-dir data/preprocessed_a \
  --output-dir data/preprocessed_b \
  --watch
```

The monitor reports PID, current file, processed and cached counts, QC
exclusions, errors, elapsed time, and ETA. Closing the monitor does not stop the
background jobs. Two jobs may run simultaneously when they use different
output directories; the lock prevents accidental concurrent writes to the same
output directory.

Progress updates are throttled to once per second to avoid sync-client churn.
TIFF reads and atomic progress, NPZ, QC, and manifest writes automatically retry
transient filesystem errors such as OneDrive file locks. A temporarily locked
progress or log file never stops image processing. Re-running the same output
directory after an interruption reuses every valid four-channel NPZ and only
processes missing items.

To process every TIFF under a folder without preparing a manifest first:

```bash
python preprocess_dataset.py \
  --input-dir data/raw_tiffs \
  --output-dir data/preprocessed_v2
```

This writes:

```text
data/preprocessed_v2/
  preprocess_run_config.json
  preprocess_qc_all.csv
  preprocessed_manifest.csv
  tensors/
    <source>_<path-hash>.npz
    <source>_<path-hash>.qc.json
```

The v2 preprocessing independently detects INS and GCG bead objects, declumps
them, filters by equivalent diameter, dilates and combines their masks, applies
nucleus diameter/border QC, morphologically cleans the cell mask, and records
blur, saturation, cell/nucleus ratio, border contact, ROI size, and object
counts. Add `--fail-on-qc` to exclude samples without a valid nucleus from the
trainable manifest while retaining every result in `preprocess_qc_all.csv`.

Train only from those frozen tensors with the separate entry point:

On Windows, the corresponding NPZ-only training UI can be opened with:

```bash
python train_preprocessed_ui.py
```

For a Streamlit dashboard with preprocessing progress, background NPZ training,
cross-run model metrics/learning-curve comparison, Grad-CAM QC previews, and
NPZ-only prediction, run:

```bash
streamlit run preprocessed_dashboard.py
```

Every dashboard path field supports direct typing plus a server-side drive,
folder, and file browser. The Manifest Builder page combines completed
`insulin|glucagon/high|low` preprocessing folders, assigns labels, groups
augmented variants by their original source ID, and writes deterministic
train/validation/test splits.

Test sets now contain equal numbers of high and low images within each hormone.
The per-class budget is 15% (or the selected test fraction) of the smaller
class's image count. Whole source groups, including their augmented variants,
stay together; the splitter selects the largest achievable equal count at or
below that budget. Validation groups retain the previous allocation for the
same dataset, fractions, and seed, and remaining groups go to training.
An error is reported if no nonempty balanced test set fits the budget.
Existing CSV splits and completed training runs are unchanged until a new
manifest is selected or the dataset-root manifest is refreshed.
The desktop training UI and dashboard default to
`training_manifest_balanced_test.csv`; the dashboard also uses this name when
preparing new training manifests. The default test fraction is 0.15.

The Training page accepts either a dataset root or an existing training-manifest
CSV. Dataset-root mode recognizes the
`insulin|glucagon/high|low/tensors` layout, rebases copied or stale OneDrive NPZ
paths to the selected local root, and prepares augmentation-safe
train/validation/test splits. The workspace default is `data/tensors_0825`.
After preparing the manifest, one click launches every selected target/model
combination sequentially through `train_preprocessed_batch.py`, with either
model-specific recommended schedules or one shared custom schedule.

Training runs publish atomic `progress.json` files and multi-run queues publish
`*_batch_progress.json`. The Streamlit Training page refreshes every three
seconds to show queue position, current target/model, epoch and batch progress,
validation AUROC, best score, ETA, device, job states, and a tail of `train.log`.
On Windows, progress and checkpoint replacement retries transient file locks;
an unavailable monitoring JSON is reported as a warning and does not terminate
model training.

Grad-CAM previews use one unaugmented `_orig` image per source group. Rotated,
flipped, and integer-pixel shifted copies are excluded, and the selected previews alternate high/low
labels when both classes are available. The QC page also hides augmented
duplicates from previews saved by older runs.

The dashboard Prediction page accepts the same dataset-root layout or an
existing prediction manifest. Dataset-root mode writes a local-path
`prediction_manifest.csv`, then calls `predict_preprocessed.py`. Prediction
reads cached NPZ tensors and refuses legacy or incomplete checkpoints so it uses
the same frozen preprocessing distribution as training.

```bash
python train_preprocessed.py \
  --manifest data/preprocessed_v2/preprocessed_manifest.csv \
  --target insulin \
  --output-dir runs/insulin_preprocessed_v2 \
  --model resnet18 \
  --image-size 224 \
  --roi-size 128 \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --num-workers 4
```

`train_preprocessed.py` forces NPZ-only loading through
`PreprocessedSecretorDataset`; the original TIFF paths are retained only for
reporting and are not read during training. Use a new preprocessing output
directory whenever preprocessing parameters change, and never mix caches from
different preprocessing versions in one experiment.

## GPU Setup

This project uses GPU automatically when PyTorch can see CUDA:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For this workstation, CUDA 13.2 PyTorch was installed with:

```bash
python -m pip install --force-reinstall torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
```

Check GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Predict with one model:

```bash
python predict.py \
  --checkpoint runs/insulin_resnet18_bf_nucleus/best_model.pt \
  --input data/new_tiffs \
  --out predictions_insulin.csv
```

Predict with both independent models into one CSV:

```bash
python predict.py \
  --insulin-checkpoint runs/insulin_resnet18_bf_nucleus/best_model.pt \
  --glucagon-checkpoint runs/glucagon_resnet18_bf_nucleus/best_model.pt \
  --input data/new_tiffs \
  --out predictions_both.csv
```

## Important Leakage Rule

Do not pass bead or autofluorescence channels to the classifier. Layers 5 and 7
create bead masks that are median-filled in BF/nucleus. Layer 2 creates an
autofluorescence/nanowell mask that is zeroed out in BF/nucleus, not
median-filled. Layer 2 uses a low-level signal threshold based on the near-black
background and high-signal range, rather than only masking the brightest top
1% pixels. It is tuned to remove broad autofluorescent/nanowell regions while
preserving pixels with clear Layer 1 nucleus signal.

If a bead overlaps the nucleus or covers a large part of the cell, exclude that
image during manual curation or keep it in a separate QC folder.

# Baseline Model Assessment

| | |
|---|---|
| **Project** | CBIS-DDSM Benign vs. Malignant Classification with MLOps |
| **Author** | Samantha Fernandez-Perez |
| **Date** | November 5, 2026 |
| **Version** | 1.0 |
| **Code reviewed** | git tag `v0-baseline` (commit `bddb033`) |
| **Purpose** | Document the existing model, record its baseline, and identify where MLflow plugs in (Milestone 1) |

---

## 1. Summary

The baseline is a ResNet18 that classifies cropped mammography lesions (ROIs) as benign or malignant. It is plain PyTorch, run from two command-line scripts, and configured from one YAML file. It was trained on an Apple M3 and writes its results to an `outputs/` folder.

At the moment it records nothing permanent. Each run overwrites the previous run's files, and the per-epoch numbers exist only in the terminal. That is the gap MLflow fills.

**Findings that shape the integration** (details in the sections below):

1. Per-epoch metrics are printed but never stored. A hook has to sit inside the training loop in `main()` to capture them (section 6).
2. `outputs/` is overwritten on every run, so there is no history and nothing ties a checkpoint to the settings that made it.
3. The config file is read but never copied into the outputs, so a result can't be traced back to its settings.
4. `metrics.json` contains nested subgroup results and `null` values. MLflow metrics must be flat numbers, so these need flattening and filtering before logging (section 7).
5. `pyproject.toml` lists no dependencies. The environment can only be reproduced from `pip freeze` (section 4).
6. Training does not enable deterministic algorithms, so repeat runs can differ slightly (section 9).

---

## 2. What the model does

| | |
|---|---|
| **Task** | Binary classification of cropped lesion images (ROI crops): 0 = benign, 1 = malignant |
| **Architecture** | torchvision ResNet18, ImageNet weights, final layer replaced by `Linear(512, 1)` (`model.py`, `build_model`) |
| **Input** | A batch of images with shape `(B, 3, 224, 224)`, `float32`. Grayscale JPEG converted to 3 channels, resized to 224×224, normalized with ImageNet mean and std |
| **Output** | One raw logit per image, shape `(B, 1)`. The code squeezes it to `(B,)`. Apply a sigmoid to get P(malignant) |
| **Loss** | `BCEWithLogitsLoss` |
| **Optimizer** | Adam, learning rate from the config |
| **Train augmentation** | Random horizontal flip, random vertical flip, random rotation up to 10° |
| **Validation and test** | Resize and normalize only |
| **Decision rule** | `prob >= threshold` means malignant. The threshold is a command-line option of `evaluate`, default 0.5 |

**Labels.** `BENIGN_WITHOUT_CALLBACK` is counted as benign. The original three-class value is kept as `pathology_raw` in `rois.csv`.

**Data splits.** The official CBIS-DDSM train/test split is used. Validation is carved from the training rows **by patient** (`GroupShuffleSplit`, `val_frac` 0.15, seeded), so no patient appears in both train and validation. The test set is not used during training; `train.py` discards the test loader (`train_dl, val_dl, _ = make_loaders(cfg)`).

| Split | ROIs | Patients | % malignant |
|---|---|---|---|
| Train | 2,412 | 1,060 | 41.5% |
| Validation | 451 | 188 | 40.1% |
| Test | 704 | 349 | 39.2% |

---

## 3. Code map

| File | Role |
|---|---|
| `configs/baseline.yaml` | Every setting (section 5) |
| `src/cbis/data.py` | `load_config`, `load_rois`, `split_train_val`, `build_transforms`, `RoiDataset`, `make_loaders` |
| `src/cbis/model.py` | `build_model(pretrained)` |
| `src/cbis/train.py` | `set_seed`, `get_device`, `train_one_epoch`, `evaluate`, `main` |
| `src/cbis/evaluate.py` | Test-set predictions, metrics, subgroup AUCs, plots. It imports `get_device` from `train.py` |
| `notebooks/01_explore.ipynb` | Metadata exploration. Builds `rois.csv` |
| `scripts/check_data.py` | Prints one batch to confirm shapes |

**Input table.** Both scripts read `processed/rois.csv` from `data_dir`. It has one row per ROI crop and these columns: `patient_id, split, label, pathology_raw, abnormality_type, density, view, laterality, roi_path`. It currently has 3,567 rows. It is built once by the notebook and is not produced by the training code.

---

## 4. Dependencies and environment

| | |
|---|---|
| **Machine** | Mac, Apple M3. Training used the `mps` device (`get_device()` returns `mps` when available, otherwise `cpu`) |
| **Python** | 3.11.18, conda environment `cbis` |

Key packages, from `pip freeze` in the `cbis` environment:

| Package | Version |
|---|---|
| torch | 2.15.0 |
| torchvision | 0.29.2 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| matplotlib | 3.11.2 |
| Pillow | 13.0.0 |
| PyYAML | 6.0.3 |
| mlflow | 3.1.4 (matches the server version) |

The environment also holds Jupyter packages. The full freeze has 162 lines.

**Gap:** `pyproject.toml` declares only the package name, version and Python requirement. Nothing is pinned, so a fresh install may pull different versions. To reproduce today's environment, run `pip freeze` in `cbis` and keep the output with the repo (a decision for Milestone 4).

---

## 5. Configuration

All settings live in `configs/baseline.yaml` and are loaded by `load_config` (`data.py`), which also expands `~` in `data_dir`. Both scripts take the path with `--config`.

| Key | Value | Meaning |
|---|---|---|
| `data_dir` | `~/code/data/cbis-ddsm` | Folder holding `csv/`, `jpeg/`, `processed/` |
| `rois_csv` | `processed/rois.csv` | Table of ROI crops, relative to `data_dir` |
| `img_size` | 224 | Square resize |
| `val_frac` | 0.15 | Share of training patients held out for validation |
| `seed` | 42 | Seeds the split, subset sample, shuffles and weights |
| `subset_size` | `null` | If set, trains on a random subset (for quick tests) |
| `batch_size` | 32 | |
| `num_workers` | 0 | DataLoader workers (0 is safest on Mac) |
| `pretrained` | `true` | ImageNet weights |
| `lr` | 0.0001 | Adam learning rate |
| `epochs` | 5 | |
| `output_dir` | `outputs` | Where results are written. **Relative to the folder you run from**, so run from the repository root |

`evaluate` adds two command-line options that are not in the config: `--checkpoint` (default `{output_dir}/best.pt`) and `--threshold` (default 0.5). Both need to be logged as parameters.

---

## 6. How to run

From the repository root, with `conda activate cbis`:

```bash
python -m cbis.train --config configs/baseline.yaml
python -m cbis.evaluate --config configs/baseline.yaml
```

Train first. Evaluate loads the checkpoint that train saved.

**What train does** (`train.py`, `main`):

1. Loads the config and seeds Python, NumPy and PyTorch (`set_seed`).
2. Builds the loaders and model, and moves the model to the device.
3. For each epoch, trains and then evaluates on validation, printing train loss, validation loss and validation AUC.
4. Saves `outputs/best.pt` (a `state_dict`) whenever validation AUC improves.
5. Prints the best validation AUC.

**What evaluate does** (`evaluate.py`, `main`):

1. Rebuilds the test split from the same config, so it matches training.
2. Loads `best.pt`, predicts probabilities on the test set, applies the threshold.
3. Writes the predictions, a metrics file and two plots.
4. Prints a summary.

Only the validation set picks the checkpoint. The test set is used only in `evaluate`, and neither it nor the threshold selects anything.

---

## 7. What gets saved

All files go to `output_dir` (`outputs/`). **Every run overwrites the previous one.**

| File | Written by | Contents |
|---|---|---|
| `best.pt` | `train` | Best-validation-AUC weights (`state_dict`, about 45 MB) |
| `test_predictions.csv` | `evaluate` | `patient_id, abnormality_type, density, view, label, prob, pred` for each test ROI (704 rows) |
| `metrics.json` | `evaluate` | Checkpoint path, threshold, UTC timestamp, `n_test`, `auc`, `accuracy`, `sensitivity`, `specificity`, `confusion_matrix`, `auc_by_abnormality_type`, `auc_by_density` |
| `roc_curve.png` | `evaluate` | ROC curve with the AUC in the legend |
| `confusion_matrix.png` | `evaluate` | 2×2 counts at the chosen threshold |

**Not saved anywhere:**
- Per-epoch train loss, validation loss and validation AUC (terminal only)
- The config used for the run
- The git commit, package versions, device, or a reference to the dataset version

---

## 8. Baseline run

**Settings:** the values in section 5 (5 epochs, lr 1e-4, seed 42, full dataset, device `mps`).

**Training log** (captured from the terminal):

| Epoch | Train loss | Val loss | Val AUC |
|---|---|---|---|
| 1 | 0.5918 | 0.6224 | 0.724 |
| 2 | 0.5433 | 0.5518 | 0.799 |
| 3 | 0.5192 | 0.5408 | 0.800 |
| 4 | 0.4694 | 0.5224 | **0.828** |
| 5 | 0.4596 | 0.6164 | 0.786 |

Best validation AUC is 0.828 at epoch 4, so `best.pt` holds the epoch-4 weights. Validation loss rises at epoch 5 while training loss keeps falling, an early sign of overfitting.

**Test results** (704 ROIs, threshold 0.5, from `metrics.json`):

| Metric | Value |
|---|---|
| AUC | 0.740 |
| Accuracy | 0.632 |
| Sensitivity | 0.775 |
| Specificity | 0.540 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[231, 197], [62, 214]]` |

| Subgroup | n | Malignant | AUC |
|---|---|---|---|
| Mass | 378 | 147 | 0.784 |
| Calcification | 326 | 129 | 0.712 |
| Density 1 | 74 | 27 | 0.890 |
| Density 2 | 287 | 111 | 0.793 |
| Density 3 | 227 | 86 | 0.686 |
| Density 4 | 114 | 50 | 0.597 |
| Density 0 | 2 | 2 | undefined (one class) |

The model does better on masses than calcifications, and AUC falls as breast density rises. These numbers are the reference point. After MLflow is added, a rerun should match them closely (see section 9 on why not exactly).

---

## 9. Reproducibility and nondeterminism

What is controlled:
- `set_seed` seeds Python's `random`, NumPy, and PyTorch (`torch.manual_seed`).
- The patient split (`GroupShuffleSplit`) and the `subset_size` sample both use `seed`.
- `num_workers` is 0, so there are no worker processes with their own random state.

What is not controlled, so repeat runs may differ slightly:
- No call to `torch.use_deterministic_algorithms(True)`, and no cuDNN settings. Some GPU kernels, including on `mps`, are not bit-for-bit repeatable.
- Results can vary across hardware and PyTorch versions.
- Random augmentation (flips, rotation) draws from the seeded PyTorch generator, so it is repeatable only if the order of operations is identical.

The goal for Milestone 3 is to **document** this, not to change the training code.

---

## 10. MLflow integration points

The plan keeps model and training logic frozen. Each point below is a place for a thin hook. All MLflow calls go through one new module, `src/cbis/tracking.py`.

Line numbers refer to tag `v0-baseline`.

### Training (`train.py`, `main`)

| # | Where | What is available there | Hook |
|---|---|---|---|
| T1 | After `cfg = load_config(...)` (line 59) | `cfg` dict, `args.config` path | Start the run. Log all config values and the config file as an artifact |
| T2 | After `set_seed` and `get_device()` (lines 60–62) | `seed`, `device` | Log `seed` and `device` as params |
| T3 | After model and optimizer are built (lines 65–67) | model, optimizer | Log architecture name, `pretrained`, loss and optimizer names |
| T4 | Inside the epoch loop, after the validation call (line 75) | `epoch`, `train_loss`, `val_loss`, `val_auc` | Log the three metrics with `step=epoch` |
| T5 | Inside the `if val_auc > best_auc:` block, after `torch.save` (line 79) | the checkpoint path | Optional: remember which epoch was best |
| T6 | After the loop, at the final print (line 81) | `best_auc`, `out_dir / "best.pt"` | Log `best_val_auc`. Log `best.pt` as an artifact and package the model (section 11) |
| T7 | Around all of `main()` | any exception | Mark the run FAILED and log the traceback as an artifact |

Because `train_one_epoch` and `evaluate` return their values to `main()`, T4 needs no change to either function. The hook goes in the loop body, next to the existing `print`.

### Evaluation (`evaluate.py`, `main`)

| # | Where | What is available there | Hook |
|---|---|---|---|
| E1 | After `cfg` and `checkpoint` are resolved (lines 133–136) | `cfg`, `args.threshold`, `checkpoint` | Start or attach to the run. Log `threshold` and `checkpoint` |
| E2 | After `metrics.json` is written (line 158) | the `metrics` dict | Log the flat metrics (below) |
| E3 | After `save_plots` (line 160) | the five files in `outputs/` | Log the two PNGs, `metrics.json` and `test_predictions.csv` as artifacts |

**Preparing `metrics.json` for MLflow.** `mlflow.log_metric` accepts only a name and a number, so the file needs a small conversion first:

- Skip `null` values (for example density 0 has `"auc": null`).
- Flatten the subgroups into names such as `test_auc`, `test_auc_mass`, `test_auc_calcification`, `test_auc_density_1`. Log subgroup sizes (`n`, `n_positive`) as metrics or tags.
- Log `confusion_matrix` as four metrics (`tn`, `fp`, `fn`, `tp`) or keep it in the JSON artifact.
- Keep the whole `metrics.json` as an artifact too, so nothing is lost.

**Open design question:** training and evaluation are separate commands. Decide in Milestone 3 whether evaluation logs into the same run as training (for example by passing a run ID) or creates its own linked run. The registered model needs its test metrics connected to it either way.

---

## 11. What to log

### Parameters
- Every key in `baseline.yaml` (section 5)
- `threshold` and `checkpoint` from `evaluate`
- `device`, `seed`, architecture (`resnet18`), loss (`BCEWithLogitsLoss`), optimizer (`Adam`)

### Metrics
- Per epoch (`step=epoch`): `train_loss`, `val_loss`, `val_auc`
- Final: `best_val_auc`
- Test: `test_auc`, `test_accuracy`, `test_sensitivity`, `test_specificity`, `test_tn/fp/fn/tp`, and AUC per abnormality type and density

### Artifacts
- `best.pt`, or better the packaged MLflow model (see below)
- `metrics.json`, `test_predictions.csv`
- `roc_curve.png`, `confusion_matrix.png`
- The config file used for the run

### Metadata (tags or params)
- Git commit hash and whether the working tree was dirty
- Dataset reference: a hash of `rois.csv` (the current file starts `ee66060e8e933d60`), its row count, and the split sizes
- Package versions: torch, torchvision, scikit-learn, numpy, mlflow
- Python version and device

### Model packaging
- `mlflow.pytorch` flavor with a **signature** (input `(B, 3, 224, 224)` float32, output `(B, 1)` float32) and an **input example**
- The saved weights are only a `state_dict`, so packaging needs `build_model(pretrained=False)` plus `load_state_dict`, the same way `evaluate.py` loads it. The model that MLflow logs should return the **logit**, since the sigmoid is applied outside the model. Document that in the registered model's description

---

## 12. Risks and open questions

| Item | Why it matters | Plan |
|---|---|---|
| Outputs overwritten each run | Two runs can't be compared from files alone | Each MLflow run gets its own artifacts. Consider run-specific output folders |
| No dependency pins | Environment isn't reproducible from the repo | Save a freeze file and document it |
| Nondeterminism | Two runs with the same config may differ | Document it. Log versions and device |
| `null` AUCs and nested JSON | Can crash `log_metric` or lose data | Flatten and filter in `tracking.py` (section 10) |
| `evaluate.py` imports from `train.py` | Importing `train` loads its imports too. There is no side effect now, but the coupling is worth knowing | Leave as is. Don't edit frozen code |
| Single threshold (0.5) chosen by default | Sensitivity and specificity depend on it | Log the threshold with every evaluation |
| Small test subgroups (density 0 has 2 cases) | AUC for tiny groups is meaningless | Log `n` next to every subgroup AUC |
| Evaluation and training are separate steps | A model's test metrics must still link to its training run | Decide the run-linking approach in Milestone 3 |

---

## 13. Next steps

1. Milestone 2 is complete: the local MLflow server (Postgres and an artifact volume) runs at `http://localhost:5001`. See [`setup.md`](setup.md).
2. Milestone 3: create `src/cbis/tracking.py` with small helpers (for example `tracked_run`, `log_epoch`, `log_artifacts`, `log_and_register_model`), then add the hooks T1–T7 and E1–E3.
3. Train twice with different settings to produce two registered model versions, and set an alias such as `champion`.

# 🩻 CBIS-DDSM + MLOps

**A plain PyTorch mammography classifier, wrapped in an MLflow tracking and registry setup, the way it's done on real teams.**

> Hi, I'm Samantha 👋 I'm a biomedical engineering grad focused on machine learning for breast cancer imaging. This is my second portfolio project (the first was BreakHis histopathology classification with fastai). The idea is simple: in industry, the model usually already exists, and the MLOps work is making it **tracked, reproducible, tested and documented without changing how it trains**. So that's exactly what I'm practicing here.

---

## The two phases

| Phase | What happens | Status |
|---|---|---|
| **A: The "existing model"** | A simple ResNet18 benign-vs-malignant classifier, built as plain PyTorch scripts and then frozen (`v0-baseline`) | ✅ done |
| **B: The MLOps layer** | Assessment, local MLflow server (Docker + Postgres), integration, model registry, tests, handover docs | 🚧 in progress (the MLflow server is running; integration is next) |

The rule for Phase B is that the model and training logic stay untouched. MLflow gets added through thin hooks, not by rewriting the training code.

---

## The task

Classify **cropped lesions (ROIs)** from the [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) mammography dataset as **benign (0)** or **malignant (1)**.

- **Model:** ResNet18 with ImageNet weights, final layer replaced by one logit, trained with `BCEWithLogitsLoss`
- **Input:** 224×224 ROI crops, grayscale converted to 3 channels, ImageNet normalization
- **Split:** the official train/test split. Validation is carved from the training set **by patient** (`GroupShuffleSplit`), so no patient appears in both train and val. The test set is only used for final evaluation.
- **Labels:** `BENIGN_WITHOUT_CALLBACK` is counted as benign. The original label is kept as `pathology_raw` in case you want to revisit that.

| Split | ROIs | Patients | % malignant |
|---|---|---|---|
| Train | 2,412 | 1,060 | 41.5% |
| Validation | 451 | 188 | 40.1% |
| Test | 704 | 349 | 39.2% |

---

## Repo layout

```
cbis-ddsm-mlops/
├── configs/baseline.yaml      # every setting: paths, image size, lr, epochs, seed, subset size
├── src/cbis/
│   ├── data.py                # config loading, patient-level split, transforms, Dataset
│   ├── model.py               # build_model(): ResNet18 with 1 output logit
│   ├── train.py               # training loop; saves the best checkpoint by validation AUC
│   └── evaluate.py            # test-set metrics, subgroup AUCs, plots
├── notebooks/01_explore.ipynb # metadata exploration; builds rois.csv
├── scripts/
│   ├── check_data.py          # prints one batch to confirm shapes
│   └── smoke_test.py          # logs a test run to the local MLflow server
├── docs/setup.md              # how to run the local MLflow environment
├── docker-compose.yml         # MLflow server + Postgres + artifact volume
├── mlflow/Dockerfile          # MLflow image + Postgres driver
└── PROJECT_PLAN.md            # the full plan, Phase A and B
```

---

## Replicate it from scratch

### 1. Environment

Tested on a Mac (Apple M3) with Python 3.11. PyTorch uses the `mps` device when it's available and falls back to CPU.

```bash
conda create -n cbis python=3.11
conda activate cbis
pip install torch torchvision pandas pillow scikit-learn pyyaml matplotlib
pip install -e .            # installs the cbis package from src/
```

The versions I used: torch 2.15.0, torchvision 0.29.2, pandas 2.3.3, scikit-learn 1.9.1, matplotlib 3.11.2, Pillow 13.0.0, PyYAML 6.0.3.

> `pyproject.toml` doesn't list dependencies yet. Install them with the `pip` line above. Pinning them properly is part of the Phase B assessment.

### 2. Data (not in git!)

1. Download **"CBIS-DDSM: Breast Cancer Image Dataset"** by awsaf49 on Kaggle (the JPEG version, about 6 GB).
2. Unzip it so you have this layout:
   ```
   <data_dir>/
   ├── csv/       # case descriptions + dicom_info.csv
   ├── jpeg/      # the images
   └── processed/ # rois.csv (you create this in the next step)
   ```
3. Set `data_dir` in `configs/baseline.yaml` to that folder (mine is `~/code/data/cbis-ddsm`).
4. Run `notebooks/01_explore.ipynb`. It cleans the metadata and writes `processed/rois.csv`: one row per ROI crop with `patient_id, split, label, pathology_raw, abnormality_type, density, view, laterality, roi_path`. It also documents the pitfalls, such as the DICOM-style paths not matching the JPEG files and one row dropped because its image doesn't exist in this download.

### 3. Sanity check (about 10 seconds)

```bash
python scripts/check_data.py
```

You should see `images: torch.Size([32, 3, 224, 224])` and a few labels. Set `subset_size` in the config (for example `300`) for fast test runs. `null` means use everything.

### 4. Train

```bash
python -m cbis.train --config configs/baseline.yaml
```

This prints train loss, validation loss and validation AUC each epoch. It saves the best weights (by validation AUC) to `outputs/best.pt`. Everything is seeded (`seed: 42`), but see the note on reproducibility below.

### 5. Evaluate

```bash
python -m cbis.evaluate --config configs/baseline.yaml
# optional: --checkpoint outputs/best.pt --threshold 0.5
```

Everything goes into `outputs/`:

| File | What it is |
|---|---|
| `metrics.json` | AUC, accuracy, sensitivity, specificity, confusion matrix, AUC by abnormality type and by density, checkpoint path, threshold, timestamp |
| `test_predictions.csv` | One row per test ROI with its probability and prediction |
| `roc_curve.png` | ROC curve with the AUC in the legend |
| `confusion_matrix.png` | 2×2 confusion matrix at the chosen threshold |

The test set never chooses the checkpoint or the threshold. The checkpoint comes from validation AUC, and the threshold is a plain command-line value.

---

## Baseline results (`v0-baseline`)

**Settings:** ResNet18 (ImageNet weights), 5 epochs, lr 1e-4 (Adam), batch size 32, 224×224, seed 42, full dataset.

**Test set (704 ROIs, threshold 0.5):**

| Metric | Value |
|---|---|
| **AUC** | **0.740** |
| Sensitivity (recall for malignant) | 0.775 |
| Specificity | 0.540 |
| Accuracy | 0.632 |

Confusion matrix (rows are actual, columns are predicted):

|  | Pred benign | Pred malignant |
|---|---|---|
| **Actual benign** | 231 | 197 |
| **Actual malignant** | 62 | 214 |

**AUC by subgroup:**

| Group | n | Malignant | AUC |
|---|---|---|---|
| Mass | 378 | 147 | 0.784 |
| Calcification | 326 | 129 | 0.712 |
| Density 1 (fatty) | 74 | 27 | 0.890 |
| Density 2 | 287 | 111 | 0.793 |
| Density 3 | 227 | 86 | 0.686 |
| Density 4 (extremely dense) | 114 | 50 | 0.597 |
| Density 0 | 2 | 2 | n/a (only one class) |

**What I take from this:** the model is decent but not great, and accuracy isn't the point of this project. The pattern is the interesting part. It does better on masses than on calcifications, and its AUC drops steadily as breast density goes up, which makes sense because dense tissue hides lesions. At threshold 0.5 it leans toward calling things malignant (high sensitivity, low specificity).

---

## Phase B: the MLflow environment

A local MLflow tracking server and model registry run in Docker (MLflow 3.1.4 and PostgreSQL 16, with a persistent artifact volume).

```bash
cp .env.example .env              # then change POSTGRES_PASSWORD if you like
docker compose up -d --build
python scripts/smoke_test.py      # logs a test run
# UI: http://localhost:5001
```

The full guide, with troubleshooting and the persistence check, is in [`docs/setup.md`](docs/setup.md). Never run `docker compose down -v`, because it deletes all your runs.

**Roadmap:**
- [x] Local MLflow environment (Docker Compose, Postgres, persistent volumes)
- [ ] Assessment doc: inputs and outputs, dependencies, MLflow integration points
- [ ] Integration through a single `tracking.py` module: params, metrics, artifacts, git hash and dataset hash
- [ ] Model registry: signature, input example, two versions, a `@champion` alias
- [ ] Tests (`pytest` on a temporary tracking URI) and handover docs

---

## Good to know

- **Reproducibility:** the seed fixes the shuffles, the starting weights and the patient split. Runs can still differ slightly across machines because of MPS/CUDA kernels, DataLoader workers and hardware. Documenting the sources of nondeterminism is part of Milestone 3.
- **Data and models stay out of git:** `.gitignore` covers `outputs/`, `*.pt` and `.env`.
- **Not for clinical use.** This is a learning and portfolio project. The model has not been validated for any medical purpose.

---

*Built by [Samantha Fernandez-Perez](https://github.com/samfern-creator).*

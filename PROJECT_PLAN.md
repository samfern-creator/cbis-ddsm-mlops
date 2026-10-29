# CBIS-DDSM + MLOps — Project Plan (prompt for Cursor)

> Paste this whole file into Cursor's chat (or reference it with `@PROJECT_PLAN.md`) at the start of each session. Then tell the agent which step you're on.

---

## Who I am and how I want help

I'm Samantha Fernandez-Perez, a biomedical engineering grad focused on machine learning for breast cancer imaging. This project practices a common real-world MLOps task: taking an existing PyTorch model and wrapping it with MLflow (tracking server, model registry, reproducibility, tests, documentation) **without changing the model's algorithm or training behavior**.

It's built the way that work happens in industry, where the model already exists and the MLOps layer has to fit around it. It also finishes my second portfolio project (my first was BreakHis histopathology classification with fastai, patient-level cross-validation, MLflow logging, and feature export).

**How I want you (the agent) to help:**
- **Coach mode.** I write the code myself. Give me the next small step, a skeleton with TODOs, hints, and links to docs. Explain *why* in plain language.
- **Do not write full files or run commands for me** unless I explicitly ask ("write it for me") or I'm stuck after trying.
- One step at a time. Wait for me to paste output before moving on.
- Keep answers short and concrete. If you spot a bug, point to the line and explain; let me fix it.
- Be honest about uncertainty. Don't invent numbers, APIs, or column names — tell me to check.

---

## Environment (already set up)

- Mac, **Apple M3**, use PyTorch **`mps`** device (fall back to `cpu`).
- Conda env **`cbis`** (Python 3.11): torch, torchvision, pandas, pillow, scikit-learn, pyyaml, matplotlib, jupyter. Run `conda activate cbis` in every new terminal.
- Repo: **`~/code/cbis-ddsm-mlops`** — its own git repo (note: `~/code` is a separate parent repo; never commit there).
- Data: **`~/data/cbis-ddsm/`** with `csv/` and `jpeg/` (Kaggle "CBIS-DDSM: Breast Cancer Image Dataset" by awsaf49, JPEG version, ~6 GB). **Data never goes in git.**
- Docker is **not installed yet** (needed in Phase B, Milestone 2).

---

## The rule that shapes everything

The core rule: wrap the existing model **without changing it**. So the project has two phases:

- **Phase A** — build a simple, plain-PyTorch "existing model" as scripts, then **freeze it** (git tag `v0-baseline`).
- **Phase B** — do the four MLOps milestones on top of it, treating Phase A code as if it belonged to another team: no changes to model/training logic, only thin integration hooks.

---

## Phase A — Build the "existing model" (no MLflow yet)

**Goal:** benign vs. malignant classification of **cropped lesions (ROIs)**, ResNet18 (ImageNet weights), 224×224, official train/test split. Accuracy is secondary; a clean, working pipeline is the goal.

### A1. Repo skeleton
```
cbis-ddsm-mlops/
├── configs/baseline.yaml      # all settings (paths, img size, epochs, lr, seed, subset size)
├── src/cbis/
│   ├── data.py                # metadata loading, path mapping, Dataset class
│   ├── model.py               # build_model()
│   ├── train.py               # train loop; reads config; saves checkpoint
│   └── evaluate.py            # metrics: AUC, confusion matrix, per-subgroup
├── notebooks/01_explore.ipynb # exploration only
├── .gitignore                 # data/, mlruns/, *.pt, *.pkl, .DS_Store
└── README.md
```
**Done when:** structure exists, `.gitignore` excludes data and model files, first commit made.

### A2. Understand the metadata (notebook)
- Load the four case-description CSVs (mass/calc × train/test) and `dicom_info.csv`.
- Key columns: patient ID, breast density, left/right breast, image view (CC/MLO), abnormality type (mass/calcification), assessment (BI-RADS), **pathology** (MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK), subtlety, and the image / cropped image / ROI mask file paths.
- **Known pitfalls to check (don't assume):**
  - Column names differ slightly between mass and calc files (e.g., density column spelling) — standardize them.
  - The CSV file paths are DICOM-style and do **not** directly match the JPEG files. Map them to `jpeg/` paths using `dicom_info.csv` (series UID + series description: full mammogram vs. cropped images vs. ROI masks). Verify a few mappings by opening the images.
  - Decide and document: `BENIGN_WITHOUT_CALLBACK` → benign.
  - Check for **patient overlap between the official train and test sets** (including across mass/calc files). Report it.
- Summaries: counts by pathology, abnormality type, density, view; image sizes of ROI crops.

**Done when:** one clean table (one row per ROI crop) with `patient_id, split, label, abnormality_type, density, view, laterality, roi_path` saved to `~/data/cbis-ddsm/processed/rois.csv`, plus a findings note.

### A3. Dataset + DataLoader (`data.py`)
- A `torch.utils.data.Dataset` that reads `rois.csv`, loads grayscale ROI → 3 channels, resizes to 224, normalizes with ImageNet stats.
- Train augmentation: flips, small rotations. No augmentation for val/test.
- Validation set: carve from the **official train split by patient** (e.g., `GroupShuffleSplit`), never from test.
- Config option `subset_size` to train on a small sample for fast iteration on the Mac.

**Done when:** I can show a batch of images with labels and confirm shapes `(B, 3, 224, 224)`.

### A4. Model + training (`model.py`, `train.py`)
- `build_model()` → torchvision ResNet18 with ImageNet weights, final layer → 2 classes (or 1 logit).
- Plain PyTorch loop: optimizer, loss, epochs from config, device = mps/cpu, `set_seed()`.
- Print train/val loss and val AUC per epoch; save best checkpoint to `outputs/`.
- Run from the command line: `python -m cbis.train --config configs/baseline.yaml`.

### A5. Evaluation (`evaluate.py`)
- Test-set AUC (ROI level), confusion matrix, and AUC **by abnormality type (mass vs. calc)** and **by breast density**.
- Save metrics to a JSON and plots to `outputs/`.

### A6. Freeze
- Run baseline end to end, record numbers in README.
- `git tag v0-baseline`. From here on, **model/training logic is frozen**.

---

## Phase B — The MLOps milestones (on top of the frozen baseline)

### Milestone 1 — Assessment (`docs/assessment.md`)
- Document: model inputs/outputs, dependencies (`pip freeze`), how to run training/eval, where the config lives, what gets saved.
- Run the baseline and record it.
- Identify **MLflow integration points**: config load, start of training, per-epoch metrics, checkpoint save, evaluation outputs.
- List what should be logged: params, metrics, artifacts, metadata.

### Milestone 2 — Local MLflow environment (Docker Compose)
- Install Docker Desktop.
- `docker-compose.yml`: MLflow tracking server + **Postgres** (run metadata) + a mounted volume for artifacts. Model registry enabled.
- `.env.example` for settings; nothing secret committed.
- **Persistence test:** `docker compose down` → `up`, confirm runs and registered models are still there.
- `docs/setup.md`: start, access UI, stop, reset, troubleshooting. Reproducible by someone else.

### Milestone 3 — Integration
- `src/cbis/tracking.py`: the **only** module that imports MLflow. Training code calls small helpers (e.g., a context manager `tracked_run(config)`, `log_epoch(metrics, step)`, `log_artifacts(dir)`, `log_and_register_model(model, signature, input_example)`).
- Log: all config values, seed, git commit hash, dataset reference (hash of `rois.csv`), package versions, device, per-epoch metrics, final eval metrics, plots, the eval JSON.
- **Model packaging:** `mlflow.pytorch` flavor with a **signature** and **input example**.
- **Registry:** register the model; train twice with different settings → **two versions**; set an alias (e.g., `@champion`); load with `models:/<name>@champion` and run a prediction.
- **Traceability:** from a registered version, find its run, params, and metrics.
- **Failure handling:** if training crashes, the run is marked FAILED and the traceback is logged as an artifact.
- **Naming conventions:** experiment `cbis-ddsm-roi-classification`; run names with date + key config.
- **Comparison:** compare runs in the UI and with `mlflow.search_runs`.
- Document sources of **nondeterminism** (seeds, MPS/cuDNN, DataLoader workers, hardware).

### Milestone 4 — Validation and handover
- `pytest` tests for `tracking.py` using a **temporary local tracking URI** (never the real server).
- End-to-end test: tiny model/data → log → register → load → predict.
- Docs: `docs/setup.md`, `docs/troubleshooting.md`, `docs/design-decisions.md` (one paragraph per choice: what, alternatives, why), README overview with results.
- Practice a 10-minute handover walkthrough.

---

## Acceptance checklist
- [ ] Existing model and workflow reviewed and documented
- [ ] Local MLflow environment starts from the instructions
- [ ] Training runs with MLflow without changing model behavior
- [ ] Runs record params, metrics, metadata, artifacts
- [ ] Multiple runs compared in UI and via API
- [ ] Model registered and retrieved as a versioned model
- [ ] Registered version traces back to its run and metadata
- [ ] Tests pass
- [ ] Setup instructions tested from scratch
- [ ] Code, config, and docs committed

---

## Current status
- [x] Repo created and `git init`
- [x] `cbis` conda env
- [x] Data downloaded to `~/data/cbis-ddsm/` (`csv/`, `jpeg/`)
- [ ] **Next: A1 (repo skeleton), then A2 (explore metadata)**

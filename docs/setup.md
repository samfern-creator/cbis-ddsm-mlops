# Local MLflow Environment: Setup Guide

| | |
|---|---|
| **Project** | CBIS-DDSM Benign vs. Malignant Classification with MLOps |
| **Author** | Samantha Fernandez-Perez |
| **Date** | November 4, 2026 |
| **Version** | 1.0 |
| **Status** | Tested on macOS (Apple M3), Docker Desktop 29.9.2, Docker Compose v5.6.1 |

---

## 1. Purpose

This guide explains how to start, use, and stop the local MLflow environment for this project. The environment gives every training run a permanent record of its settings, metrics, and output files, so results can be compared and traced back later. It runs entirely on your own machine and doesn't need a cloud account.

## 2. What runs

The environment is defined in `docker-compose.yml` and has three parts:

| Component | What it does | Where its data lives |
|---|---|---|
| **MLflow server** (`mlflow`) | Web interface and API for logging and viewing runs, plus the model registry | — |
| **PostgreSQL** (`postgres`) | Stores run information: parameters, metrics, tags, and registered models | Docker volume `pgdata` |
| **Artifact store** | Stores files logged by runs, such as plots, model weights, and reports | Docker volume `mlartifacts` |

Both volumes are managed by Docker, so stopping or rebuilding the containers doesn't erase any runs.

**Versions:** the MLflow server and client are both pinned to **3.1.4**. This is the newest release that installed on Python 3.9 in my testing, which keeps the setup compatible with older model environments. The client and server should always use matching versions.

## 3. Prerequisites

- **Docker Desktop**, installed and running (the whale icon in the menu bar)
- **A Python environment with the MLflow client**, for logging runs:
  ```bash
  conda activate cbis
  pip install "mlflow==3.1.4"
  ```

## 4. First-time setup

From the repository root:

1. Create your local settings file from the template:
   ```bash
   cp .env.example .env
   ```
2. Optionally, open `.env` and change `POSTGRES_PASSWORD`. The `.env` file is listed in `.gitignore` and must never be committed.

## 5. Start the environment

```bash
docker compose up -d --build
```

Check that both services are running:

```bash
docker compose ps
```

`postgres` should show **healthy**, and `mlflow` should show **running**. The MLflow server waits for the database to be ready before it starts.

Open the interface at **http://localhost:5001**.

## 6. Check that everything works

Run the smoke test:

```bash
python scripts/smoke_test.py
```

It logs one run, `first-run`, to the experiment `smoke-test`. In the interface, confirm that the run shows:

- the parameter `lr = 0.001`
- a `loss` metric that decreases over 5 steps
- the artifact `hello.txt`

The test leaves a `hello.txt` in the working folder. You can delete it.

## 7. Log runs from your own code

Point the client at the local server before logging:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("your-experiment-name")
```

## 8. Stop and restart

Stop the environment. Your runs are kept:

```bash
docker compose down
```

Start it again:

```bash
docker compose up -d
```

> ⚠️ **Never run `docker compose down -v`.** The `-v` flag deletes the volumes, which permanently erases every run, metric, artifact, and registered model.

**Persistence check (passed November 4, 2026):** after `docker compose down` and `docker compose up -d`, the `first-run` run was still available with its parameter, metric history, and artifact.

## 9. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `bind: address already in use` on port 5000 | macOS AirPlay Receiver uses port 5000 | The project maps the interface to port **5001** instead. Alternatively, turn off AirPlay Receiver in System Settings → General → AirDrop & Handoff. |
| The page at `localhost:5001` doesn't load | The containers aren't running, or MLflow is still starting | Run `docker compose ps`. If needed, read the logs with `docker compose logs mlflow --tail 30`. |
| `docker: command not found`, or it can't connect to the Docker daemon | Docker Desktop isn't running | Open Docker Desktop and wait until the whale icon stops animating. |
| Runs don't appear in the interface | The client is logging somewhere else | Make sure your code calls `mlflow.set_tracking_uri("http://localhost:5001")`. |
| Errors that mention the MLflow version | The client and server versions differ | Check with `pip show mlflow`; the client should be 3.1.4. |

## 10. Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | Defines the MLflow server, PostgreSQL, and the two volumes |
| `mlflow/Dockerfile` | Official MLflow 3.1.4 image plus the PostgreSQL driver (`psycopg2-binary`) |
| `.env.example` | Template for database settings (committed) |
| `.env` | Your local settings (not committed) |
| `scripts/smoke_test.py` | Logs one test run to confirm the environment works |

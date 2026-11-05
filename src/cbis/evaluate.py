import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from cbis.data import RoiDataset, build_transforms, load_config, load_rois, split_train_val
from cbis.model import build_model
from cbis.train import get_device

PRED_COLUMNS = ["patient_id", "abnormality_type", "density", "view", "label", "prob", "pred"]
GROUP_COLUMNS = ["abnormality_type", "density"]


def load_test_data(cfg):
    """Rebuild the test split exactly as in training; return (test_df, DataLoader)."""
    _, _, test_df = split_train_val(load_rois(cfg), cfg)
    ds = RoiDataset(test_df, cfg["data_dir"], build_transforms(cfg["img_size"], train=False))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    return test_df, dl


@torch.no_grad()
def predict(model, loader, device):
    """Return the malignant probability (sigmoid of the logit) for every row, in order."""
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


def safe_auc(labels, probs):
    """AUC, or None when only one class is present (AUC is undefined then)."""
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probs))


def auc_by_group(df, column):
    """AUC (plus n and number of positives) for each value of `column`."""
    out = {}
    for key, g in df.groupby(column, dropna=False):
        out[str(key)] = {
            "n": int(len(g)),
            "n_positive": int(g.label.sum()),
            "auc": safe_auc(g.label, g.prob),
        }
    return out


def compute_metrics(df, threshold):
    """Overall AUC, threshold metrics, confusion matrix, and AUC per subgroup."""
    tn, fp, fn, tp = confusion_matrix(df.label, df.pred, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(df)),
        "auc": safe_auc(df.label, df.prob),
        "accuracy": float(accuracy_score(df.label, df.pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "auc_by_abnormality_type": auc_by_group(df, "abnormality_type"),
        "auc_by_density": auc_by_group(df, "density"),
    }


def save_plots(df, metrics, out_dir):
    """Save roc_curve.png and confusion_matrix.png into out_dir."""
    fpr, tpr, _ = roc_curve(df.label, df.prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    cm = np.array(metrics["confusion_matrix"])
    names = ["benign", "malignant"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=names)
    ax.set_yticks([0, 1], labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {metrics['threshold']})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def fmt(x):
    """Format a metric for printing; None (undefined) shows as n/a."""
    return "n/a" if x is None else f"{x:.3f}"


def print_summary(metrics):
    """Print overall numbers and the per-subgroup AUC tables."""
    print(f"\nTest set: n={metrics['n_test']}  threshold={metrics['threshold']}")
    print(f"AUC:         {fmt(metrics['auc'])}")
    print(f"Sensitivity: {fmt(metrics['sensitivity'])}")
    print(f"Specificity: {fmt(metrics['specificity'])}")
    for title, key in [("abnormality_type", "auc_by_abnormality_type"), ("density", "auc_by_density")]:
        print(f"\nAUC by {title}:")
        print(f"  {'group':<14}{'n':>6}{'pos':>6}{'auc':>8}")
        for group, m in metrics[key].items():
            print(f"  {group:<14}{m['n']:>6}{m['n_positive']:>6}{fmt(m['auc']):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: {output_dir}/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"

    device = get_device()
    print("device:", device)
    test_df, test_dl = load_test_data(cfg)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()

    df = test_df.copy()
    df["prob"] = predict(model, test_dl, device)
    df["pred"] = (df.prob >= args.threshold).astype(int)
    df[PRED_COLUMNS].to_csv(out_dir / "test_predictions.csv", index=False)

    metrics = {
        "checkpoint": str(checkpoint),
        "threshold": args.threshold,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **compute_metrics(df, args.threshold),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if metrics["auc"] is not None:
        save_plots(df, metrics, out_dir)

    print_summary(metrics)
    print(f"\nSaved predictions, metrics.json and plots to {out_dir}/")


if __name__ == "__main__":
    main()

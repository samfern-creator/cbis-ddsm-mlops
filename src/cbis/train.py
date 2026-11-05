import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cbis.data import load_config, make_loaders
from cbis.model import build_model


def set_seed(seed):
    """Make runs repeatable (same seed -> same shuffles and starting weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)          # (B, 1) -> (B,)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (mean loss, AUC) on a loader."""
    model.eval()
    total, probs, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        total += loss_fn(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    auc = roc_auc_score(labels, probs)
    return total / len(loader.dataset), auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print("device:", device)

    train_dl, val_dl, _ = make_loaders(cfg)   # test stays untouched until A5
    model = build_model(cfg["pretrained"]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_auc = evaluate(model, val_dl, loss_fn, device)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.3f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_dir / "best.pt")

    print(f"best val_auc: {best_auc:.3f}  ->  {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

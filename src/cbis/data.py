from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path):
    """Read the YAML config and expand ~ in data_dir."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(cfg["data_dir"]).expanduser()
    return cfg


def load_rois(cfg):
    """Read rois.csv; optionally keep a random subset for fast runs."""
    df = pd.read_csv(cfg["data_dir"] / cfg["rois_csv"])
    if cfg.get("subset_size"):
        df = df.sample(n=cfg["subset_size"], random_state=cfg["seed"])
    return df


def split_train_val(df, cfg):
    """Split the official TRAIN rows into train/val BY PATIENT. Test is untouched."""
    train_df = df[df.split == "train"]
    test_df = df[df.split == "test"]
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["val_frac"], random_state=cfg["seed"])
    train_idx, val_idx = next(gss.split(train_df, groups=train_df.patient_id))
    return (
        train_df.iloc[train_idx].reset_index(drop=True),
        train_df.iloc[val_idx].reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_transforms(img_size, train):
    """Train: resize + flips + small rotation. Val/test: resize only. Then normalize."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


class RoiDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.data_dir / row.roi_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.float32)
        return img, label


def make_loaders(cfg):
    """Return train, val, test DataLoaders."""
    df = load_rois(cfg)
    train_df, val_df, test_df = split_train_val(df, cfg)

    size = cfg["img_size"]
    train_ds = RoiDataset(train_df, cfg["data_dir"], build_transforms(size, train=True))
    val_ds = RoiDataset(val_df, cfg["data_dir"], build_transforms(size, train=False))
    test_ds = RoiDataset(test_df, cfg["data_dir"], build_transforms(size, train=False))

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)
    return train_dl, val_dl, test_dl

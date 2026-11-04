from cbis.data import load_config, make_loaders

cfg = load_config("configs/baseline.yaml")
cfg["subset_size"] = 300                     # quick test only

train_dl, val_dl, test_dl = make_loaders(cfg)
print("batches:", len(train_dl), len(val_dl), len(test_dl))

x, y = next(iter(train_dl))
print("images:", x.shape, x.dtype)           # expect torch.Size([32, 3, 224, 224])
print("labels:", y[:10])

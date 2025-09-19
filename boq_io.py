import os
import pandas as pd
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    data_root = os.path.expandvars(cfg.get("data_root", "./data"))
    return {"data_root": data_root}

def path_in_data(*parts, cfg=None):
    if cfg is None:
        cfg = load_config()
    return os.path.join(cfg["data_root"], *parts)

def load_csv(filename, cfg=None, **read_csv_kwargs):
    if cfg is None:
        cfg = load_config()
    full_path = os.path.join(cfg["data_root"], filename)
    return pd.read_csv(full_path, **read_csv_kwargs)

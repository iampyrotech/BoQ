import os
import yaml
import pandas as pd

def _read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def load_config():
    # 1) per-user override (gitignored)
    user_cfg = _read_yaml("config.local.yaml")
    # 2) team defaults (committed)
    base_cfg = _read_yaml("config.yaml")
    # merge: user overrides base
    cfg = {**base_cfg, **user_cfg}
    data_root = cfg.get("data_root", "./data")
    return {"data_root": data_root}

def path_in_data(*parts, cfg=None):
    if cfg is None:
        cfg = load_config()
    return os.path.join(cfg["data_root"], *parts)

def load_csv(filename, cfg=None, **kwargs):
    if cfg is None:
        cfg = load_config()
    full = os.path.join(cfg["data_root"], filename)
    return pd.read_csv(full, **kwargs)

import os
import re
import pandas as pd
import yaml

_PTN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")  # matches ${VAR} or ${VAR:-default}

def _expand_with_default(s: str) -> str:
    """
    Expand ${VAR} or ${VAR:-default}. If VAR unset/empty, use default.
    Falls back to os.path.expandvars for $VAR and ${VAR}.
    """
    def repl(m):
        var = m.group(1)
        default = m.group(2)
        val = os.getenv(var)
        if val is None or val == "":
            return default if default is not None else ""
        return val
    # First handle ${VAR:-default}
    s2 = _PTN.sub(repl, s)
    # Then handle plain $VAR / ${VAR}
    return os.path.expandvars(s2)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("data_root", "./data")
    data_root = _expand_with_default(str(raw))
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

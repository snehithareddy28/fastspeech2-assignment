"""Single-file config loader (new, not in ming024).

ming024 splits configuration into preprocess.yaml / model.yaml / train.yaml.
The assignment asks for one config.yaml per language, so each language folder
holds a single YAML with top-level sections. This module splits it back into
the three dicts that the upstream code expects, so model/dataset code can stay
close to upstream.

Relative paths in the YAML are resolved against the repository root.
"""
import copy
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p):
    if p is None or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(REPO_ROOT, p))


def load_raw_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["path"] = {k: _abs(v) for k, v in cfg["path"].items()}
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def load_config(path):
    """Return (raw_cfg, (preprocess_config, model_config, train_config))."""
    cfg = load_raw_config(path)
    language = cfg["language"]

    preprocess_config = {
        "dataset": cfg["dataset"],
        "language": language,
        "path": cfg["path"],
        "preprocessing": cfg["preprocessing"],
        "data": cfg.get("data", {}),
    }
    # upstream code reads preprocessing.text.language
    preprocess_config["preprocessing"].setdefault("text", {})
    preprocess_config["preprocessing"]["text"]["language"] = language

    model_config = copy.deepcopy(cfg["model"])
    model_config["symbol_set"] = language

    # anneal_steps may be given as fractions of total_step (e.g. [0.6, 0.8, 0.9]),
    # so shortening a run only needs total_step to change.
    opt = cfg["train"]["optimizer"]
    total = cfg["train"]["step"]["total_step"]
    if all(0 < a <= 1 for a in opt["anneal_steps"]):
        opt["anneal_steps"] = [int(round(a * total)) for a in opt["anneal_steps"]]

    train_config = {
        "path": cfg["path"],
        "optimizer": cfg["train"]["optimizer"],
        "step": cfg["train"]["step"],
        "misc": cfg["train"].get("misc", {}),
    }
    return cfg, (preprocess_config, model_config, train_config)

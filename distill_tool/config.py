from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ENGINE_COMMIT = "915bbf294bb74c859f0b41f1c23bc395014ea6679"
TRAIN_SHA256 = "70481af82f89ac31a119ef108371066786f504028b256ec1c23412112829acfa"


def load_recipe(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("Recipe must be a mapping with version: 1")
    allowed = {"version", "workspace", "engine", "data", "input", "training", "methods", "evaluation", "models", "workflow"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown recipe keys: {sorted(set(cfg) - allowed)}")
    for key in ("workspace", "engine", "data"):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    shape = cfg.get("input", {})
    if set(shape) != {"height", "width", "channels"}:
        raise ValueError("input requires height, width, channels")
    if shape["channels"] != 3 or any(type(shape[k]) is not int or shape[k] < 32 for k in ("height", "width")):
        raise ValueError("Only RGB with integer height and width >= 32 is supported")
    t = cfg["training"]
    if int(t["epochs"]) < 1 or int(t["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if t.get("device", "cpu") != "cpu" and "," in str(t["device"]):
        raise ValueError("This release runs one GPU per process; DDP is not advertised as supported")
    if not isinstance(cfg.get("methods"), dict):
        raise ValueError("methods must be a mapping of registered method names to parameters")
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in ("none", "baseline") for name in cfg["methods"]):
        raise ValueError("Method names must be identifiers and cannot be none/baseline")
    flow = cfg.get("workflow", {})
    if set(flow) - {"train_teacher", "baseline", "evaluate", "benchmark", "export"}:
        raise ValueError("Unknown workflow option")
    if any(type(v) is not bool for v in flow.values()):
        raise ValueError("workflow options must be boolean")
    if not cfg["methods"] and not flow.get("baseline", True):
        raise ValueError("Select a student baseline or at least one distillation method")
    if int(t.get("workers", 8)) < 0:
        raise ValueError("workers must be nonnegative")
    if "models" in cfg:
        if set(cfg["models"]) != {"teacher", "student"}:
            raise ValueError("models requires teacher and student")
        for role, spec in cfg["models"].items():
            if set(spec) - {"architecture", "weights"} or spec.get("architecture") not in {"yolov5" + n for n in "nsmlx"}:
                raise ValueError(f"Unsupported {role} model; select YOLOv5 n/s/m/l/x")
            if spec.get("weights"):
                value = Path(spec["weights"]).expanduser()
                spec["weights"] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
        if cfg["methods"] and not flow.get("train_teacher", True) and not cfg["models"]["teacher"].get("weights"):
            raise ValueError("An already trained teacher checkpoint is required when teacher training is disabled")
    cfg["recipe_path"] = str(path)
    return cfg


def fingerprint(cfg):
    clean = {k: v for k, v in cfg.items() if k != "recipe_path"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def network_shape(height, width, stride=32):
    return ((height + stride - 1) // stride * stride, (width + stride - 1) // stride * stride)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)

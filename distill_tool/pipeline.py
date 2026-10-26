from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .config import ENGINE_COMMIT, emit, fingerprint, network_shape, write_json
from .data import audit_dataset, resolve_split
from .engine.patch import generate_trainer


def provenance(cfg):
    from importlib.metadata import version, PackageNotFoundError
    root = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        code.update(file.relative_to(root).as_posix().encode())
        code.update(file.read_bytes())
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "PyYAML", "onnx", "onnxruntime"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    summary = Path(cfg["data"]).parent / "summary.json"
    return {"tool_source_sha256": code.hexdigest(), "engine_commit": ENGINE_COMMIT,
            "engine_train_sha256": hashlib.sha256((Path(cfg["engine"]) / "train.py").read_bytes()).hexdigest(),
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "recipe_sha256": fingerprint(cfg), "dataset": json.loads(summary.read_text()) if summary.exists() else None,
            "dataset_yaml_sha256": hashlib.sha256(Path(cfg["data"]).read_bytes()).hexdigest(),
            "model_checkpoints_sha256": {role: hashlib.sha256(Path(spec["weights"]).read_bytes()).hexdigest()
                for role, spec in cfg.get("models", {}).items() if spec.get("weights") and Path(spec["weights"]).is_file()},
            "pretrained_sha256": {n: hashlib.sha256((Path(cfg["engine"]) / (n + ".pt")).read_bytes()).hexdigest()
                                  for n in ("yolov5m", "yolov5n") if (Path(cfg["engine"]) / (n + ".pt")).exists()}}


def stages(cfg):
    methods = cfg.get("methods", {})
    flow = cfg.get("workflow", {})
    teacher, baseline, prefix = stage_names(cfg)
    return ([teacher] if flow.get("train_teacher", True) and methods else []) + ([baseline] if flow.get("baseline", True) else []) + [prefix + name for name in methods]


def stage_names(cfg):
    return ("teacher", "student_baseline", "student_") if "models" in cfg else ("teacher_m", "student_n_baseline", "student_n_")


def stage_method(stage, cfg=None):
    if cfg is not None:
        teacher, baseline, prefix = stage_names(cfg)
        return "none" if stage in (teacher, baseline) else stage[len(prefix):]
    if stage in ("teacher_m", "student_n_baseline", "teacher", "student_baseline"):
        return "none"
    return stage[len("student_n_"):] if stage.startswith("student_n_") else stage[len("student_"):]


def environment(cfg):
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([root, cfg["engine"], env.get("PYTHONPATH", "")])
    env["DISTILL_INPUT_HW"] = f"{cfg['input']['height']},{cfg['input']['width']}"
    env["DISTILL_SEED"] = str(cfg["training"]["seed"])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    return env


def prepare_engine_dataset(cfg):
    """Materialize absolute image lists so engine cwd cannot change path meaning."""
    work = Path(cfg["workspace"]) / "dataset"
    work.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load(Path(cfg["data"]).read_text(encoding="utf-8"))
    normalized = {"path": str(work.resolve()), "names": source["names"], "nc": len(source["names"])}
    for split in ("train", "val", "test"):
        if not source.get(split) or (split == "test" and not cfg.get("workflow", {}).get("evaluate", True)):
            continue
        manifest = work / (split + ".txt")
        manifest.write_text("\n".join(str(p.absolute()) for p in resolve_split(cfg["data"], split)) + "\n", encoding="utf-8")
        normalized[split] = str(manifest.resolve())
    target = work / "dataset.yaml"
    target.write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def commands(cfg, stage, resume=False):
    engine, work = Path(cfg["engine"]), Path(cfg["workspace"])
    t = cfg["training"]
    method = stage_method(stage, cfg)
    config = dict(cfg["methods"].get(method, {}))
    weight = config.pop("weight", 1.0)
    teacher_stage, _, _ = stage_names(cfg)
    role = "teacher" if stage == teacher_stage else "student"
    spec = cfg.get("models", {}).get(role, {})
    model_name = spec.get("architecture", "yolov5m" if role == "teacher" else "yolov5n")
    weights = spec.get("weights") or str(engine / (model_name + ".pt"))
    nh, nw = network_shape(cfg["input"]["height"], cfg["input"]["width"])
    normalized_data = work / "dataset/dataset.yaml"
    args = [sys.executable, str(engine / "train_distill_tool.py"), "--data", str(normalized_data) if normalized_data.exists() else cfg["data"],
            "--weights", weights, "--cfg", str(engine / "models" / (model_name + ".yaml")),
            "--imgsz", str(max(nh, nw)), "--batch-size", str(t["batch_size"]), "--epochs", str(t["epochs"]),
            "--patience", str(t["epochs"]), "--device", str(t["device"]), "--workers", str(t.get("workers", 8)),
            "--seed", str(t["seed"]), "--hyp", str(engine / "data/hyps/hyp.scratch-low.yaml"),
            "--project", str(work / "train"), "--name", stage, "--noautoanchor", "--noplots",
            "--kd-method", method, "--kd-weight", str(weight), "--kd-config", json.dumps(config)]
    if method != "none":
        teacher_weights = str(work / "train" / teacher_stage / "weights/best.pt") if cfg.get("workflow", {}).get("train_teacher", True) else cfg["models"]["teacher"]["weights"]
        args += ["--teacher-weights", teacher_weights]
    if resume:
        args += ["--resume", str(work / "train" / stage / "weights/resume.pt")]
    return args


def plan(cfg):
    return {"recipe_sha256": fingerprint(cfg), "external_hwc": [cfg["input"]["height"], cfg["input"]["width"], 3],
            "network_nchw": [cfg["training"]["batch_size"], 3, *network_shape(cfg["input"]["height"], cfg["input"]["width"])],
            "stages": [{"name": name, "depends_on": [stage_names(cfg)[0]] if stage_method(name, cfg) != "none" and cfg.get("workflow", {}).get("train_teacher", True) else [],
                        "argv": commands(cfg, name)} for name in stages(cfg)],
            "test_policy": "Test is accessed only after all training recipes are frozen; deployment selection uses validation."}


def run_stage(cfg, stage, resume=False):
    if stage not in stages(cfg):
        raise ValueError(f"Unknown stage {stage}")
    work, engine = Path(cfg["workspace"]), Path(cfg["engine"])
    identity_path = work / "stage_manifests" / (stage + ".json")
    if identity_path.exists() and json.loads(identity_path.read_text())["recipe_sha256"] != fingerprint(cfg):
        raise ValueError("Stage recipe changed; choose a new workspace instead of resuming different settings")
    run_dir = work / "train" / stage
    marker = run_dir / "complete.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        if done["recipe_sha256"] != fingerprint(cfg):
            raise ValueError("Existing stage belongs to a different recipe; choose a new workspace")
        if done.get("weights_sha256") and hashlib.sha256((run_dir / "weights/best.pt").read_bytes()).hexdigest() != done["weights_sha256"]:
            raise ValueError("Completed stage weights changed after training")
        emit("stage_skipped", stage=stage, reason="already complete")
        return
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Partial stage exists: {run_dir}. Inspect logs, then use --resume.")
    if resume and not (run_dir / "weights/resume.pt").is_file():
        raise FileNotFoundError("No complete resumable checkpoint exists")
    prepare_engine_dataset(cfg)
    generate_trainer(engine)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    env = environment(cfg)
    args = commands(cfg, stage, resume)
    record = {"stage": stage, "status": "running", "recipe_sha256": fingerprint(cfg), "argv": args}
    write_json(work / "status.json", record)
    write_json(identity_path, {"recipe_sha256": fingerprint(cfg), "stage": stage})
    with (work / "logs" / (stage + ".log")).open("a" if resume else "w", encoding="utf-8") as log:
        proc = subprocess.Popen(args, cwd=engine, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = proc.pid
        write_json(work / "status.json", record)
        emit("stage_started", stage=stage, pid=proc.pid)
        last = None
        while proc.poll() is None:
            progress = run_dir / "progress.json"
            if progress.exists():
                try:
                    row = json.loads(progress.read_text())
                    if row.get("epoch") != last:
                        emit("progress", stage=stage, **row)
                        last = row.get("epoch")
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(2)
    if proc.returncode:
        record.update(status="failed", exit_code=proc.returncode)
        write_json(work / "status.json", record)
        raise RuntimeError(f"{stage} failed with {proc.returncode}; see {work / 'logs' / (stage + '.log')}")
    if not (run_dir / "weights/best.pt").exists():
        raise RuntimeError("Trainer finished without a best checkpoint")
    write_json(marker, {"recipe_sha256": fingerprint(cfg), "epochs": cfg["training"]["epochs"],
                       "weights_sha256": hashlib.sha256((run_dir / "weights/best.pt").read_bytes()).hexdigest()})
    emit("stage_complete", stage=stage)


def import_engine(cfg):
    root = str(Path(__file__).resolve().parent.parent)
    for path in (cfg["engine"], root):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.update({k: v for k, v in environment(cfg).items() if k in ("DISTILL_INPUT_HW", "DISTILL_SEED", "CUBLAS_WORKSPACE_CONFIG")})


def evaluate(cfg):
    import_engine(cfg)
    import torch
    from models.experimental import attempt_load
    from utils.general import check_dataset
    from val import run
    from .engine.data import create_fixed_dataloader
    audit_dataset(cfg["data"])
    work = Path(cfg["workspace"])
    out = work / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "comparison.json").exists():
        saved = json.loads((out / "comparison.json").read_text())
        for row in saved:
            if hashlib.sha256(Path(row["weights"]).read_bytes()).hexdigest() != row["weights_sha256"]:
                raise ValueError("Test results belong to different weights; use a new workspace")
        return saved
    data = check_dataset(str(prepare_engine_dataset(cfg)))
    device = torch.device("cpu" if cfg["training"]["device"] == "cpu" else f"cuda:{cfg['training']['device']}")
    rows = []
    for stage in stages(cfg):
        weight = work / "train" / stage / "weights/best.pt"
        model = attempt_load(str(weight), device=device, inplace=True, fuse=True).eval()
        loader, _ = create_fixed_dataloader(data["test"], max(network_shape(cfg["input"]["height"], cfg["input"]["width"])),
            cfg["evaluation"].get("batch_size", 64), 32, workers=cfg["training"].get("workers", 8), prefix="test: ")
        metrics, maps, _ = run(data, model=model, dataloader=loader, imgsz=max(network_shape(cfg["input"]["height"], cfg["input"]["width"])),
            batch_size=cfg["evaluation"].get("batch_size", 64), half=device.type == "cuda", plots=False,
            conf_thres=0.001, iou_thres=0.6, max_det=300, save_dir=out, verbose=True)
        row = {"model": stage, "weights": str(weight), "weights_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "precision": float(metrics[0]), "recall": float(metrics[1]), "mAP50": float(metrics[2]), "mAP50_95": float(metrics[3]),
            "per_class_mAP50_95": {data["names"][i]: float(v) for i, v in enumerate(maps)}}
        rows.append(row)
        write_json(out / (stage + ".json"), row)
        emit("model_evaluated", **row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline = next((r for r in rows if r["model"] == stage_names(cfg)[1]), None)
    for row in rows:
        row["delta_mAP50_95"] = row["mAP50_95"] - baseline["mAP50_95"] if baseline else None
    write_json(out / "comparison.json", rows)
    keys = [k for k in rows[0] if k != "per_class_mAP50_95"]
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def validation_score(run_dir):
    with (Path(run_dir) / "results.csv").open() as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    best = max(rows, key=lambda r: .1 * r["metrics/mAP_0.5"] + .9 * r["metrics/mAP_0.5:0.95"])
    return best["metrics/mAP_0.5:0.95"]


def release(cfg, skip_export=False):
    import_engine(cfg)
    import torch
    from models.experimental import attempt_load
    from .deploy import export_onnx, infer, PaddedStudent
    import cv2
    work = Path(cfg["workspace"])
    dest = work / "deploy"
    dest.mkdir(parents=True, exist_ok=True)
    teacher_stage, baseline_stage, _ = stage_names(cfg)
    scores = {name: validation_score(work / "train" / name) for name in stages(cfg) if name != teacher_stage}
    distilled = [s for s in scores if s != baseline_stage]
    best_distilled = max(distilled, key=scores.get) if distilled else None
    recommended = max(scores, key=scores.get)
    outputs = {}
    for name in stages(cfg):
        source = work / "train" / name / "weights/best.pt"
        model = attempt_load(str(source), device=torch.device("cpu"), inplace=True, fuse=False).float().eval()
        target = dest / (name + ".pt")
        # A deploy file contains only the detector; no teacher, loss, optimizer or hooks.
        torch.save({"model": model, "input_hwc": [cfg["input"]["height"], cfg["input"]["width"], 3]}, target)
        if not skip_export:
            outputs[name] = export_onnx(target, dest / (name + ".onnx"), cfg["input"]["height"], cfg["input"]["width"])
    image_path = resolve_split(cfg["data"], "val")[0]
    image = cv2.imread(str(image_path))
    detector = attempt_load(str(dest / (recommended + ".pt")), device=torch.device("cpu"), inplace=True, fuse=True).float().eval()
    detections = infer(PaddedStudent(detector, cfg["input"]["height"], cfg["input"]["width"]), image,
        cfg["input"]["height"], cfg["input"]["width"], torch.device("cpu"))
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.imwrite(str(dest / "smoke.jpg"), image)
    result = {"selection_basis": "validation only", "validation_scores": scores,
              "best_distilled": best_distilled, "recommended_student": recommended, "exports": outputs,
              "external_hwc": [cfg["input"]["height"], cfg["input"]["width"], 3],
              "network_hw": list(network_shape(cfg["input"]["height"], cfg["input"]["width"])),
              "smoke_detections": len(detections)}
    write_json(dest / "release.json", result)
    return result

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .config import ENGINE_COMMIT, emit, load_recipe, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="distill-tool", description="YOLOv5 training, distillation and fair evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "doctor", "run", "train", "evaluate", "benchmark", "release", "status"):
        p = sub.add_parser(name)
        p.add_argument("recipe")
        if name in ("run", "train"):
            p.add_argument("--resume", action="store_true")
        if name == "train":
            p.add_argument("--stage", required=True)
        if name == "release":
            p.add_argument("--skip-export", action="store_true")
    for name in ("predict", "export"):
        p = sub.add_parser(name)
        p.add_argument("recipe")
        p.add_argument("--weights", required=True)
        p.add_argument("--output", required=True)
        if name == "predict":
            p.add_argument("--source", required=True)
    sub.add_parser("methods")
    p = sub.add_parser("ui", help="Open the general training interface")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--state", default="~/.yolo-distill")
    p.add_argument("--no-browser", action="store_true")
    p = sub.add_parser("bootstrap")
    p.add_argument("--engine", default="vendor/yolov5")
    p = sub.add_parser("prepare-data")
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--legacy-subset")
    args = parser.parse_args(argv)
    try:
        if args.command == "ui":
            from .web import serve
            serve(port=args.port, state=args.state, open_browser=not args.no_browser)
            return
        if args.command == "methods":
            from .methods import discover
            print(json.dumps({name: cls.__doc__ or cls.__name__ for name, cls in discover().items()}, indent=2))
            return
        if args.command == "bootstrap":
            path = Path(args.engine).resolve()
            if path.exists():
                raise FileExistsError("Engine directory exists; refusing to overwrite it")
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--branch", "v7.0", "--depth", "1", "https://github.com/ultralytics/yolov5.git", str(path)], check=True)
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if actual != ENGINE_COMMIT:
                raise ValueError(f"Unexpected YOLOv5 revision {actual}")
            emit("engine_ready", path=str(path), commit=actual)
            return
        if args.command == "prepare-data":
            from .data import prepare_hagrid
            prepare_hagrid(args.images, args.annotations, args.output, args.legacy_subset)
            return
        cfg = load_recipe(args.recipe)
        from . import pipeline as p
        if args.command == "plan":
            print(json.dumps(p.plan(cfg), indent=2, ensure_ascii=False))
        elif args.command == "doctor":
            from .data import audit_dataset
            from .engine.patch import generate_trainer
            counts = audit_dataset(cfg["data"], required_splits=("train", "val", "test") if cfg.get("workflow", {}).get("evaluate", True) else ("train", "val"))
            trainer = generate_trainer(cfg["engine"])
            p.import_engine(cfg)
            import torch
            emit("doctor", counts=counts, trainer=str(trainer), torch=torch.__version__, cuda=torch.cuda.is_available(),
                 external_hwc=[cfg["input"]["height"], cfg["input"]["width"], 3],
                 network_hw=list(p.network_shape(cfg["input"]["height"], cfg["input"]["width"])))
        elif args.command == "status":
            path = Path(cfg["workspace"]) / "status.json"
            print(path.read_text() if path.exists() else json.dumps({"status": "not_started"}))
        elif args.command == "train":
            p.run_stage(cfg, args.stage, args.resume)
        elif args.command == "evaluate":
            p.evaluate(cfg)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            benchmark(cfg)
        elif args.command == "release":
            p.release(cfg, args.skip_export)
        elif args.command in ("predict", "export"):
            p.import_engine(cfg)
            from .deploy import export_onnx, infer, PaddedStudent
            if args.command == "export":
                emit("export_complete", **export_onnx(args.weights, args.output, cfg["input"]["height"], cfg["input"]["width"]))
            else:
                import cv2
                import torch
                from models.experimental import attempt_load
                image = cv2.imread(args.source)
                if image is None:
                    raise ValueError("Cannot decode input image")
                model = attempt_load(args.weights, device=torch.device("cpu"), inplace=True, fuse=True).float().eval()
                wrapper = PaddedStudent(model, cfg["input"]["height"], cfg["input"]["width"])
                detections = infer(wrapper, image, cfg["input"]["height"], cfg["input"]["width"], torch.device("cpu"))
                write_json(args.output, {"boxes_xyxy_conf_class": detections.tolist(), "source": args.source,
                                       "coordinate_space": "original image pixels"})
                emit("prediction_complete", output=args.output, detections=len(detections))
        elif args.command == "run":
            from .data import audit_dataset
            from .benchmark import benchmark
            audit_dataset(cfg["data"], required_splits=("train", "val", "test") if cfg.get("workflow", {}).get("evaluate", True) else ("train", "val"))
            work = Path(cfg["workspace"])
            work.mkdir(parents=True, exist_ok=True)
            prov = p.provenance(cfg)
            saved_prov = work / "provenance.json"
            if saved_prov.exists():
                previous = json.loads(saved_prov.read_text())
                keys = ("tool_source_sha256", "recipe_sha256", "engine_train_sha256", "dataset_yaml_sha256", "pretrained_sha256", "model_checkpoints_sha256", "dataset")
                if any(previous.get(key) != prov.get(key) for key in keys):
                    raise ValueError("Code, recipe or data changed for this workspace; choose a fresh workspace")
            write_json(work / "resolved_recipe.json", cfg)
            write_json(saved_prov, prov)
            for stage in p.stages(cfg):
                partial = (work / "train" / stage).exists() and not (work / "train" / stage / "complete.json").exists()
                p.run_stage(cfg, stage, args.resume and partial)
            for phase, fn in (("test", p.evaluate), ("benchmark", benchmark), ("release", p.release)):
                option = {"test": "evaluate", "benchmark": "benchmark", "release": "export"}[phase]
                if not cfg.get("workflow", {}).get(option, True):
                    continue
                write_json(work / "status.json", {"status": "running", "phase": phase})
                fn(cfg)
            from .report import write_report
            write_report(cfg)
            write_json(work / "status.json", {"status": "complete", "report": str(work / "REPORT.md")})
            emit("suite_complete", report=str(work / "REPORT.md"))
    except Exception as exc:
        if "cfg" in locals() and args.command == "run":
            write_json(Path(cfg["workspace"]) / "status.json", {"status": "failed", "phase": locals().get("phase", "training"),
                "error": str(exc), "error_type": type(exc).__name__})
        emit("error", kind=type(exc).__name__, message=str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

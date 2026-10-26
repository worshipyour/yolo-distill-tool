"""Interleaved warm GPU measurements with explicit latency scopes."""
import json
import statistics
import time
from pathlib import Path

from .config import emit, write_json


def benchmark(cfg):
    from .pipeline import import_engine, stages
    import_engine(cfg)
    import cv2
    import numpy as np
    import torch
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    from .data import resolve_split
    from .deploy import PaddedStudent, infer, preprocess
    if cfg["training"]["device"] == "cpu":
        raise ValueError("Acceptance speed comparison requires a CUDA GPU; CPU inference remains available")
    device = torch.device(f"cuda:{cfg['training']['device']}")
    h, w = cfg["input"]["height"], cfg["input"]["width"]
    work = Path(cfg["workspace"])
    file = work / "evaluation/runtime.json"
    if file.exists():
        return json.loads(file.read_text())
    # Reuse validation images only. Test data is not needed for runtime tuning.
    image = cv2.imread(str(resolve_split(cfg["data"], "val")[0]))
    raw, _ = preprocess(image, h, w)
    x = raw.to(device).half() / 255
    inputs = {1: x, 128: x.repeat(128, 1, 1, 1).contiguous()}
    models = {}
    for name in stages(cfg):
        model = attempt_load(str(work / "train" / name / "weights/best.pt"), device=device, inplace=True, fuse=True).eval().half()
        models[name] = PaddedStudent(model, h, w)
    config = cfg["evaluation"]
    repeats, rounds, warmup = (int(config.get(k, v)) for k, v in
        (("benchmark_repeats", 100), ("benchmark_rounds", 3), ("benchmark_warmup", 50)))
    if min(repeats, rounds, warmup) <= 0:
        raise ValueError("Benchmark repeats, rounds and warmup must be positive")
    def sync():
        torch.cuda.synchronize(device)
    def measure(fn, count):
        times = []
        for _ in range(count):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1000)
        return times
    raw_results = {name: {"forward_b1": [], "forward_b128": [], "forward_nms_b1": [], "decoded_image_b1": []}
                   for name in models}
    with torch.inference_mode():
        for model in models.values():
            for _ in range(warmup):
                model(inputs[1])
                model(inputs[128])
        sync()
        names = list(models)
        for round_index in range(rounds):
            order = names[round_index:] + names[:round_index]
            for name in order:
                model = models[name]
                result = raw_results[name]
                result["forward_b1"] += measure(lambda: model(inputs[1]), repeats)
                result["forward_b128"] += measure(lambda: model(inputs[128]), repeats)
                result["forward_nms_b1"] += measure(lambda: non_max_suppression(model(inputs[1]), 0.25, 0.45, max_det=300), repeats)
                result["decoded_image_b1"] += measure(lambda: infer(model, image, h, w, device, half=True), repeats)
                emit("benchmark_block", model=name, round=round_index + 1)
    results = {}
    for name, timings in raw_results.items():
        results[name] = {key: {"median_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95))}
                         for key, values in timings.items()}
        results[name]["batch128_forward_images_per_second"] = 128000 / results[name]["forward_b128"]["median_ms"]
        results[name]["parameters"] = sum(p.numel() for p in models[name].parameters())
    report = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
              "external_hwc": [h, w, 3], "dtype": "fp16", "rounds": rounds, "repeats_per_round": repeats,
              "warmup_per_model_per_shape": warmup, "scopes": {
                  "forward_b1": "preloaded normalized RGB tensor -> stride padding + detector forward; CUDA synchronized wall time",
                  "forward_b128": "same, batch 128; no NMS or preprocessing",
                  "forward_nms_b1": "preloaded tensor -> internal padding + forward + NMS",
                  "decoded_image_b1": "decoded CPU BGR image -> letterbox + normalize + host to device + internal padding + forward + NMS + box rescale; excludes disk decode"},
              "results": results, "raw_ms": raw_results}
    write_json(file, report)
    emit("benchmark_complete", path=str(file))
    return report

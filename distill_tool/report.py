import json
from pathlib import Path

from .config import network_shape


def write_report(cfg):
    from .pipeline import stages, stage_names, validation_score
    work = Path(cfg["workspace"])
    def read(relative, default):
        path = work / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    rows = read("evaluation/comparison.json", [])
    runtime = read("evaluation/runtime.json", {})
    release = read("deploy/release.json", {})
    baseline = next((r for r in rows if r["model"] == stage_names(cfg)[1]), None)
    h, w = cfg["input"]["height"], cfg["input"]["width"]
    nh, nw = network_shape(h, w)
    lines = ["# YOLOv5 训练与蒸馏报告", "", f"输入：高 {h} × 宽 {w} × RGB；模型内部补边至 {nh} × {nw}。", ""]
    for role, spec in cfg.get("models", {}).items():
        if role == "teacher" and not cfg["methods"]:
            continue
        structure = "从已有教师权重读取结构" if role == "teacher" and not cfg.get("workflow", {}).get("train_teacher", True) else spec['architecture']
        lines.append(f"- {role}: {structure}；初始化权重：{spec.get('weights') or '官方预训练权重'}。")
    lines += ["", "## 训练", "", "|阶段|轮数|验证集 best mAP50-95|", "|---|---:|---:|"]
    for name in stages(cfg):
        score = validation_score(work / "train" / name)
        lines.append(f"|{name}|{cfg['training']['epochs']}|{score:.6f}|")
    if rows:
        lines += ["", "## 独立测试", "", "模型选择依据验证集，测试指标不参与模型选择。", "",
                  "|模型|mAP50|mAP50-95|相对学生基线|单张前向+NMS中位数/ms|", "|---|---:|---:|---:|---:|"]
        for row in rows:
            name = row["model"]
            latency = runtime.get("results", {}).get(name, {}).get("forward_nms_b1", {}).get("median_ms")
            delta = f"{row['mAP50_95'] - baseline['mAP50_95']:+.6f}" if baseline else "未运行基线"
            lines.append(f"|{name}|{row['mAP50']:.6f}|{row['mAP50_95']:.6f}|{delta}|{latency:.3f}|" if latency is not None else
                         f"|{name}|{row['mAP50']:.6f}|{row['mAP50_95']:.6f}|{delta}|未测速|")
    else:
        lines += ["", "本任务未执行独立测试；验证分数不能作为独立测试结果。"]
    if runtime:
        lines += ["", f"测速：{runtime['gpu']}，FP16，{runtime['rounds']}轮，每轮{runtime['repeats_per_round']}次，CUDA同步计时。",
                  "单图前向+NMS不含CPU预处理，其他计时边界见 evaluation/runtime.json。"]
    if release:
        lines += ["", f"按验证集选择的部署学生：{release['recommended_student']}。", "学生部署不包含教师和蒸馏模块。", ""]
        for name, export in release["exports"].items():
            lines.append(f"- {name}: ONNX Runtime核对={export['onnxruntime_verified']}；最大绝对差={export['max_abs_error']:.6g}。")
    lines += ["", "单次训练结果可能受随机性影响；蒸馏方法接入成功不保证精度提升。"]
    path = work / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

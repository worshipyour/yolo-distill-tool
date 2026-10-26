"""Durable, detached training jobs for the local UI. No shell command assembly."""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil
import yaml

from .catalog import MODELS, method_catalog
from .config import load_recipe, write_json


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tail(path, limit=16000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def build_recipe(values, workspace, engine):
    """Map the general form to the same recipe consumed by the CLI."""
    def integer(name, default, low, high):
        value = values.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} 必须是 {low}–{high} 的整数")
        return value
    data = Path(str(values.get("data", ""))).expanduser()
    if not str(values.get("data", "")).strip() or not data.is_file():
        raise ValueError("请选择执行机器上存在的数据集 YAML 文件")
    data = data.resolve()
    dataset = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not all(dataset.get(k) for k in ("train", "val", "names")):
        raise ValueError("数据集 YAML 必须包含 train、val 和 names")
    method = str(values.get("method", "response_kd"))
    if method != "none" and method not in method_catalog():
        raise ValueError("未安装该蒸馏方法插件")
    device = str(values.get("device", "0")).strip()
    if device != "cpu" and not re.fullmatch(r"\d+", device):
        raise ValueError("请选择单张 GPU 编号，或 cpu")
    models = {}
    for role, default in (("teacher", "yolov5m"), ("student", "yolov5n")):
        architecture = values.get(role, default)
        if architecture not in MODELS:
            raise ValueError("当前引擎支持 YOLOv5 n/s/m/l/x")
        weights = str(values.get(role + "_weights", "")).strip()
        if weights and not Path(weights).expanduser().is_file():
            raise ValueError(f"{role} 权重文件不存在")
        models[role] = {"architecture": architecture, "weights": str(Path(weights).expanduser().resolve()) if weights else ""}
    train_teacher = values.get("train_teacher", True)
    if type(train_teacher) is not bool:
        raise ValueError("train_teacher 必须为布尔值")
    if method != "none" and not train_teacher and not models["teacher"]["weights"]:
        raise ValueError("使用已有教师时，请选择已在目标数据集上训练好的教师权重")
    params = values.get("parameters", method_catalog().get(method, {}).get("parameters", {}))
    if not isinstance(params, dict) or any(not isinstance(k, str) or k.startswith("_") for k in params):
        raise ValueError("方法参数必须是名称到参数值的映射")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("方法参数不能包含 NaN 或无穷大")
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    finite(params)
    if "weight" in params and (not isinstance(params["weight"], (int, float)) or params["weight"] < 0):
        raise ValueError("蒸馏损失权重必须非负")
    flow = {key: values.get(key, False) for key in ("evaluate", "benchmark", "export")}
    flow.update(train_teacher=train_teacher and method != "none", baseline=values.get("baseline", True) or method == "none")
    if any(type(v) is not bool for v in flow.values()):
        raise ValueError("工作流选项必须为布尔值")
    if flow["evaluate"] and not dataset.get("test"):
        raise ValueError("启用独立测试需要数据集 YAML 提供 test；仅训练只需要 train/val")
    if flow["benchmark"] and device == "cpu":
        raise ValueError("当前统一测速需要 CUDA GPU")
    return {
        "version": 1, "workspace": str(Path(workspace).resolve()), "engine": str(Path(engine).expanduser().resolve()), "data": str(data),
        "models": models,
        "input": {"height": integer("height", 640, 32, 4096), "width": integer("width", 640, 32, 4096), "channels": 3},
        "training": {"epochs": integer("epochs", 50, 1, 10000), "batch_size": integer("batch_size", 16, 1, 4096),
                     "device": device, "workers": integer("workers", 4, 0, 128), "seed": integer("seed", 42, 0, 2147483647)},
        "methods": {} if method == "none" else {method: params}, "workflow": flow,
        "evaluation": {"batch_size": integer("eval_batch", 16, 1, 4096), "benchmark_rounds": 3, "benchmark_repeats": 100, "benchmark_warmup": 50},
    }


class JobManager:
    def __init__(self, state):
        self.state = Path(state).expanduser().resolve()
        self.root = self.state / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = Path(os.environ.get("YOLO_DISTILL_ENGINE", str(self.state / "engines" / "yolov5-v7"))).expanduser().resolve()
        self.lock = threading.RLock()
        self.processes = {}

    def directory(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{16}", job_id):
            raise ValueError("无效任务 ID")
        path = self.root / job_id
        if not (path / "job.json").is_file():
            raise FileNotFoundError("任务不存在")
        return path

    def _process(self, directory):
        info = read_json(directory / "process.json", {})
        try:
            process = psutil.Process(info["pid"])
            args = process.cmdline()
            if abs(process.create_time() - info["created"]) < .01 and "distill_tool.jobs" in args and str(directory) in args and process.status() != psutil.STATUS_ZOMBIE:
                return process
        except (KeyError, psutil.Error):
            pass
        return None

    def list(self):
        records = [self.describe(path.name) for path in self.root.iterdir() if path.is_dir() and (path / "job.json").is_file()]
        return sorted(records, key=lambda row: row["created"], reverse=True)

    def describe(self, job_id, logs=False):
        directory = self.directory(job_id)
        owned = self.processes.get(job_id)
        if owned is not None:
            owned.poll()  # Reap completed detached children while the service stays up.
        row = read_json(directory / "job.json", {})
        row["alive"] = self._process(directory) is not None
        if row.get("status") in ("starting", "checking", "running") and not row["alive"]:
            if not (owned and owned.poll() is None) and time.time() - row.get("updated", row["created"]) > 5:
                row["status"] = "interrupted"
        cfg = read_json(directory / "resolved.json", {})
        work = Path(cfg.get("workspace", str(directory / "output")))
        row["workspace"] = str(work)
        row["pipeline"] = read_json(work / "status.json", {})
        stage = row["pipeline"].get("stage")
        row["progress"] = read_json(work / "train" / stage / "progress.json", {}) if stage else {}
        row["epochs"] = cfg.get("training", {}).get("epochs")
        row["report_ready"] = (work / "REPORT.md").is_file()
        if logs:
            row["log"] = tail(directory / "console.log")
            row["trainer_log"] = tail(work / "logs" / (stage + ".log")) if stage else ""
        return row

    def _spawn(self, directory, resume=False):
        args = [sys.executable, "-m", "distill_tool.jobs", "worker", str(directory)]
        if resume:
            args.append("--resume")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent.parent), env.get("PYTHONPATH", "")])
        options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        with (directory / "console.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **options)
        self.processes[directory.name] = process
        write_json(directory / "process.json", {"pid": process.pid, "created": psutil.Process(process.pid).create_time()})

    def start(self, values):
        with self.lock:
            if any(row["alive"] or row["status"] == "starting" for row in self.list()):
                raise ValueError("已有任务运行中，请完成或停止后再启动下一项")
            job_id = uuid.uuid4().hex[:16]
            name = str(values.get("name", "")).strip()[:80] or "训练任务"
            output = str(values.get("output", "")).strip() or str(Path.cwd() / "runs")
            workspace = Path(output).expanduser().resolve() / (time.strftime("%Y%m%d-%H%M%S") + "-" + job_id[:6])
            cfg = build_recipe(values, workspace, values.get("engine") or self.engine)
            directory = self.root / job_id
            directory.mkdir()
            file = directory / "recipe.yaml"
            file.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            load_recipe(file)
            write_json(directory / "resolved.json", cfg)
            write_json(directory / "job.json", {"id": job_id, "name": name, "status": "starting", "created": time.time(), "updated": time.time(), "error": None})
            self._spawn(directory)
            return self.describe(job_id)

    def resume(self, job_id):
        with self.lock:
            directory = self.directory(job_id)
            if any(row["alive"] or row["status"] == "starting" for row in self.list()):
                raise ValueError("已有任务正在运行")
            row = read_json(directory / "job.json")
            if row["status"] == "complete":
                raise ValueError("该任务已完成")
            row.update(status="starting", updated=time.time(), error=None)
            write_json(directory / "job.json", row)
            self._spawn(directory, resume=True)
            return self.describe(job_id)

    def stop(self, job_id):
        with self.lock:
            directory = self.directory(job_id)
            process = self._process(directory)
            if process is None:
                raise ValueError("任务未运行，或无法核对任务进程身份")
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=True)
            else:
                if os.getpgid(process.pid) != process.pid:
                    raise ValueError("任务进程组不匹配，未执行停止")
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=4)
                except psutil.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            row = read_json(directory / "job.json")
            row.update(status="stopped", updated=time.time(), error="用户已停止任务，可从最近一个完整 checkpoint 恢复")
            write_json(directory / "job.json", row)
            return self.describe(job_id)


def worker(directory, resume=False):
    directory = Path(directory).resolve()
    row = read_json(directory / "job.json")
    def update(status, error=None):
        row.update(status=status, updated=time.time(), error=error)
        write_json(directory / "job.json", row)
    try:
        update("checking")
        cfg = load_recipe(directory / "recipe.yaml")
        print("检查 Python / PyTorch / 训练设备…", flush=True)
        probe = "import torch, torchvision; d=" + repr(str(cfg["training"]["device"])) + "; assert d == 'cpu' or (torch.cuda.is_available() and int(d) < torch.cuda.device_count()), 'Selected CUDA device is not available'; print('torch', torch.__version__, 'device', d)"
        subprocess.run([sys.executable, "-c", probe], check=True)
        engine = Path(cfg["engine"])
        if not engine.exists():
            print("首次运行：下载固定版本 YOLOv5 引擎…", flush=True)
            subprocess.run([sys.executable, "-m", "distill_tool.cli", "bootstrap", "--engine", str(engine)], check=True)
        if not (engine / "train.py").is_file():
            raise ValueError("引擎目录缺少 train.py，请选择固定 v7.0 引擎目录或新的空路径")
        print("启动真实训练流程，完整日志和 checkpoint 会保存到：" + cfg["workspace"], flush=True)
        update("running")
        args = [sys.executable, "-m", "distill_tool.cli", "run", str(directory / "recipe.yaml")]
        if resume:
            args.append("--resume")
        subprocess.run(args, check=True)
        update("complete")
    except Exception as exc:
        print("任务失败：" + str(exc), flush=True)
        update("failed", str(exc) + "。请查看日志；依赖未安装时先运行安装脚本。")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "worker":
        raise SystemExit("Internal worker command")
    raise SystemExit(worker(sys.argv[2], resume="--resume" in sys.argv[3:]))

"""Public UI -> durable job -> CLI boundaries, without a GPU dependency."""
import json
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import psutil
import pytest
import yaml

from distill_tool.config import load_recipe, write_json
from distill_tool.jobs import JobManager, build_recipe, worker
from distill_tool.web import make_server


@pytest.fixture
def dataset(tmp_path):
    file = tmp_path / "dataset.yaml"
    file.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": ["object"]}), encoding="utf-8")
    return file


def test_selected_architectures_and_external_teacher_reach_real_trainer_arguments(tmp_path, dataset):
    from distill_tool.pipeline import commands, stages
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"checkpoint")
    cfg = build_recipe({"data": str(dataset), "teacher": "yolov5l", "student": "yolov5s", "teacher_weights": str(teacher),
                        "train_teacher": False, "baseline": False, "method": "fgd", "height": 320, "width": 328}, tmp_path / "run", tmp_path / "engine")
    assert stages(cfg) == ["student_fgd"]
    args = commands(cfg, "student_fgd")
    assert args[args.index("--teacher-weights") + 1] == str(teacher)
    assert Path(args[args.index("--cfg") + 1]).name == "yolov5s.yaml"
    assert Path(args[args.index("--weights") + 1]).name == "yolov5s.pt"
    assert args[args.index("--kd-method") + 1] == "fgd"


def test_standard_training_does_not_create_teacher_or_require_test_split(tmp_path, dataset):
    from distill_tool.pipeline import stages
    cfg = build_recipe({"data": str(dataset), "method": "none", "baseline": False}, tmp_path / "run", tmp_path / "engine")
    file = tmp_path / "recipe.yaml"
    file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = load_recipe(file)
    assert stages(loaded) == ["student_baseline"]
    assert loaded["methods"] == {}


def test_installed_method_plugin_is_listed_and_keeps_its_identifier(tmp_path, dataset, monkeypatch):
    from types import SimpleNamespace
    from distill_tool import catalog
    from distill_tool.pipeline import commands, stages
    monkeypatch.setattr(catalog, "entry_points", lambda **_: [SimpleNamespace(name="n_custom"), SimpleNamespace(name="../bad")])
    assert "n_custom" in catalog.method_catalog() and "../bad" not in catalog.method_catalog()
    cfg = build_recipe({"data": str(dataset), "method": "n_custom", "parameters": {"weight": 1.0}}, tmp_path / "run", tmp_path / "engine")
    args = commands(cfg, stages(cfg)[-1])
    assert args[args.index("--kd-method") + 1] == "n_custom"


@pytest.mark.parametrize("extra,match", [({"train_teacher": False}, "教师权重"), ({"evaluate": True}, "test"),
    ({"device": "cpu", "benchmark": True}, "CUDA"), ({"parameters": {"weight": float("nan")}}, "NaN"),
    ({"method": "not_installed"}, "插件"), ({"batch_size": -1}, "batch_size")])
def test_unrunnable_jobs_are_rejected_before_spawning(tmp_path, dataset, extra, match):
    with pytest.raises(ValueError, match=match):
        build_recipe({"data": str(dataset), **extra}, tmp_path / "run", tmp_path / "engine")


def test_one_click_api_persists_and_dispatches_job_and_rejects_cross_origin(tmp_path, dataset, monkeypatch):
    dispatched = []
    monkeypatch.setattr(JobManager, "_spawn", lambda self, directory, resume=False: dispatched.append((directory, resume)))
    server = make_server(0, tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        html = urllib.request.urlopen(base).read().decode()
        token = re.search(r'name="distill-token" content="([^"]+)"', html).group(1)
        values = {"data": str(dataset), "method": "fgd", "student": "yolov5s", "output": str(tmp_path / "runs")}
        def start(origin):
            return urllib.request.urlopen(urllib.request.Request(base + "/api/start", data=json.dumps(values).encode(),
                headers={"X-Distill-Token": token, "Content-Type": "application/json", "Origin": origin}))
        with pytest.raises(urllib.error.HTTPError) as error:
            start("https://untrusted.invalid")
        assert error.value.code == 403 and not dispatched
        with start(base) as response:
            job = json.load(response)
        assert len(dispatched) == 1
        recipe = load_recipe(dispatched[0][0] / "recipe.yaml")
        assert recipe["models"]["student"]["architecture"] == "yolov5s"
        assert list(recipe["methods"]) == ["fgd"]
        assert job["status"] == "starting"
        with pytest.raises(urllib.error.HTTPError) as duplicate:
            start(base)
        assert duplicate.value.code == 400 and len(dispatched) == 1
        with pytest.raises(urllib.error.HTTPError) as missing_token:
            urllib.request.urlopen(base + "/api/jobs")
        assert missing_token.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_stop_refuses_a_pid_that_does_not_belong_to_worker(tmp_path):
    manager = JobManager(tmp_path)
    directory = manager.root / ("a" * 16)
    directory.mkdir()
    write_json(directory / "job.json", {"id": directory.name, "status": "running", "created": 1})
    process = psutil.Process()
    write_json(directory / "process.json", {"pid": process.pid, "created": process.create_time()})
    with pytest.raises(ValueError, match="身份"):
        manager.stop(directory.name)
    assert process.is_running()


def test_worker_records_cli_failure_and_resumes_using_saved_recipe(tmp_path, dataset, monkeypatch):
    directory = tmp_path / "job"
    directory.mkdir()
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "train.py").write_text("# fixture")
    cfg = build_recipe({"data": str(dataset), "method": "none", "device": "cpu"}, tmp_path / "run", engine)
    (directory / "recipe.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_json(directory / "job.json", {"id": "fixture", "name": "fixture", "created": 1})
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if "run" in args:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", run)
    assert worker(directory, resume=True) == 1
    assert calls[-1] == [sys.executable, "-m", "distill_tool.cli", "run", str(directory / "recipe.yaml"), "--resume"]
    assert json.loads((directory / "job.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_generic_dataset_lists_and_nested_folders_have_same_geometry_inputs(tmp_path):
    from distill_tool.data import audit_dataset
    from distill_tool.pipeline import prepare_engine_dataset
    for split in ("train", "val"):
        image = tmp_path / "images" / split / "nested" / "same_name.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(split.encode())
        label = tmp_path / "labels" / split / "nested" / "same_name.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n")
    data = tmp_path / "dataset.yaml"
    data.write_text(yaml.safe_dump({"train": ["images/train"], "val": "images/val", "names": ["object"]}), encoding="utf-8")
    assert audit_dataset(data, required_splits=("train", "val")) == {"train": 1, "val": 1}
    cfg = {"data": str(data), "workspace": str(tmp_path / "run"), "workflow": {"evaluate": False}}
    normalized = yaml.safe_load(prepare_engine_dataset(cfg).read_text())
    train_image = Path(normalized["train"]).read_text().strip()
    assert Path(train_image).is_absolute() and "images" in Path(train_image).parts
    assert "test" not in normalized

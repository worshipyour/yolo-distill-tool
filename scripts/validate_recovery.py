"""GPU acceptance: train two short epochs, replay epoch two from epoch-one state.

Use a small mechanical-check recipe; this is not an accuracy experiment.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_tool.config import load_recipe, write_json
from distill_tool.engine.patch import generate_trainer
from distill_tool.pipeline import commands, environment, import_engine


def verify(workspace):
    import torch
    saved = torch.load(workspace / "train/student_n_fgd/weights/resume.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1
    assert saved["distillation"]["method"] == "fgd" and saved["distillation"]["state_dict"]
    assert len(saved["optimizer"]["param_groups"]) == 4
    assert '"event": "distillation_restored"' in (workspace / "resume.log").read_text()
    result = {"passed": True, "epoch": saved["epoch"] + 1, "method": "fgd",
              "method_state_tensors": len(saved["distillation"]["state_dict"]),
              "optimizer_param_groups": len(saved["optimizer"]["param_groups"]), "scope": "recovery_test_only"}
    write_json(workspace / "RESULT.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    import_engine(cfg)
    original_work = Path(cfg["workspace"])
    workspace = Path(args.workspace).resolve()
    if args.verify_only:
        verify(workspace)
        return
    if workspace.exists():
        raise FileExistsError("Use a fresh recovery-check workspace")
    cfg["workspace"] = str(workspace)
    cfg["training"]["epochs"] = 2
    teacher = workspace / "train/teacher_m/weights/best.pt"
    teacher.parent.mkdir(parents=True)
    shutil.copy2(original_work / "train/teacher_m/weights/best.pt", teacher)
    trainer = generate_trainer(cfg["engine"])
    stage = "student_n_fgd"
    argv = commands(cfg, stage) + ["--save-period", "1"]
    env = environment(cfg)
    with (workspace / "initial.log").open("w") as log:
        subprocess.run(argv, cwd=cfg["engine"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = workspace / "train" / stage / "weights/epoch0.pt"
    with (workspace / "resume.log").open("w") as log:
        subprocess.run([sys.executable, str(trainer), "--resume", str(checkpoint)], cwd=cfg["engine"], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    verify(workspace)


if __name__ == "__main__":
    main()

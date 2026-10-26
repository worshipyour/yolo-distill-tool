"""Check real YOLOv5 teacher/student gradient and serialization boundaries on CPU."""
import argparse
import copy
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_tool.config import load_recipe, write_json
from distill_tool.pipeline import import_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    import_engine(cfg)
    import torch
    import yaml
    from models.yolo import Model
    from distill_tool.engine.runtime import Distiller
    torch.set_num_threads(2)
    data = yaml.safe_load(Path(cfg["data"]).read_text())
    names = dict(enumerate(data["names"])) if isinstance(data["names"], list) else data["names"]
    results = {}
    for method in ("response_kd", "fgd"):
        student = Model(str(Path(cfg["engine"]) / "models/yolov5n.yaml"), ch=3, nc=len(names)).float().train()
        student.names = names
        initial_keys = set(student.state_dict())
        options = dict(cfg["methods"][method])
        options.pop("weight")
        controller = Distiller(student, method, str(Path(cfg["workspace"]) / "train/teacher_m/weights/best.pt"), options, torch.device("cpu"))
        imgs = torch.rand(2, 3, 320, 352)
        targets = torch.tensor([[0., 0., .5, .5, .2, .2], [1., 1., .4, .4, .15, .2]])
        _, loss = controller.forward(student, imgs, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
        assert all(not p.requires_grad and p.grad is None for p in controller.teacher.parameters())
        assert all(not module._forward_pre_hooks for module in student.modules())
        assert set(student.state_dict()) == initial_keys
        if method == "fgd":
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in controller.parameters())
        torch.save({"model": copy.deepcopy(student).half()}, io.BytesIO())
        results[method] = {"passed": True, "loss": float(loss.detach()), "teacher_gradient": False,
                           "student_gradient": True, "student_state_keys_unchanged": True,
                           "student_serializable": True, "training_hooks_remaining": 0}
    write_json(args.output, results)


if __name__ == "__main__":
    main()

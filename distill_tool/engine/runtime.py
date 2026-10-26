"""Scoped feature hooks keep training-only state out of student checkpoints."""
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path

import torch

from distill_tool.config import emit, write_json
from distill_tool.methods import build
from distill_tool.methods.base import KDInputs


def unwrap(model):
    return model.module if hasattr(model, "module") else model


class FeatureTap(AbstractContextManager):
    def __init__(self, model):
        self.head = unwrap(model).model[-1]
        self.features = []
        self.handle = None

    def _capture(self, module, inputs):
        # Detect mutates its input list, so capture a separate list of tensor refs.
        self.features = list(inputs[0])

    def __enter__(self):
        self.handle = self.head.register_forward_pre_hook(self._capture)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False


class Distiller:
    def __init__(self, student, method, teacher_weights, config, device):
        self.name = method
        self.config = config
        self.device = device
        self.teacher = self.method = None
        self.total = 0.0
        self.batches = 0
        self.components = {}
        self._shape_logged = False
        if method == "none":
            return
        from models.experimental import attempt_load
        if not teacher_weights:
            raise ValueError("A distillation method requires teacher weights")
        self.teacher = attempt_load(teacher_weights, device=device, inplace=True, fuse=False).float().eval()
        self.teacher.requires_grad_(False)
        s_head, t_head = unwrap(student).model[-1], self.teacher.model[-1]
        def names(model):
            value = model.names
            return list(value.values()) if isinstance(value, dict) else list(value)
        if s_head.nc != t_head.nc or names(unwrap(student)) != names(self.teacher):
            raise ValueError("Teacher and student class count/order must match")
        if s_head.nl != t_head.nl or not torch.equal(s_head.stride, t_head.stride):
            raise ValueError("Teacher/student detection strides must match")
        if method == "response_kd" and not torch.allclose(s_head.anchors, t_head.anchors, atol=1e-4):
            raise ValueError("Response KD requires identical anchor geometry; use --noautoanchor for the suite")
        sc, tc = [m.in_channels for m in s_head.m], [m.in_channels for m in t_head.m]
        self.method = build(method, sc, tc, **config).to(device)

    def parameters(self):
        return list(self.method.parameters()) if self.method is not None else []

    def forward(self, model, imgs, targets):
        if not self._shape_logged:
            emit("actual_input", external_hw=os.environ["DISTILL_INPUT_HW"], network_nchw=list(imgs.shape), method=self.name)
            self._shape_logged = True
        if self.teacher is None:
            return model(imgs), imgs.new_zeros(())
        use_features = self.method.requires_features
        if use_features:
            with FeatureTap(model) as student_tap:
                pred = model(imgs)
            with torch.no_grad(), FeatureTap(self.teacher) as teacher_tap:
                teacher_out = self.teacher(imgs)
            sf, tf = student_tap.features, teacher_tap.features
        else:
            pred = model(imgs)
            with torch.no_grad():
                teacher_out = self.teacher(imgs)
            sf, tf = [], []
        teacher_pred = teacher_out[1] if isinstance(teacher_out, tuple) else teacher_out
        # FGD weighted sums are evaluated in float32 even during AMP detector training.
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            loss, terms = self.method(KDInputs(pred, teacher_pred, sf, tf, targets, tuple(imgs.shape[-2:])))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {self.name} loss")
        self.total += float(loss.detach())
        self.batches += 1
        for key, value in terms.items():
            self.components[key] = self.components.get(key, 0.0) + float(value)
        return pred, loss

    def checkpoint(self):
        return {"method": self.name, "config": self.config,
                "state_dict": self.method.state_dict() if self.method is not None else {}}

    def restore(self, checkpoint):
        saved = checkpoint.get("distillation")
        if saved is None or saved["method"] != self.name or saved["config"] != self.config:
            raise ValueError("Resume checkpoint distillation method/config does not match the recipe")
        if self.method is not None:
            self.method.load_state_dict(saved["state_dict"], strict=True)
        emit("distillation_restored", method=self.name, state_tensors=len(saved["state_dict"]))

    def log_epoch(self, save_dir, epoch, results):
        row = {"epoch": epoch + 1, "method": self.name, "kd_loss": self.total / max(self.batches, 1),
               "kd_terms": {k: v / max(self.batches, 1) for k, v in self.components.items()},
               "validation": [float(v) for v in results], "batches": self.batches}
        with (Path(save_dir) / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        write_json(Path(save_dir) / "progress.json", row)
        emit("epoch_complete", **row)
        self.total, self.batches, self.components = 0.0, 0, {}


def make_distiller(model, opt, device):
    config = json.loads(opt.kd_config)
    return Distiller(model, opt.kd_method, opt.teacher_weights, config, device)

import copy
import importlib.util
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from distill_tool.config import load_recipe, network_shape
from distill_tool.deploy import PaddedStudent, infer
from distill_tool.engine.patch import generate_trainer
from distill_tool.engine.runtime import FeatureTap


class MutableHead(nn.Module):
    def forward(self, inputs):
        inputs[0] = inputs[0].unsqueeze(-1)
        return inputs[0]


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([MutableHead()])

    def forward(self, x):
        return self.model[-1]([x])


def test_scoped_hook_copies_detect_input_list_and_is_removed_on_error():
    model = TinyDetector()
    with FeatureTap(model) as tap:
        model(torch.zeros(1, 3, 5, 7))
    assert tap.features[0].ndim == 4
    assert not model.model[-1]._forward_pre_hooks
    with pytest.raises(RuntimeError):
        with FeatureTap(model):
            raise RuntimeError("interrupted forward")
    assert not model.model[-1]._forward_pre_hooks
    torch.save(copy.deepcopy(model), io.BytesIO())


class ShapeProbe(nn.Module):
    def forward(self, x):
        assert tuple(x.shape) == (1, 3, 320, 352)
        torch.testing.assert_close(x[..., 328:], torch.full_like(x[..., 328:], 114 / 255))
        return x, None


def test_external_320x328_is_padded_inside_deployment_graph():
    wrapper = PaddedStudent(ShapeProbe())
    assert not wrapper.training and not wrapper.model.training
    out = wrapper(torch.zeros(1, 3, 320, 328))
    assert out.shape == (1, 3, 320, 352)
    assert network_shape(320, 328) == (320, 352)
    with pytest.raises(ValueError, match="Expected"):
        wrapper(torch.zeros(1, 3, 320, 320))


class DetectionProbe(nn.Module):
    def forward(self, x):
        # One confident prediction ensures scale_boxes performs its in-place update.
        return x.new_tensor([[[100., 100., 50., 50., .9, .9]]])


def test_predict_scales_nonempty_inference_detections():
    import numpy as np
    image = np.zeros((640, 656, 3), dtype=np.uint8)
    detections = infer(PaddedStudent(DetectionProbe()), image, 320, 328, torch.device('cpu'))
    assert detections.shape == (1, 6)
    torch.testing.assert_close(detections[0, :4], torch.tensor([150., 150., 250., 250.]))


def test_pinned_trainer_generation_compiles_and_rejects_changed_upstream(tmp_path):
    source = Path(__file__).resolve().parents[1] / "vendor/yolov5/train.py"
    if not source.exists():
        pytest.skip("bootstrap engine first")
    (tmp_path / "train.py").write_bytes(source.read_bytes())
    generated = generate_trainer(tmp_path)
    compile(generated.read_text(encoding="utf-8"), str(generated), "exec")
    (tmp_path / "train.py").write_text("unreviewed engine")
    with pytest.raises(ValueError, match="pinned"):
        generate_trainer(tmp_path)


def test_fixed_shape_label_scaling_preserves_pixel_box(monkeypatch):
    engine = Path(__file__).resolve().parents[1] / "vendor/yolov5"
    if not engine.exists():
        pytest.skip("bootstrap engine first")
    monkeypatch.syspath_prepend(str(engine))
    from distill_tool.engine.data import pad_canvas
    import numpy as np
    image = np.zeros((320, 328, 3), dtype=np.uint8)
    label = np.array([[2, .5, .5, .2, .3]], dtype=np.float32)
    padded, changed = pad_canvas(image, label, (320, 328))
    assert padded.shape == (320, 352, 3)
    np.testing.assert_allclose(changed[:, [1, 3]] * 352, label[:, [1, 3]] * 328, atol=1e-5)
    np.testing.assert_allclose(changed[:, [2, 4]], label[:, [2, 4]])
    assert (padded[:, 328:] == 114).all()


def test_resume_rejects_recipe_changes_before_launching_process(tmp_path):
    import json
    from distill_tool.pipeline import run_stage
    recipe = Path(__file__).resolve().parents[1] / "recipes/hagrid_320x328.yaml"
    cfg = load_recipe(recipe)
    cfg["workspace"] = str(tmp_path)
    marker = tmp_path / "stage_manifests/teacher_m.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"recipe_sha256": "different"}))
    with pytest.raises(ValueError, match="recipe changed"):
        run_stage(cfg, "teacher_m", resume=True)

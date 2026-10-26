import io
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from distill_tool.methods.base import KDInputs
from distill_tool.methods.fgd import FGDLevel, FGD, foreground_masks
from distill_tool.methods.response import ResponseKD


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_masks_exactly_match_previous_rasterization(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(123)
    targets = torch.rand(1500, 6)
    targets[:, 0] = torch.arange(len(targets)) % 12
    targets[:, 2:4] = targets[:, 2:4] * 1.5 - .25
    targets[0, 4] = 0
    targets[1, 5] = -.1
    targets[2, 2:] = torch.tensor([.5, .5, 1., 1.])
    for height, width in ((5, 7), (40, 44)):
        expected = torch.zeros(12, height, width)
        for box in targets.tolist():
            image = int(box[0])
            cx, cy, bw, bh = box[2:]
            if bw <= 0 or bh <= 0:
                continue
            x0 = max(0, min(width - 1, int((cx - bw / 2) * width // 1)))
            y0 = max(0, min(height - 1, int((cy - bh / 2) * height // 1)))
            x1 = max(0, min(width - 1, math.ceil((cx + bw / 2) * width)))
            y1 = max(0, min(height - 1, math.ceil((cy + bh / 2) * height)))
            region = expected[image, y0:y1 + 1, x0:x1 + 1]
            region.copy_(torch.maximum(region, region.new_tensor(1. / ((x1+1-x0)*(y1+1-y0)))))
        expected_bg = (expected == 0).float()
        expected_bg /= expected_bg.sum((1, 2), keepdim=True).clamp_min(1)
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            fg, bg = foreground_masks(targets.to(device), 12, height, width, torch.device(device))
        finally:
            torch.use_deterministic_algorithms(previous)
        torch.testing.assert_close(fg.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(bg.cpu(), expected_bg, rtol=0, atol=0)


def original_fgd():
    source = (Path(__file__).parent / "reference/fgd_original.py").read_text()
    source = source.replace("from mmcv.cnn import constant_init, kaiming_init", "")
    source = source.replace("from ..builder import DISTILL_LOSSES", "")
    def constant_init(module, val):
        nn.init.constant_(module.weight, val)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    def kaiming_init(module, mode):
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity="relu")
        nn.init.zeros_(module.bias)
    scope = dict(DISTILL_LOSSES=SimpleNamespace(register_module=lambda: lambda cls: cls),
                 constant_init=constant_init, kaiming_init=kaiming_init)
    exec(compile(source, "fgd_original.py", "exec"), scope)
    return scope["FeatureLoss"]


def test_fgd_matches_original_loss_and_feature_gradients():
    torch.manual_seed(5)
    official = original_fgd()(4, 8, "reference")
    standalone = FGDLevel(4, 8)
    standalone.load_state_dict(official.state_dict())
    s1 = torch.randn(2, 4, 7, 9, requires_grad=True)
    s2 = s1.detach().clone().requires_grad_()
    teacher = torch.randn(2, 8, 7, 9, requires_grad=True)
    targets = torch.tensor([[0, 1, .43, .42, .25, .23], [1, 2, .62, .6, .18, .25]])
    boxes = []
    for i in range(2):
        row = targets[i, 2:]
        cx, cy, w, h = row
        boxes.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2])[None] * torch.tensor([288., 224., 288., 224.]))
    expected = official(s1, teacher.detach(), boxes, [{"img_shape": (224, 288)}] * 2)
    actual, terms = standalone(s2, teacher, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(s2.grad, s1.grad, rtol=2e-5, atol=2e-6)
    assert teacher.grad is None
    for name, p in standalone.named_parameters():
        reference = dict(official.named_parameters())[name]
        if p.grad is not None:
            torch.testing.assert_close(p.grad, reference.grad, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("targets", [torch.empty(0, 6), torch.tensor([[0., 0., .5, .5, 1., 1.]])])
def test_fgd_empty_or_border_targets_are_finite(targets):
    layer = FGDLevel(4, 8)
    loss, _ = layer(torch.randn(1, 4, 5, 7, requires_grad=True), torch.randn(1, 8, 5, 7), targets)
    loss.backward()
    assert torch.isfinite(loss)
    fg, bg = foreground_masks(targets, 1, 5, 7, torch.device("cpu"))
    assert torch.isfinite(fg).all() and torch.isfinite(bg).all()
    assert ((fg > 0) & (bg > 0)).sum() == 0


def test_response_kd_updates_logits_without_backpropagating_teacher_or_boxes():
    s = torch.randn(2, 3, 5, 7, 15, requires_grad=True)
    t = torch.randn(2, 3, 5, 7, 15, requires_grad=True)
    method = ResponseKD([], [])
    loss, terms = method(KDInputs([s], [t], [], [], torch.empty(0, 6), (40, 56)))
    loss.backward()
    assert torch.isfinite(loss) and s.grad[..., 4:].abs().sum() > 0
    assert s.grad[..., :4].abs().sum() == 0
    assert t.grad is None


def test_fgd_checkpoint_restores_method_and_optimizer_state():
    torch.manual_seed(7)
    original = FGD([4], [8])
    optimizer = torch.optim.SGD(original.parameters(), lr=.01, momentum=.9)
    s, t = torch.randn(1, 4, 5, 7), torch.randn(1, 8, 5, 7)
    inputs = KDInputs([], [], [s], [t], torch.tensor([[0., 0., .5, .5, .4, .4]]), (40, 56))
    loss, _ = original(inputs)
    loss.backward()
    optimizer.step()
    stream = io.BytesIO()
    torch.save({"method": original.state_dict(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    restored = FGD([4], [8])
    restored.load_state_dict(saved["method"])
    opt2 = torch.optim.SGD(restored.parameters(), lr=.01, momentum=.9)
    opt2.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(restored(inputs)[0], original(inputs)[0])
    for model, opt in ((original, optimizer), (restored, opt2)):
        opt.zero_grad()
        model(inputs)[0].backward()
        opt.step()
    for a, b in zip(original.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)

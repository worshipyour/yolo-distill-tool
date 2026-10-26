# FGD algorithm adapted from https://github.com/yzd-v/FGD
# Original implementation distributed under Apache-2.0; see THIRD_PARTY.md.
# Changes: standalone PyTorch, YOLO normalized boxes, stable empty/clipped masks,
# configurable multi-level construction and explicit component logging.
import torch
from torch import nn
from torch.nn import functional as F

from . import register
from .base import DistillationMethod


def foreground_masks(targets, batch, height, width, device):
    """FGD's inverse-area foreground mask and normalized complementary background."""
    fg = torch.zeros(batch, height, width, device=device, dtype=torch.float32)
    # Batch rectangle rasterization avoids thousands of tiny CUDA launches.
    # Float64 coordinates preserve the old Python arithmetic at pixel boundaries.
    boxes = targets.detach().to(device=device, dtype=torch.float64)
    ys = torch.arange(height, device=device)[None, :, None]
    xs = torch.arange(width, device=device)[None, None, :]
    # Bound temporary memory even for dense annotations or large input sizes.
    chunk_size = max(1, 2_000_000 // (height * width))
    for part in boxes.split(chunk_size):
        if not len(part):
            continue
        cx, cy, bw, bh = part[:, 2:].unbind(1)
        x0 = ((cx - bw / 2) * width).floor().clamp(0, width - 1)
        x1 = ((cx + bw / 2) * width).ceil().clamp(0, width - 1)
        y0 = ((cy - bh / 2) * height).floor().clamp(0, height - 1)
        y1 = ((cy + bh / 2) * height).ceil().clamp(0, height - 1)
        inside = ((xs >= x0[:, None, None]) & (xs <= x1[:, None, None]) &
                  (ys >= y0[:, None, None]) & (ys <= y1[:, None, None]) &
                  (bw[:, None, None] > 0) & (bh[:, None, None] > 0))
        area = (1. / ((x1 + 1 - x0) * (y1 + 1 - y0))).float()
        values = inside.float() * area[:, None, None]
        image_indices = part[:, 0].long()[:, None, None].expand_as(values)
        fg.scatter_reduce_(0, image_indices, values, reduce="amax", include_self=True)
    bg = (fg == 0).float()
    bg /= bg.sum((1, 2), keepdim=True).clamp_min(1)
    return fg, bg


class FGDLevel(nn.Module):
    def __init__(self, student_channels, teacher_channels, temperature=0.5,
                 alpha=0.001, beta=0.0005, gamma=0.001, relation=0.000005):
        super().__init__()
        if temperature <= 0:
            raise ValueError("FGD temperature must be positive")
        self.temperature = float(temperature)
        self.weights = (float(alpha), float(beta), float(gamma), float(relation))
        c = teacher_channels
        self.align = nn.Conv2d(student_channels, c, 1) if student_channels != c else nn.Identity()
        self.conv_mask_s, self.conv_mask_t = nn.Conv2d(c, 1, 1), nn.Conv2d(c, 1, 1)
        def add_branch():
            return nn.Sequential(nn.Conv2d(c, c // 2, 1), nn.LayerNorm([c // 2, 1, 1]),
                                 nn.ReLU(inplace=True), nn.Conv2d(c // 2, c, 1))
        self.channel_add_conv_s, self.channel_add_conv_t = add_branch(), add_branch()
        for layer in (self.conv_mask_s, self.conv_mask_t):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for branch in (self.channel_add_conv_s, self.channel_add_conv_t):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def attention(self, x):
        n, c, h, w = x.shape
        value = x.abs()
        spatial = (F.softmax(value.mean(1).flatten(1) / self.temperature, 1) * (h * w)).view(n, h, w)
        channel = F.softmax(value.mean((2, 3)) / self.temperature, 1) * c
        return spatial, channel

    def pool(self, x, conv):
        n, c, h, w = x.shape
        weights = F.softmax(conv(x).view(n, 1, h * w), dim=2)
        return torch.bmm(x.view(n, c, h * w), weights.transpose(1, 2)).view(n, c, 1, 1)

    def forward(self, student, teacher, targets):
        student = self.align(student.float())
        teacher = teacher.detach().float()
        if student.shape != teacher.shape:
            raise ValueError("FGD aligned feature shapes differ")
        n, _, h, w = teacher.shape
        st, ct = self.attention(teacher)
        ss, cs = self.attention(student)
        fg, bg = foreground_masks(targets, n, h, w, teacher.device)
        # Algebraically identical to the official square-root weighted MSE.
        squared = (student - teacher).square() * st[:, None] * ct[:, :, None, None]
        fg_loss = (squared * fg[:, None]).sum() / n
        bg_loss = (squared * bg[:, None]).sum() / n
        attention_loss = ((ss - st).abs().sum() + (cs - ct).abs().sum()) / n
        out_s = student + self.channel_add_conv_s(self.pool(student, self.conv_mask_s))
        out_t = teacher + self.channel_add_conv_t(self.pool(teacher, self.conv_mask_t))
        relation_loss = F.mse_loss(out_s, out_t, reduction="sum") / n
        terms = torch.stack([fg_loss, bg_loss, attention_loss, relation_loss])
        total = (terms * terms.new_tensor(self.weights)).sum()
        return total, terms


@register("fgd")
class FGD(DistillationMethod):
    requires_features = True

    def __init__(self, student_channels, teacher_channels, **kwargs):
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError("Different numbers of FGD feature levels")
        self.levels = nn.ModuleList(FGDLevel(s, t, **kwargs) for s, t in zip(student_channels, teacher_channels))

    def forward(self, inputs):
        losses, terms = [], []
        for layer, s, t in zip(self.levels, inputs.student_features, inputs.teacher_features):
            loss, part = layer(s, t, inputs.targets)
            losses.append(loss)
            terms.append(part.detach())
        total = torch.stack(losses).sum()
        values = torch.stack(terms).sum(0)
        return total, dict(zip(("foreground", "background", "attention", "relation"), values))

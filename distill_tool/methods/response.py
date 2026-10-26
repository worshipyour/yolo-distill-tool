"""Classic temperature-scaled soft-target KD, adapted to YOLO sigmoid heads.

This is a response-distillation baseline, not an official reproduction of a
detector-specific paper. No raw box-logit imitation is performed. Matching
anchor geometry, strides, classes and grid ordering is checked by the engine.
"""
import torch
from torch.nn import functional as F

from . import register
from .base import DistillationMethod


@register("response_kd")
class ResponseKD(DistillationMethod):
    def __init__(self, student_channels, teacher_channels, temperature=2.0,
                 confidence=0.25, background_weight=0.05, object_weight=1.0, class_weight=1.0):
        super().__init__()
        if temperature <= 0 or not 0 < confidence < 1 or not 0 <= background_weight <= 1:
            raise ValueError("Invalid response KD temperature/confidence/background_weight")
        self.temperature = float(temperature)
        self.confidence = float(confidence)
        self.background_weight = float(background_weight)
        self.object_weight = float(object_weight)
        self.class_weight = float(class_weight)

    def forward(self, inputs):
        if len(inputs.student_predictions) != len(inputs.teacher_predictions):
            raise ValueError("Different teacher/student detection levels")
        losses = []
        obj_terms, cls_terms = [], []
        for s, t in zip(inputs.student_predictions, inputs.teacher_predictions):
            if s.shape != t.shape:
                raise ValueError(f"Response KD needs aligned predictions, got {s.shape}, {t.shape}")
            s, t = s.float(), t.detach().float()
            temp = self.temperature
            fg = (t[..., 4].sigmoid() >= self.confidence).float()
            bg = 1 - fg
            # Separate positive/negative means prevent numerous background cells
            # from dominating a hand-detection image with very few positive cells.
            obj = F.binary_cross_entropy_with_logits(s[..., 4] / temp,
                    (t[..., 4] / temp).sigmoid(), reduction="none") * temp ** 2
            obj = (obj * fg).sum() / fg.sum().clamp_min(1) + self.background_weight * (obj * bg).sum() / bg.sum().clamp_min(1)
            cls = F.binary_cross_entropy_with_logits(s[..., 5:] / temp,
                    (t[..., 5:] / temp).sigmoid(), reduction="none").mean(-1) * temp ** 2
            cls = (cls * fg).sum() / fg.sum().clamp_min(1)
            losses.append(self.object_weight * obj + self.class_weight * cls)
            obj_terms.append(obj.detach())
            cls_terms.append(cls.detach())
        total = torch.stack(losses).mean()
        return total, {"object": torch.stack(obj_terms).mean(), "class": torch.stack(cls_terms).mean()}

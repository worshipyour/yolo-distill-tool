from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class KDInputs:
    student_predictions: list
    teacher_predictions: list
    student_features: list
    teacher_features: list
    targets: torch.Tensor  # [image_index, class, cx, cy, w, h], normalized to padded network input
    input_hw: tuple


class DistillationMethod(nn.Module):
    requires_features = False

    def forward(self, inputs: KDInputs):
        """Return an unweighted scalar mean-per-image loss and a term dictionary."""
        raise NotImplementedError

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import network_shape


class PaddedStudent(nn.Module):
    """Accept normalized RGB NCHW at the external size; stride-pad inside the graph."""
    def __init__(self, model, height=320, width=328):
        super().__init__()
        self.model = model
        self.height, self.width = height, width
        nh, nw = network_shape(height, width)
        self.pad = (0, nw - width, 0, nh - height)
        # The wrapper itself must be eval too: ONNX export restores the wrapper's
        # previous mode recursively, which would otherwise put Detect into training.
        self.eval()

    def forward(self, image):
        if not torch.jit.is_tracing() and tuple(image.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"Expected RGB NCHW at {self.height}x{self.width}")
        image = F.pad(image, self.pad, value=114 / 255)
        result = self.model(image)
        return result[0] if isinstance(result, tuple) else result


def preprocess(image_bgr, height, width):
    from utils.augmentations import letterbox
    canvas, ratio, pad = letterbox(image_bgr, (height, width), auto=False, scaleup=False)
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0)
    return tensor, (ratio, pad)


def infer(model, image_bgr, height, width, device, half=False, conf=0.25, iou=0.45):
    from utils.general import non_max_suppression, scale_boxes
    raw, meta = preprocess(image_bgr, height, width)
    x = raw.to(device).to(torch.float16 if half else torch.float32) / 255
    with torch.inference_mode():
        detections = non_max_suppression(model(x), conf, iou, max_det=300)[0]
        # YOLOv5 rescales boxes in place. Keep this operation in inference mode
        # because NMS returned an inference tensor.
        if len(detections):
            scale_boxes((height, width), detections[:, :4], image_bgr.shape, ratio_pad=meta)
    return detections


def export_onnx(weight, output, height=320, width=328, verify=True):
    import onnx
    from models.experimental import attempt_load
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = PaddedStudent(attempt_load(str(weight), device=torch.device("cpu"), inplace=True, fuse=True).float().eval(), height, width)
    # Normalized RGB floating point input; NMS is a separate, documented operation.
    x = torch.zeros(1, 3, height, width)
    with torch.inference_mode():
        torch.onnx.export(model, x, str(output), input_names=["images"], output_names=["predictions"],
                          opset_version=13, do_constant_folding=True)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    if shape != [1, 3, height, width]:
        raise AssertionError(f"Exported graph input is {shape}")
    max_error = 0.0
    if verify:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        generator = torch.Generator().manual_seed(42)
        for _ in range(3):
            sample = torch.rand((1, 3, height, width), generator=generator)
            with torch.inference_mode():
                expected = model(sample).numpy()
            actual = session.run(None, {"images": sample.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {"path": str(output), "input_nchw": shape, "network_hw": list(network_shape(height, width)),
            "dtype": "float32", "input_order": "RGB", "range": [0, 1], "nms_in_graph": False,
            "onnxruntime_verified": verify, "max_abs_error": max_error}

"""UI metadata stays lightweight; selecting a method does not import CUDA."""
from importlib.metadata import entry_points
import re

MODELS = ["yolov5" + size for size in "nsmlx"]
BUILTINS = {
    "response_kd": {
        "label": "Response KD · 经典输出蒸馏",
        "description": "模仿教师的目标置信度与类别软输出，保留原检测损失。",
        "parameters": {"weight": 0.05, "temperature": 2.0, "confidence": 0.25,
                       "background_weight": 0.05, "object_weight": 1.0, "class_weight": 1.0},
    },
    "fgd": {
        "label": "FGD · 特征蒸馏",
        "description": "对齐前景/背景特征、空间与通道注意力，以及全局关系。",
        "parameters": {"weight": 0.005, "temperature": 0.5, "alpha": 0.001,
                       "beta": 0.0005, "gamma": 0.001, "relation": 0.000005},
    },
}


def method_catalog():
    methods = {name: dict(value) for name, value in BUILTINS.items()}
    for ep in entry_points(group="distill_tool.methods"):
        if ep.name not in methods and re.fullmatch(r"[a-z][a-z0-9_]*", ep.name) and ep.name not in ("none", "baseline"):
            methods[ep.name] = {"label": ep.name + " · 已安装插件", "description": "使用插件默认参数，或在高级选项中填写插件参数。",
                                "parameters": {"weight": 1.0}, "plugin": True}
    return methods

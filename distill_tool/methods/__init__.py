"""Methods add training losses without changing the deployed student graph."""

from importlib.metadata import entry_points

REGISTRY = {}


def register(name):
    def decorate(cls):
        if name in REGISTRY:
            raise ValueError(f"Duplicate method: {name}")
        REGISTRY[name] = cls
        return cls
    return decorate


def discover():
    from . import response, fgd  # noqa: F401
    for ep in entry_points(group="distill_tool.methods"):
        if ep.name not in REGISTRY:
            REGISTRY[ep.name] = ep.load()
    return REGISTRY


def build(name, student_channels, teacher_channels, **kwargs):
    methods = discover()
    if name not in methods:
        raise ValueError(f"Unknown method {name!r}; available: {', '.join(methods)}")
    return methods[name](student_channels, teacher_channels, **kwargs)

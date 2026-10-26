from pathlib import Path

import pytest


@pytest.fixture
def yolov5_engine(monkeypatch):
    """Match the CLI's explicit engine path setup in a fresh checkout."""
    engine = Path(__file__).resolve().parents[1] / "vendor" / "yolov5"
    if not (engine / "train.py").is_file():
        pytest.skip("bootstrap engine first")
    monkeypatch.syspath_prepend(str(engine))
    return engine

from pathlib import Path

import pytest
import yaml

from distill_tool.data import audit_dataset, labels_for


def test_shared_validation_and_test_are_rejected(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"placeholder")
    cfg = {"path": str(tmp_path), "train": "images", "val": "images", "test": "images", "names": ["hand"]}
    file = tmp_path / "bad.yaml"
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="leakage"):
        audit_dataset(file, check_labels=False)


def test_ha_grid_bbox_is_converted_from_top_left_xywh_to_yolo_center():
    result = labels_for({"bboxes": [[.1, .2, .4, .2]], "labels": ["peace"]})
    assert [float(v) for v in result.split()] == pytest.approx([0, .3, .3, .4, .2])


def test_invalid_annotation_lengths_and_empty_selected_labels_are_rejected():
    with pytest.raises(ValueError):
        labels_for({"bboxes": [[0, 0, 1, 1]], "labels": []})
    with pytest.raises(ValueError):
        labels_for({"bboxes": [[0, 0, 1, 1]], "labels": ["no_gesture"]})


def test_content_duplicates_are_replaced_without_changing_balanced_counts(tmp_path, monkeypatch):
    import io
    import json
    import random
    import zipfile
    from PIL import Image
    import distill_tool.data as module
    monkeypatch.setattr(module, "CLASSES", ["peace"])
    colors = []
    for color in ("red", "green", "blue"):
        stream = io.BytesIO()
        Image.new("RGB", (40, 40), color).save(stream, format="JPEG")
        colors.append(stream.getvalue())
    ann_file, image_file = tmp_path / "ann.zip", tmp_path / "images.zip"
    with zipfile.ZipFile(ann_file, "w") as ann, zipfile.ZipFile(image_file, "w") as images:
        for split_index, split in enumerate(("train", "val", "test")):
            names = [split + "0", split + "1"]
            order = sorted(names)
            random.Random(3 + split_index * 10000).shuffle(order)
            records = {}
            for name in names:
                records[name] = {"bboxes": [[.1, .2, .4, .2]], "labels": ["peace"], "user_id": split + "-subject"}
                payload = colors[max(0, split_index - 1)] if name == order[0] else colors[split_index]
                images.writestr("HaGRID/peace/" + name + ".jpg", payload)
            ann.writestr("annotations/" + split + "/peace.json", json.dumps(records))
    result = module.prepare_hagrid(image_file, ann_file, tmp_path / "prepared", seed=3,
                                  train_per_class=1, val_per_class=1, test_per_class=1)
    summary = json.loads((result.parent / "summary.json").read_text())
    assert summary["counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["duplicate_candidates_skipped"] >= 2
    records = [json.loads(x) for x in (result.parent / "manifest.jsonl").read_text().splitlines()]
    assert len({r["sha256"] for r in records}) == 3


def test_image_alias_keeps_yolo_label_lookup_in_dataset_tree(tmp_path):
    import os
    raw = tmp_path / "raw"
    raw.mkdir()
    data = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        source = raw / (split + ".jpg")
        source.write_bytes(split.encode())
        image = data / "images" / split / (split + ".jpg")
        label = data / "labels" / split / (split + ".txt")
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        try:
            image.symlink_to(source)
        except OSError:
            pytest.skip("OS does not allow unprivileged symlinks")
        label.write_text("0 0.5 0.5 0.25 0.25\n")
        (data / (split + ".txt")).write_text(str(image) + "\n")
    file = data / "dataset.yaml"
    file.write_text(yaml.safe_dump({"path": str(data), "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": ["hand"]}))
    assert audit_dataset(file) == {"train": 1, "val": 1, "test": 1}

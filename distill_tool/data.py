"""Balanced HaGRID conversion with official split and participant separation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from .config import emit, write_json

CLASSES = ["peace", "two_up", "stop_inverted", "three", "stop", "rock", "three2", "dislike", "four", "mute"]


def labels_for(record, classes=CLASSES):
    boxes, labels = record.get("bboxes", []), record.get("labels", [])
    if len(boxes) != len(labels):
        raise ValueError("Annotation bbox/label lengths differ")
    rows = []
    for name, box in zip(labels, boxes):
        if name not in classes:
            continue
        x, y, w, h = map(float, box)
        x1, y1, x2, y2 = max(0., x), max(0., y), min(1., x + w), min(1., y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid HaGRID box {box}")
        rows.append(f"{classes.index(name)} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
    if not rows:
        raise ValueError("Image has no selected-class boxes")
    return "\n".join(rows) + "\n"


def prepare_hagrid(images, annotations, output, legacy_subset=None, seed=20261024,
                   train_per_class=8000, val_per_class=1000, test_per_class=2000):
    images, annotations, output = Path(images), Path(annotations), Path(output).resolve()
    legacy_subset = Path(legacy_subset) if legacy_subset else None
    settings = dict(seed=seed, counts={"train": train_per_class, "val": val_per_class, "test": test_per_class},
                    classes=CLASSES, images_bytes=images.stat().st_size, annotations_bytes=annotations.stat().st_size,
                    legacy_subset=str(legacy_subset) if legacy_subset else None)
    summary_path = output / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old["settings"] != settings:
            raise ValueError("Output already contains a different dataset; choose a new output directory")
        audit_dataset(output / "dataset.yaml")
        return output / "dataset.yaml"
    legacy = {sp: {c: [] for c in CLASSES} for sp in ("train", "test")}
    if legacy_subset:
        with (legacy_subset / "manifest.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] in legacy and row["class"] in CLASSES:
                    legacy[row["split"]][row["class"]].append(row["image_id"])
    legacy_test_hashes = set()
    if legacy_subset:
        for cls, image_ids in legacy["test"].items():
            for image_id in image_ids:
                legacy_test_hashes.add(hashlib.sha256((legacy_subset / "test" / cls / (image_id + ".jpg")).read_bytes()).hexdigest())
    plans = []
    users, ids = {s: set() for s in settings["counts"]}, {s: set() for s in settings["counts"]}
    with zipfile.ZipFile(annotations) as ann, zipfile.ZipFile(images) as archive:
        image_index = {}
        for info in archive.infolist():
            p = PurePosixPath(info.filename)
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name in CLASSES:
                image_index[(p.parent.name, p.stem)] = info
        for split_index, (split, count) in enumerate(settings["counts"].items()):
            for ci, cls in enumerate(CLASSES):
                records = json.loads(ann.read(f"annotations/{split}/{cls}.json"))
                if split == "train" and legacy["train"][cls]:
                    first = sorted(legacy["train"][cls])
                    if len(first) != count:
                        raise ValueError("Legacy train count differs from requested count")
                    remaining = sorted(set(records) - set(first))
                    random.Random(seed + ci).shuffle(remaining)
                    selected = first + remaining
                else:
                    excluded = set(legacy["test"][cls]) if split == "test" else set()
                    pool = sorted(set(records) - excluded)
                    if len(pool) < count:
                        raise ValueError(f"Insufficient unseen {split}/{cls}: {len(pool)} < {count}")
                    random.Random(seed + split_index * 10000 + ci).shuffle(pool)
                    selected = pool
                for image_id in selected:
                    record = records[image_id]
                    if not record.get("user_id"):
                        raise ValueError("Missing participant ID; cannot establish subject-disjoint splits")
                    users[split].add(record["user_id"])
                    ids[split].add(image_id)
                    info = image_index[(cls, image_id)]
                    plans.append((split, cls, image_id, record, info))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if users[a] & users[b] or ids[a] & ids[b]:
                raise ValueError(f"Official split overlap: {a}/{b}")
        output.mkdir(parents=True, exist_ok=True)
        manifests = {sp: [] for sp in settings["counts"]}
        hashes = {}
        accepted = {(s, c): 0 for s in settings["counts"] for c in CLASSES}
        selected_users = {s: set() for s in settings["counts"]}
        duplicates_skipped = 0
        manifest_path = output / "manifest.jsonl"
        with manifest_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as mf:
            for index, (split, cls, image_id, record, info) in enumerate(plans):
                if accepted[(split, cls)] >= settings["counts"][split]:
                    continue
                stem = cls + "__" + image_id
                image_dest = output / "images" / split / (stem + ".jpg")
                label_dest = output / "labels" / split / (stem + ".txt")
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                source = legacy_subset / split / cls / (image_id + ".jpg") if legacy_subset else None
                payload = source.read_bytes() if source and source.exists() else archive.read(info)
                digest = hashlib.sha256(payload).hexdigest()
                if digest in hashes or (split == "test" and digest in legacy_test_hashes):
                    duplicates_skipped += 1
                    continue
                if not image_dest.exists():
                    if source and source.exists():
                        image_dest.symlink_to(source.resolve())
                    else:
                        image_dest.with_suffix(".jpg.part").write_bytes(payload)
                        image_dest.with_suffix(".jpg.part").replace(image_dest)
                if image_dest.stat().st_size != info.file_size:
                    raise ValueError(f"Image size mismatch: {image_dest}")
                if hashlib.sha256(image_dest.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Existing image content mismatch: {image_dest}")
                hashes[digest] = split
                labels = labels_for(record)
                if label_dest.exists() and label_dest.read_text() != labels:
                    raise ValueError(f"Refusing to overwrite conflicting labels: {label_dest}")
                label_dest.write_text(labels)
                relative = image_dest.relative_to(output).as_posix()
                manifests[split].append(str(image_dest))
                accepted[(split, cls)] += 1
                selected_users[split].add(record["user_id"])
                row = dict(split=split, source_class=cls, image_id=image_id, user_id=record["user_id"],
                           sha256=digest, image=relative)
                mf.write(json.dumps(row) + "\n")
                completed = sum(accepted.values())
                if completed % 1000 == 0:
                    emit("data_progress", completed=completed, total=sum(settings['counts'].values()) * len(CLASSES))
        for (split, cls), count in accepted.items():
            if count != settings["counts"][split]:
                raise ValueError(f"Insufficient unique images in {split}/{cls}: {count}")
        manifest_path.with_suffix(".jsonl.tmp").replace(manifest_path)
    data = {"path": str(output), "nc": len(CLASSES), "names": CLASSES}
    for split, paths in manifests.items():
        (output / (split + ".txt")).write_text("\n".join(paths) + "\n")
        data[split] = split + ".txt"
    (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    write_json(summary_path, {"settings": settings, "counts": {k: len(v) for k, v in manifests.items()},
        "participants": {k: len(v) for k, v in selected_users.items()}, "split_overlap": 0,
        "duplicate_candidates_skipped": duplicates_skipped,
        "legacy_test_excluded": bool(legacy_subset), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    audit_dataset(output / "dataset.yaml")
    emit("data_ready", path=str(output / "dataset.yaml"))
    return output / "dataset.yaml"


def resolve_split(data_yaml, split):
    data_yaml = Path(data_yaml).resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    sources = data[split] if isinstance(data[split], list) else [data[split]]
    images = []
    extensions = {".jpg", ".png", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    for source in sources:
        p = Path(source)
        p = (root / p).absolute() if not p.is_absolute() else p
        if p.is_dir():
            images.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in extensions))
        elif p.suffix.lower() in extensions:
            images.append(p)
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = Path(line.strip())
                base = p.parent if line.strip().startswith("./") else root
                images.append(entry if entry.is_absolute() else (base / entry).absolute())
    return images


def audit_dataset(data_yaml, check_labels=True, required_splits=("train", "val", "test")):
    data_yaml = Path(data_yaml).resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("names"), (list, dict)) or not data["names"]:
        raise ValueError("Dataset YAML requires a nonempty names list or class-index mapping")
    splits = {sp: resolve_split(data_yaml, sp) for sp in required_splits}
    paths_sets = {sp: {str(p.resolve()) for p in paths} for sp, paths in splits.items()}
    pairs = [(a, b) for i, a in enumerate(splits) for b in list(splits)[i + 1:]]
    for a, b in pairs:
        if paths_sets[a] & paths_sets[b]:
            raise ValueError(f"Data leakage: {a} and {b} share images")
    nc = len(data["names"])
    manifest = data_yaml.parent / "manifest.jsonl"
    if manifest.exists():
        subject_sets = {s: set() for s in splits}
        hash_sets = {s: set() for s in splits}
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row["split"] in splits:
                subject_sets[row["split"]].add(row["user_id"])
                hash_sets[row["split"]].add(row["sha256"])
        for a, b in pairs:
            if subject_sets[a] & subject_sets[b] or hash_sets[a] & hash_sets[b]:
                raise ValueError(f"Participant/content leakage: {a}/{b}")
        summary = data_yaml.parent / "summary.json"
        if summary.exists() and json.loads(summary.read_text())["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("Dataset manifest changed since preparation")
    for split, images in splits.items():
        if len(images) != len({str(p) for p in images}) or not images:
            raise ValueError(f"Duplicate or empty {split} list")
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(image)
            if check_labels:
                label = Path(str(image).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
                if not label.exists():
                    continue  # YOLO permits background images with no label file.
                for line in label.read_text().splitlines():
                    values = [float(v) for v in line.split()]
                    if len(values) != 5 or int(values[0]) != values[0] or not 0 <= values[0] < nc:
                        raise ValueError(f"Invalid YOLO class/row: {label}")
                    if not all(__import__('math').isfinite(v) for v in values) or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                        raise ValueError(f"Invalid YOLO coordinates: {label}")
    return {sp: len(p) for sp, p in splits.items()}

"""Make a balanced mechanical-check suite without touching final test images."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill_tool.config import load_recipe, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_recipe(args.recipe)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh smoke directory")
    output.mkdir(parents=True)
    data_path = Path(cfg["data"])
    original = yaml.safe_load(data_path.read_text())
    rows = [json.loads(line) for line in (data_path.parent / "manifest.jsonl").read_text().splitlines()]
    chosen = []
    for split, count in (("train", 20), ("val", 10), ("test", 10)):
        for cls in original["names"]:
            def eligible(row):
                if row["source_class"] != cls:
                    return False
                if split == "train":
                    return row["split"] == "train"
                bucket = int(hashlib.sha256(row["user_id"].encode()).hexdigest(), 16) % 2
                return row["split"] == "val" and bucket == (0 if split == "val" else 1)
            pool = [r for r in rows if eligible(r)]
            samples = random.Random(42).sample(pool, count)
            for row in samples:
                row = dict(row)
                row["split"] = split
                # Preserve the dataset image alias: YOLO derives labels by
                # replacing /images/ with /labels/. Resolving the image symlink
                # would incorrectly move label lookup into the raw image store.
                row["image"] = str(data_path.parent / row["image"])
                chosen.append(row)
    manifest = output / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in chosen) + "\n")
    data = dict(path=str(output), names=original["names"], nc=len(original["names"]))
    for split in ("train", "val", "test"):
        (output / (split + ".txt")).write_text("\n".join(r["image"] for r in chosen if r["split"] == split) + "\n")
        data[split] = split + ".txt"
    (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    write_json(output / "summary.json", {"scope": "mechanical_smoke_only", "final_test_accessed": False,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    cfg.pop("recipe_path")
    cfg["workspace"] = str(output / "run")
    cfg["data"] = str(output / "dataset.yaml")
    cfg["training"].update(epochs=2, batch_size=8, workers=0)
    cfg["evaluation"].update(batch_size=8, benchmark_rounds=1, benchmark_repeats=3, benchmark_warmup=3)
    (output / "recipe.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(output / "recipe.yaml")


if __name__ == "__main__":
    main()

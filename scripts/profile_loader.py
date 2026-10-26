"""Measure the augmented training loader without reading the test split."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--workers', default='8,16,32,48')
    p.add_argument('--steps', type=int, default=192)
    args = p.parse_args()
    from distill_tool.config import load_recipe
    from distill_tool.pipeline import import_engine
    cfg = load_recipe(args.recipe)
    import_engine(cfg)
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.dataloaders import LoadImagesAndLabels, seed_worker
    from utils.general import check_dataset
    from distill_tool.engine.data import FixedShapeDataset
    torch.set_num_threads(2)
    data = check_dataset(cfg['data'])
    hyp = yaml.safe_load((Path(cfg['engine']) / 'data/hyps/hyp.scratch-low.yaml').read_text())
    dataset = FixedShapeDataset(data['train'], 352, args.batch, augment=True, hyp=hyp,
                               rect=False, cache_images=False, stride=32)
    results = []
    for workers in map(int, args.workers.split(',')):
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=workers,
                            pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
        it = iter(loader)
        warmup = 2 * workers
        if warmup + args.steps > len(loader):
            raise ValueError('Training split too small for the requested warmup and timed steps')
        for _ in range(warmup):
            next(it)
        start = time.perf_counter()
        for _ in range(args.steps):
            images, targets, _, _ = next(it)
        seconds = time.perf_counter() - start
        row = {'workers': workers, 'batch': args.batch, 'images_per_second': args.batch*args.steps/seconds,
               'steps': args.steps, 'warmup': warmup,
               'scope': 'train loader with augmentation, pin_memory; excludes GPU; timed window exceeds prefetch capacity'}
        results.append(row)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        # Release this iterator's workers before trying a different worker count.
        del it, loader


if __name__ == '__main__':
    main()

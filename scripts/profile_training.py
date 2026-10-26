"""Measure AMP training capacity; this does not score or tune on the test split."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--batches', default='64,128,256,384')
    p.add_argument('--case', choices=['teacher_m', 'baseline', 'response_kd', 'fgd'])
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--warmup', type=int, default=6)
    p.add_argument('--memory-gib', type=float, default=34)
    args = p.parse_args()
    if not args.case:
        results = []
        for case in ('teacher_m', 'baseline', 'response_kd', 'fgd'):
            for batch in map(int, args.batches.split(',')):
                child = subprocess.run([sys.executable, __file__, '--recipe', args.recipe,
                    '--teacher', args.teacher, '--device', str(args.device), '--output', args.output + '.case.json',
                    '--batches', str(batch), '--case', case, '--steps', str(args.steps),
                    '--warmup', str(args.warmup), '--memory-gib', str(args.memory_gib)], capture_output=True, text=True)
                try:
                    row = json.loads(Path(args.output + '.case.json').read_text())
                    if row.get('case') != case or row.get('batch') != batch:
                        raise ValueError('stale child output')
                except Exception:
                    row = {'case': case, 'batch': batch, 'status': 'error', 'stderr': child.stderr[-2000:]}
                results.append(row)
                Path(args.output).write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
        return
    from distill_tool.config import load_recipe
    from distill_tool.pipeline import import_engine
    cfg = load_recipe(args.recipe)
    import_engine(cfg)
    import torch
    import yaml
    from models.yolo import Model
    from utils.general import init_seeds
    from utils.loss import ComputeLoss
    from utils.torch_utils import ModelEMA, smart_optimizer
    from distill_tool.engine.runtime import Distiller
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    init_seeds(42, deterministic=True)
    batch = int(args.batches)
    row = {'case': args.case, 'batch': batch, 'device': torch.cuda.get_device_name(device),
           'shape': [batch, 3, 320, 352], 'warmup': args.warmup, 'steps': args.steps,
           'scope': 'synthetic images, four boxes/image, AMP forward+loss+backward+SGD+clip+EMA; excludes loader'}
    try:
        names = yaml.safe_load(Path(cfg['data']).read_text())['names']
        model = Model(str(Path(cfg['engine']) / 'models' / ('yolov5m.yaml' if args.case == 'teacher_m' else 'yolov5n.yaml')),
                      ch=3, nc=len(names)).to(device).train()
        hyp = yaml.safe_load((Path(cfg['engine']) / 'data/hyps/hyp.scratch-low.yaml').read_text())
        optimizer = smart_optimizer(model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'] * batch / 64)
        hyp['cls'] *= len(names) / 80
        hyp['obj'] *= (352 / 640)**2
        hyp['label_smoothing'] = 0.
        model.nc, model.hyp, model.names = len(names), hyp, names
        method = args.case if args.case in cfg['methods'] else 'none'
        options = dict(cfg['methods'].get(method, {}))
        weight = options.pop('weight', 1.)
        controller = Distiller(model, method, args.teacher, options, device)
        if controller.parameters():
            optimizer.add_param_group({'params': controller.parameters(), 'weight_decay': 0.})
        ema = ModelEMA(model)
        loss_fn = ComputeLoss(model)
        scaler = torch.cuda.amp.GradScaler()
        images = torch.rand(batch, 3, 320, 352, device=device)
        targets = torch.tensor([[i, i % len(names), .25 + .5*(j%2), .25 + .5*(j//2), .2, .2]
                                for i in range(batch) for j in range(4)], device=device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred, kd = controller.forward(model, images, targets)
                loss, _ = loss_fn(pred, targets)
                loss = loss + weight * kd * batch
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + controller.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            return float(loss.detach())

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            last_loss = step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        row.update(status='ok', images_per_second=batch*args.steps/seconds,
                   step_seconds=seconds/args.steps, loss=last_loss,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
    except torch.cuda.OutOfMemoryError:
        row.update(status='oom', memory_limit_gib=args.memory_gib)
    Path(args.output).write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

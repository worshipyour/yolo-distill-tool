"""Generate a small pinned adapter; keep the official engine files unmodified."""
import hashlib
from pathlib import Path

from distill_tool.config import TRAIN_SHA256


def generate_trainer(engine):
    engine = Path(engine)
    source = (engine / "train.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != TRAIN_SHA256:
        raise ValueError("Engine train.py differs from the pinned v7.0 source; refusing an unreviewed patch")
    s = source.decode()
    def replace(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise ValueError(f"YOLOv5 adapter anchor mismatch: {old[:80]!r}")
        s = s.replace(old, new, 1)
    replace("from utils.dataloaders import create_dataloader\n", "from distill_tool.engine.data import create_fixed_dataloader as create_dataloader\nfrom distill_tool.engine.runtime import make_distiller\n")
    # Setup after model metadata/class attributes are set; optimizer/EMA are already ready.
    replace("    # Start training\n", """    distiller = make_distiller(model, opt, device)
    if distiller.parameters():
        optimizer.add_param_group({'params': distiller.parameters(), 'weight_decay': 0.0,
                                   'lr': hyp['lr0'], 'initial_lr': hyp['lr0']})
        scheduler.base_lrs.append(hyp['lr0'])
        scheduler.lr_lambdas.append(lf)
    if resume:
        resume_kd = torch.load(weights, map_location=device)
        distiller.restore(resume_kd)
    # Start training
""")
    # Extra method parameters must exist before smart_resume restores optimizer groups.
    # Move optimizer restoration until after distiller setup, retaining upstream epoch/EMA restore.
    replace("            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)\n",
            "            resume_optimizer = ckpt['optimizer']\n            ckpt['optimizer'] = None\n            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)\n            best_fitness = ckpt['best_fitness']\n")
    replace("        distiller.restore(resume_kd)\n", "        distiller.restore(resume_kd)\n        optimizer.load_state_dict(resume_optimizer)\n")
    replace("                pred = model(imgs)  # forward\n                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size\n",
            "                targets_device = targets.to(device)\n                pred, kd_loss = distiller.forward(model, imgs, targets_device)\n                loss, loss_items = compute_loss(pred, targets_device)\n                loss = loss + opt.kd_weight * kd_loss * imgs.shape[0]\n")
    replace("                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients\n",
            "                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + distiller.parameters(), max_norm=10.0)\n")
    replace("        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers\n", "        lr = [x['lr'] for x in optimizer.param_groups[:3]]  # upstream logger schema\n")
    replace("            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)\n",
            "            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)\n            distiller.log_epoch(save_dir, epoch, results)\n")
    replace("                    'optimizer': optimizer.state_dict(),\n", "                    'optimizer': optimizer.state_dict(),\n                    'distillation': distiller.checkpoint(),\n                    'scaler': scaler.state_dict(),\n")
    replace("                torch.save(ckpt, last)\n", "                torch.save(ckpt, w / 'resume.pt')  # retained optimizer + KD state\n                torch.save(ckpt, last)\n")
    replace("    callbacks.run('on_train_start')\n", "    if resume and resume_kd.get('scaler'):\n        scaler.load_state_dict(resume_kd['scaler'])\n    init_seeds(opt.seed + 1 + RANK, deterministic=True)\n    callbacks.run('on_train_start')\n")
    replace("    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')\n", """    parser.add_argument('--kd-method', type=str, default='none')
    parser.add_argument('--teacher-weights', type=str, default='')
    parser.add_argument('--kd-weight', type=float, default=1.0)
    parser.add_argument('--kd-config', type=str, default='{}')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
""")
    target = engine / "train_distill_tool.py"
    target.write_text(s, encoding="utf-8")
    return target

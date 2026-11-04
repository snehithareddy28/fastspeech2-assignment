"""FastSpeech 2 training loop.

Based on ming024 train.py. MODIFIED:
  * one config.yaml per language (fastspeech2/config.py) instead of 3 files.
  * resumable: `latest.pth.tar` is written every `latest_step` steps (model,
    optimizer, step, epoch, RNG state) and training resumes from it
    automatically; numbered checkpoints every `save_step` (only the newest
    `keep_last` kept) plus `best.pth.tar` by validation total loss.
  * losses appended to CSV files (train_log.csv / val_log.csv) so plots can be
    regenerated without TensorBoard; TensorBoard is optional.
  * `--max_hours` wall-clock budget: saves and exits cleanly before a cloud
    session limit.
  * single-GPU (no nn.DataParallel), DataLoader workers, optional fp16 AMP,
    fixed seeds, `--device` flag so two runs can share one machine.
  * no `quit()` at the end; returns normally.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config  # noqa: E402
from dataset import Dataset  # noqa: E402
from evaluate import evaluate  # noqa: E402
from model import FastSpeech2Loss  # noqa: E402
from utils.model import get_model, get_param_num, load_checkpoint  # noqa: E402
from utils.tools import to_device  # noqa: E402

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _truncate_csv(path, max_step):
    """On resume, drop log rows written after the checkpoint we resume from."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= max_step]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(kept)


def save_ckpt(path, model, optimizer, step, epoch, extra=None):
    tmp = path + ".tmp"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer._optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        state.update(extra)
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a killed session never leaves a half file


def find_resume(ckpt_dir):
    latest = os.path.join(ckpt_dir, "latest.pth.tar")
    if os.path.exists(latest):
        return latest
    numbered = glob.glob(os.path.join(ckpt_dir, "[0-9]*.pth.tar"))
    if numbered:
        return max(numbered, key=lambda p: int(re.findall(r"(\d+)\.pth\.tar$", p)[0]))
    return None


def main(config_path, restore=None, fresh=False, max_hours=None, device_str=None):
    raw_cfg, configs = load_config(config_path)
    preprocess_config, model_config, train_config = configs
    misc = train_config["misc"]
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # torch >= 2.11 rejects set_device(torch.device("cuda")) without an index
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    seed = misc.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ckpt_dir = train_config["path"]["ckpt_path"]
    log_dir = train_config["path"]["log_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        f.write(open(config_path, encoding="utf-8").read())

    # Data
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = misc.get("group_size", 4)  # >1 enables length sorting inside groups
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=misc.get("num_workers", 4),
        persistent_workers=misc.get("num_workers", 4) > 0,
    )

    # Model
    ckpt_path = restore
    if ckpt_path is None and not fresh:
        ckpt_path = find_resume(ckpt_dir)
    if ckpt_path:
        print("Resuming from", ckpt_path)
    model, optimizer, step0 = get_model(ckpt_path, configs, device, train=True)
    epoch = 1
    best_val = float("inf")
    if ckpt_path:
        ck = load_checkpoint(ckpt_path, "cpu")
        epoch = int(ck.get("epoch", 1))
        best_val = float(ck.get("best_val", float("inf")))
        rng = ck.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception:
                    pass
        del ck
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    print("Train utterances:", len(dataset))

    train_csv = os.path.join(log_dir, "train_log.csv")
    val_csv = os.path.join(log_dir, "val_log.csv")
    _truncate_csv(train_csv, step0)
    _truncate_csv(val_csv, step0)

    tb = None
    if misc.get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(log_dir, "tensorboard"))
        except Exception as e:
            print("TensorBoard disabled:", e)

    use_amp = bool(misc.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = step0 + 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    s = train_config["step"]
    total_step, log_step, val_step, save_step = (
        s["total_step"], s["log_step"], s["val_step"], s["save_step"])
    latest_step = s.get("latest_step", 1000)
    keep_last = s.get("keep_last", 3)
    t_start = time.time()
    deadline = t_start + max_hours * 3600 if max_hours else None

    def checkpoint_now(cur_step, cur_epoch, is_best=False):
        extra = {"best_val": best_val}
        save_ckpt(os.path.join(ckpt_dir, "latest.pth.tar"), model, optimizer, cur_step, cur_epoch, extra)
        if is_best:
            save_ckpt(os.path.join(ckpt_dir, "best.pth.tar"), model, optimizer, cur_step, cur_epoch, extra)

    if step > total_step:
        print("Already trained to", step0)
        return

    outer_bar = tqdm(total=total_step, desc="Training", position=0, initial=step0)
    model.train()
    t_log = time.time()
    while True:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    output = model(*(batch[2:]))
                    losses = Loss(batch, output)
                total_loss = losses[0] / grad_acc_step
                if not torch.isfinite(total_loss):
                    print("Non-finite loss at step {}, skipping batch".format(step))
                    optimizer.zero_grad()
                    continue
                scaler.scale(total_loss).backward()
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer._update_learning_rate()
                    scaler.step(optimizer._optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    lv = [l.item() for l in losses]
                    lr = optimizer._optimizer.param_groups[0]["lr"]
                    sps = log_step / max(time.time() - t_log, 1e-6)
                    t_log = time.time()
                    _append_csv(train_csv, ["step"] + LOSS_NAMES + ["lr", "epoch", "steps_per_sec",
                                                                    "elapsed_h"],
                                [step] + ["%.5f" % v for v in lv] +
                                ["%.3e" % lr, epoch, "%.2f" % sps,
                                 "%.3f" % ((time.time() - t_start) / 3600)])
                    outer_bar.write(
                        "Step {}/{}, Total {:.4f}, Mel {:.4f}, PostNet {:.4f}, Pitch {:.4f}, "
                        "Energy {:.4f}, Duration {:.4f}, lr {:.2e}, {:.2f} it/s".format(
                            step, total_step, *lv, lr, sps))
                    if tb:
                        for n, v in zip(LOSS_NAMES, lv):
                            tb.add_scalar("train/" + n, v, step)

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(model, step, configs, device=device)
                    model.train()
                    outer_bar.write(message)
                    _append_csv(val_csv, ["step"] + LOSS_NAMES,
                                [step] + ["%.5f" % v for v in val_losses])
                    if tb:
                        for n, v in zip(LOSS_NAMES, val_losses):
                            tb.add_scalar("val/" + n, v, step)
                    if val_losses[0] < best_val:
                        best_val = val_losses[0]
                        checkpoint_now(step, epoch, is_best=True)

                if step % save_step == 0:
                    save_ckpt(os.path.join(ckpt_dir, "{}.pth.tar".format(step)), model, optimizer,
                              step, epoch, {"best_val": best_val})
                    numbered = sorted(glob.glob(os.path.join(ckpt_dir, "[0-9]*.pth.tar")),
                                      key=lambda p: int(re.findall(r"(\d+)\.pth\.tar$", p)[0]))
                    for old in numbered[:-keep_last]:
                        os.remove(old)

                if step % latest_step == 0:
                    checkpoint_now(step, epoch)

                if step >= total_step:
                    checkpoint_now(step, epoch)
                    outer_bar.write("Reached total_step {}".format(total_step))
                    return
                if deadline and time.time() > deadline:
                    checkpoint_now(step, epoch)
                    outer_bar.write("Time budget reached at step {}; saved latest.pth.tar".format(step))
                    return
                step += 1
                outer_bar.update(1)
        epoch += 1


def cli(default_config=None):
    parser = argparse.ArgumentParser(description="Train FastSpeech 2")
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--restore", default=None, help="checkpoint to resume from "
                        "(default: <ckpt_path>/latest.pth.tar if present)")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="wall-clock budget; save and exit when exceeded")
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu")
    args = parser.parse_args()
    main(args.config, args.restore, args.fresh, args.max_hours, args.device)


if __name__ == "__main__":
    cli()

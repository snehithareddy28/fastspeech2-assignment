"""Export a small inference checkpoint (fp16 weights only) plus the metadata
needed to run inference without the training data.

    python scripts/export_checkpoint.py English [--ckpt English/checkpoints/best.pth.tar]

Writes into <Lang>/checkpoints/:
  <lang>_step<N>_infer_fp16.pth.tar   model weights in fp16 (~half the fp32 size)
  stats.json, speakers.json           pitch/energy normalisation stats, speaker map
  train_speaker_counts.json           used to pick the default inference speaker
"""
import argparse
import json
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))
from config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["English", "Telugu"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    _, (pc, mc, tc) = load_config(args.config or os.path.join(ROOT, args.lang, "config.yaml"))
    cdir = tc["path"]["ckpt_path"]
    src = args.ckpt or os.path.join(cdir, "best.pth.tar")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    step = int(ck.get("step", 0))
    state = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()}
    out = os.path.join(cdir, "{}_step{}_infer_fp16.pth.tar".format(args.lang.lower(), step))
    torch.save({"model": state, "step": step, "source": os.path.basename(src),
                "best_val": ck.get("best_val")}, out)
    pp = pc["path"]["preprocessed_path"]
    for fn in ["stats.json", "speakers.json"]:
        shutil.copy(os.path.join(pp, fn), os.path.join(cdir, fn))
    counts = {}
    with open(os.path.join(pp, "train.txt"), encoding="utf-8") as f:
        for line in f:
            s = line.split("|")[1]
            counts[s] = counts.get(s, 0) + 1
    with open(os.path.join(cdir, "train_speaker_counts.json"), "w") as f:
        json.dump(counts, f, indent=1)
    print("Wrote", out, "({:.1f} MB)".format(os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()

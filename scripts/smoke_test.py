"""End-to-end smoke test on synthetic data (no downloads, no MFA).

Builds a small fake corpus per language (harmonic "speech" + fake TextGrids
with realistic utterance lengths), then runs feature extraction, a short
training run, a resume, inference with plots, loss plots and checkpoint
export. Prints the measured training speed (it/s), which is what
train.step.total_step should be based on.

    python scripts/smoke_test.py --steps 300 [--lang te en] [--out /tmp/smoke]
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

SENTS = {
    "te": ["పులి అడవిలో నిశ్శబ్దంగా నడిచింది.", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
           "ఈ సమావేశం ఉదయం ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
           "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."],
    "en": ["The researchers will conduct an experiment to evaluate the proposed method.",
           "The orchestra's conduct during the ceremony was praised by the audience.",
           "The meeting will start at ten, so everyone should arrive on time.",
           "The movement gained considerable momentum during the proceedings."],
}


def make_corpus(lang, out, n_utts, seed=0):
    import tgt

    if lang == "en":
        from text.english import text_to_phones
        speakers = ["spkA", "spkB", "spkC"]
        cfg_path = os.path.join(ROOT, "English", "config.yaml")
    else:
        from text.telugu import text_to_phones
        speakers = ["te_female"]
        cfg_path = os.path.join(ROOT, "Telugu", "config.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    root = os.path.join(out, lang)
    for k in ["raw_path", "textgrid_path", "preprocessed_path", "ckpt_path", "log_path",
              "result_path", "lexicon_path"]:
        cfg["path"][k] = os.path.join(root, k)
    sr = 22050
    rng = np.random.default_rng(seed)
    rows = ["basename\tspeaker\tsplit"]
    for i in range(n_utts):
        spk = speakers[i % len(speakers)]
        # 1-3 sentences per utterance -> 3-12 s, like the real data after chunking
        ph = []
        for _ in range(rng.integers(1, 4)):
            p, _, _ = text_to_phones(SENTS[lang][rng.integers(len(SENTS[lang]))])
            ph += [x for x in p if x != "sp"] + ["sp"]
        ph = ph[:-1]
        dur = [rng.uniform(0.15, 0.35) if p == "sp" else rng.uniform(0.05, 0.11) for p in ph]
        T = 0.2 + sum(dur) + 0.2
        t = np.arange(int(T * sr)) / sr
        f0 = 140 + 30 * np.sin(2 * np.pi * 0.5 * t) + 25 * speakers.index(spk)
        wav = 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / sr) + 0.01 * rng.standard_normal(len(t))
        uid = "u%04d" % i
        d = os.path.join(cfg["path"]["raw_path"], spk)
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, uid + ".wav"), (wav * 30000).astype(np.int16), sr)
        open(os.path.join(d, uid + ".lab"), "w", encoding="utf-8").write("synthetic\n")
        tg = tgt.TextGrid()
        tier = tgt.IntervalTier(0, T, "phones")
        tier.add_interval(tgt.Interval(0, 0.2, ""))
        cur = 0.2
        for p, dd in zip(ph, dur):
            tier.add_interval(tgt.Interval(cur, cur + dd, "" if p == "sp" else p))
            cur += dd
        tier.add_interval(tgt.Interval(cur, T, ""))
        tg.add_tier(tier)
        td = os.path.join(cfg["path"]["textgrid_path"], spk)
        os.makedirs(td, exist_ok=True)
        tgt.io.write_to_file(tg, os.path.join(td, uid + ".TextGrid"), format="long")
        rows.append("%s\t%s\t%s" % (uid, spk, "val" if i % 8 == 0 else "train"))
    open(os.path.join(cfg["path"]["raw_path"], "split.tsv"), "w").write("\n".join(rows) + "\n")
    return cfg


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--utts", type=int, default=160)
    ap.add_argument("--lang", nargs="+", default=["te", "en"])
    ap.add_argument("--out", default=os.path.join(ROOT, "smoke_out"))
    ap.add_argument("--amp", action="store_true", help="also time a short fp16 AMP run")
    args = ap.parse_args()
    py = sys.executable
    report = {}
    for lang in args.lang:
        L = "English" if lang == "en" else "Telugu"
        cfg = make_corpus(lang, args.out, args.utts)
        s = cfg["train"]["step"]
        k = max(5, args.steps // 10)
        s.update(total_step=args.steps, log_step=k, val_step=2 * k, save_step=args.steps,
                 latest_step=2 * k)
        cfg_file = os.path.join(args.out, lang + "_config.yaml")
        yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"), allow_unicode=True)
        run([py, L + "/preprocess.py", "features", "--config", cfg_file, "--workers", "4"])
        half = args.steps // 2
        cfg["train"]["step"]["total_step"] = half
        yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"), allow_unicode=True)
        run([py, L + "/train.py", "--config", cfg_file])           # first half
        cfg["train"]["step"]["total_step"] = args.steps
        yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"), allow_unicode=True)
        run([py, L + "/train.py", "--config", cfg_file])           # resume to the end
        log = os.path.join(cfg["path"]["log_path"], "train_log.csv")
        rows = list(csv.DictReader(open(log)))
        sps = [float(r["steps_per_sec"]) for r in rows[2:]] or [float(rows[-1]["steps_per_sec"])]
        report[L] = {"median_it_per_s": round(float(np.median(sps)), 2),
                     "last_step": int(rows[-1]["step"])}
        if args.amp:
            cfg["train"]["misc"]["amp"] = True
            cfg["path"]["ckpt_path"] += "_amp"
            cfg["path"]["log_path"] += "_amp"
            cfg["train"]["step"]["total_step"] = 150
            yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"), allow_unicode=True)
            run([py, L + "/train.py", "--config", cfg_file, "--fresh"])
            rows = list(csv.DictReader(open(os.path.join(cfg["path"]["log_path"], "train_log.csv"))))
            report[L]["amp_median_it_per_s"] = round(
                float(np.median([float(r["steps_per_sec"]) for r in rows[1:]])), 2)
            report[L]["amp_last_total_loss"] = rows[-1]["total"]
            cfg["train"]["misc"]["amp"] = False
            cfg["path"]["ckpt_path"] = cfg["path"]["ckpt_path"][:-4]
            cfg["path"]["log_path"] = cfg["path"]["log_path"][:-4]
            yaml.safe_dump(cfg, open(cfg_file, "w", encoding="utf-8"), allow_unicode=True)
        run([py, L + "/inference.py", "--config", cfg_file,
             "--out", os.path.join(args.out, lang + "_results")])
    print("\nSMOKE TEST PASSED")
    for L, r in report.items():
        print(L, r)


if __name__ == "__main__":
    main()

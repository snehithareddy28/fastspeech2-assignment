"""Telugu (IndicTTS Telugu) data preparation and feature extraction.

Stages (run from the repo root):
  python Telugu/preprocess.py prepare     # download, select ~6 h of one speaker, split, wav/lab + lexicon
  bash scripts/align.sh telugu             # train an MFA acoustic model on this subset, then align
  python Telugu/preprocess.py features    # mel, durations, pitch, energy -> data/Telugu/preprocessed

Selection strategy (see config.yaml `data:` and the report):
  * IndicTTS Telugu has two speakers, identified by the `gender` label
    (female / male). A single speaker is used for stability; `speaker: auto`
    picks the one with more usable audio after filtering.
  * keep utterances of min_dur..max_dur seconds whose normalised transcript
    is pure Telugu script (no Latin letters) and for which the rule-based G2P
    covers every character;
  * random sample (fixed seed) up to train_hours + val_hours;
  * ~1/6 of the audio goes to validation, grouped so identical transcripts
    never appear in both splits.
"""
import argparse
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2"))

from config import load_config  # noqa: E402

CONFIG = os.path.join(HERE, "config.yaml")


def prepare(cfg):
    from tqdm import tqdm

    from prep_common import (audio_info, decode_audio, load_hf_rows, split_by_text_groups,
                             stable_id, write_manifests, write_utt)
    from text.telugu import is_telugu_text, normalize_telugu, word_to_phones, words_for_alignment

    d = cfg["data"]
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    raw_dir = cfg["path"]["raw_path"]
    ds = load_hf_rows(d["hf_dataset"], d["hf_split"], d["audio_column"],
                      token=os.environ.get("HF_TOKEN"), cache_dir=d.get("cache_dir"))
    print(ds)
    gender_feat = ds.features.get("gender")
    names = getattr(gender_feat, "names", None)
    genders = [names[g] if names is not None else str(g) for g in ds["gender"]]
    texts = ds[d["text_column"]]

    stats = defaultdict(int)
    cands = []
    for i in tqdm(range(len(ds)), desc="scanning"):
        t = (texts[i] or "").strip()
        stats["rows"] += 1
        dur, orig_sr = audio_info(ds[i][d["audio_column"]])
        stats["hours_all"] += dur / 3600
        stats["hours_" + genders[i]] += dur / 3600
        if not (d["min_dur"] <= dur <= d["max_dur"]):
            stats["drop_duration"] += 1
            continue
        norm = normalize_telugu(t)
        if not norm or not is_telugu_text(norm):
            stats["drop_non_telugu_chars"] += 1
            continue
        lab = words_for_alignment(norm)
        try:
            for w in lab.split():
                word_to_phones(w)
        except ValueError:
            stats["drop_g2p_unsupported"] += 1
            continue
        if len(lab.split()) < d.get("min_words", 2):
            stats["drop_too_few_words"] += 1
            continue
        cands.append({"row_index": i, "speaker": genders[i], "dur": dur, "orig_text": t,
                      "norm_text": norm, "lab": lab, "orig_sr": orig_sr})

    hours = defaultdict(float)
    for c in cands:
        hours[c["speaker"]] += c["dur"] / 3600
    spk = d.get("speaker", "auto")
    if spk == "auto":
        spk = max(hours, key=hours.get)
    print("Usable hours per speaker:", dict(hours), "-> using", spk)
    pool = [c for c in cands if c["speaker"] == spk]
    rnd = random.Random(d["seed"])
    rnd.shuffle(pool)
    budget = (d["train_hours"] + d["val_hours"]) * 3600
    items, acc = [], 0.0
    for c in pool:
        if acc >= budget:
            break
        items.append(c)
        acc += c["dur"]
    speaker_name = "te_" + spk
    for c in items:
        c["speaker"] = speaker_name
        c["uid"] = "te_" + stable_id(d["hf_dataset"], c["row_index"])
    items = split_by_text_groups(items, d["val_hours"] / (d["train_hours"] + d["val_hours"]),
                                 d["seed"])

    print("Decoding and writing {} utterances ...".format(len(items)))
    for c in tqdm(items):
        wav, _ = decode_audio(ds[c["row_index"]][d["audio_column"]], sr)
        write_utt(raw_dir, c["speaker"], c["uid"], wav, sr, c["lab"])

    words = sorted(set(w for c in items for w in c["lab"].split()))
    entries = [(w, " ".join(word_to_phones(w))) for w in words]
    phone_counts = defaultdict(int)
    for c in items:
        for w in c["lab"].split():
            for p in word_to_phones(w):
                phone_counts[p] += 1
    h = lambda which: round(sum(c["dur"] for c in items if c["split"] == which) / 3600, 3)
    summary = {
        "dataset": d["hf_dataset"], "speaker_used": spk,
        "usable_hours_per_speaker": {k: round(v, 3) for k, v in hours.items()},
        "filters": {k: d[k] for k in ["min_dur", "max_dur"]},
        "counts": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()},
        "train_utts": sum(c["split"] == "train" for c in items),
        "val_utts": sum(c["split"] == "val" for c in items),
        "train_hours": h("train"), "val_hours": h("val"),
        "original_sample_rates": sorted(set(c["orig_sr"] for c in items)),
        "lexicon_words": len(words),
        "phone_counts": dict(sorted(phone_counts.items(), key=lambda kv: -kv[1])),
    }
    write_manifests(raw_dir, cfg["path"]["result_path"], items, summary, entries,
                    cfg["path"]["lexicon_path"])


def features(cfg_path, workers):
    import json
    import shutil

    from preprocessor.preprocessor import Preprocessor

    _, (pc, mc, tc) = load_config(cfg_path)
    summary = Preprocessor(pc).build_from_path(num_workers=workers)
    res = os.path.join(pc["path"]["result_path"], "others")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "features_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for fn in ["speakers.json", "stats.json", "skipped.tsv", "train.txt", "val.txt"]:
        shutil.copy(os.path.join(pc["path"]["preprocessed_path"], fn),
                    os.path.join(res, "preprocessed_" + fn))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "features"])
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    raw_cfg, _ = load_config(args.config)
    if args.stage == "prepare":
        prepare(raw_cfg)
    else:
        features(args.config, args.workers)

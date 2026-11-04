"""English (Svarah) data preparation and feature extraction.

Stages (run from the repo root):
  python English/preprocess.py prepare    # download Svarah, select ~6 h, split, write wav/lab + lexicon
  bash scripts/align.sh english            # Montreal Forced Aligner (separate conda env)
  python English/preprocess.py features   # mel, durations, pitch, energy -> data/English/preprocessed

Selection strategy (see config.yaml `data:` and the report):
  * Svarah has no speaker-id column. Speakers are recovered from the audio
    file name prefix when it looks like an id, else from the tuple of speaker
    metadata columns (gender, age group, L1, state, district, qualification,
    job, occupation). Which one was used is recorded in the summary.
  * keep utterances of min_dur..max_dur seconds whose transcript only uses
    plain English characters (drops tagged/noisy transcripts);
  * keep speakers with at least min_speaker_minutes, then take speakers in
    decreasing order of available audio until train_hours + val_hours is
    reached (fewer speakers with more data each -> better speaker embeddings);
  * per speaker, ~1/6 of the audio goes to validation, grouped so identical
    transcripts never appear in both splits.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fastspeech2"))

from config import load_config  # noqa: E402

CONFIG = os.path.join(HERE, "config.yaml")
_ALLOWED = re.compile(r"^[A-Za-z0-9 .,'?!;:\-\"()%&/’]+$")
_META = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district",
         "highest_qualification", "job_category", "occupation_domain"]


def speaker_keys(ds, audio_col):
    """Return (keys, method). Try file-name prefix first, then metadata tuple."""
    paths = [a.get("path") or "" for a in ds[audio_col]]
    prefixes = [re.split(r"[_\-]", os.path.basename(p))[0] if p else "" for p in paths]
    n_unique = len(set(prefixes))
    print("Example audio paths:", paths[:3], "| unique prefixes:", n_unique)
    if all(prefixes) and 30 <= n_unique <= 1000:
        return ["spk" + p for p in prefixes], "audio file name prefix"
    from prep_common import stable_id
    cols = {c: ds[c] for c in _META if c in ds.column_names}
    keys = []
    for i in range(len(ds)):
        keys.append("spk" + stable_id(*[cols[c][i] for c in cols]))
    print("Unique metadata-tuple speakers:", len(set(keys)))
    return keys, "metadata tuple ({})".format(", ".join(cols))


def prepare(cfg):
    from prep_common import (audio_info, decode_audio, load_hf_rows, split_by_text_groups,
                             stable_id, write_manifests, write_utt)
    from text.english import build_lexicon_entries, normalize_english, words_for_alignment

    d = cfg["data"]
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    raw_dir = cfg["path"]["raw_path"]
    ds = load_hf_rows(d["hf_dataset"], d["hf_split"], d["audio_column"],
                      token=os.environ.get("HF_TOKEN"), cache_dir=d.get("cache_dir"))
    print(ds)
    keys, spk_method = speaker_keys(ds, d["audio_column"])
    texts = ds[d["text_column"]]
    genders = ds["gender"] if "gender" in ds.column_names else [None] * len(ds)
    durs = ds["duration"] if "duration" in ds.column_names else None

    stats = defaultdict(int)
    cands = []
    for i in range(len(ds)):
        t = (texts[i] or "").strip()
        dur = float(durs[i]) if durs is not None else audio_info(ds[i][d["audio_column"]])[0]
        stats["rows"] += 1
        stats["hours_all"] += dur / 3600
        if not (d["min_dur"] <= dur <= d["max_dur"]):
            stats["drop_duration"] += 1
            continue
        if not t or not _ALLOWED.match(t):
            stats["drop_charset"] += 1
            continue
        lab = words_for_alignment(t)
        if len(lab.split()) < d.get("min_words", 2):
            stats["drop_too_few_words"] += 1
            continue
        if d.get("gender") and genders[i] and genders[i].lower() != d["gender"]:
            stats["drop_gender"] += 1
            continue
        cands.append({"row_index": i, "speaker": keys[i], "dur": dur, "orig_text": t,
                      "norm_text": normalize_english(t, for_alignment=True), "lab": lab,
                      "gender": genders[i]})
    stats["hours_after_filters"] = sum(c["dur"] for c in cands) / 3600

    by_spk = defaultdict(list)
    for c in cands:
        by_spk[c["speaker"]].append(c)
    spk_hours = {s: sum(c["dur"] for c in v) / 3600 for s, v in by_spk.items()}
    eligible = [s for s in spk_hours if spk_hours[s] * 60 >= d["min_speaker_minutes"]]
    eligible.sort(key=lambda s: (-spk_hours[s], s))
    budget = d["train_hours"] + d["val_hours"]
    chosen, acc = [], 0.0
    for s in eligible:
        if acc >= budget:
            break
        chosen.append(s)
        acc += spk_hours[s]
    items = [c for s in chosen for c in by_spk[s]]
    if acc > budget * 1.02:
        # trim the overshoot randomly from the last speaker's utterances
        import random
        rnd = random.Random(d["seed"])
        last = [c for c in items if c["speaker"] == chosen[-1]]
        rnd.shuffle(last)
        excess = (acc - budget) * 3600
        drop = set()
        for c in last:
            if excess <= 0 or len(last) - len(drop) <= 10:
                break
            drop.add(id(c))
            excess -= c["dur"]
        items = [c for c in items if id(c) not in drop]
    for c in items:
        c["uid"] = "en_" + stable_id(d["hf_dataset"], c["row_index"])
    items = split_by_text_groups(items, d["val_hours"] / budget, d["seed"])

    print("Decoding and writing {} utterances ...".format(len(items)))
    from tqdm import tqdm
    for c in tqdm(items):
        wav, orig_sr = decode_audio(ds[c["row_index"]][d["audio_column"]], sr)
        c["orig_sr"] = orig_sr
        write_utt(raw_dir, c["speaker"], c["uid"], wav, sr, c["lab"])

    words = [w for c in items for w in c["lab"].split()]
    entries, oov = build_lexicon_entries(words, cfg["path"]["base_lexicon_path"])
    h = lambda which: round(sum(c["dur"] for c in items if c["split"] == which) / 3600, 3)
    summary = {
        "dataset": d["hf_dataset"], "speaker_id_method": spk_method,
        "filters": {k: d[k] for k in ["min_dur", "max_dur", "min_speaker_minutes"]},
        "counts": dict(stats),
        "speakers_total_after_filters": len(by_spk), "speakers_eligible": len(eligible),
        "speakers_chosen": len(chosen),
        "chosen_speakers": {s: round(spk_hours[s] * 60, 1) for s in chosen},
        "train_utts": sum(c["split"] == "train" for c in items),
        "val_utts": sum(c["split"] == "val" for c in items),
        "train_hours": h("train"), "val_hours": h("val"),
        "original_sample_rates": sorted(set(c["orig_sr"] for c in items)),
        "lexicon_words": len(set(words)), "lexicon_oov_g2p": len(oov),
        "oov_examples": oov[:40],
    }
    write_manifests(raw_dir, cfg["path"]["result_path"], items, summary, entries,
                    cfg["path"]["lexicon_path"])


def features(cfg_path, workers):
    from preprocessor.preprocessor import Preprocessor
    import json
    import shutil

    _, (pc, mc, tc) = load_config(cfg_path)
    summary = Preprocessor(pc).build_from_path(num_workers=workers)
    res = pc["path"]["result_path"]
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

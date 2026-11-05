"""Shared helpers for dataset download, subsampling and train/val split (new).

Pipeline per language (called from <Language>/preprocess.py):
  1. load the Hugging Face dataset with audio decoding disabled (raw bytes),
  2. compute duration and normalised text for every row, apply filters,
  3. choose speakers / utterances up to the train+val hour budget,
  4. split into train and val by *transcript groups* (identical normalised
     transcripts always land in the same split, so no sentence is shared),
  5. write 22.05 kHz mono 16-bit wavs + .lab transcripts in the MFA layout
     raw/<speaker>/<id>.wav|.lab, plus split.tsv, selection.tsv and the MFA
     lexicon.
"""
import csv
import hashlib
import io
import json
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf


def load_hf_rows(name, split, audio_col, token=None, cache_dir=None):
    from datasets import Audio, load_dataset

    ds = load_dataset(name, split=split, token=token, cache_dir=cache_dir)
    ds = ds.cast_column(audio_col, Audio(decode=False))
    return ds


def audio_info(audio_field):
    """(duration_s, sample_rate) without decoding the full file."""
    b = audio_field.get("bytes")
    if b is None:
        info = sf.info(audio_field["path"])
    else:
        info = sf.info(io.BytesIO(b))
    return info.frames / info.samplerate, info.samplerate


def decode_audio(audio_field, target_sr):
    import librosa

    b = audio_field.get("bytes")
    data, sr = sf.read(io.BytesIO(b) if b is not None else audio_field["path"], dtype="float32",
                       always_2d=True)
    wav = data.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return wav, sr


def write_utt(raw_dir, speaker, uid, wav, sr, lab_text, max_wav_value=32768.0):
    """Peak-normalise (as upstream ljspeech.prepare_align did) and write wav + lab."""
    d = os.path.join(raw_dir, speaker)
    os.makedirs(d, exist_ok=True)
    peak = np.max(np.abs(wav)) + 1e-8
    wav = wav / peak * (max_wav_value - 1) * 0.97
    sf.write(os.path.join(d, uid + ".wav"), wav.astype(np.int16), sr, subtype="PCM_16")
    with open(os.path.join(d, uid + ".lab"), "w", encoding="utf-8") as f:
        f.write(lab_text + "\n")


def split_by_text_groups(items, val_fraction, seed, key_fn=lambda it: it["speaker"]):
    """Assign 'train'/'val' so that val gets ~val_fraction of the duration *within
    each stratum* (speaker), and identical normalised transcripts never cross
    the split. items: list of dicts with 'dur', 'norm_text', 'speaker'."""
    rng = random.Random(seed)
    # transcript groups are global (a sentence read by two speakers must not
    # appear in both splits)
    group_of = {}
    groups = defaultdict(list)
    for i, it in enumerate(items):
        g = it["norm_text"].lower()
        groups[g].append(i)
        group_of[i] = g
    by_stratum = defaultdict(list)
    for g, idxs in groups.items():
        # stratum = speaker of first member; most groups have one member
        by_stratum[key_fn(items[idxs[0]])].append(g)
    assign = {}
    for stratum in sorted(by_stratum):
        gs = sorted(by_stratum[stratum])
        rng.shuffle(gs)
        total = sum(items[i]["dur"] for g in gs for i in groups[g])
        target, acc = total * val_fraction, 0.0
        for g in gs:
            gd = sum(items[i]["dur"] for i in groups[g])
            if acc < target and acc + gd <= target * 1.5 + 1e-9:
                assign[g] = "val"
                acc += gd
            else:
                assign[g] = "train"
    for i, it in enumerate(items):
        it["split"] = assign[group_of[i]]
    # sanity: no transcript overlap
    tr = set(it["norm_text"].lower() for it in items if it["split"] == "train")
    va = set(it["norm_text"].lower() for it in items if it["split"] == "val")
    assert not (tr & va), "transcript overlap between train and val"
    return items


def stable_id(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


def write_manifests(raw_dir, results_dir, items, summary, lexicon_entries, lexicon_path):
    os.makedirs(raw_dir, exist_ok=True)
    results_dir = os.path.join(results_dir, "others")  # extras beside the required files
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(raw_dir, "split.tsv"), "w", encoding="utf-8") as f:
        f.write("basename\tspeaker\tsplit\n")
        for it in items:
            f.write("{}\t{}\t{}\n".format(it["uid"], it["speaker"], it["split"]))
    # selection.tsv is small and committed to Results/ for reproducibility
    cols = ["uid", "speaker", "split", "dur", "row_index", "orig_text", "norm_text"]
    for path in [os.path.join(raw_dir, "selection.tsv"),
                 os.path.join(results_dir, "data_selection.tsv")]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols)
            for it in items:
                w.writerow([it["uid"], it["speaker"], it["split"], "%.3f" % it["dur"],
                            it["row_index"], it["orig_text"], it["norm_text"]])
    with open(os.path.join(results_dir, "data_selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for w_, pron in lexicon_entries:
            f.write("{}\t{}\n".format(w_, pron))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

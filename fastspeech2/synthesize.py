"""Inference and variance-adaptor plots.

Replaces ming024 synthesize.py. MODIFIED / NEW:
  * language frontends: text/english.py (normalisation + g2p_en with POS-based
    heteronym choice) and text/telugu.py (normalisation + rule-based G2P).
  * reads a TSV of test sentences (id, text) and keeps the text exactly as
    given; normalisation happens inside the frontend and is logged.
  * per utterance it saves: wav, predicted duration / pitch (F0, Hz) /
    energy plots, an overview figure (mel + F0 + energy), and a JSON with
    every predicted value plus the phone sequence and G2P source per word.
  * speaker selection by name or "auto" (speaker with most training audio).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch  # import before matplotlib: avoids a c10.dll init failure on Windows
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.io import wavfile  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.model import get_model, get_vocoder, vocoder_infer  # noqa: E402


def frontend(lang):
    if lang == "en":
        from text.english import text_to_phones
    elif lang == "te":
        from text.telugu import text_to_phones
    else:
        raise ValueError(lang)
    return text_to_phones


def pick_speaker(preprocess_config, name):
    pp = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(pp, "speakers.json")) as f:
        speaker_map = json.load(f)
    if name and name != "auto":
        return name, speaker_map[name]
    counts_file = os.path.join(pp, "train_speaker_counts.json")
    if os.path.exists(counts_file):
        with open(counts_file) as f:
            counts = json.load(f)
    else:
        counts = {}
        with open(os.path.join(pp, "train.txt"), encoding="utf-8") as f:
            for line in f:
                spk = line.split("|")[1]
                counts[spk] = counts.get(spk, 0) + 1
    best = max(sorted(counts), key=counts.get)
    return best, speaker_map[best]


def _word_spans(word_phones, romanize=False):
    """(label, first_phone, end_phone, source) per word. For Telugu the label is
    the phone string, because matplotlib cannot shape Telugu script (vowel signs
    and conjuncts render in the wrong order)."""
    spans, i = [], 0
    for w, ph, src in word_phones:
        label = "".join(ph) if romanize else w
        spans.append((label, i, i + len(ph), src))
        i += len(ph)
    return spans


def _decorate_words(ax, spans, edges_x, fp, y=1.01):
    for w, a, b, src in spans:
        if src == "pause":
            continue
        x0, x1 = edges_x[a], edges_x[b]
        ax.axvline(x0, color="0.6", lw=0.6, ls="--")
        ax.text((x0 + x1) / 2, y, w.strip("{}"), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, rotation=30, fontproperties=fp)


def make_plots(uid, text, rec, out_root, lang, frame_s):
    fp = None
    phones = rec["phones"]
    dur = np.array(rec["duration_frames"], dtype=float)
    f0 = np.array(rec["pitch_hz"])
    en = np.array(rec["energy"])
    spans = _word_spans(rec["word_phones"], romanize=(lang == "te"))
    n = len(phones)
    idx = np.arange(n)
    edges_idx = np.arange(n + 1) - 0.5
    if lang == "te":
        title = "{} (phones per word): {}".format(uid, " ".join(
            sp[0] if sp[3] != "pause" else "," for sp in spans))
    else:
        title = "{}: {}".format(uid, text)

    def _title(ax):
        ax.set_title(title, fontsize=8, fontproperties=fp, pad=28)

    width = min(20, max(8, n * 0.2))

    # Duration: one bar per phone
    fig, ax = plt.subplots(figsize=(width, 3.6))
    colors = ["0.75" if p in ("sp", "spn", "sil") else "tab:blue" for p in phones]
    ax.bar(idx, dur * frame_s * 1000, color=colors)
    ax.set_xticks(idx)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("predicted duration (ms)")
    sec = ax.secondary_yaxis("right", functions=(lambda ms: ms / (frame_s * 1000),
                                                 lambda fr: fr * frame_s * 1000))
    sec.set_ylabel("frames")
    _decorate_words(ax, spans, edges_idx, fp)
    _title(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "duration", uid + ".png"), dpi=150)
    plt.close(fig)

    # Pitch / energy on a time axis: phone-level predictions held over each phone
    t_edges = np.concatenate([[0], np.cumsum(dur)]) * frame_s
    centers = (t_edges[:-1] + t_edges[1:]) / 2
    for key, vals, ylabel, color in [("pitch", f0, "predicted F0 (Hz)", "tab:red"),
                                     ("energy", en, "predicted energy (L2 norm of STFT frame)",
                                      "tab:purple")]:
        fig, ax = plt.subplots(figsize=(width, 3.6))
        ax.stairs(vals, t_edges, color=color, lw=1.8, baseline=None)
        ax.plot(centers, vals, "o", color=color, ms=2.5)
        for p, c, d in zip(phones, centers, dur):
            if d > 0:
                ax.text(c, 0.0, p, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6, rotation=90, color="0.3")
        ax.set_xlim(0, t_edges[-1] if t_edges[-1] > 0 else 1)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = max(0.15 * (hi - lo), 0.05 * abs(hi) + 1e-3)
        ax.set_ylim(lo - 1.6 * pad, hi + pad)  # extra room at the bottom for phone labels
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        _decorate_words(ax, spans, t_edges, fp)
        _title(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, key, uid + ".png"), dpi=150)
        plt.close(fig)

    # Overview: mel with F0 and energy overlays
    mel = np.array(rec["_mel"])
    fig, ax = plt.subplots(figsize=(width, 3.4))
    ax.imshow(mel, origin="lower", aspect="auto", extent=[0, mel.shape[1] * frame_s, 0, mel.shape[0]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax2 = ax.twinx()
    ax2.stairs(f0, t_edges, color="tomato", lw=1.5, baseline=None, label="F0 (Hz)")
    ax2.set_ylabel("F0 (Hz)", color="tomato")
    ax3 = ax.twinx()
    ax3.stairs(en, t_edges, color="white", lw=1.0, baseline=None, label="energy")
    ax3.set_yticks([])
    _decorate_words(ax, spans, t_edges, fp)
    _title(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "overview", uid + ".png"), dpi=150)
    plt.close(fig)


def synthesize_sentences(config_path, ckpt, sentences_tsv, out_dir, speaker="auto",
                         controls=(1.0, 1.0, 1.0), device=None):
    raw_cfg, configs = load_config(config_path)
    preprocess_config, model_config, train_config = configs
    lang = preprocess_config["language"]
    if not os.path.exists(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")):
        # no training data on this machine: use the metadata exported next to the checkpoint
        preprocess_config["path"]["preprocessed_path"] = train_config["path"]["ckpt_path"]
        print("Using stats/speakers from", train_config["path"]["ckpt_path"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_model(ckpt, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    to_phones = frontend(lang)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    p_min, p_max, p_mean, p_std = stats["pitch"]
    e_min, e_max, e_mean, e_std = stats["energy"]
    spk_name, spk_id = pick_speaker(preprocess_config, speaker)
    print("Speaker:", spk_name, "id", spk_id)

    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    frame_s = hop / sr
    wav_dir = os.path.join(out_dir, "test_generated_audios")
    plot_root = os.path.join(out_dir, "variance_adaptor_plots")
    for d in [wav_dir] + [os.path.join(plot_root, k) for k in
                          ["duration", "pitch", "energy", "overview"]]:
        os.makedirs(d, exist_ok=True)

    sentences = []
    with open(sentences_tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and not row[0].startswith("#"):
                sentences.append((row[0], row[1]))

    summary = []
    for uid, text in sentences:
        phones, norm, word_phones = to_phones(text, verbose=False)
        seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", [], lang))
        print("\n[{}] {}\n  normalised: {}\n  phones: {}".format(uid, text, norm, " ".join(phones)))
        speakers = torch.tensor([spk_id], dtype=torch.long, device=device)
        texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
        src_lens = torch.tensor([len(seq)], device=device)
        with torch.no_grad():
            out = model(speakers, texts, src_lens, len(seq),
                        p_control=controls[0], e_control=controls[1], d_control=controls[2])
        mel_post = out[1][0, : out[9][0]].float()
        p_pred = out[2][0, : len(seq)].float().cpu().numpy()
        e_pred = out[3][0, : len(seq)].float().cpu().numpy()
        log_d = out[4][0, : len(seq)].float().cpu().numpy()
        d_round = out[5][0, : len(seq)].float().cpu().numpy().astype(int)

        wav = vocoder_infer(mel_post.transpose(0, 1).unsqueeze(0), vocoder, model_config,
                            preprocess_config, lengths=[int(out[9][0]) * hop])[0]
        wavfile.write(os.path.join(wav_dir, uid + ".wav"), sr, wav)

        rec = {
            "id": uid, "text": text, "normalized_text": norm, "speaker": spk_name,
            "phones": phones,
            "word_phones": [(w, p, s) for w, p, s in word_phones],
            "duration_frames": d_round.tolist(),
            "duration_ms": (d_round * frame_s * 1000).round(1).tolist(),
            "log_duration_pred": log_d.round(4).tolist(),
            "pitch_norm": p_pred.round(4).tolist(),
            "pitch_hz": (p_pred * p_std + p_mean).round(2).tolist(),
            "energy_norm": e_pred.round(4).tolist(),
            "energy": (e_pred * e_std + e_mean).round(4).tolist(),
            "total_frames": int(out[9][0]),
            "total_seconds": round(int(out[9][0]) * frame_s, 3),
            "checkpoint": os.path.basename(ckpt) if ckpt else None,
            "controls": list(controls),
        }
        rec["_mel"] = mel_post.transpose(0, 1).cpu().numpy()
        make_plots(uid, text, rec, plot_root, lang, frame_s)
        del rec["_mel"]
        with open(os.path.join(plot_root, uid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        summary.append({
            "id": uid, "seconds": rec["total_seconds"], "n_phones": len(phones),
            "mean_f0_hz": round(float(np.mean(rec["pitch_hz"])), 1),
            "min_f0_hz": round(float(np.min(rec["pitch_hz"])), 1),
            "max_f0_hz": round(float(np.max(rec["pitch_hz"])), 1),
            "zero_duration_phones": int(np.sum(d_round == 0)),
            "normalized_text": norm,
        })
    with open(os.path.join(out_dir, "test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("\nWrote", wav_dir, "and", plot_root)
    return summary


def cli(default_config=None, default_sentences=None, default_out=None, default_ckpt_dir=None):
    parser = argparse.ArgumentParser(description="FastSpeech 2 inference + variance plots")
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint (default: best.pth.tar, else latest.pth.tar, else "
                             "*_infer_fp16.pth.tar in the language's checkpoints/ folder)")
    parser.add_argument("--sentences", default=default_sentences)
    parser.add_argument("--text", default=None, help="synthesize one sentence instead (id 'custom')")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--speaker", default="auto")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        _, (pc, mc, tc) = load_config(args.config)
        d = tc["path"]["ckpt_path"]
        import glob
        for cand in [os.path.join(d, "best.pth.tar"), os.path.join(d, "latest.pth.tar")] + \
                sorted(glob.glob(os.path.join(d, "*_infer_fp16.pth.tar"))):
            if os.path.exists(cand):
                ckpt = cand
                break
    if ckpt is None:
        sys.exit("No checkpoint found; pass --ckpt")
    print("Checkpoint:", ckpt)

    sentences = args.sentences
    if args.text:
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
        tmp.write("custom\t{}\n".format(args.text))
        tmp.close()
        sentences = tmp.name
        args.out = os.path.join(args.out, "custom")
    synthesize_sentences(args.config, ckpt, sentences, args.out, args.speaker,
                         (args.pitch_control, args.energy_control, args.duration_control),
                         args.device)


if __name__ == "__main__":
    cli()

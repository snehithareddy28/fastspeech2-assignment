"""Fallback aligner: CTC forced alignment with a multilingual wav2vec2 model.

Used only if Montreal Forced Aligner cannot align the Telugu data
(`bash scripts/align.sh telugu-ctc`). Writes TextGrids in the same layout as
MFA (data/<Lang>/TextGrid/<speaker>/<id>.TextGrid, tiers "words" and
"phones", silences as empty intervals), so `preprocess.py features` runs
unchanged.

Method
  * acoustic model: MMS-300m fine-tuned for forced alignment over romanised
    characters (HF: MahmoudAshraf/mms-300m-1130-forced-aligner, the
    1130-language MMS aligner; vocabulary a-z and apostrophe, 20 ms frames);
  * each Telugu phone from text/telugu.py is written as its romanisation
    (tx -> t, aa -> aa, sh -> sh, ...), the utterance becomes one token
    sequence, and the best CTC path is found with a Viterbi pass
    (implemented here, so no dependency on torchaudio's forced_align);
  * a phone spans the frames of its letters; gaps inside a word go to the
    preceding phone; a gap of >= min_pause seconds between words becomes a
    silence interval, shorter gaps are split at the midpoint.

    python scripts/ctc_align.py --config Telugu/config.yaml [--limit N] [--device cuda]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

from config import load_config  # noqa: E402

MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# romanisation of the phone set in text/telugu.py (letters the CTC model knows)
ROMAN = {
    "a": "a", "aa": "aa", "i": "i", "ii": "ii", "u": "u", "uu": "uu", "e": "e", "ee": "ee",
    "ai": "ai", "o": "o", "oo": "oo", "au": "au",
    "k": "k", "kh": "kh", "g": "g", "gh": "gh", "ng": "n", "c": "c", "ch": "ch", "j": "j",
    "jh": "jh", "ny": "n", "tx": "t", "txh": "th", "dx": "d", "dxh": "dh", "nx": "n",
    "t": "t", "th": "th", "d": "d", "dh": "dh", "n": "n", "p": "p", "ph": "ph", "b": "b",
    "bh": "bh", "m": "m", "y": "y", "r": "r", "rx": "r", "l": "l", "lx": "l", "zh": "l",
    "v": "v", "sh": "sh", "sx": "sh", "s": "s", "h": "h", "ts": "ts", "dz": "dz",
}


def viterbi_ctc(logp, tokens, blank=0):
    """Best CTC path. logp: (T, V) log-probs; tokens: list[int] without blanks.
    Returns (T,) array of indices into the blank-extended sequence, or None."""
    T = logp.shape[0]
    ext = [blank]
    for tk in tokens:
        ext += [tk, blank]
    S = len(ext)
    ext_t = torch.tensor(ext, dtype=torch.long)
    emit = logp[:, ext_t]  # (T, S)
    neg = torch.tensor(-1e30, dtype=logp.dtype)
    # transitions from s-2 allowed only onto a non-blank that differs from ext[s-2]
    skip_ok = torch.zeros(S, dtype=torch.bool)
    for s in range(2, S):
        skip_ok[s] = ext[s] != blank and ext[s] != ext[s - 2]
    alpha = torch.full((S,), -1e30, dtype=logp.dtype)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]
    back = torch.zeros((T, S), dtype=torch.int8)
    for t in range(1, T):
        stay = alpha
        prev1 = torch.cat([neg.view(1), alpha[:-1]])
        prev2 = torch.where(skip_ok, torch.cat([neg.view(1).repeat(2), alpha[:-2]]), neg)
        cand = torch.stack([stay, prev1, prev2])  # (3, S)
        best, arg = cand.max(0)
        alpha = best + emit[t]
        back[t] = arg.to(torch.int8)
    end_states = [S - 1, S - 2] if S > 1 else [0]
    s = max(end_states, key=lambda k: alpha[k].item())
    if alpha[s].item() < -1e29:
        return None
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    if path[0] > 1:
        return None
    return path


def align_utterance(wav16, words, model, vocab, device, min_pause):
    """words: list of (word, [phones]). Returns (duration_s, word_items, phone_items)."""
    x = torch.from_numpy(wav16).float()
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device)).logits[0].float().cpu()
    logp = torch.log_softmax(logits, dim=-1)
    T = logp.shape[0]
    dur = len(wav16) / 16000.0
    fdur = dur / T

    tokens, owner = [], []  # owner: (word_idx, phone_idx) per token
    for wi, (_, phones) in enumerate(words):
        for pi, p in enumerate(phones):
            for ch in ROMAN[p]:
                tokens.append(vocab[ch])
                owner.append((wi, pi))
    path = viterbi_ctc(logp, tokens, blank=vocab["<blank>"])
    if path is None:
        return None

    # frame span of every token (states 2j+1)
    tok_start = np.full(len(tokens), -1)
    tok_end = np.full(len(tokens), -1)
    for t, s in enumerate(path):
        if s % 2 == 1:
            j = s // 2
            if tok_start[j] < 0:
                tok_start[j] = t
            tok_end[j] = t + 1
    if (tok_start < 0).any():
        return None

    # phone spans
    spans = {}
    for j, key in enumerate(owner):
        a, b = spans.get(key, (tok_start[j], tok_end[j]))
        spans[key] = (min(a, tok_start[j]), max(b, tok_end[j]))
    phone_items, word_items = [], []
    word_bounds = []
    for wi, (w, phones) in enumerate(words):
        ws = [list(spans[(wi, pi)]) for pi in range(len(phones))]
        for k in range(len(ws) - 1):  # close gaps inside the word
            ws[k][1] = ws[k + 1][0]
        word_bounds.append(ws)
    for wi in range(len(word_bounds) - 1):  # gaps between words
        a_end = word_bounds[wi][-1][1]
        b_start = word_bounds[wi + 1][0][0]
        if (b_start - a_end) * fdur < min_pause:
            mid = (a_end + b_start) // 2
            word_bounds[wi][-1][1] = mid
            word_bounds[wi + 1][0][0] = mid
    cur = 0
    for wi, (w, phones) in enumerate(words):
        ws = word_bounds[wi]
        if ws[0][0] > cur:
            phone_items.append((cur * fdur, ws[0][0] * fdur, ""))
            word_items.append((cur * fdur, ws[0][0] * fdur, ""))
        for (a, b), p in zip(ws, phones):
            phone_items.append((a * fdur, b * fdur, p))
        word_items.append((ws[0][0] * fdur, ws[-1][1] * fdur, w))
        cur = ws[-1][1]
    if cur < T:
        phone_items.append((cur * fdur, dur, ""))
        word_items.append((cur * fdur, dur, ""))
    return dur, word_items, phone_items


def write_textgrid(path, dur, word_items, phone_items):
    import tgt

    tg = tgt.TextGrid()
    for name, items in [("words", word_items), ("phones", phone_items)]:
        tier = tgt.IntervalTier(0, dur, name)
        for a, b, lab in items:
            if b - a > 1e-6:
                tier.add_interval(tgt.Interval(a, min(b, dur), lab))
        tg.add_tier(tier)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tgt.io.write_to_file(tg, path, format="long")


def main():
    import librosa
    from tqdm import tqdm
    from transformers import AutoModelForCTC

    from text.telugu import word_to_phones

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "Telugu", "config.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_pause", type=float, default=0.15)
    args = ap.parse_args()
    _, (pc, _, _) = load_config(args.config)
    raw, out = pc["path"]["raw_path"], pc["path"]["textgrid_path"]

    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(args.device).eval()
    from transformers import AutoTokenizer
    vocab = AutoTokenizer.from_pretrained(MODEL_ID).get_vocab()

    items = []
    for spk in sorted(os.listdir(raw)):
        d = os.path.join(raw, spk)
        if os.path.isdir(d):
            items += [(spk, f[:-4]) for f in sorted(os.listdir(d)) if f.endswith(".wav")]
    items = items[: args.limit] if args.limit else items
    ok, failed = 0, []
    for spk, uid in tqdm(items):
        with open(os.path.join(raw, spk, uid + ".lab"), encoding="utf-8") as f:
            words = [(w, word_to_phones(w)) for w in f.read().split()]
        wav, _ = librosa.load(os.path.join(raw, spk, uid + ".wav"), sr=16000)
        res = align_utterance(wav, words, model, vocab, args.device, args.min_pause)
        if res is None:
            failed.append(uid)
            continue
        write_textgrid(os.path.join(out, spk, uid + ".TextGrid"), *res)
        ok += 1
    print("CTC-aligned {} / {} utterances; failed: {}".format(ok, len(items), len(failed)))


if __name__ == "__main__":
    main()

"""Feature extraction: mel, phone durations, phone-level pitch and energy.

Based on ming024 preprocessor/preprocessor.py. MODIFIED:
  * train/val split comes from a fixed split file written at data-selection
    time (raw_path/split.tsv) instead of a random val_size cut, so the split is
    reproducible and documented, and identical transcripts never straddle it.
  * TextGrid phones tier found by name suffix (MFA 2/3 naming), and MFA 2/3's
    empty-label silence intervals are mapped to "sp" (upstream only knew
    sil/sp/spn and would emit "" as a phone).
  * utterances with phones outside the language's symbol table are skipped and
    listed in skipped.tsv instead of crashing training later.
  * multiprocessing over utterances; librosa.load called with explicit sr.
  * pyworld F0 range configurable (f0_floor/f0_ceil) for each speaker set.
  * bug fix: upstream used `pitch`/`energy` from the previous utterance when a
    TextGrid was missing.
  * writes a summary.json (hours, counts, skipped) for the report.
"""
import json
import os
from multiprocessing import Pool

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text.symbols import get_symbols

_WORKER = {}


def _init_worker(config):
    _WORKER["p"] = Preprocessor(config)


def _work(item):
    speaker, basename = item
    try:
        return item, _WORKER["p"].process_utterance(speaker, basename)
    except Exception as e:  # keep going, report later
        return item, ("error", repr(e))


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.tg_dir = config["path"]["textgrid_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.lang = config["language"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        pcfg = config["preprocessing"]["pitch"]
        self.f0_floor = pcfg.get("f0_floor", 71.0)
        self.f0_ceil = pcfg.get("f0_ceil", 800.0)
        self.valid_phones = set(s[1:] for s in get_symbols(self.lang) if s.startswith("@"))

        assert pcfg["feature"] in ["phoneme_level", "frame_level"]
        assert config["preprocessing"]["energy"]["feature"] in ["phoneme_level", "frame_level"]
        self.pitch_phoneme_averaging = pcfg["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        self.pitch_normalization = pcfg["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def read_split(self):
        split = {}
        with open(os.path.join(self.in_dir, "split.tsv"), encoding="utf-8") as f:
            next(f)
            for line in f:
                basename, speaker, which = line.rstrip("\n").split("\t")[:3]
                split[(speaker, basename)] = which
        return split

    def build_from_path(self, num_workers=4):
        import shutil
        for d in ["mel", "pitch", "energy", "duration"]:
            # start clean: normalize() rewrites files in place, so stale files
            # from a previous run would be normalised twice
            shutil.rmtree(os.path.join(self.out_dir, d), ignore_errors=True)
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)

        split = self.read_split()
        items, missing_tg = [], []
        for (speaker, basename) in sorted(split):
            tg = os.path.join(self.tg_dir, speaker, "{}.TextGrid".format(basename))
            if os.path.exists(tg):
                items.append((speaker, basename))
            else:
                missing_tg.append((speaker, basename))
        print("Utterances in split: {}, with TextGrid: {}, missing TextGrid: {}".format(
            len(split), len(items), len(missing_tg)))

        speakers = sorted(set(s for s, _ in split))
        speaker_map = {s: i for i, s in enumerate(speakers)}

        print("Processing Data ...")
        out = {"train": [], "val": []}
        n_frames = {"train": 0, "val": 0}
        skipped = [(s, b, "no TextGrid (alignment failed)") for s, b in missing_tg]
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        with Pool(num_workers, initializer=_init_worker, initargs=(self.config,)) as pool:
            for (speaker, basename), ret in tqdm(
                pool.imap_unordered(_work, items, chunksize=8), total=len(items)
            ):
                if ret is None or ret[0] == "error" or ret[0] == "skip":
                    reason = "unknown" if ret is None else ret[1]
                    skipped.append((speaker, basename, reason))
                    continue
                info, pitch, energy, n = ret
                which = split[(speaker, basename)]
                out[which].append(info)
                n_frames[which] += n
                # statistics only from training data
                if which == "train":
                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        else:
            pitch_mean, pitch_std = 0, 1
        if self.energy_normalization:
            energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        else:
            energy_mean, energy_std = 0, 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std)

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speaker_map))
        stats = {
            "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
            "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
        }
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            f.write(json.dumps(stats))

        for which, fname in [("train", "train.txt"), ("val", "val.txt")]:
            with open(os.path.join(self.out_dir, fname), "w", encoding="utf-8") as f:
                for m in sorted(out[which]):
                    f.write(m + "\n")
        with open(os.path.join(self.out_dir, "skipped.tsv"), "w", encoding="utf-8") as f:
            f.write("speaker\tbasename\treason\n")
            for s, b, r in skipped:
                f.write("{}\t{}\t{}\n".format(s, b, r))

        to_h = lambda n: n * self.hop_length / self.sampling_rate / 3600
        summary = {
            "train_utts": len(out["train"]), "val_utts": len(out["val"]),
            "train_hours_trimmed": round(to_h(n_frames["train"]), 3),
            "val_hours_trimmed": round(to_h(n_frames["val"]), 3),
            "skipped": len(skipped), "speakers": len(speaker_map), "stats": stats,
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return summary

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(self.tg_dir, speaker, "{}.TextGrid".format(basename))

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        tier = None
        for t in textgrid.tiers:
            if t.name == "phones" or t.name.endswith("phones"):
                tier = t
                break
        phone, duration, start, end = self.get_alignment(tier)
        if start >= end or len(phone) == 0:
            return ("skip", "empty alignment")
        bad = sorted(set(p for p in phone if p not in self.valid_phones))
        if bad:
            return ("skip", "phones not in symbol table: " + " ".join(bad))
        text = "{" + " ".join(phone) + "}"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)

        # Read raw text (the original transcript, kept for logging)
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return ("skip", "no voiced frames")

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        if mel_spectrogram.shape[1] < sum(duration) or len(pitch) < sum(duration):
            return ("skip", "alignment longer than audio")

        if self.pitch_phoneme_averaging:
            # perform linear interpolation over unvoiced frames
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        np.save(os.path.join(self.out_dir, "duration",
                             "{}-duration-{}.npy".format(speaker, basename)), duration)
        np.save(os.path.join(self.out_dir, "pitch",
                             "{}-pitch-{}.npy".format(speaker, basename)), pitch)
        np.save(os.path.join(self.out_dir, "energy",
                             "{}-energy-{}.npy".format(speaker, basename)), energy)
        np.save(os.path.join(self.out_dir, "mel",
                             "{}-mel-{}.npy".format(speaker, basename)), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text.strip()

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # MODIFIED: MFA>=2 marks silence with an empty label
                phones.append("sp" if p in ("", "sil") else p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

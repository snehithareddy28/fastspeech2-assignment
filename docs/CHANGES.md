# Changes to ming024/FastSpeech2

Upstream: https://github.com/ming024/FastSpeech2, commit `d4e79eb` (2021-07-08), vendored
unmodified into `fastspeech2/` in the first commit of this repository. Every later change is
visible with:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD -- fastspeech2/
```

The model architecture (encoder, variance adaptor, length regulator, decoder, PostNet,
losses, LR schedule) is unchanged. Changes are in data handling, text frontends,
compatibility and training infrastructure.

## 1. Compatibility with current libraries

| File | Change | Why |
|---|---|---|
| `audio/stft.py` | `pad_center(..., size=)`, `librosa.filters.mel(sr=, n_fft=, n_mels=, fmin=, fmax=)` | librosa >= 0.10 made these keyword-only |
| `audio/stft.py` | removed hard-coded `.cuda()` in `STFT.transform` | preprocessing crashed without a GPU |
| `audio/audio_processing.py` | `pad_center(..., size=)` | librosa >= 0.10 |
| `utils/model.py` | `torch.load(..., weights_only=False, map_location=...)` | torch >= 2.6 default rejects these checkpoints |
| `utils/model.py` | `model.requires_grad_(False)` | upstream assigned an attribute (no effect) |
| `utils/model.py` | HiFi-GAN path relative to the package, auto-unzip | upstream required running from the repo root and manual unzip |

## 2. Bug fixes

| File | Bug |
|---|---|
| `model/modules.py` | energy predictor received `p_control` instead of `e_control`, so `--energy_control` did nothing |
| `preprocessor/preprocessor.py` | MFA >= 2 writes silences as empty intervals; upstream read TextGrids without empty intervals, which drops silence frames from the duration sum, and would emit `""` as a phone otherwise. Now read with `include_empty_intervals=True` and mapped to `sp` |
| `preprocessor/preprocessor.py` | stale `pitch`/`energy` from the previous utterance used when a TextGrid was missing |
| `preprocessor/preprocessor.py` | re-running normalised feature files twice; output dirs are now cleared first |
| `text/__init__.py` | unknown phones were silently dropped (input/duration length mismatch); now raise |
| `utils/tools.py` | energy targets were passed to the loss as float64 (pitch was cast to float32); on CUDA the backward pass fails with "Found dtype Double but expected Float" |
| `preprocessor/preprocessor.py` | worker processes now use the "spawn" start method: with Linux "fork" they inherited an initialised OpenMP/PyTorch thread pool and deadlocked on the first STFT |
| `train.py` | `torch.cuda.set_device` gets an explicit index (torch 2.11 rejects a bare "cuda" device) |

## 3. Multi-language support (Telugu)

| File | Change |
|---|---|
| `text/symbols.py`, `text/__init__.py` | symbol tables per language (`en`: upstream ARPAbet table without pinyin; `te`: Telugu phone set). `text_to_sequence(text, cleaners, lang)` |
| `transformer/Models.py` | encoder vocabulary size from the language's table (`model_config["symbol_set"]`) |
| `text/telugu.py` (new) | Telugu number/time/ordinal normalisation and rule-based G2P (inherent vowel, virama, matras, homorganic anusvara, visarga) |
| `text/english.py` (new) | English normalisation (times, AM/PM, dates, years, ordinals) and G2P with g2p_en on the whole sentence, so its POS-based homograph table resolves heteronyms. Upstream looked words up one at a time in a lexicon (first pronunciation always wins) |
| `text/pinyin.py`, `preprocessor/{ljspeech,aishell3,libritts}.py`, `prepare_align.py`, `preprocess.py` | removed (Mandarin / other datasets, replaced by `<Language>/preprocess.py`) |

## 4. Data pipeline

| File | Change |
|---|---|
| `prep_common.py` (new) | Hugging Face loading without audio decoding, resampling to 22.05 kHz, transcript-grouped train/val split (no shared sentences), manifests |
| `preprocessor/preprocessor.py` | split taken from `raw/split.tsv` (fixed at selection time) instead of a random `val_size` cut; pitch/energy statistics from training data only; multiprocessing; configurable pyworld F0 range; utterances with phones outside the symbol table are skipped and listed; `summary.json` |
| `dataset.py` | passes the language to `text_to_sequence` |

## 5. Training and inference

| File | Change |
|---|---|
| `config.py` (new) | one `config.yaml` per language, split into upstream's three dicts; paths relative to repo root |
| `train.py` | resumable (`latest.pth.tar` with optimizer, step, epoch, RNG; `best.pth.tar` by validation loss; numbered checkpoints with retention), CSV loss logs, `--max_hours` wall-clock budget, `--device`, DataLoader workers, optional fp16 AMP, non-finite loss guard, no `nn.DataParallel`, no `quit()` |
| `evaluate.py` | returns the loss values for logging; cached validation loader |
| `synthesize.py` | rewritten: test-sentence TSV, language frontends, per-utterance duration/pitch/energy/overview plots and JSON of all predictions, speaker selection, runs from exported checkpoint metadata when the training data is absent |

## 6. Alignment fallback (new)

`scripts/ctc_align.py`: CTC forced alignment with the multilingual MMS-300m forced-alignment
model over romanised Telugu phones, Viterbi implemented in the script, MFA-compatible
TextGrids. Run with `bash scripts/align.sh telugu-ctc` if MFA training/alignment fails.

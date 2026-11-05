# FastSpeech 2 for English and Telugu

Two separately trained FastSpeech 2 text-to-speech models, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
([paper](https://arxiv.org/abs/2006.04558)):

| | English | Telugu |
|---|---|---|
| Data | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, many speakers) | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (one speaker used) |
| Amount | ~5 h train / ~1 h validation, no overlap | ~5 h train / ~1 h validation, no overlap |
| Text frontend | normalisation + g2p_en (POS-based heteronyms), ARPAbet | normalisation + rule-based G2P, 50-phone ASCII set |
| Alignment | MFA, pretrained `english_us_arpa` acoustic model | MFA acoustic model trained on the Telugu subset |
| Speakers | multi-speaker (speaker embedding) | single speaker |
| Vocoder | pretrained HiFi-GAN "universal" (not fine-tuned) | same |

Technical report: [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf).
All changes to the upstream code: [`docs/CHANGES.md`](docs/CHANGES.md).

## Repository layout

```
English/  train.py  preprocess.py  inference.py  config.yaml  test_sentences.tsv  checkpoints/
Telugu/   train.py  preprocess.py  inference.py  config.yaml  test_sentences.tsv  checkpoints/
Results/<Lang>/
    training_loss.png  validation_loss.png  train_vs_val.png  loss_summary.json
    test_generated_audios/<ID>.wav
    variance_adaptor_plots/{duration,pitch,energy,overview}/<ID>.png  and <ID>.json
    data_selection.tsv  data_selection_summary.json  features_summary.json  logs/
fastspeech2/   vendored ming024 code (first commit = pristine upstream), patched
scripts/       MFA setup + alignment, pipeline runner, loss plots, checkpoint export, report build
Report/        Technical_Report.md -> Technical_Report.pdf
docs/CHANGES.md
```

The language folders hold thin entry points; the shared, modified ming024 code lives
in `fastspeech2/`. Each language has one `config.yaml` (upstream used three files).

## Checkpoints

| Model | File in repo (fp16 weights, Git LFS) | Full training checkpoint (fp32 + optimizer) |
|---|---|---|
| English | `English/checkpoints/english_step10000_infer_fp16.pth.tar` | [english_best_step10000.pth.tar](https://github.com/snehithareddy28/fastspeech2-assignment/releases/download/v1.0/english_best_step10000.pth.tar) |
| Telugu | `Telugu/checkpoints/telugu_step14000_infer_fp16.pth.tar` | [telugu_best_step14000.pth.tar](https://github.com/snehithareddy28/fastspeech2-assignment/releases/download/v1.0/telugu_best_step14000.pth.tar) |

`stats.json`, `speakers.json` and `train_speaker_counts.json` next to each checkpoint let
`inference.py` run without the training data. Clone with `git lfs install` first, or download
the files from the [v1.0 release](https://github.com/snehithareddy28/fastspeech2-assignment/releases/tag/v1.0) (SHA-256 sums attached). Both are the
lowest-validation-loss checkpoints; to use a full checkpoint, save it as
`<Lang>/checkpoints/best.pth.tar`.

## Reproducing

Tested target: Linux, Python 3.10 to 3.12, one or two NVIDIA GPUs (Kaggle T4 x2 / Colab).
All commands run from the repository root.

### 0. Environment

```bash
pip install -r requirements.txt            # keep the preinstalled CUDA torch on Colab/Kaggle
bash scripts/setup_mfa.sh                  # Montreal Forced Aligner in its own micromamba env
export HF_TOKEN=...                         # Svarah is gated: accept its terms on Hugging Face first
```

### 1. Data selection, alignment, features (CPU; ~1 to 2 h per language)

```bash
bash scripts/run_pipeline.sh data english   # = the three commands below
#   python English/preprocess.py prepare     download, filter, select ~6 h, split, wav/lab, lexicon
#   bash scripts/align.sh english            MFA align with english_us_arpa
#   python English/preprocess.py features    mel, durations, pitch, energy
bash scripts/run_pipeline.sh data telugu    # same; align.sh trains a Telugu MFA model first
```

These stages need no GPU, so on Kaggle they can run in a CPU session to save GPU quota.

### 2. Training (GPU)

```bash
bash scripts/run_pipeline.sh train-seq 5.1   # English then Telugu, 5.1 h each (what was used)
# or one language at a time:
python English/train.py --max_hours 5.1
python Telugu/train.py  --max_hours 5.1
```

On a 4-core Kaggle T4 box, two parallel runs (`train-both`) were CPU-bound at about 1.1 it/s
each versus about 3.8 it/s for one run, so the languages were trained one after the other.

Training resumes automatically from `<Lang>/checkpoints/latest.pth.tar` (saved every 1000
steps), so a killed session loses at most 1000 steps: rerun the same command. Losses are
appended to `Results/<Lang>/logs/{train,val}_log.csv`.

**Choosing `total_step`.** The trainer prints `it/s`. After ~500 steps set
`train.step.total_step` in the config to about `it/s x 3600 x GPU hours` (the LR anneal points
are fractions of it). Default: 100k steps at batch 16.

### 3. Results

```bash
bash scripts/run_pipeline.sh results english   # loss plots, fp16 export, test-sentence inference
bash scripts/run_pipeline.sh results telugu
python scripts/build_report.py                  # Report/Technical_Report.md -> PDF
```

Single sentence: `python English/inference.py --text "Any sentence."` (writes to
`Results/English/custom/`). English re-joins a word split by a stray space when the joined
form is a dictionary word ("rhyth m" -> "rhythm"); pass `--no_join` to turn this off.

## Test sentences

Kept exactly as in the assignment PDF (`English/test_sentences.tsv`,
`Telugu/test_sentences.tsv`); normalisation happens inside the frontend and the normalised
text and phone sequence for each sentence are saved in
`Results/<Lang>/variance_adaptor_plots/<ID>.json`. Note that T3 in the PDF reads
"పులి అడ విలో ..." with a space inside "అడవిలో", the Telugu counterpart of E3's "rhyth m";
it is kept. `Results/English/test_generated_audios/E3.wav` uses the split-word rule;
the unjoined version ("rye-th em") is kept in `Results/English/e3_without_join/` for comparison.
E2a was regenerated after a frontend fix for possessives ("orchestra's" had its "'s" read as
the letter s); the earlier output is in `Results/English/e2a_before_clitic_fix/`.

#!/usr/bin/env bash
# One-command stages for a Linux GPU box (Kaggle / Colab / rented).
#   bash scripts/run_pipeline.sh data english     # download+select, MFA align, features  (CPU)
#   bash scripts/run_pipeline.sh data telugu
#   bash scripts/run_pipeline.sh train english [max_hours] [cuda:0]
#   bash scripts/run_pipeline.sh train-both [max_hours]   # English on cuda:0, Telugu on cuda:1
#   bash scripts/run_pipeline.sh results english  # loss plots, fp16 export, test-sentence inference
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
stage="${1:?stage}"; lang="${2:-}"
# record the exact environment used for this stage (requirements.txt has lower bounds only)
mkdir -p Results
{ echo "# $(date -u +%FT%TZ) stage=$stage $lang"; python --version; nvidia-smi -L 2>/dev/null || true
  python -m pip freeze; } > "Results/environment_${stage}${lang:+_$lang}.txt" 2>&1 || true
cap() { echo "${1^}"; }
case "$stage" in
  data)
    L=$(cap "$lang")
    python "$L/preprocess.py" prepare
    [ -x .mfa_env/bin/mfa ] || bash scripts/setup_mfa.sh
    if [ "$lang" = "telugu" ]; then
      # if MFA cannot train/align Telugu, fall back to the CTC aligner automatically
      bash scripts/align.sh telugu || { echo "MFA failed for Telugu -> CTC fallback"; rm -rf data/Telugu/TextGrid; bash scripts/align.sh telugu-ctc; }
    else
      bash scripts/align.sh "$lang"
    fi
    python "$L/preprocess.py" features
    ;;
  train)
    L=$(cap "$lang"); H="${3:-}"; DEV="${4:-cuda:0}"
    mkdir -p "Results/$L/logs"
    python "$L/train.py" --device "$DEV" ${H:+--max_hours "$H"} 2>&1 | tee -a "Results/$L/logs/train_stdout.log"
    ;;
  train-both)
    H="${2:-}"
    mkdir -p Results/English/logs Results/Telugu/logs
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    # two GPUs: one language each; one GPU: both processes share it (each needs < 8 GB)
    G2=$([ "$NGPU" -ge 2 ] && echo 1 || echo 0)
    echo "GPUs: $NGPU -> English on cuda:0, Telugu on cuda:$G2"
    CUDA_VISIBLE_DEVICES=0 python English/train.py ${H:+--max_hours "$H"} > Results/English/logs/train_stdout.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$G2 python Telugu/train.py ${H:+--max_hours "$H"} > Results/Telugu/logs/train_stdout.log 2>&1 &
    wait
    ;;
  results)
    L=$(cap "$lang")
    python scripts/plot_losses.py "$L"
    python scripts/export_checkpoint.py "$L"
    python "$L/inference.py"
    ;;
  *) echo "unknown stage $stage"; exit 1;;
esac

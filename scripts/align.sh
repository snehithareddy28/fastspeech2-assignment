#!/usr/bin/env bash
# Forced alignment with Montreal Forced Aligner.
#   bash scripts/align.sh english   # pretrained english_us_arpa acoustic model, our lexicon
#   bash scripts/align.sh telugu    # train an acoustic model on the Telugu subset, then align
#   bash scripts/align.sh telugu-ctc  # FALLBACK: CTC forced alignment (MMS wav2vec2), no MFA needed
# Requires: python <Lang>/preprocess.py prepare  (writes data/<Lang>/raw and lexicon.txt)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LANG_ARG="${1:?usage: align.sh english|telugu|telugu-ctc}"
if [ "$LANG_ARG" = "telugu-ctc" ]; then
  D="$ROOT/data/Telugu"
  pip install -q "transformers>=4.40"
  python "$ROOT/scripts/ctc_align.py" --config "$ROOT/Telugu/config.yaml" 2>&1 | tee "$D/ctc_align.log"
  n_wav=$(find "$D/raw" -name '*.wav' | wc -l); n_tg=$(find "$D/TextGrid" -name '*.TextGrid' | wc -l)
  mkdir -p "$ROOT/Results/Telugu"
  echo "Aligned $n_tg / $n_wav utterances (CTC fallback aligner)" | tee "$ROOT/Results/Telugu/alignment_coverage.txt"
  exit 0
fi
MFA="${MFA:-$ROOT/.mfa_env/bin/mfa}"
# MFA shells out to Kaldi/OpenFst binaries (fstcompile, ...) that live next to it
export PATH="$(dirname "$MFA"):$PATH"
ALIGN_CFG="$ROOT/scripts/mfa_align_config.yaml"
J="${JOBS:-$(nproc)}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$ROOT/mfa_models}"
mkdir -p "$MFA_ROOT_DIR"
"$MFA" version

case "$LANG_ARG" in
  english)
    D="$ROOT/data/English"
    "$MFA" model download acoustic english_us_arpa
    # Our lexicon = all LibriSpeech/CMU variants for in-vocabulary words + g2p_en for OOVs,
    # so MFA picks the acoustically best variant (heteronyms) and never sees an OOV.
    "$MFA" align --clean -j "$J" --config_path "$ALIGN_CFG" \
      "$D/raw" "$D/lexicon.txt" english_us_arpa "$D/TextGrid" 2>&1 | tee "$D/mfa_align.log"
    ;;
  telugu)
    D="$ROOT/data/Telugu"
    # No pretrained Telugu MFA model exists: train one (mono -> tri -> LDA/MLLT -> SAT)
    # on the selected ~6 h with the rule-based G2P lexicon, then align with it.
    # --single_speaker: MFA splits jobs by speaker, so with one speaker it silently uses
    # 1 job instead of $J (observed: ~3.5 h for training on one core).
    "$MFA" train --clean -j "$J" --single_speaker \
      "$D/raw" "$D/lexicon.txt" "$D/mfa_telugu_acoustic.zip" 2>&1 | tee "$D/mfa_train.log"
    "$MFA" align --clean -j "$J" --single_speaker --config_path "$ALIGN_CFG" \
      "$D/raw" "$D/lexicon.txt" "$D/mfa_telugu_acoustic.zip" "$D/TextGrid" 2>&1 | tee "$D/mfa_align.log"
    ;;
  *) echo "unknown language $LANG_ARG"; exit 1;;
esac

n_wav=$(find "$D/raw" -name '*.wav' | wc -l)
n_tg=$( (find "$D/TextGrid" -name '*.TextGrid' 2>/dev/null || true) | wc -l)
echo "Aligned $n_tg / $n_wav utterances" | tee "$D/alignment_coverage.txt"
[ "$n_tg" -gt 0 ] || { echo "MFA produced no TextGrids"; exit 1; }
mkdir -p "$ROOT/Results/${LANG_ARG^}"
cp "$D/alignment_coverage.txt" "$ROOT/Results/${LANG_ARG^}/alignment_coverage.txt"
"$MFA" version > "$ROOT/Results/${LANG_ARG^}/mfa_version.txt"

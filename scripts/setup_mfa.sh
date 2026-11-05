#!/usr/bin/env bash
# Install Montreal Forced Aligner into its own environment with micromamba
# (MFA's Kaldi/OpenFst dependencies are conda-only). Linux x86_64.
#   bash scripts/setup_mfa.sh            -> env at ./.mfa_env, binary ./.mfa_env/bin/mfa
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${MFA_ENV:-$ROOT/.mfa_env}"
MM="$ROOT/.micromamba/bin/micromamba"
if [ ! -x "$MM" ]; then
  mkdir -p "$ROOT/.micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT/.micromamba" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$ROOT/.micromamba"
if [ ! -x "$ENV/bin/mfa" ]; then
  "$MM" create -y -p "$ENV" -c conda-forge "montreal-forced-aligner=${MFA_VERSION:-3}" python=3.11
fi
mkdir -p "$ROOT/Results/others"
export PATH="$ENV/bin:$PATH"
"$ENV/bin/mfa" version | tee "$ROOT/Results/others/mfa_version.txt"
command -v fstcompile >/dev/null || { echo "OpenFst binaries missing from $ENV/bin"; exit 1; }
echo "MFA ready: $ENV/bin/mfa"

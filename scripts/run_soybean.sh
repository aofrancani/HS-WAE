#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
VARIANT="${1:-hswae}"
if (( $# )); then shift; fi
cd "$ROOT/HS-AE"
MODULE=main_stats
case "$VARIANT" in
  hswae) DEFAULT_CONFIG=configs/dwt_attn.yaml ;;
  raw) cd v0; DEFAULT_CONFIG=configs/no-dwt_reduced.yaml ;;
  dwt) DEFAULT_CONFIG=configs/dwt_controlled.yaml ;;
  norecon) DEFAULT_CONFIG=configs/dwt_attn_norecon.yaml ;;
  curves) MODULE=main; DEFAULT_CONFIG=configs/dwt_attn_all_epochs.yaml ;;
  latent) MODULE=main_stats_latent; DEFAULT_CONFIG=configs/dwt_attn.yaml ;;
  *) echo 'Usage: run_soybean.sh {hswae|raw|dwt|norecon|curves|latent} [module options]' >&2; exit 2 ;;
esac
exec "$PYTHON" -m "$MODULE" --mode train --config "${CONFIG:-$DEFAULT_CONFIG}" "$@"

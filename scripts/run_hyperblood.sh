#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
VARIANT="${1:-hswae}"
if (( $# )); then shift; fi
cd "$ROOT/HSI_Blood_mc"
case "$VARIANT" in
  hswae) MODULE=main_stats; DEFAULT_CONFIG=configs/dwt_attn.yaml ;;
  raw) MODULE=v0.main_stats; DEFAULT_CONFIG=v0/configs/no-dwt.yaml ;;
  *) echo 'Usage: run_hyperblood.sh {hswae|raw} [module options]' >&2; exit 2 ;;
esac
exec "$PYTHON" -m "$MODULE" --mode train --config "${CONFIG:-$DEFAULT_CONFIG}" "$@"

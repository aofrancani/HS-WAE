#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET="${1:-}"
if (( $# )); then shift; fi
case "$DATASET" in
  soybean) cd "$ROOT/HS-AE" ;;
  hyperblood) cd "$ROOT/HSI_Blood_mc" ;;
  *) echo 'Usage: run_controlled.sh {soybean|hyperblood} --variant ... --band-region ... [options]' >&2; exit 2 ;;
esac
exec "$PYTHON" -m scripts.run_wtam_definitive --config "${CONFIG:-configs/wtam_definitive.yaml}" "$@"

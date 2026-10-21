#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-}"
if (( $# )); then shift; fi
case "$DATASET" in
  soybean) cd "$ROOT/HS-AE" ;;
  hyperblood) cd "$ROOT/HSI_Blood_mc" ;;
  *) echo 'Usage: analyze_controlled.sh {soybean|hyperblood} --run-dir ... [options]' >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" -m scripts.analyze_wtam_definitive "$@"

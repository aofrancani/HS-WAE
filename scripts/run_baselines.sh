#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/HS-AE"
exec "${PYTHON:-python}" -m scripts.run_baselines --config "${CONFIG:-configs/dwt_attn.yaml}" "$@"

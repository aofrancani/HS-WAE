#!/usr/bin/env bash
# Full paper-sized training, never run as part of smoke validation.
# DRY_RUN=1 prints commands. GPU defaults can be overridden with GPU and DEVICE.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-}"
METHOD="${2:-hswae}"
run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
case "$DATASET" in
  soybean)
    for band in nir re rgb vis uv swir-1 swir-2a swir-2b nir-swir-1 re-nir re-swir-1 vis-nir vis-re vis-swir-1; do
      for day in {1..7}; do run bash "$ROOT/scripts/run_soybean.sh" "$METHOD" --band_region "$band" --days "$day" --devices "${GPU:-0}"; done
    done ;;
  hyperblood)
    for band in all vis vis-nir rgb re nir re-nir; do run bash "$ROOT/scripts/run_hyperblood.sh" "$METHOD" --band_region "$band" --devices "${GPU:-0}"; done ;;
  controlled-soybean|controlled-hyperblood)
    target="${DATASET#controlled-}"
    if [[ "$target" == soybean ]]; then regions=(nir); days=(1 2 3 4 5 6 7); variants=(raw_ae dwt_concat dwt_static_gate dwt_dynamic_gate wtam_full)
    else regions=(all vis vis-nir rgb re nir re-nir); days=(none); variants=(dwt_concat dwt_static_gate dwt_dynamic_gate wtam_full); fi
    for band in "${regions[@]}"; do for day in "${days[@]}"; do for seed in {2..26}; do for variant in "${variants[@]}"; do
      args=(--variant "$variant" --band-region "$band" --seed "$seed" --exp-idx "$((seed-2))" --device "${DEVICE:-cuda:0}" --outdir "results/wtam_definitive/$target")
      condition="$band"
      if [[ "$day" != none ]]; then args+=(--days "$day"); condition="${band}_days_${day}"; fi
      run bash "$ROOT/scripts/run_controlled.sh" "$target" "${args[@]}"
      if [[ "$variant" != raw_ae ]]; then
        printf -v rep '%02d' "$((seed-2))"
        run bash "$ROOT/scripts/analyze_controlled.sh" "$target" --run-dir "results/wtam_definitive/$target/$variant/$condition/run_${rep}_seed_${seed}" --device "${DEVICE:-cuda:0}"
      fi
    done; done; done; done ;;
  baselines)
    for day in {1..7}; do
      run bash "$ROOT/scripts/run_baselines.sh" --band_region nir --days "$day" --feature_mode raw --models rf svm xgb mlp cnn1d --n_exp 25 --seed 2 --device "${DEVICE:-cuda:0}"
    done ;;
  reductions)
    for feature in pca95 pca64 selectk; do
      for day in {1..7}; do run bash "$ROOT/scripts/run_baselines.sh" --band_region nir --days "$day" --feature_mode "$feature" --models rf svm xgb mlp --n_exp 25 --seed 2 --device "${DEVICE:-cuda:0}"; done
    done ;;
  *) echo 'Usage: paper_sweep.sh {soybean|hyperblood|controlled-soybean|controlled-hyperblood|baselines|reductions} [hswae|raw|norecon]' >&2; exit 2 ;;
esac

# Execution guide

Use the [README](README.md) for installation, standard training, checkpoint evaluation and TensorBoard. Run the Bash commands below from the HS-WAE root. Set `PYTHON=/path/to/python` to select an interpreter.

## Native Python entry points

Change to the indicated directory before running its command:

| Working directory | Repeated-run command |
| --- | --- |
| `HS-AE` | `python -m main_stats --mode train --config configs/dwt_attn.yaml --band_region nir --days 2 --devices 0` |
| `HS-AE/v0` | `python -m main_stats --mode train --config configs/no-dwt_reduced.yaml --band_region nir --days 2 --devices 0` |
| `HSI_Blood_mc` | `python -m main_stats --mode train --config configs/dwt_attn.yaml --band_region vis --devices 0` |
| `HSI_Blood_mc` | `python -m v0.main_stats --mode train --config v0/configs/no-dwt.yaml --band_region vis --devices 0` |

Use `main` instead of `main_stats` for a single run with a checkpoint (`v0.main` for HyperBlood Raw). Repeated-run drivers use 25 training seeds (2–26) with a fixed data split. Their incidental fallback checkpoints are not the selected best checkpoints; use single-run or controlled training when retaining weights for inference.

## Local configuration

```bash
python -m tools.configure --cpu --num-workers 0 \
  --soybean-root /path/to/HS-Class --blood-csv /path/to/HSI_Blood.csv
CONFIG=configs/local/dwt_attn.yaml bash scripts/run_soybean.sh hswae --band_region nir --days 2
CONFIG=v0/configs/local/no-dwt.yaml bash scripts/run_hyperblood.sh raw --band_region vis
```

Omit the data arguments to use the repository's `data/` layout. Add `--overwrite` to replace existing local configs. Omit `--devices 0` when using CPU configs.

## Ablations and baselines

RF, SVM, XGBoost, MLP, and 1D-CNN are implemented in [run_baselines.py](HS-AE/scripts/run_baselines.py) for the soybean experiments; launch them with `scripts/run_baselines.sh`.

```bash
bash scripts/run_soybean.sh dwt --band_region nir --days 2 --devices 0
bash scripts/run_soybean.sh norecon --band_region nir --days 2 --devices 0
bash scripts/run_soybean.sh curves --band_region nir --days 2 --devices 0
bash scripts/run_baselines.sh --band_region nir --days 2 --feature_mode raw \
  --models rf svm xgb mlp cnn1d --n_exp 25 --seed 2 --device cuda:0
```

Feature modes: `raw`, `pca95`, `pca64`, `selectk`. Use CNN1D with raw wavelengths. To print a full campaign without starting training:

```bash
DRY_RUN=1 bash scripts/paper_sweep.sh soybean hswae
```

Other campaigns: `hyperblood`, `controlled-soybean`, `controlled-hyperblood`, `baselines`, `reductions`. Remove `DRY_RUN=1` to execute.

## Controlled WTAM runs

```bash
bash scripts/run_controlled.sh soybean --variant wtam_full --band-region nir --days 2 \
  --seed 2 --exp-idx 0 --device cuda:0 --outdir results/wtam_definitive/soybean
bash scripts/analyze_controlled.sh soybean \
  --run-dir results/wtam_definitive/soybean/wtam_full/nir_days_2/run_00_seed_2 --device cuda:0
bash scripts/run_controlled.sh hyperblood --variant dwt_concat --band-region vis \
  --seed 2 --exp-idx 0 --device cuda:0 --outdir results/wtam_definitive/hyperblood
```

Variants: `raw_ae`, `dwt_concat`, `dwt_static_gate`, `dwt_dynamic_gate`, `wtam_full`. Use `dwt_concat` for HyperBlood DWT-only runs. Existing output runs are protected from overwrite. The legacy `finalize_wtam_definitive.py` requires original server paths; use `analyze_controlled.sh` for portable per-run analysis.

## TensorBoard and saved plots

```bash
tensorboard --logdir . --port 6006
```

Open [http://localhost:6006](http://localhost:6006). For logs outside this repository, use `--logdir /path/to/logs`. Controlled WTAM runs save outputs without TensorBoard logging:

```bash
python -m tools.plot_run \
  --runs results/wtam_definitive/soybean/wtam_full/nir_days_2/run_00_seed_2 \
  --labels HS-WAE --outdir figures/soybean
python -m tools.plot_training --logdir /path/to/tensorboard/run --output figures/training.png
```

Use a module's `--help` for additional options. Older optional figure scripts may require LaTeX; these two plotting tools do not.

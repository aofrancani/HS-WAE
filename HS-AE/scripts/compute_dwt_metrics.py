"""
Compute DWT-based difference metrics per band and per day
using the difference between mean Control and Damaged curves.

Metrics:
  - MSE_A, MSE_D, Weighted_Total (classic)
  - JS_divergence_bits  (energy proportion shift)
  - Wasserstein_agg     (distribution shift across coefficients)

Usage:
    python -m scripts.compute_dwt_metrics_mean_diff --band nir --days 3 4 5 6 7
"""

import yaml
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from tqdm import tqdm
from lightning.pytorch import seed_everything
from dataloader import build_datamodule
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# -----------------------
# Configurable parameters
# -----------------------
wavelet = "db4"
boundary = "periodization"
W_A, W_D = 0.5, 0.5  # weights for approximation/detail
seed = 2

USE_LOG_ENERGY = True  # log1p before normalizing energies
INCLUDE_APPROX = True  # include A_L in energy vector
WASSERSTEIN_ABS = True  # use absolute values of coeffs for Wasserstein
WASSERSTEIN_AGG = "len_weighted"  # mean | len_weighted | energy_weighted

save_dir = Path("logs/dwt_metrics")
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------
# Helper functions
# -----------------------
def normalize_days_arg(days_input):
    """Normalize user-provided day inputs into canonical format ['day_03', ...]."""
    if not days_input:
        return None
    if len(days_input) == 1 and "," in days_input[0]:
        days_input = [d.strip() for d in days_input[0].split(",") if d.strip()]
    normalized = []
    for d in days_input:
        d = d.strip().lower()
        if d.startswith("day_"):
            normalized.append(d)
        else:
            try:
                d_int = int(d)
                normalized.append(f"day_{d_int:02d}")
            except ValueError:
                raise ValueError(f"Invalid day format: '{d}'")
    return normalized


def load_data(config_path: Path, data_path: Path | None = None):
    """Load datamodule from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if data_path:
        cfg["data"]["csv_path"] = str(data_path)
    dm = build_datamodule(cfg)
    dm.setup(stage="test")
    return dm, cfg


def wavedec(x: np.ndarray):
    """DWT decomposition."""
    return pywt.wavedec(x, wavelet=wavelet, mode=boundary)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((a - b) ** 2))


# ============================================================
# --- Energy and distribution-based metrics
# ============================================================
def level_energies(coeffs, include_approx=True):
    """Compute energy per level from DWT coefficients."""
    parts = coeffs if include_approx else coeffs[1:]
    return np.array([np.sum(c**2) for c in parts], dtype=float)


def class_mean_energy_vector(X, use_log=True, include_approx=True):
    """Compute mean per-level energy vector (and proportions) over all samples in X."""
    Es = []
    for xi in X:
        coeffs = wavedec(xi)
        e = level_energies(coeffs, include_approx=include_approx)
        if use_log:
            e = np.log1p(e)
        Es.append(e)
    Emean = np.nanmean(np.stack(Es, axis=0), axis=0)
    p = Emean / np.clip(Emean.sum(), 1e-12, None)
    return p, Emean


def js_energy_proportions(X_ctrl, X_dmg, base=2.0):
    """Jensen–Shannon divergence between energy proportions."""
    p_ctrl, _ = class_mean_energy_vector(X_ctrl, USE_LOG_ENERGY, INCLUDE_APPROX)
    p_dmg, _ = class_mean_energy_vector(X_dmg, USE_LOG_ENERGY, INCLUDE_APPROX)
    p_ctrl = p_ctrl / np.clip(p_ctrl.sum(), 1e-12, None)
    p_dmg = p_dmg / np.clip(p_dmg.sum(), 1e-12, None)
    js = jensenshannon(p_ctrl, p_dmg, base=base)
    return float(js), p_ctrl, p_dmg


def wasserstein_agg_on_coeffs(
    X_ctrl, X_dmg, use_abs=True, include_approx=True, agg="len_weighted"
):
    """Aggregate Wasserstein distances across all DWT levels."""
    c0 = wavedec(X_ctrl[0])
    levels = range(len(c0)) if include_approx else range(1, len(c0))

    dists = []
    weights_len = []
    weights_energy = []

    for li in levels:
        ctrl_coeffs = np.concatenate([wavedec(x)[li] for x in X_ctrl])
        dmg_coeffs = np.concatenate([wavedec(x)[li] for x in X_dmg])

        if use_abs:
            ctrl_vals = np.abs(ctrl_coeffs)
            dmg_vals = np.abs(dmg_coeffs)
        else:
            ctrl_vals = ctrl_coeffs
            dmg_vals = dmg_coeffs

        w = wasserstein_distance(ctrl_vals, dmg_vals)
        dists.append(w)

        weights_len.append(len(ctrl_vals) + len(dmg_vals))
        weights_energy.append(np.sum(ctrl_vals**2) + np.sum(dmg_vals**2))

    dists = np.array(dists, dtype=float)
    if agg == "mean":
        score = float(np.mean(dists))
    elif agg == "len_weighted":
        w = np.array(weights_len, float)
        score = float(np.average(dists, weights=w + 1e-12))
    elif agg == "energy_weighted":
        w = np.array(weights_energy, float)
        score = float(np.average(dists, weights=w + 1e-12))
    else:
        raise ValueError("agg must be 'mean', 'len_weighted', or 'energy_weighted'.")

    return score, dists


# ============================================================
# --- Core computation
# ============================================================
def compute_dwt_diff_between_means(X: np.ndarray, y: np.ndarray):
    """Compute DWT difference between mean Control and mean Damaged curves."""
    X_ctrl, X_dmg = X[y == 0], X[y == 1]
    if len(X_ctrl) == 0 or len(X_dmg) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Classic MSE-based
    mean_ctrl = np.mean(X_ctrl, axis=0)
    mean_dmg = np.mean(X_dmg, axis=0)
    coeffs_ctrl = wavedec(mean_ctrl)
    coeffs_dmg = wavedec(mean_dmg)

    A_ctrl, A_dmg = coeffs_ctrl[0], coeffs_dmg[0]
    D_ctrl, D_dmg = np.concatenate(coeffs_ctrl[1:]), np.concatenate(coeffs_dmg[1:])
    mse_A = mse(A_ctrl, A_dmg)
    mse_D = mse(D_ctrl, D_dmg)
    total_mse = W_A * mse_A + W_D * mse_D

    # JS divergence
    js, p_ctrl, p_dmg = js_energy_proportions(X_ctrl, X_dmg)

    # Wasserstein aggregation
    w_agg, w_levels = wasserstein_agg_on_coeffs(
        X_ctrl,
        X_dmg,
        use_abs=WASSERSTEIN_ABS,
        include_approx=INCLUDE_APPROX,
        agg=WASSERSTEIN_AGG,
    )

    return mse_A, mse_D, total_mse, js, w_agg


# ============================================================
# --- Main per-band-day routine
# ============================================================
def compute_dwt_per_band_day(
    band: str,
    logs_dir: str = "logs",
    config_base: str = "configs",
    data_dir: str | None = None,
    selected_days: list[str] | None = None,
):
    logs_path = Path(logs_dir)
    dwt_dir = logs_path / "dwt"

    days = sorted([p.name for p in (dwt_dir / band).iterdir() if p.is_dir()])
    days = [d for d in days if d not in ("day_01", "day_02")]
    if selected_days:
        days = [d for d in days if d in selected_days]

    seed_everything(seed, workers=True)
    summary_rows = []

    tqdm_desc = f"[{band.upper()}] Computing mean DWT metrics ({', '.join(days)})"
    for day in tqdm(days, desc=tqdm_desc, ncols=100):
        config_path = Path(config_base) / "dwt" / band / f"{day}.yaml"
        if not config_path.exists():
            print(f"⚠️ Config not found: {config_path}")
            continue

        dm, _ = load_data(config_path, data_path=data_dir)
        test_ds = dm.test_ds
        X_test = test_ds.X.numpy()
        y_test = test_ds.y.numpy()

        mse_A, mse_D, total_mse, js, w_agg = compute_dwt_diff_between_means(
            X_test, y_test
        )

        summary_rows.append(
            {
                "band": band,
                "day": day,
                "MSE_A": mse_A,
                "MSE_D": mse_D,
                "Weighted_Total": total_mse,
                "JS_divergence_bits": js,
                "Wasserstein_agg": w_agg,
                "Num_Samples": len(X_test),
            }
        )

    # --- Save summary per band ---
    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values("day", inplace=True)
    summary_path = save_dir / f"dwt_metrics_summary_{band}.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n✅ DWT mean-curve metrics computed for band={band}")
    print(df_summary.round(6))
    print(f"📄 Saved summary to: {summary_path.resolve()}")


# ============================================================
# --- CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute DWT-based difference metrics per band and per day (mean-curve version)"
    )
    parser.add_argument(
        "--band", type=str, required=True, help="Spectral band (e.g., nir, vis, re)"
    )
    parser.add_argument(
        "--days",
        nargs="+",
        default=["3", "4", "5", "6", "7"],
        help="List of days (e.g., 3 4 5 6 7)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Optional dataset path override"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Logs directory (where checkpoints are stored)",
    )
    parser.add_argument(
        "--config_base", type=str, default="configs", help="Base config directory"
    )

    args = parser.parse_args()
    selected_days = normalize_days_arg(args.days)

    compute_dwt_per_band_day(
        band=args.band,
        logs_dir=args.logs_dir,
        config_base=args.config_base,
        data_dir=args.data_dir,
        selected_days=selected_days,
    )

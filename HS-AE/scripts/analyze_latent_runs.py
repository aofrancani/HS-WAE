from __future__ import annotations

"""
Analyze repeated-run full-dimensional latent representations extracted by
extract_latent_runs.py.

Primary outputs:
  - per_run_latent_metrics.csv
  - run_counts.csv
  - summary_latent_metrics.csv (mean, SD, bootstrap 95% CI)
  - paired_comparisons.csv (paired differences, CI, Wilcoxon, Holm, rank-biserial)
  - paired_dwt_vs_hswae.csv
  - nir_publication_summary.csv
  - figures/*.pdf and *.png
  - latent_analysis_report.md

No model training or checkpoint loading occurs here.
"""

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

# Matplotlib is used only for saved publication figures.
import matplotlib.pyplot as plt


PRIMARY_METRICS = [
    "silhouette_64d",
    "davies_bouldin_64d",
    "fisher_ratio_64d",
    "centroid_distance_64d",
    "within_rms_64d",
    "separation_ratio_64d",
]

METRIC_DIRECTION = {
    "silhouette_64d": "higher",
    "davies_bouldin_64d": "lower",
    "calinski_harabasz_64d": "higher",
    "fisher_ratio_64d": "higher",
    "centroid_distance_64d": "higher",
    "within_rms_64d": "lower",
    "within_rms_control_64d": "lower",
    "within_rms_damaged_64d": "lower",
    "separation_ratio_64d": "higher",
    "mean_interclass_distance_64d": "higher",
    "mean_intraclass_control_64d": "lower",
    "mean_intraclass_damaged_64d": "lower",
    "silhouette_zscore64d": "higher",
    "davies_bouldin_zscore64d": "lower",
    "fisher_ratio_zscore64d": "higher",
    "centroid_distance_zscore64d": "higher",
    "within_rms_zscore64d": "lower",
    "within_rms_control_zscore64d": "lower",
    "within_rms_damaged_zscore64d": "lower",
    "separation_ratio_zscore64d": "higher",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze extracted full latent representations.")
    p.add_argument(
        "--root",
        default="results/latent",
        help="Directory created by extract_latent_runs.py.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Analysis output directory. Default: <root>/analysis.",
    )
    p.add_argument(
        "--expected-runs",
        type=int,
        default=25,
        help="Expected independent runs per method/band/day; used for diagnostics.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for 95%% confidence intervals.",
    )
    p.add_argument("--seed", type=int, default=2026, help="Analysis RNG seed.")
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Optional band filter for analysis.",
    )
    p.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=None,
        help="Optional day filter for analysis.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "dwt", "hswae"],
        choices=["raw", "dwt", "hswae"],
    )
    return p.parse_args()


def load_manifest(root: Path) -> pd.DataFrame:
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        rows = []
        for info_path in root.rglob("run_info.json"):
            rows.append(json.loads(info_path.read_text(encoding="utf-8")))
        if not rows:
            raise FileNotFoundError(
                f"No manifest.csv or run_info.json files found under {root}"
            )
        df = pd.DataFrame(rows)

    # Normalize days generated as string representations by CSV serialization.
    def parse_day(v: Any) -> Optional[int]:
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float) and not np.isnan(v):
            return int(v)
        if isinstance(v, str):
            nums = [int(x) for x in __import__("re").findall(r"\d+", v)]
            if len(nums) == 1:
                return nums[0]
        return None

    if "days" in df.columns:
        df["day"] = df["days"].apply(parse_day)
    elif "day" not in df.columns:
        df["day"] = np.nan

    if "version_number" not in df.columns:
        df["version_number"] = np.nan
    return df


def finite_rows(Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(Z).all(axis=1) & np.isfinite(y)
    return Z[mask], y[mask].astype(int)


def zscore_features(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (Z - mu) / sd


def fisher_ratio(Z: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) != 2:
        return np.nan
    z0, z1 = Z[y == classes[0]], Z[y == classes[1]]
    if len(z0) < 2 or len(z1) < 2:
        return np.nan
    mu0, mu1 = z0.mean(axis=0), z1.mean(axis=0)
    s0 = np.var(z0, axis=0, ddof=1).sum()
    s1 = np.var(z1, axis=0, ddof=1).sum()
    den = s0 + s1
    num = float(np.sum((mu1 - mu0) ** 2))
    return float(num / den) if den > 0 else np.nan


def centroid_metrics(Z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    classes = np.unique(y)
    if len(classes) != 2:
        return {
            "centroid_distance": np.nan,
            "within_rms": np.nan,
            "within_rms_control": np.nan,
            "within_rms_damaged": np.nan,
            "separation_ratio": np.nan,
        }
    z0, z1 = Z[y == classes[0]], Z[y == classes[1]]
    mu0, mu1 = z0.mean(axis=0), z1.mean(axis=0)
    centroid_distance = float(np.linalg.norm(mu1 - mu0))

    rms0 = float(np.sqrt(np.mean(np.sum((z0 - mu0) ** 2, axis=1)))) if len(z0) else np.nan
    rms1 = float(np.sqrt(np.mean(np.sum((z1 - mu1) ** 2, axis=1)))) if len(z1) else np.nan
    within_rms = float(np.nanmean([rms0, rms1]))
    ratio = centroid_distance / within_rms if np.isfinite(within_rms) and within_rms > 0 else np.nan
    return {
        "centroid_distance": centroid_distance,
        "within_rms": within_rms,
        "within_rms_control": rms0,
        "within_rms_damaged": rms1,
        "separation_ratio": float(ratio),
    }


def mean_pairwise_distances(Z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    classes = np.unique(y)
    if len(classes) != 2:
        return {
            "mean_interclass_distance": np.nan,
            "mean_intraclass_control": np.nan,
            "mean_intraclass_damaged": np.nan,
        }
    z0, z1 = Z[y == classes[0]], Z[y == classes[1]]
    inter = float(cdist(z0, z1).mean()) if len(z0) and len(z1) else np.nan
    intra0 = float(pdist(z0).mean()) if len(z0) >= 2 else np.nan
    intra1 = float(pdist(z1).mean()) if len(z1) >= 2 else np.nan
    return {
        "mean_interclass_distance": inter,
        "mean_intraclass_control": intra0,
        "mean_intraclass_damaged": intra1,
    }


def safe_metric(fn, *args) -> float:
    try:
        return float(fn(*args))
    except Exception:
        return np.nan


def compute_metrics(Z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    Z, y = finite_rows(Z, y)
    out: Dict[str, float] = {
        "n_samples": int(len(y)),
        "n_control": int(np.sum(y == np.min(y))) if len(y) else 0,
        "n_damaged": int(np.sum(y == np.max(y))) if len(y) else 0,
        "latent_dim": int(Z.shape[1]) if Z.ndim == 2 else 0,
    }

    if len(np.unique(y)) < 2 or len(y) < 3:
        for name in METRIC_DIRECTION:
            out[name] = np.nan
        return out

    out["silhouette_64d"] = safe_metric(silhouette_score, Z, y)
    out["davies_bouldin_64d"] = safe_metric(davies_bouldin_score, Z, y)
    out["calinski_harabasz_64d"] = safe_metric(calinski_harabasz_score, Z, y)
    out["fisher_ratio_64d"] = fisher_ratio(Z, y)

    c = centroid_metrics(Z, y)
    out["centroid_distance_64d"] = c["centroid_distance"]
    out["within_rms_64d"] = c["within_rms"]
    out["within_rms_control_64d"] = c["within_rms_control"]
    out["within_rms_damaged_64d"] = c["within_rms_damaged"]
    out["separation_ratio_64d"] = c["separation_ratio"]

    d = mean_pairwise_distances(Z, y)
    out["mean_interclass_distance_64d"] = d["mean_interclass_distance"]
    out["mean_intraclass_control_64d"] = d["mean_intraclass_control"]
    out["mean_intraclass_damaged_64d"] = d["mean_intraclass_damaged"]

    # Sensitivity analysis: same full 64-D space after unsupervised coordinate-wise scaling.
    Zs = zscore_features(Z)
    out["silhouette_zscore64d"] = safe_metric(silhouette_score, Zs, y)
    out["davies_bouldin_zscore64d"] = safe_metric(davies_bouldin_score, Zs, y)
    out["fisher_ratio_zscore64d"] = fisher_ratio(Zs, y)
    cs = centroid_metrics(Zs, y)
    out["centroid_distance_zscore64d"] = cs["centroid_distance"]
    out["within_rms_zscore64d"] = cs["within_rms"]
    out["within_rms_control_zscore64d"] = cs["within_rms_control"]
    out["within_rms_damaged_zscore64d"] = cs["within_rms_damaged"]
    out["separation_ratio_zscore64d"] = cs["separation_ratio"]
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    return bootstrap_mean_ci(diff, n_boot, rng)


def rank_biserial_paired(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d), method="average")
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom > 0 else 0.0


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out.tolist()

    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted_sorted = []
    running = 0.0
    for rank_i, idx in enumerate(order):
        adj = (m - rank_i) * p[idx]
        running = max(running, adj)
        adjusted_sorted.append(min(1.0, running))
    for idx, adj in zip(order, adjusted_sorted):
        out[idx] = adj
    return out.tolist()


def assign_replicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign deterministic within-condition replicate indices.

    Prefer version_number when available; otherwise checkpoint mtime/path order.
    This is used only when a trustworthy varying seed is not available.
    """
    df = df.copy()
    sort_cols = [c for c in ["method", "band", "day", "version_number", "checkpoint_mtime", "run_dir"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="stable")
    df["replicate_index"] = df.groupby(["method", "band", "day"], dropna=False).cumcount()
    return df


def choose_pair_key(per_run: pd.DataFrame, method_a: str, method_b: str, band: str, day: int) -> str:
    sub = per_run[
        (per_run["band"] == band)
        & (per_run["day"] == day)
        & (per_run["method"].isin([method_a, method_b]))
    ]
    if "config_seed" in sub.columns:
        a = sub[sub["method"] == method_a]["config_seed"].dropna()
        b = sub[sub["method"] == method_b]["config_seed"].dropna()
        # Only trust seed pairing if seeds actually vary and are unique within each method.
        if a.nunique() >= 2 and b.nunique() >= 2 and a.is_unique and b.is_unique:
            if set(a.tolist()) == set(b.tolist()):
                return "config_seed"
    return "replicate_index"


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(digest[:8], "little") % 1_000_000
    return int(base_seed + offset)


def summarize(per_run: pd.DataFrame, metric_cols: Sequence[str], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (method, band, day), g in per_run.groupby(["method", "band", "day"], dropna=False):
        for metric in metric_cols:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            rng = np.random.default_rng(stable_seed(seed, method, band, day, metric))
            ci_lo, ci_hi = bootstrap_mean_ci(finite, n_boot, rng)
            rows.append(
                {
                    "method": method,
                    "band": band,
                    "day": day,
                    "metric": metric,
                    "direction": METRIC_DIRECTION.get(metric, "higher"),
                    "n_runs": len(finite),
                    "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "median": float(np.median(finite)) if len(finite) else np.nan,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "min": float(np.min(finite)) if len(finite) else np.nan,
                    "max": float(np.max(finite)) if len(finite) else np.nan,
                    "fraction_negative_silhouette": float(np.mean(finite < 0)) if metric.startswith("silhouette") and len(finite) else np.nan,
                    "number_negative_silhouette": int(np.sum(finite < 0)) if metric.startswith("silhouette") else 0,
                }
            )
    return pd.DataFrame(rows)


def paired_tests(
    per_run: pd.DataFrame,
    metric_cols: Sequence[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    method_pairs = [("dwt", "hswae"), ("raw", "dwt"), ("raw", "hswae")]
    rows = []

    for band in sorted(per_run["band"].dropna().unique()):
        for day in sorted(per_run["day"].dropna().unique()):
            for method_a, method_b in method_pairs:
                a0 = per_run[(per_run["method"] == method_a) & (per_run["band"] == band) & (per_run["day"] == day)]
                b0 = per_run[(per_run["method"] == method_b) & (per_run["band"] == band) & (per_run["day"] == day)]
                if a0.empty or b0.empty:
                    continue
                key = choose_pair_key(per_run, method_a, method_b, band, int(day))

                for metric in metric_cols:
                    if metric not in per_run.columns:
                        continue
                    a = a0[[key, metric]].rename(columns={metric: "a"})
                    b = b0[[key, metric]].rename(columns={metric: "b"})
                    merged = a.merge(b, on=key, how="inner").dropna(subset=["a", "b"])
                    if merged.empty:
                        continue
                    av = merged["a"].to_numpy(dtype=float)
                    bv = merged["b"].to_numpy(dtype=float)
                    diff = av - bv
                    finite = np.isfinite(diff)
                    av, bv, diff = av[finite], bv[finite], diff[finite]
                    if len(diff) == 0:
                        continue

                    try:
                        if np.allclose(diff, 0):
                            w_stat, p_raw = 0.0, 1.0
                        else:
                            w = wilcoxon(av, bv, alternative="two-sided", zero_method="wilcox")
                            w_stat, p_raw = float(w.statistic), float(w.pvalue)
                    except Exception:
                        w_stat, p_raw = np.nan, np.nan

                    rng = np.random.default_rng(stable_seed(seed, band, day, method_a, method_b, metric))
                    ci_lo, ci_hi = paired_bootstrap_ci(diff, n_boot, rng)
                    rb = rank_biserial_paired(diff)

                    direction = METRIC_DIRECTION.get(metric, "higher")
                    mean_diff = float(np.mean(diff))
                    if direction == "higher":
                        winner = method_a if mean_diff > 0 else method_b if mean_diff < 0 else "tie"
                    else:
                        winner = method_a if mean_diff < 0 else method_b if mean_diff > 0 else "tie"

                    rows.append(
                        {
                            "band": band,
                            "day": int(day),
                            "metric": metric,
                            "direction": direction,
                            "method_a": method_a,
                            "method_b": method_b,
                            "pair_key": key,
                            "n_pairs": len(diff),
                            "mean_a": float(np.mean(av)),
                            "std_a": float(np.std(av, ddof=1)) if len(av) > 1 else np.nan,
                            "mean_b": float(np.mean(bv)),
                            "std_b": float(np.std(bv, ddof=1)) if len(bv) > 1 else np.nan,
                            "mean_diff_a_minus_b": mean_diff,
                            "median_diff_a_minus_b": float(np.median(diff)),
                            "diff_ci95_low": ci_lo,
                            "diff_ci95_high": ci_hi,
                            "wilcoxon_stat": w_stat,
                            "p_raw": p_raw,
                            "rank_biserial_a_minus_b": rb,
                            "winner_by_mean": winner,
                        }
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "band", "day", "metric", "direction", "method_a", "method_b",
            "pair_key", "n_pairs", "mean_a", "std_a", "mean_b", "std_b",
            "mean_diff_a_minus_b", "median_diff_a_minus_b", "diff_ci95_low",
            "diff_ci95_high", "wilcoxon_stat", "p_raw",
            "rank_biserial_a_minus_b", "winner_by_mean",
            "p_holm_across_days", "significant_0.05",
        ])

    # Holm correction across DAYS for each band x metric x method pair.
    out["p_holm_across_days"] = np.nan
    group_cols = ["band", "metric", "method_a", "method_b"]
    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        out.loc[idx, "p_holm_across_days"] = holm_adjust(out.loc[idx, "p_raw"].tolist())
    out["significant_0.05"] = out["p_holm_across_days"] < 0.05
    return out


def make_counts(per_run: pd.DataFrame, expected: int) -> pd.DataFrame:
    counts = (
        per_run.groupby(["method", "band", "day"], dropna=False)
        .size()
        .reset_index(name="n_runs")
    )
    counts["expected_runs"] = expected
    counts["complete"] = counts["n_runs"] == expected
    return counts


def format_mean_sd(mean: float, std: float, decimals: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make_nir_publication_summary(summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    metrics = ["silhouette_64d", "davies_bouldin_64d", "separation_ratio_64d"]
    rows = []
    nir = summary[(summary["band"] == "nir") & (summary["metric"].isin(metrics))]
    for day in sorted(nir["day"].dropna().unique()):
        for metric in metrics:
            row: Dict[str, Any] = {"day": int(day), "metric": metric}
            for method in ["raw", "dwt", "hswae"]:
                s = nir[(nir["day"] == day) & (nir["metric"] == metric) & (nir["method"] == method)]
                if not s.empty:
                    r = s.iloc[0]
                    row[f"{method}_mean"] = r["mean"]
                    row[f"{method}_std"] = r["std"]
                    row[f"{method}_ci95_low"] = r["ci95_low"]
                    row[f"{method}_ci95_high"] = r["ci95_high"]
                    row[f"{method}_mean_sd"] = format_mean_sd(r["mean"], r["std"])
            p = paired[
                (paired["band"] == "nir")
                & (paired["day"] == day)
                & (paired["metric"] == metric)
                & (paired["method_a"] == "dwt")
                & (paired["method_b"] == "hswae")
            ]
            if not p.empty:
                pr = p.iloc[0]
                row["dwt_vs_hswae_p_holm"] = pr["p_holm_across_days"]
                row["dwt_vs_hswae_rank_biserial"] = pr["rank_biserial_a_minus_b"]
                row["dwt_minus_hswae_mean_diff"] = pr["mean_diff_a_minus_b"]
                row["winner_by_mean"] = pr["winner_by_mean"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, band: str, outdir: Path) -> None:
    sub = summary[(summary["metric"] == metric) & (summary["band"] == band)].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method in ["raw", "dwt", "hswae"]:
        g = sub[sub["method"] == method].sort_values("day")
        if g.empty:
            continue
        x = g["day"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        lo = g["ci95_low"].to_numpy(dtype=float)
        hi = g["ci95_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=method)
        ax.fill_between(x, lo, hi, alpha=0.15)

    ax.set_xlabel("Acquisition day")
    ax.set_ylabel(metric.replace("_64d", "").replace("_", " ").title())
    ax.set_xticks(sorted(sub["day"].dropna().unique()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    stem = f"{band}_{metric}"
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    outdir: Path,
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    per_run: pd.DataFrame,
    expected: int,
) -> None:
    lines: List[str] = []
    lines.append("# Full Latent-Space Analysis Report")
    lines.append("")
    lines.append("All clustering/separation metrics in the primary analysis were computed directly in the original full-dimensional latent space (not on t-SNE coordinates).")
    lines.append("")

    incomplete = counts[~counts["complete"]]
    lines.append("## Run completeness")
    if incomplete.empty:
        lines.append(f"All discovered method × band × day conditions contain exactly {expected} runs.")
    else:
        lines.append(f"Some conditions do not contain exactly {expected} runs. Review `run_counts.csv` before using paired inference.")
        lines.append("")
        lines.append("```text")
        lines.append(incomplete.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## NIR: DWT-only vs HS-WAE")
    focus = paired[
        (paired["band"] == "nir")
        & (paired["method_a"] == "dwt")
        & (paired["method_b"] == "hswae")
        & (paired["metric"].isin(["silhouette_64d", "davies_bouldin_64d", "separation_ratio_64d"]))
    ].copy()
    if focus.empty:
        lines.append("No paired NIR DWT-only vs HS-WAE results were available.")
    else:
        cols = [
            "day",
            "metric",
            "n_pairs",
            "mean_a",
            "mean_b",
            "mean_diff_a_minus_b",
            "diff_ci95_low",
            "diff_ci95_high",
            "p_holm_across_days",
            "rank_biserial_a_minus_b",
            "winner_by_mean",
        ]
        lines.append("```text")
        lines.append(focus[cols].sort_values(["metric", "day"]).to_string(index=False))
        lines.append("```")
    lines.append("")

    neg = summary[(summary["metric"] == "silhouette_64d") & (summary["fraction_negative_silhouette"] > 0)]
    lines.append("## Negative silhouette diagnostics")
    if neg.empty:
        lines.append("No analyzed condition had negative full-space silhouette values across the extracted runs.")
    else:
        lines.append("The following conditions had at least one negative full-space silhouette value. These should not be described as well-separated clusters without qualification.")
        lines.append("")
        lines.append("```text")
        lines.append(
            neg[["method", "band", "day", "n_runs", "mean", "std", "fraction_negative_silhouette", "number_negative_silhouette"]]
            .sort_values(["band", "day", "method"])
            .to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    lines.append("## Checkpoint compatibility")
    if "saved_accuracy_matches" in per_run.columns:
        match = per_run["saved_accuracy_matches"].astype(str).str.lower() == "true"
        mismatches = per_run[~match & per_run["saved_test_accuracy"].notna()]
    else:
        mismatches = per_run.iloc[0:0]
    if mismatches.empty:
        lines.append("Extracted predictions reproduce all available saved test accuracies.")
    else:
        lines.append(
            f"{len(mismatches)} extracted runs do not reproduce their saved test accuracy "
            "with the current dataset. The stored model predictions were retained unchanged."
        )
        lines.append("")
        lines.append("```text")
        cols = ["day", "source_root", "saved_test_accuracy", "extracted_test_accuracy"]
        lines.append(mismatches[cols].to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("- `silhouette_64d`: higher is better; negative values indicate substantial overlap/misassignment relative to class labels.")
    lines.append("- `davies_bouldin_64d`: lower is better.")
    lines.append("- `separation_ratio_64d`: centroid distance divided by average within-class RMS dispersion; higher is better.")
    lines.append("- `fisher_ratio_64d`: squared between-class mean distance relative to pooled within-class variance; higher is better.")
    lines.append("- `*_zscore64d` metrics are sensitivity analyses after coordinate-wise standardization of the same full latent vectors; they are not t-SNE metrics.")
    lines.append("- Paired inference uses the stored seed when it is demonstrably unique and varying across runs; otherwise it uses deterministic within-condition replicate order. Check `pair_key` before reporting paired tests.")

    (outdir / "latent_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir else root / "analysis"
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(root)
    manifest = manifest[manifest["method"].isin(args.methods)].copy()
    if args.bands is not None:
        manifest = manifest[manifest["band"].isin(args.bands)]
    if args.days is not None:
        manifest = manifest[manifest["day"].isin(args.days)]

    rows: List[Dict[str, Any]] = []
    for _, r in manifest.iterrows():
        npz_path = Path(r["npz_path"])
        if not npz_path.exists():
            warnings.warn(f"Missing extracted NPZ: {npz_path}")
            continue
        data = np.load(npz_path)
        Z = data["latent"]
        y = data["y_true"]
        metrics = compute_metrics(Z, y)
        row = r.to_dict()
        row.update(metrics)
        rows.append(row)

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        raise RuntimeError("No extracted runs could be analyzed.")

    per_run = assign_replicate_index(per_run)
    per_run.to_csv(outdir / "per_run_latent_metrics.csv", index=False)

    counts = make_counts(per_run, args.expected_runs)
    counts.to_csv(outdir / "run_counts.csv", index=False)

    metric_cols = [c for c in METRIC_DIRECTION if c in per_run.columns]
    summary = summarize(per_run, metric_cols, args.bootstrap, args.seed)
    summary.to_csv(outdir / "summary_latent_metrics.csv", index=False)

    paired = paired_tests(per_run, metric_cols, args.bootstrap, args.seed)
    paired.to_csv(outdir / "paired_comparisons.csv", index=False)

    dwt_hs = paired[(paired["method_a"] == "dwt") & (paired["method_b"] == "hswae")].copy()
    dwt_hs.to_csv(outdir / "paired_dwt_vs_hswae.csv", index=False)

    nir_table = make_nir_publication_summary(summary, paired)
    nir_table.to_csv(outdir / "nir_publication_summary.csv", index=False)

    # Save alpha summaries when WTAM alpha is available in per-sample CSVs.
    alpha_rows = []
    for _, r in manifest[manifest["method"] == "hswae"].iterrows():
        csv_path = Path(r["samples_csv"])
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
        if not alpha_cols:
            continue
        for alpha_col in alpha_cols:
            for label, g in df.groupby("y_true"):
                vals = pd.to_numeric(g[alpha_col], errors="coerce").dropna().to_numpy()
                alpha_rows.append(
                    {
                        "method": "hswae",
                        "band": r["band"],
                        "day": r["day"],
                        "version_number": r.get("version_number"),
                        "config_seed": r.get("config_seed"),
                        "alpha_level": alpha_col.replace("alpha_", ""),
                        "y_true": int(label),
                        "n_samples": len(vals),
                        "mean_alpha": float(np.mean(vals)) if len(vals) else np.nan,
                        "std_alpha": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    }
                )
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        alpha_df.to_csv(outdir / "wtam_alpha_per_run_class_summary.csv", index=False)
        alpha_summary_rows = []
        for keys, group in alpha_df.groupby(["band", "day", "alpha_level", "y_true"]):
            values = group["mean_alpha"].to_numpy(dtype=float)
            rng = np.random.default_rng(stable_seed(args.seed, "alpha", *keys))
            ci_low, ci_high = bootstrap_mean_ci(values, args.bootstrap, rng)
            alpha_summary_rows.append({
                "band": keys[0], "day": keys[1], "alpha_level": keys[2],
                "y_true": keys[3], "n_runs": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)), "min": float(np.min(values)),
                "max": float(np.max(values)), "ci95_low": ci_low, "ci95_high": ci_high,
            })
        pd.DataFrame(alpha_summary_rows).to_csv(
            outdir / "wtam_alpha_summary.csv", index=False
        )

    # Primary NIR figures and, if desired, the same metrics for every available band.
    available_bands = sorted(summary["band"].dropna().unique())
    for band in available_bands:
        for metric in ["silhouette_64d", "davies_bouldin_64d", "separation_ratio_64d"]:
            plot_metric(summary, metric, band, figdir)

    write_report(outdir, counts, summary, paired, per_run, args.expected_runs)

    print("\n" + "=" * 80)
    print(f"Analyzed runs: {len(per_run)}")
    print(f"Conditions: {len(counts)}")
    print(f"Output directory: {outdir.resolve()}")
    print("Key files:")
    for name in [
        "run_counts.csv",
        "per_run_latent_metrics.csv",
        "summary_latent_metrics.csv",
        "paired_dwt_vs_hswae.csv",
        "nir_publication_summary.csv",
        "latent_analysis_report.md",
    ]:
        print(f"  - {outdir / name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

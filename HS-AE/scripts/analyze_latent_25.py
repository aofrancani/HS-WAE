"""Analyze paired 25-run DWT-only vs HS-WAE full latent-space results."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

PRIMARY_METRICS = [
    "test/acc",
    "test/f1",
    "test/auroc",
    "silhouette_64d",
    "davies_bouldin_64d",
    "separation_ratio_64d",
    "fisher_ratio_64d",
]
# Orientation used for inferential effect/difference: positive = HS-WAE better.
DIRECTION = {
    "test/acc": +1,
    "test/f1": +1,
    "test/auroc": +1,
    "silhouette_64d": +1,
    "davies_bouldin_64d": -1,
    "separation_ratio_64d": +1,
    "fisher_ratio_64d": +1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/latent_25")
    p.add_argument("--dwt-root", default=None)
    p.add_argument("--hswae-root", default=None)
    p.add_argument("--band", default="nir")
    p.add_argument("--days", nargs="+", type=int, default=list(range(1, 8)))
    p.add_argument("--dwt-method", default="dwt")
    p.add_argument("--hswae-method", default="dwt_attn")
    p.add_argument("--expected-runs", type=int, default=25)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260910)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def find_condition_file(root: Path, method: str, band: str, day: int) -> Path:
    candidates = list((root / method / band).glob(f"day_{day:02d}/metrics_latent_runs.csv"))
    if not candidates:
        candidates = list((root / method / band).glob(f"*{day:02d}*/metrics_latent_runs.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one metrics file for {method}/{band}/day {day}; found {candidates}")
    return candidates[0]


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = x[idx].mean(axis=1)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm family-wise error correction, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def rank_biserial_paired(oriented_diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; positive favors HS-WAE."""
    d = np.asarray(oriented_diff, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    denom = r_plus + r_minus
    return float((r_plus - r_minus) / denom) if denom else 0.0


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def sample_std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def main():
    a = parse_args()
    root = Path(a.root)
    dwt_root = Path(a.dwt_root) if a.dwt_root else root
    hswae_root = Path(a.hswae_root) if a.hswae_root else root
    outdir = Path(a.outdir) if a.outdir else root / "analysis_25"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.bootstrap_seed)

    condition_rows, all_frames = [], []
    paired_rows = []

    for day in a.days:
        fd = find_condition_file(dwt_root, a.dwt_method, a.band, day)
        fh = find_condition_file(hswae_root, a.hswae_method, a.band, day)
        d = pd.read_csv(fd)
        h = pd.read_csv(fh)
        d["source_file"] = str(fd); h["source_file"] = str(fh)
        all_frames.extend([d, h])

        for method, df in [(a.dwt_method, d), (a.hswae_method, h)]:
            condition_rows.append({
                "method": method, "band": a.band, "day": day,
                "n_runs": len(df),
                "n_unique_exp_idx": df["exp_idx"].nunique(),
                "n_unique_seed": df["seed"].nunique(),
                "latent_dims": ";".join(map(str, sorted(df["latent_dim"].dropna().astype(int).unique()))),
                "n_negative_silhouette": int((df["silhouette_64d"] < 0).sum()),
                "fraction_negative_silhouette": float((df["silhouette_64d"] < 0).mean()),
                "expected_runs": a.expected_runs,
                "complete": len(df) == a.expected_runs,
            })

        # Pair by BOTH exp_idx and seed; this prevents accidental misalignment.
        key = ["exp_idx", "seed"]
        merged = d.merge(h, on=key, suffixes=("_dwt", "_hswae"), validate="one_to_one")
        if len(merged) != a.expected_runs:
            print(f"[WARNING] Day {day}: expected {a.expected_runs} pairs, found {len(merged)}")

        for metric in PRIMARY_METRICS:
            xd = merged[f"{metric}_dwt"].to_numpy(float)
            xh = merged[f"{metric}_hswae"].to_numpy(float)
            raw_diff = xh - xd
            oriented = DIRECTION[metric] * raw_diff
            valid = np.isfinite(oriented)
            xd, xh, raw_diff, oriented = xd[valid], xh[valid], raw_diff[valid], oriented[valid]

            if len(oriented) > 0 and np.any(oriented != 0):
                stat, p = wilcoxon(oriented, zero_method="wilcox", alternative="two-sided", method="auto")
                p = float(p); stat = float(stat)
            else:
                stat, p = np.nan, 1.0
            ci_lo, ci_hi = bootstrap_mean_ci(oriented, a.bootstrap, rng)

            paired_rows.append({
                "band": a.band, "day": day, "metric": metric,
                "n_pairs": len(oriented),
                "mean_dwt": float(np.mean(xd)), "std_dwt": sample_std(xd),
                "mean_hswae": float(np.mean(xh)), "std_hswae": sample_std(xh),
                "raw_mean_hswae_minus_dwt": float(np.mean(raw_diff)),
                "oriented_mean_difference_positive_favors_hswae": float(np.mean(oriented)),
                "oriented_median_difference": float(np.median(oriented)),
                "oriented_ci95_low": ci_lo, "oriented_ci95_high": ci_hi,
                "wilcoxon_stat": stat, "p_raw": p,
                "rank_biserial_positive_favors_hswae": rank_biserial_paired(oriented),
                "hswae_better_count": int(np.sum(oriented > 0)),
                "tie_count": int(np.sum(oriented == 0)),
                "dwt_better_count": int(np.sum(oriented < 0)),
            })

    counts = pd.DataFrame(condition_rows)
    counts.to_csv(outdir / "run_counts_25.csv", index=False)
    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv(outdir / "all_per_run_metrics.csv", index=False)

    paired = pd.DataFrame(paired_rows)
    # Holm correction across the seven days separately for each metric.
    paired["p_holm_across_days"] = np.nan
    for metric, idx in paired.groupby("metric").groups.items():
        ids = list(idx)
        paired.loc[ids, "p_holm_across_days"] = holm_adjust(
            paired.loc[ids, "p_raw"].to_numpy(float)
        )
    paired["significant_holm_0_05"] = paired["p_holm_across_days"] < .05
    paired.to_csv(outdir / "paired_dwt_vs_hswae_25.csv", index=False)

    # Descriptive summary of every stored metric.
    metric_cols = [c for c in all_df.columns if c.endswith("_64d") or c.endswith("zscore64d")]
    desc_rows = []
    for (method, days), g in all_df.groupby(["method", "days"]):
        try:
            day = int(str(days).split(",")[0])
        except Exception:
            day = days
        for metric in metric_cols:
            x = g[metric].dropna().to_numpy(float)
            if not len(x): continue
            lo, hi = bootstrap_mean_ci(x, a.bootstrap, rng)
            desc_rows.append({"method": method, "band": a.band, "day": day, "metric": metric,
                              "n": len(x), "mean": np.mean(x), "std": sample_std(x),
                              "median": np.median(x), "min": np.min(x), "max": np.max(x),
                              "ci95_low": lo, "ci95_high": hi,
                              "number_negative_silhouette": int(np.sum(x < 0)) if "silhouette" in metric else np.nan,
                              "fraction_negative_silhouette": float(np.mean(x < 0)) if "silhouette" in metric else np.nan})
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(outdir / "summary_latent_metrics_25.csv", index=False)

    # One global paired test per metric: average each seed across days first (n=25).
    global_rows = []
    for metric in PRIMARY_METRICS:
        pieces = []
        for day in a.days:
            fd = pd.read_csv(find_condition_file(dwt_root, a.dwt_method, a.band, day))[["exp_idx","seed",metric]]
            fh = pd.read_csv(find_condition_file(hswae_root, a.hswae_method, a.band, day))[["exp_idx","seed",metric]]
            m = fd.merge(fh, on=["exp_idx","seed"], suffixes=("_dwt","_hswae"))
            m["day"] = day
            m["oriented"] = DIRECTION[metric] * (m[f"{metric}_hswae"] - m[f"{metric}_dwt"])
            pieces.append(m[["exp_idx","seed","day","oriented"]])
        gg = pd.concat(pieces).groupby(["exp_idx","seed"], as_index=False)["oriented"].mean()
        x = gg["oriented"].to_numpy(float)
        stat, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto") if np.any(x != 0) else (np.nan, 1.0)
        lo, hi = bootstrap_mean_ci(x, a.bootstrap, rng)
        global_rows.append({"metric":metric,"n_seed_pairs":len(x),"mean_oriented_difference":np.mean(x),
                            "median_oriented_difference":np.median(x),"ci95_low":lo,"ci95_high":hi,
                            "wilcoxon_stat":stat,"p_raw":p,
                            "rank_biserial_positive_favors_hswae":rank_biserial_paired(x),
                            "hswae_better_seed_count":int(np.sum(x>0)),"dwt_better_seed_count":int(np.sum(x<0)),
                            "tie_seed_count":int(np.sum(x==0))})
    global_df = pd.DataFrame(global_rows)
    global_df["p_holm_across_metrics"] = holm_adjust(global_df["p_raw"].to_numpy(float))
    global_df.to_csv(outdir / "global_seed_averaged_dwt_vs_hswae.csv", index=False)

    # Compact publication table: mean±SD + corrected p/effect for primary metrics.
    pub = paired.copy()
    pub["DWT-only"] = [fmt(m,s) for m,s in zip(pub.mean_dwt, pub.std_dwt)]
    pub["HS-WAE"] = [fmt(m,s) for m,s in zip(pub.mean_hswae, pub.std_hswae)]
    pub = pub[["day","metric","DWT-only","HS-WAE","oriented_mean_difference_positive_favors_hswae","oriented_ci95_low","oriented_ci95_high",
               "p_holm_across_days","rank_biserial_positive_favors_hswae","hswae_better_count","dwt_better_count","tie_count"]]
    pub.to_csv(outdir / "publication_table_latent_25.csv", index=False)

    # Plots for the two most interpretable metrics.
    for metric, ylabel in [("silhouette_64d", "Silhouette score (64-D)"),
                            ("davies_bouldin_64d", "Davies–Bouldin index (64-D)")]:
        fig, ax = plt.subplots(figsize=(6.4,4.2))
        for method, label in [(a.dwt_method,"DWT-only"),(a.hswae_method,"HS-WAE")]:
            means=[]; lows=[]; highs=[]
            for day in a.days:
                row = desc[(desc.method==method)&(desc.day==day)&(desc.metric==metric)]
                means.append(float(row.iloc[0]["mean"])); lows.append(float(row.iloc[0]["ci95_low"])); highs.append(float(row.iloc[0]["ci95_high"]))
            means=np.asarray(means); lows=np.asarray(lows); highs=np.asarray(highs)
            ax.errorbar(a.days, means, yerr=np.vstack([means-lows, highs-means]), marker="o", capsize=3, label=label)
        ax.set_xlabel("Acquisition day"); ax.set_ylabel(ylabel); ax.set_xticks(a.days); ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(outdir/"figures"/f"nir_{metric}_dwt_vs_hswae.pdf", bbox_inches="tight"); plt.close(fig)

    # Human-readable report, intentionally conservative.
    primary = paired[paired["metric"].isin(PRIMARY_METRICS)].copy()
    significant = primary[primary["significant_holm_0_05"]]
    favors_hswae = significant[significant["oriented_mean_difference_positive_favors_hswae"] > 0]
    favors_dwt = significant[significant["oriented_mean_difference_positive_favors_hswae"] < 0]
    no_difference = primary[~primary["significant_holm_0_05"]]
    neg = desc[(desc["metric"] == "silhouette_64d") & (desc["number_negative_silhouette"] > 0)]

    def day_metric_list(frame):
        if frame.empty:
            return "None."
        return ", ".join(
            f"Day {int(r.day)} ({r.metric})" for r in frame.itertuples()
        ) + "."

    lines = [
        "# 25-run full latent-space analysis", "",
        f"Band: {a.band}",
        f"Expected paired runs per day: {a.expected_runs}", "",
        "Primary metrics were calculated directly in the original 64-D latent space. t-SNE is not used for inference.", "",
        "## Completeness", "",
        text_table(counts), "",
        "## Paired DWT-only vs HS-WAE", "",
        text_table(paired[["day","metric","n_pairs","mean_dwt","mean_hswae","oriented_mean_difference_positive_favors_hswae","oriented_ci95_low","oriented_ci95_high","p_holm_across_days","rank_biserial_positive_favors_hswae"]]), "",
        "## Holm-adjusted conclusions", "",
        f"Significantly favoring HS-WAE: {day_metric_list(favors_hswae)}",
        f"Significantly favoring DWT-only: {day_metric_list(favors_dwt)}",
        f"No significant difference: {day_metric_list(no_difference)}", "",
        "## Negative Silhouette occurrences", "",
        ("None." if neg.empty else text_table(neg[["method", "day", "number_negative_silhouette", "fraction_negative_silhouette"]])), "",
        "## Global seed-averaged comparison", "",
        text_table(global_df), "",
        "Interpretation: all inferential differences are oriented so positive values favor HS-WAE. For Davies–Bouldin, the sign is reversed because lower is better.",
    ]
    (outdir / "latent_25_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved analysis to {outdir}")


if __name__ == "__main__":
    main()

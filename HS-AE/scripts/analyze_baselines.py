#!/usr/bin/env python3
"""Aggregate baseline results and optionally compare against archived HS-WAE stats."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def bootstrap_ci(x, n_boot=10000, seed=12345):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid): return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m-rank)*p[idx])
        running = max(running, adj)
        out[idx] = running
    return out


def rank_biserial_from_diffs(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0: return 0.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    wp = ranks[d > 0].sum(); wn = ranks[d < 0].sum()
    return float((wp-wn)/(wp+wn)) if wp+wn else 0.0


def collect(root: Path):
    frames = []
    for p in root.glob("*/*/day_*/**/metrics_runs.csv"):
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception as e:
            print("Skip", p, e)
    if not frames:
        raise SystemExit(f"No metrics_runs.csv found under {root}")
    return pd.concat(frames, ignore_index=True)


def hswae_file(root: Path, band: str, day: int):
    # Support common naming in this project.
    candidates = [
        root / f"metrics_stats_{band}_day_{day:02d}.csv",
        root / f"metrics_stats_{band}_day_{day}.csv",
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def normalize_hswae(df):
    aliases = {
        "accuracy": ["test/acc", "test_acc", "accuracy", "acc"],
        "f1": ["test/f1", "test_f1", "f1"],
        "auroc": ["test/auroc", "test_auroc", "auroc", "auc"],
        "precision": ["test/precision", "test_precision", "precision"],
        "recall": ["test/recall", "test_recall", "recall"],
        "fpr": ["test/fpr", "test_fpr", "fpr"],
    }
    out = df.copy()
    for target, opts in aliases.items():
        for o in opts:
            if o in out.columns:
                out[target] = out[o]
                break
    if "exp_idx" not in out: out["exp_idx"] = np.arange(len(out))
    return out


def main(args):
    root = Path(args.root)
    out = root / "analysis"
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(exist_ok=True)
    df = collect(root)

    # Completeness
    counts = df.groupby(["feature_mode","band_region","day","model"]).agg(
        n_runs=("seed","count"), n_seeds=("seed","nunique")
    ).reset_index()
    counts["expected_runs"] = args.expected_runs
    counts["complete"] = counts["n_runs"].eq(args.expected_runs)
    counts.to_csv(out/"run_counts.csv", index=False)

    metrics = ["accuracy","f1","auroc","precision","recall","fpr"]
    summary_rows = []
    for keys, g in df.groupby(["feature_mode","band_region","day","model"]):
        for metric in metrics:
            x = g[metric].dropna().to_numpy(float)
            if len(x)==0: continue
            lo, hi = bootstrap_ci(x, args.bootstrap, args.bootstrap_seed)
            summary_rows.append({
                "feature_mode":keys[0],"band_region":keys[1],"day":keys[2],"model":keys[3],"metric":metric,
                "n":len(x),"mean":x.mean(),"std":x.std(ddof=1) if len(x)>1 else 0.0,
                "median":np.median(x),"ci95_low":lo,"ci95_high":hi,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out/"summary_metrics.csv", index=False)

    # Feature-reduction provenance (PCA dimensions / explained variance / selected bands).
    tx_rows=[]
    import json
    for pth in root.glob("*/*/day_*/feature_transform.json"):
        try:
            d=json.loads(pth.read_text())
            parts=pth.relative_to(root).parts
            tx_rows.append({"feature_mode":parts[0],"band_region":parts[1],"day":int(parts[2].split("_")[-1]),**d})
        except Exception as e:
            print("Could not read transform",pth,e)
    if tx_rows:
        pd.DataFrame(tx_rows).to_csv(out/"feature_transform_summary.csv",index=False)

    reduction = summary[(summary.metric.isin(["accuracy","f1","auroc"])) & (summary.feature_mode.isin(["raw","pca95","pca64","selectk"]))].copy()
    reduction["mean_sd"] = reduction.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    reduction.to_csv(out/"reduction_comparison_table.csv", index=False)

    # Compact NIR table for early detection Days 1-3.
    early = summary[(summary.band_region=="nir") & (summary.day.isin([1,2,3])) & (summary.metric.isin(["accuracy","f1","auroc"]))].copy()
    early["mean_sd"] = early.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    early.to_csv(out/"nir_early_days_table.csv", index=False)

    paired_rows=[]
    if args.hswae_root:
        hroot=Path(args.hswae_root)
        for (fm,band,day,model), g in df.groupby(["feature_mode","band_region","day","model"]):
            hp=hswae_file(hroot, band, int(day))
            if hp is None: continue
            h=normalize_hswae(pd.read_csv(hp))
            if "seed" not in h.columns: continue
            merged=g.merge(h[["exp_idx","seed"]+[m for m in metrics if m in h.columns]], on=["exp_idx","seed"], suffixes=("_base","_hswae"))
            for metric in ["accuracy","f1","auroc"]:
                a=f"{metric}_base"; b=f"{metric}_hswae"
                if a not in merged or b not in merged: continue
                # positive = HS-WAE favored (except FPR, not analyzed here)
                diff=merged[b].to_numpy(float)-merged[a].to_numpy(float)
                lo,hi=bootstrap_ci(diff,args.bootstrap,args.bootstrap_seed)
                try:
                    p=float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue) if np.any(diff!=0) else 1.0
                except ValueError: p=1.0
                paired_rows.append({"feature_mode":fm,"band_region":band,"day":day,"model":model,"metric":metric,
                    "n_pairs":len(diff),"baseline_mean":merged[a].mean(),"hswae_mean":merged[b].mean(),
                    "mean_diff_hswae_minus_baseline":diff.mean(),"ci95_low":lo,"ci95_high":hi,"p_raw":p,
                    "rank_biserial":rank_biserial_from_diffs(diff),
                    "baseline_zero_sd":bool(np.isclose(merged[a].std(ddof=1),0.0))})
        paired=pd.DataFrame(paired_rows)
        if len(paired):
            # Holm within each feature/model/metric across 7 days.
            paired["p_holm"] = np.nan
            for _, idx in paired.groupby(["feature_mode","band_region","model","metric"]).groups.items():
                idx=list(idx); paired.loc[idx,"p_holm"]=holm_adjust(paired.loc[idx,"p_raw"].to_numpy())
            paired.to_csv(out/"paired_vs_hswae.csv", index=False)

    # Publication-style NIR raw temporal figures.
    for metric, ylabel in [("accuracy","Accuracy"),("f1","F1-score"),("auroc","AUROC")]:
        q=summary[(summary.feature_mode=="raw")&(summary.band_region=="nir")&(summary.metric==metric)].copy()
        if q.empty: continue
        plt.figure(figsize=(7,4.5))
        for model,g in q.groupby("model"):
            g=g.sort_values("day")
            yerr=np.vstack([g["mean"]-g["ci95_low"],g["ci95_high"]-g["mean"]])
            plt.errorbar(g["day"],g["mean"],yerr=yerr,marker="o",capsize=3,label=model.upper())
        plt.xlabel("Acquisition day"); plt.ylabel(ylabel); plt.xticks(sorted(q.day.unique())); plt.ylim(0,1.03)
        plt.legend(ncol=2,frameon=False); plt.tight_layout()
        plt.savefig(figs/f"nir_raw_{metric}_baselines.pdf",bbox_inches="tight")
        plt.savefig(figs/f"nir_raw_{metric}_baselines.png",dpi=300,bbox_inches="tight")
        plt.close()

    # Report
    lines=["# Baseline Analysis","",f"Rows analyzed: {len(df)}",""]
    incomplete=counts[~counts.complete]
    lines.append(f"Incomplete conditions: {len(incomplete)}")
    if len(incomplete): lines.append(incomplete.to_string(index=False))
    lines += ["","## Recommended interpretation","- Use NIR Days 1–3 as the primary early-detection baseline comparison.",
              "- Use the full Day 1–7 NIR curves as temporal robustness evidence.",
              "- PCA transforms were fit on training data only; test data were never used for feature extraction or tuning.",
              "- SVM may be effectively deterministic under a fixed split; do not interpret repeated identical SVM values as stochastic uncertainty."]
    (out/"baseline_analysis_report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Analysis saved to",out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/baselines")
    p.add_argument("--hswae-root",default=None,help="e.g. logs/stats/dwt_attn")
    p.add_argument("--expected-runs",type=int,default=25)
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--bootstrap-seed",type=int,default=20260910)
    main(p.parse_args())

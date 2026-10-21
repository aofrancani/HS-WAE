from __future__ import annotations

"""
Optional qualitative t-SNE stability plots from ALREADY EXTRACTED full latent vectors.

Important: this script does not compute or report clustering-quality metrics on t-SNE.
Its purpose is only to show how the qualitative 2-D visualization changes with t-SNE
random initialization while the underlying latent vectors remain fixed.
"""

import argparse
import json
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate t-SNE sensitivity plots from saved full latents.")
    p.add_argument("--root", default="results/latent")
    p.add_argument("--method", required=True, choices=["raw", "dwt", "hswae"])
    p.add_argument("--band", default="nir")
    p.add_argument("--day", type=int, required=True)
    p.add_argument(
        "--replicate-index",
        type=int,
        default=12,
        help="0-based replicate after sorting runs by version/path; 12 is the median index for 25 runs.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 21, 42, 84])
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def parse_day(v: Any):
    import re
    nums = [int(x) for x in re.findall(r"\d+", str(v))]
    return nums[0] if len(nums) == 1 else None


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest = pd.read_csv(root / "manifest.csv")
    if "days" in manifest.columns:
        manifest["day"] = manifest["days"].apply(parse_day)
    elif "day" not in manifest.columns:
        raise KeyError("Manifest contains neither 'days' nor 'day'")
    sub = manifest[
        (manifest["method"] == args.method)
        & (manifest["band"] == args.band)
        & (manifest["day"] == args.day)
    ].copy()
    if sub.empty:
        raise RuntimeError("No matching extracted runs found.")

    sort_cols = [c for c in ["replicate_id", "version_number", "checkpoint_mtime", "run_dir"] if c in sub.columns]
    sub = sub.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if args.replicate_index < 0 or args.replicate_index >= len(sub):
        raise IndexError(f"replicate-index {args.replicate_index} out of range for {len(sub)} runs")

    row = sub.iloc[args.replicate_index]
    data = np.load(Path(row["npz_path"]))
    Z = data["latent"]
    y = data["y_true"]

    # t-SNE requires perplexity < n_samples.
    perplexity = min(args.perplexity, max(2.0, len(y) - 1.0))

    outdir = Path(args.outdir) if args.outdir else root / "analysis" / "tsne_stability"
    outdir.mkdir(parents=True, exist_ok=True)

    coords_rows: List[pd.DataFrame] = []
    for seed in args.seeds:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        P = tsne.fit_transform(Z)
        coords = pd.DataFrame(
            {
                "tsne_1": P[:, 0],
                "tsne_2": P[:, 1],
                "y_true": y,
                "tsne_seed": seed,
                "test_order": np.arange(len(y)),
            }
        )
        coords_rows.append(coords)

        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        for label in sorted(np.unique(y)):
            mask = y == label
            ax.scatter(P[mask, 0], P[mask, 1], s=28, label=f"Class {label}")
        ax.set_xlabel(r"t-SNE$_1$")
        ax.set_ylabel(r"t-SNE$_2$")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        stem = f"{args.method}_{args.band}_day_{args.day:02d}_rep_{args.replicate_index:02d}_tsne_seed_{seed}"
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    pd.concat(coords_rows, ignore_index=True).to_csv(
        outdir / f"{args.method}_{args.band}_day_{args.day:02d}_rep_{args.replicate_index:02d}_coordinates.csv",
        index=False,
    )

    provenance = {
        "method": args.method,
        "band": args.band,
        "day": args.day,
        "replicate_index": args.replicate_index,
        "source_run_dir": row["run_dir"],
        "source_checkpoint": row["checkpoint_path"],
        "perplexity": perplexity,
        "tsne_seeds": args.seeds,
        "note": "Qualitative sensitivity visualization only. No clustering metrics are computed on t-SNE coordinates.",
    }
    (outdir / f"{args.method}_{args.band}_day_{args.day:02d}_rep_{args.replicate_index:02d}_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Saved t-SNE stability outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Compose an existing four-seed t-SNE coordinate table into a 2x2 figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_COORDINATES = (
    "results/latent/analysis/tsne_stability/"
    "hswae_nir_day_02_rep_00_coordinates.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", default=DEFAULT_COORDINATES)
    parser.add_argument("--seeds", nargs=4, type=int, default=[0, 21, 42, 84])
    parser.add_argument("--outdir", default="results/latent/analysis/tsne_stability")
    parser.add_argument("--stem", default="hswae_nir_day_02_rep_00_tsne_seed_grid")
    args = parser.parse_args()

    source = Path(args.coordinates)
    frame = pd.read_csv(source)
    required = {"tsne_1", "tsne_2", "y_true", "tsne_seed", "test_order"}
    if not required.issubset(frame.columns):
        raise KeyError(f"Missing columns: {sorted(required - set(frame.columns))}")
    available = set(frame["tsne_seed"].astype(int))
    if not set(args.seeds).issubset(available):
        raise ValueError(f"Requested seeds {args.seeds}; available seeds {sorted(available)}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    colors = {0: "#299029", 1: "#d62728"}
    markers = {0: "o", 1: "x"}
    names = {0: "Control", 1: "Damaged"}
    handles = []

    for panel, (ax, seed) in enumerate(zip(axes.flat, args.seeds)):
        selected = frame[frame["tsne_seed"].astype(int) == seed].sort_values("test_order")
        for label in sorted(selected["y_true"].unique()):
            group = selected[selected["y_true"] == label]
            kwargs = {
                "s": 34,
                "marker": markers.get(int(label), "o"),
                "color": colors.get(int(label), "#1f77b4"),
                "alpha": 0.88,
                "linewidths": 1.1,
            }
            if markers.get(int(label), "o") == "o":
                kwargs.update(edgecolors="white", linewidths=0.45)
            artist = ax.scatter(group["tsne_1"], group["tsne_2"],
                                label=names.get(int(label), f"Class {label}"), **kwargs)
            if panel == 0:
                handles.append(artist)
        ax.set_title(f"t-SNE seed = {seed}", fontsize=11)
        ax.set_xlabel(r"t-SNE$_1$")
        ax.set_ylabel(r"t-SNE$_2$")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("HS-WAE latent space — Soybean NIR, Day 2", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{args.stem}.pdf"
    png = outdir / f"{args.stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(pdf.resolve())
    print(png.resolve())


if __name__ == "__main__":
    main()

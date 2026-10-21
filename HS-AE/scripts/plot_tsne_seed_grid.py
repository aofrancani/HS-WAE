from __future__ import annotations

"""Plot a 2x2 t-SNE random-seed sensitivity grid from one fixed latent array."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

DEFAULT_SOURCE = (
    "results/latent_25/dwt_attn/nir/day_02/"
    "run_00_seed_2/test_latent_outputs.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x2 t-SNE seed-sensitivity figure from fixed latent vectors."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", nargs=4, type=int, default=[0, 21, 42, 84])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--init", choices=["random", "pca"], default="random")
    parser.add_argument("--outdir", default="figures/tsne_seed_stability")
    parser.add_argument("--stem", default="hswae_nir_day_02_tsne_seed_stability")
    return parser.parse_args()


def neighborhood_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a) - 1)
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.load(source, allow_pickle=True)
    if "latent" not in data.files or "y_true" not in data.files:
        raise KeyError("Source NPZ must contain 'latent' and 'y_true'.")
    latent = np.asarray(data["latent"], dtype=np.float64)
    labels = np.asarray(data["y_true"], dtype=int)
    if latent.ndim != 2 or len(latent) != len(labels):
        raise ValueError(f"Invalid latent/label shapes: {latent.shape}, {labels.shape}")
    perplexity = min(float(args.perplexity), float(len(labels) - 1))

    embeddings: dict[int, np.ndarray] = {}
    coordinate_rows = []
    for seed in args.seeds:
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init=args.init,
            learning_rate="auto",
            method="barnes_hut",
        ).fit_transform(latent)
        embeddings[seed] = embedding
        coordinate_rows.append(pd.DataFrame({
            "tsne_seed": seed,
            "test_order": np.arange(len(labels)),
            "y_true": labels,
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
        }))

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    colors = {0: "#299029", 1: "#d62728"}
    markers = {0: "o", 1: "x"}
    names = {0: "Control", 1: "Damaged"}
    handles = []
    for panel, (ax, seed) in enumerate(zip(axes.flat, args.seeds)):
        embedding = embeddings[seed]
        for label in sorted(np.unique(labels)):
            mask = labels == label
            kwargs = dict(s=34, marker=markers.get(label, "o"), color=colors.get(label),
                          alpha=0.88, linewidths=1.1)
            if markers.get(label, "o") == "o":
                kwargs.update(edgecolors="white", linewidths=0.45)
            artist = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                                label=names.get(label, f"Class {label}"), **kwargs)
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
    csv = outdir / f"{args.stem}_coordinates.csv"
    diagnostics_path = outdir / f"{args.stem}_stability_diagnostics.csv"
    provenance_path = outdir / f"{args.stem}_provenance.json"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.concat(coordinate_rows, ignore_index=True).to_csv(csv, index=False)

    diagnostics = []
    for seed_a, seed_b in combinations(args.seeds, 2):
        a, b = embeddings[seed_a], embeddings[seed_b]
        _, _, disparity = procrustes(a, b)
        diagnostics.append({
            "seed_a": seed_a,
            "seed_b": seed_b,
            "pairwise_distance_spearman": float(spearmanr(pdist(a), pdist(b)).statistic),
            "knn_overlap_k5": neighborhood_overlap(a, b, 5),
            "knn_overlap_k10": neighborhood_overlap(a, b, 10),
            "procrustes_disparity": float(disparity),
        })
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)

    provenance = {
        "source_npz": str(source.resolve()),
        "fixed_model_exp_idx": int(data["exp_idx"]) if "exp_idx" in data.files else None,
        "fixed_model_seed": int(data["seed"]) if "seed" in data.files else None,
        "dataset": "soybean",
        "method": "HS-WAE",
        "band": "nir",
        "day": 2,
        "n_test": int(len(labels)),
        "latent_dimension": int(latent.shape[1]),
        "class_counts": {str(v): int((labels == v).sum()) for v in np.unique(labels)},
        "tsne_seeds": args.seeds,
        "perplexity": perplexity,
        "init": args.init,
        "learning_rate": "auto",
        "method_implementation": "sklearn.manifold.TSNE, Barnes-Hut",
        "interpretation": "Qualitative random-initialization sensitivity only; no inferential separation metric is computed on t-SNE coordinates.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pdf": str(pdf.resolve()), "png": str(png.resolve()),
        "coordinates": str(csv.resolve()), "diagnostics": str(diagnostics_path.resolve()),
        "provenance": str(provenance_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

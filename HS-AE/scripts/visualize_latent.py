import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Optional
import lightning.pytorch as pl


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
from models.build_model import build_model
from dataloader import build_datamodule
from utils import find_checkpoint, resolve_run_name

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


def class_stats(name: str, ds):
    labels, counts = np.unique(ds.y.numpy(), return_counts=True)
    print(f"{name} class distribution:")
    for lbl, cnt in zip(labels, counts):
        print(f"  Class {lbl}: {cnt}")
    print(f"  Total: {len(ds)}\n")


def _extract_latent_from_batch(
    model: torch.nn.Module,
    batch: any,
    device: torch.device,
    dwt_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (latent, label) for one batch, for DWT and non-DWT modes."""

    if dwt_enabled:
        x_raw, levels, y = batch
        levels = [lvl.to(device) for lvl in levels]
        y = y.to(device)

        if hasattr(model, "wavelet_attn") and model.wavelet_attn is not None:
            out = model(levels, compute_loss=False)
            z = out["latent"]
        else:
            x_vec = torch.cat(levels, dim=1)
            if hasattr(model, "ae") and hasattr(model.ae, "encode"):
                z = model.ae.encode(x_vec)
            elif hasattr(model, "encode"):
                z = model.encode(x_vec)
            elif hasattr(model, "encoder"):
                z = model.encoder(x_vec)
            else:
                raise RuntimeError("Model has no encoder method for DWT path.")

    else:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        if hasattr(model, "ae") and hasattr(model.ae, "encode"):
            z = model.ae.encode(x)
        elif hasattr(model, "encode"):
            z = model.encode(x)
        elif hasattr(model, "encoder"):
            z = model.encoder(x)
        else:
            raise RuntimeError("Model has no encoder method for non-DWT path.")

    return z.cpu().numpy(), y.cpu().numpy()


def visualize_latent_space(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    crops: List[str],
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 60,
) -> pd.DataFrame:
    """
    Visualize the latent space of an autoencoder using t-SNE.

    Args:
        model: Trained model with encode() method
        datamodule: LightningDataModule containing the data
        crops: List of crop names
        save_path: Optional path to save the plot
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device

    # Collect latent representations and metadata
    latent_vectors = []
    labels = []
    crop_indices = []
    dataloader = datamodule.test_dataloader()
    dwt_enabled = bool(
        getattr(datamodule, "cfg", None)
        and getattr(datamodule.cfg, "dwt_enabled", False)
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            z_batch, y_batch = _extract_latent_from_batch(
                model, batch, device, dwt_enabled
            )
            latent_vectors.append(z_batch)
            labels.extend(y_batch.tolist())
            crop_idx = [int(idx) // 2 for idx in y_batch.tolist()]
            crop_indices.extend(crop_idx)

    # Process data
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    tsne = TSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    )
    latent_projected = tsne.fit_transform(latent_vectors)

    # Get label mapping
    label_names = datamodule.get_label_mapping()

    # Create DataFrame
    df_dict = {
        "x": latent_projected[:, 0],
        "y": latent_projected[:, 1],
        "Label": [label_names[label] for label in labels],
        "Crop": [crops[idx] for idx in crop_indices],
        "Label_id": labels,  # Keep numeric labels for coloring
    }
    if n_components == 3:
        df_dict["z"] = latent_projected[:, 2]

    return pd.DataFrame(df_dict)


def plot_latent_space(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    save_path: Optional[str] = None,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 10,
) -> None:
    """
    Plot the latent space visualization.

    Args:
        df: DataFrame with columns ['x', 'y', 'z'(optional), 'Label', 'Crop']
        n_components: Number of t-SNE components (2 or 3)
        perplexity: t-SNE perplexity parameter
        save_path: Path to save the plot
        title_fontsize: Font size for plot titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend text
    """
    # plt.style.use("default")
    # sns.set_theme(style="whitegrid")

    # Check if there's only one crop type
    unique_crops = df["Crop"].unique()
    single_crop = len(unique_crops) == 1
    crop_name = unique_crops[0] if single_crop else None
    marker_size = 40

    # Define consistent styles for Control and Damaged
    markers = {"Control": "o", "Damaged": "x"}
    colors = {"Control": "#2ca02c", "Damaged": "#d62728"}  # green  # red

    if n_components == 2:
        if single_crop:
            fig, ax = plt.subplots(figsize=(6, 5))
            # Filter DataFrame first
            # df = df[df["x"] < -4.0]
            # Plot each class separately with consistent style
            for label in df["Label"].unique():
                mask = df["Label"] == label

                is_control = "Control" in label
                style = "Control" if is_control else "Damaged"

                ax.scatter(
                    df.loc[mask, "x"],
                    df.loc[mask, "y"],
                    c=colors[style],
                    marker=markers[style],
                    s=marker_size,
                    label=label,
                )

            plot_title = f"Latent Space - {crop_name.capitalize()}"
            # ax.set_title(plot_title, fontsize=title_fontsize, pad=10)
            ax.set_xlabel(r"$t$-SNE$_1$", fontsize=label_fontsize + 2)
            ax.set_ylabel(r"$t$-SNE$_2$", fontsize=label_fontsize + 2)
            ax.legend(
                # title="Labels",
                # bbox_to_anchor=(1.02, 1),
                loc="upper right",
                # borderaxespad=1,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )
            ax.tick_params(axis="both", which="major", labelsize=label_fontsize)
            ax.grid(True, alpha=0.5)
            # Spines (plot borders) with same transparency
            for spine in ax.spines.values():
                spine.set_edgecolor("black")  # you can also use "0.0" for gray scale
                spine.set_alpha(0.3)
                spine.set_linewidth(1)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Plot 1: Color by crop type
            plt.scatter(
                data=df,
                x="x",
                y="y",
                hue="Crop",
                style="Label",
                ax=ax1,
                palette="deep",
                s=marker_size,
            )
            ax1.set_title("Latent Space by Crop Type (2D)", fontsize=title_fontsize)
            ax1.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax1.set_ylabel("t-SNE 2", fontsize=label_fontsize)
            ax1.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

            # Plot 2: Color by class label
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="Label",
                style="Crop",
                ax=ax2,
                palette="Set2",
                s=marker_size,
            )
            ax2.set_title("Latent Space by Class Label (2D)", fontsize=title_fontsize)
            ax2.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax2.set_ylabel("t-SNE 2", fontsize=label_fontsize)
            ax2.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

    elif n_components == 3:
        if single_crop:
            # Single 3D plot for one crop
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            unique_labels = sorted(df["Label"].unique())
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = df["Label"] == label
                ax.scatter(
                    df.loc[mask, "x"],
                    df.loc[mask, "y"],
                    df.loc[mask, "z"],
                    c=[color],
                    label=label,
                    marker="o",
                )

            ax.set_title(
                f"Latent Space by Class Label - {crop_name.capitalize()} (3D)",
                fontsize=title_fontsize,
            )
            ax.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax.set_ylabel("t-SNE 2", fontsize=label_fontsize)
            ax.set_zlabel("t-SNE 3", fontsize=label_fontsize)
            ax.legend(
                title="Labels",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )
        else:
            fig = plt.figure(figsize=(20, 8))

            # Plot 1: Color by crop type
            ax1 = fig.add_subplot(121, projection="3d")
            colors = {
                "corn": "#1f77b4",  # blue
                "soy": "#ff1a0e",  # orange
                "cotton": "#286928",  # green
            }

            # Create scatter plot for each crop separately
            for crop in df["Crop"].unique():
                mask = df["Crop"] == crop
                ax1.scatter(
                    df.loc[mask, "x"],
                    df.loc[mask, "y"],
                    df.loc[mask, "z"],
                    c=colors[crop],
                    label=crop,
                    marker="o",
                )

            ax1.set_title("Latent Space by Crop Type (3D)", fontsize=title_fontsize)
            ax1.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax1.set_ylabel("t-SNE 2", fontsize=label_fontsize)
            ax1.set_zlabel("t-SNE 3", fontsize=label_fontsize)
            ax1.legend(
                title="Crops",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

            # Plot 2: Color by class label
            ax2 = fig.add_subplot(122, projection="3d")
            unique_labels = sorted(df["Label"].unique())
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = df["Label"] == label
                ax2.scatter(
                    df.loc[mask, "x"],
                    df.loc[mask, "y"],
                    df.loc[mask, "z"],
                    c=[color],
                    label=label,
                    marker="o",
                )

            ax2.set_title("Latent Space by Class Label (3D)", fontsize=title_fontsize)
            ax2.set_xlabel("t-SNE 1", fontsize=label_fontsize)
            ax2.set_ylabel("t-SNE 2", fontsize=label_fontsize)
            ax2.set_zlabel("t-SNE 3", fontsize=label_fontsize)
            ax2.legend(
                title="Labels",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

    plt.tight_layout()

    if save_path:
        # Add crop name to filename if single crop
        path = Path(save_path)
        stem = f"{path.stem}_{n_components}d_p{int(perplexity)}"
        if single_crop:
            stem = f"{stem}_{crop_name}"
        new_path = path.parent / f"{stem}.pdf"

        plt.savefig(
            new_path,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
            metadata={"Creator": "Matplotlib"},
            backend="pgf",  # Use PGF backend for best LaTeX rendering
        )
        print(f"[Visualize] Saved plot to {new_path}")

    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize latent space from trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/crops/scc_exp.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help='Checkpoint name (e.g. "sae-114.ckpt") or "best" for best checkpoint',
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of t-SNE components (default: 2)",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter (default: 30.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override devices, e.g. "auto", "0", "0,1"',
    )
    # Parse arguments
    args = parser.parse_args()

    # Load config and resolve paths
    config_path = Path(args.config)
    ckpt_name = args.ckpt

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Visualize] Using device: {device}")

    # Resolve run name and find best checkpoint
    save_dir = Path(cfg["logging"]["save_dir"]) / config_path.parent.name
    run_name = resolve_run_name(cfg, config_path)
    checkpoint_path = find_checkpoint(ckpt_name, run_dir=save_dir)
    print(f"[Visualize] Using checkpoint: {checkpoint_path}")

    # Load and fix state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix from state dict keys
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            fixed_state_dict[k[6:]] = v  # Remove first 6 chars ('model.')
        else:
            fixed_state_dict[k] = v

    # Load data
    datamodule = build_datamodule(cfg)
    print(f"[Data] bands={datamodule.input_dim} classes={datamodule.num_classes}")

    # Load model
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    model = build_model(
        model_name,
        model_cfg,
        input_dim=datamodule.input_dim,
        num_classes=datamodule.num_classes,
        dm=datamodule,
    )

    # Load fixed state dict
    model.load_state_dict(fixed_state_dict)
    model = model.to(device)
    model.eval()

    # Visualize latent space
    df_tsne = visualize_latent_space(
        model=model,
        datamodule=datamodule,
        n_components=args.n_components,
        crops=cfg["data"]["crops"],
        perplexity=args.perplexity,
    )

    # Build output filename with consistent naming convention:
    # tsne_<band>_day_<day_idx>_<config_name>.pdf
    band_name = (
        config_path.parent.name
        if config_path.parent.name not in ("", None)
        else "unknown"
    )
    day_name = (
        config_path.parent.parent.name
        if config_path.parent.parent.name not in ("", None)
        else "day_00"
    )
    config_name = (
        config_path.parent.parent.parent.name
        if config_path.parent.parent.parent.name not in ("", None)
        else config_path.stem
    )

    # Normalize day prefix
    if not day_name.startswith("day_"):
        day_name = f"day_{day_name}"

    output_name = f"tsne_{band_name}_{day_name}_{config_name}.pdf"
    save_path = Path(checkpoint_path).parent.parent / output_name

    # Create visualization
    plot_latent_space(
        df_tsne,
        args.n_components,
        args.perplexity,
        save_path,
        title_fontsize=14,
        label_fontsize=14,
        legend_fontsize=10,
    )

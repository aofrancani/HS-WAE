import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from typing import List, Tuple, Optional, Dict
import random
import yaml

from models.build_model import build_model
from dataloader import build_datamodule
from utils import find_checkpoint, resolve_run_name
import pywt  # for inverse DWT

random.seed(512)

# try:
#     plt.rcParams.update(
#         {
#             "text.usetex": True,
#             "font.family": "serif",
#         }
#     )
# except Exception as e:
#     print(f"[Plot] LaTeX not available, using default fonts. ({e})")

# Define font sizes
label_size = 12
linewidth = 1
title_size = 12
fontsize = 12


# =========================
# Helpers (sampling)
# =========================
def get_random_samples(
    dataloader,
    n_control: int = 5,
    n_damaged: int = 5,
    device: str | torch.device = "cuda",
):
    """
    Supports:
      - non-DWT: batch = (x, y)
      - DWT token mode: batch = (x_raw, levels, y)

    Returns:
      x_model: Tensor(B,D) OR List[Tensor(B,n_l)]  (what goes into the model)
      y_sel: Tensor(B,)
      sel: selected indices
      x_raw_sel: Tensor(B,bands) or None
    """
    all_x = []
    all_y = []
    all_xraw = []  # only for DWT mode

    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
            all_x.append(x)
            all_y.append(y)
        elif len(batch) == 3:
            x_raw, x, y = batch
            all_xraw.append(x_raw)
            all_x.append(x)
            all_y.append(y)
        else:
            raise ValueError(f"Unexpected batch format len={len(batch)}")

    y_all = torch.cat(all_y, dim=0)

    control_idx = (y_all == 0).nonzero(as_tuple=False).squeeze(1)
    damaged_idx = (y_all == 1).nonzero(as_tuple=False).squeeze(1)

    control_pick = random.sample(control_idx.tolist(), min(n_control, len(control_idx)))
    damaged_pick = random.sample(damaged_idx.tolist(), min(n_damaged, len(damaged_idx)))
    sel = control_pick + damaged_pick

    x0 = all_x[0]
    if isinstance(x0, list):
        # DWT: x is list of level tensors
        K = len(x0)
        levels_all = []
        for l in range(K):
            levels_all.append(torch.cat([xb[l] for xb in all_x], dim=0))  # (N, n_l)
        x_sel = [lvl[sel].to(device) for lvl in levels_all]

        x_raw_all = torch.cat(all_xraw, dim=0)  # (N, bands)
        x_raw_sel = x_raw_all[sel].to(device)
    else:
        # non-DWT: x is tensor
        x_all = torch.cat(all_x, dim=0)  # (N, D)
        x_sel = x_all[sel].to(device)
        x_raw_sel = None

    y_sel = y_all[sel].to(device)
    return x_sel, y_sel, sel, x_raw_sel


# =========================
# Helpers (ranges & (de)norm)
# =========================
def valid_mask_from_ranges(
    wavelengths_nm: np.ndarray,
    keep_range: Tuple[int, int],
    exclude_ranges: List[Tuple[int, int]],
) -> np.ndarray:
    """Boolean mask True for wavelengths we keep (outside excluded bands)."""
    w = wavelengths_nm
    keep = (w >= keep_range[0]) & (w <= keep_range[1])
    if exclude_ranges:
        for a, b in exclude_ranges:
            keep &= ~((w >= a) & (w <= b))
    return keep


def denormalize_spectra(
    spectra: torch.Tensor, standardizer: Dict[str, np.ndarray]
) -> torch.Tensor:
    """Denormalize (feature-wise)."""
    mean = torch.from_numpy(standardizer["mean"]).to(spectra.device)
    std = torch.from_numpy(standardizer["std"]).to(spectra.device)
    return spectra * std + mean


def unstandardize_levels(
    levels: List[torch.Tensor],
    stats: List[Dict[str, np.ndarray]],
) -> List[torch.Tensor]:
    out = []
    for l, x in enumerate(levels):
        mean = torch.from_numpy(stats[l]["mean"]).to(x.device)
        std = torch.from_numpy(stats[l]["std"]).to(x.device)
        out.append(x * std + mean)
    return out


def split_by_lengths(x_vec: torch.Tensor, lengths: List[int]) -> List[torch.Tensor]:
    parts = []
    idx = 0
    for L in lengths:
        parts.append(x_vec[:, idx : idx + L])
        idx += L
    return parts


def unweight_levels(
    weighted_levels: List[torch.Tensor],
    alpha: torch.Tensor,
    eps: float = 1e-6,
) -> List[torch.Tensor]:
    """
    weighted_levels[l] = coeffs[l] * alpha[:, l]
    => coeffs[l] ≈ weighted_levels[l] / max(alpha[:,l], eps)
    """
    coeffs = []
    for l, wl in enumerate(weighted_levels):
        a = alpha[:, l].unsqueeze(1).clamp_min(eps)
        coeffs.append(wl / a)
    return coeffs


def inv_dwt_from_levels(
    levels_np: List[np.ndarray],
    wavelet: str,
    boundary: str,
    include_approx: bool,
    band_count: int,
) -> np.ndarray:
    """
    levels_np: list [A_L, D_L, ..., D_1] if include_approx
               else [D_L, ..., D_1] and we inject zeros for A_L
    """
    if include_approx:
        coeffs = levels_np
    else:
        # inject placeholder approx of correct length (PyWavelets needs it)
        # easiest: create dummy to infer approx length:
        dummy = np.zeros(band_count, dtype=np.float32)
        c = pywt.wavedec(dummy, wavelet=wavelet, level=None, mode=boundary)
        approx_len = len(c[0])
        coeffs = [np.zeros(approx_len, dtype=np.float32)] + levels_np

    x = pywt.waverec(coeffs, wavelet=wavelet, mode=boundary)
    return x[:band_count].astype(np.float32)


# =========================
# NEW: DWT inverse helpers
# =========================
def dwt_template_lengths(
    orig_len: int, wavelet: str, level, boundary: str
) -> list[int]:
    """Return lengths of [cA_L, cD_L, ..., cD_1] using a dummy signal."""
    dummy = np.zeros(orig_len, dtype=np.float32)
    coeffs = pywt.wavedec(dummy, wavelet=wavelet, level=level, mode=boundary)
    return [len(c) for c in coeffs]  # [len(cA_L), len(cD_L), ..., len(cD_1)]


def unflatten_to_coeffs(
    flat: np.ndarray, lengths: list[int], include_approx: bool
) -> list[np.ndarray]:
    """Split flattened vector into [cA_L, cD_L, ..., cD_1]."""
    parts = []
    idx = 0
    if include_approx:
        L0 = lengths[0]
        parts.append(flat[idx : idx + L0])
        idx += L0
    else:
        parts.append(np.zeros(lengths[0], dtype=flat.dtype))  # placeholder approx

    for Lk in lengths[1:]:
        parts.append(flat[idx : idx + Lk])
        idx += Lk
    return parts


def inv_dwt_batch(
    X_flat: np.ndarray,
    band_count: int,
    wavelet: str,
    level,
    boundary: str,
    include_approx: bool,
) -> np.ndarray:
    """Inverse DWT for batch of flattened coeffs -> spectra."""
    lens = dwt_template_lengths(band_count, wavelet, level, boundary)
    out = []
    for i in range(X_flat.shape[0]):
        coeffs = unflatten_to_coeffs(X_flat[i], lens, include_approx)
        x = pywt.waverec(coeffs, wavelet=wavelet, mode=boundary)
        out.append(x[:band_count])  # ensure same length
    return np.stack(out, axis=0).astype(np.float32)


# =========================
# Plotters
# =========================
def plot_reconstructions(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    labels: torch.Tensor,
    wavelengths: List[float],
    exclude_ranges: List[Tuple[int, int]] = [(1350, 1440), (1800, 1980)],
    keep_range: Tuple[int, int] = (390, 2300),
    save_path: Optional[Path] = None,
    *,
    already_denormalized: bool = False,
    standardizer: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Plot original vs reconstructed spectra in a 2x2 grid with excluded regions.

    If `already_denormalized` is False, you must pass `standardizer` to denormalize.
    If True, data is assumed to be reflectance already (no denorm applied).
    """
    # wavelengths as increasing array
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1:
        wl = wl.ravel()
    if not np.all(np.diff(wl) > 0):
        order = np.argsort(wl)
        wl = wl[order]
        original = original[:, order]
        reconstructed = reconstructed[:, order]

    # pick 2 control + 2 damaged
    control_mask = labels == 0
    damaged_mask = labels == 1
    control_orig = original[control_mask][:2]
    control_recon = reconstructed[control_mask][:2]
    damaged_orig = original[damaged_mask][:2]
    damaged_recon = reconstructed[damaged_mask][:2]

    # denormalize if needed
    if not already_denormalized:
        if standardizer is None:
            raise ValueError("standardizer is required when already_denormalized=False")
        control_orig = denormalize_spectra(control_orig, standardizer)
        control_recon = denormalize_spectra(control_recon, standardizer)
        damaged_orig = denormalize_spectra(damaged_orig, standardizer)
        damaged_recon = denormalize_spectra(damaged_recon, standardizer)

    # valid intervals (break lines over excluded)
    intervals = []
    prev_end = keep_range[0]
    for a, b in exclude_ranges:
        if b <= keep_range[0] or a >= keep_range[1]:
            continue
        intervals.append((max(prev_end, keep_range[0]), min(a, keep_range[1])))
        prev_end = b
    intervals.append((max(prev_end, keep_range[0]), keep_range[1]))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    legend_lines, legend_labels = [], []
    rows = [
        (control_orig, control_recon, "Control"),
        (damaged_orig, damaged_recon, "Damaged"),
    ]

    for row, (orig, recon, title_prefix) in enumerate(rows):
        for col in range(2):
            ax = axes[row, col]

            # shade excluded regions
            for a, b in exclude_ranges:
                a_clipped = max(a, keep_range[0])
                b_clipped = min(b, keep_range[1])
                if a_clipped < b_clipped:
                    ax.axvspan(a_clipped, b_clipped, color="gray", alpha=0.2, zorder=0)

            # plot per valid interval
            for start, end in intervals:
                mask = (wl >= start) & (wl <= end)
                (line_o,) = ax.plot(
                    wl[mask],
                    orig[col].detach().cpu().numpy()[mask],
                    color="black",
                    linestyle="-",
                    linewidth=linewidth,
                    zorder=2,
                )
                (line_r,) = ax.plot(
                    wl[mask],
                    recon[col].detach().cpu().numpy()[mask],
                    color="red",
                    linestyle="--",
                    linewidth=linewidth,
                    zorder=2,
                )

            if row == 0 and col == 0:
                legend_lines = [line_o, line_r]
                legend_labels = ["Original", "Reconstructed"]

            ax.set_title(f"{title_prefix} Sample {col+1}", fontsize=title_size)
            ax.set_xlabel("Wavelength (nm)", fontsize=label_size)
            ax.set_ylabel("Reflectance", fontsize=label_size)
            ax.tick_params(axis="both", labelsize=fontsize)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.set_xlim(keep_range)

    axes[0, 1].legend(
        legend_lines, legend_labels, loc="upper right", fontsize=fontsize - 2
    )
    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(
            save_path.with_suffix(".pdf"),
            format="pdf",
            bbox_inches="tight",
            # backend="pgf",
            metadata={"Creator": "Matplotlib"},
        )
        print(f"[Plot] Saved reconstructions to {save_path}")

    plt.show()
    plt.close()


def plot_single_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    label: int,
    wavelengths: List[float],
    *,
    already_denormalized: bool = False,
    standardizer: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[Path] = None,
    sample_id: int = 0,
) -> None:
    """Plot a single sample."""
    if not already_denormalized:
        if standardizer is None:
            raise ValueError("standardizer is required when already_denormalized=False")
        original = denormalize_spectra(original, standardizer)
        reconstructed = denormalize_spectra(reconstructed, standardizer)

    plt.figure(figsize=(8, 5))
    plt.plot(
        wavelengths,
        original.detach().cpu().numpy(),
        color="black",
        linestyle="-",
        label="Original",
        linewidth=2,
    )
    plt.plot(
        wavelengths,
        reconstructed.detach().cpu().numpy(),
        color="red",
        linestyle="-",
        label="Reconstructed",
        linewidth=2,
    )
    class_name = "Control" if label == 0 else "Damaged"
    plt.title(f"{class_name} Sample", fontsize=12)
    plt.xlabel("Wavelength (nm)", fontsize=10)
    plt.ylabel("Reflectance", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved reconstruction to {save_path}")
    plt.close()


# =========================
# Main
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot reconstructions from a trained model"
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
        help='Checkpoint name (e.g. "sae-114.ckpt") or "best"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override device, e.g. "auto", "0", "0,1"',
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device:
        device = "cuda:" + str(args.device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Visualize] Using device: {device}")

    # Resolve run + checkpoint
    save_dir = Path(cfg["logging"]["save_dir"]) / config_path.parent.name
    run_name = resolve_run_name(cfg, config_path)
    checkpoint_path = find_checkpoint(args.ckpt, run_dir=save_dir)
    print(f"[Load] Using checkpoint: {checkpoint_path}")

    # Output dir
    log_root = Path(cfg["logging"]["save_dir"])
    run_name = resolve_run_name(cfg, config_path)
    run_dir = log_root / run_name
    plots_dir = run_dir / "plots" / "reconstructions"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Saving plots to {plots_dir}")

    # Load checkpoint (Lightning)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix if present
    fixed_state_dict = {}
    for k, v in state_dict.items():
        fixed_state_dict[k[6:]] = v if k.startswith("model.") else v

    # Data & model
    datamodule = build_datamodule(cfg)
    print(
        f"[Data] input_dim={datamodule.input_dim} num_classes={datamodule.num_classes}"
    )

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    model = build_model(
        model_name,
        model_cfg,
        input_dim=datamodule.input_dim,
        num_classes=datamodule.num_classes,
        dm=datamodule,
    )
    model.load_state_dict(fixed_state_dict)
    model = model.to(device)
    model.eval()

    # Samples
    x_samp, y_samp, _, x_raw_samp = get_random_samples(
        datamodule.test_dataloader(), n_control=5, n_damaged=5, device=device
    )

    with torch.no_grad():
        standardizer = datamodule.get_standardizer()

    if standardizer is None:
        raise ValueError("No standardization stats found in datamodule!")

    dm = datamodule

    with torch.no_grad():
        if isinstance(x_samp, list):
            # ----------------------------
            # DWT token mode
            # Model recon is already spectrum now
            # ----------------------------
            if x_raw_samp is None:
                raise ValueError("Expected x_raw_samp for DWT mode, got None.")

            out = model(x_samp, y=None, compute_loss=False)
            reconstructed = out["recon"]  # (B, bands)

            # original spectrum is x_raw_samp (maybe standardized, maybe not)
            original = x_raw_samp

            # If YOU standardized x_raw in dataloader, denormalize here.
            # That requires a dict standardizer for the bands (NOT the per-level list).
            # Suggested: dm.standardizer_stats_raw (see section 4 below).
            if (
                hasattr(dm, "standardizer_stats_raw")
                and dm.standardizer_stats_raw is not None
            ):
                original = denormalize_spectra(original, dm.standardizer_stats_raw)
                reconstructed = denormalize_spectra(
                    reconstructed, dm.standardizer_stats_raw
                )

            samples_reflectance = original
            reconstructed_reflectance = reconstructed
            labels = y_samp

        else:
            # ----------------------------
            # Non-DWT mode (unchanged)
            # ----------------------------
            if not isinstance(standardizer, dict):
                raise ValueError(
                    "Expected global standardizer stats (dict) for non-DWT mode."
                )

            mean = torch.from_numpy(standardizer["mean"]).to(device)
            std = torch.from_numpy(standardizer["std"]).to(device)

            x_unstd = x_samp * std + mean
            out = model(x_samp, y=None, compute_loss=False)
            xhat_unstd = out["recon"] * std + mean

            samples_reflectance = x_unstd
            reconstructed_reflectance = xhat_unstd
            labels = y_samp

        # 3) Wavelengths and ranges
        wavelengths = [int(w) for w in datamodule.kept_wavelengths_nm]
        exclude_ranges = cfg["data"]["preprocess"]["exclude_ranges"]
        keep_range = tuple(cfg["data"]["preprocess"]["keep_range"])

    # 4) Plot
    plot_reconstructions(
        original=samples_reflectance,
        reconstructed=reconstructed_reflectance,
        labels=labels,
        wavelengths=wavelengths,
        exclude_ranges=exclude_ranges,
        keep_range=keep_range,
        save_path=plots_dir / "plot_reconstructions",
        already_denormalized=True,
        standardizer=standardizer,  # ignored when already_denormalized=True
    )


if __name__ == "__main__":
    main()

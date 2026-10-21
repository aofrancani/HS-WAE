#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi

import sys

sys.path.append(str(Path(__file__).parent / "src"))
import ds_load  # type: ignore
from ds_load import get_data, get_anno, get_rgb, IMAGES  # type: ignore


CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "blood",
    2: "ketchup",
    3: "artificial blood",
    4: "beetroot juice",
    5: "poster paint",
    6: "tomato concentrate",
    7: "acrylic paint",
    8: "uncertain blood",
}

COLOR_MAP: Dict[str, str] = {
    "acrylic paint": "#1f77b4",
    "artificial blood": "#8c564b",
    "beetroot juice": "#9467bd",
    "blood": "#d62728",
    "ketchup": "#ff9896",
    "poster paint": "#2ca02c",
    "tomato concentrate": "#ff7f0e",
    "uncertain blood": "#7f7f7f",
    "unknown": "#7f7f7f",
}

DEFAULT_HIGHLIGHT = "#FFD54F"

label_size = 14
linewidth = 1.5
title_size = 12
fontsize = 12


def set_latex_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": fontsize,
            "axes.labelsize": label_size,
            "axes.titlesize": title_size,
            "legend.fontsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "lines.linewidth": linewidth,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_scene_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)


def class_color(class_name: str) -> str:
    return COLOR_MAP.get(class_name, COLOR_MAP["unknown"])


def display_class_name(class_name: str) -> str:
    return class_name[:1].upper() + class_name[1:] if class_name else class_name


def label_connected_components(
    mask: np.ndarray,
    connectivity: int = 2,
) -> Tuple[np.ndarray, int]:
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if connectivity == 1:
        structure = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        )
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    labeled, n = ndi.label(mask, structure=structure)
    return labeled, int(n)


def component_props(
    labeled: np.ndarray,
    comp_id: int,
) -> Tuple[int, Tuple[int, int, int, int], Tuple[float, float]]:
    coords = np.column_stack(np.where(labeled == comp_id))
    n_pix = int(coords.shape[0])

    rmin = int(coords[:, 0].min())
    rmax = int(coords[:, 0].max())
    cmin = int(coords[:, 1].min())
    cmax = int(coords[:, 1].max())

    centroid_r = float(coords[:, 0].mean())
    centroid_c = float(coords[:, 1].mean())

    return n_pix, (rmin, cmin, rmax, cmax), (centroid_r, centroid_c)


def mean_spectrum_for_component(
    data: np.ndarray,
    labeled: np.ndarray,
    comp_id: int,
) -> np.ndarray:
    mask = labeled == comp_id
    return data[mask].mean(axis=0)


def extract_scene_blobs(
    scene_name: str,
    remove_uncertain_blood: bool = True,
    connectivity: int = 2,
    min_pixels: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    data, wav = get_data(scene_name, remove_bands=True, clean=True)
    anno = get_anno(
        scene_name,
        remove_uncertain_blood=remove_uncertain_blood,
        clean=True,
    )
    rgb = get_rgb(data, wav)

    if data.shape[:2] != anno.shape:
        raise ValueError(
            f"[{scene_name}] Spatial mismatch: data {data.shape[:2]} vs anno {anno.shape}"
        )

    blobs: List[dict] = []
    local_blob_id = 0

    present_classes = sorted(int(x) for x in np.unique(anno).tolist())
    present_classes = [c for c in present_classes if c != 0]

    for cls in present_classes:
        class_mask = anno == cls
        labeled, n = label_connected_components(class_mask, connectivity=connectivity)

        for comp_id in range(1, n + 1):
            n_pix, bbox, centroid = component_props(labeled, comp_id)
            if n_pix < min_pixels:
                continue

            mu = mean_spectrum_for_component(data, labeled, comp_id)
            local_blob_id += 1
            cname = CLASS_NAMES.get(cls, "unknown")

            blobs.append(
                {
                    "local_blob_id": local_blob_id,
                    "class_id": cls,
                    "class_name": cname,
                    "n_pixels": n_pix,
                    "centroid": centroid,
                    "bbox": bbox,
                    "spectrum": mu,
                    "color": class_color(cname),
                }
            )

    return data, wav, rgb, anno, blobs


def choose_representative_blobs(
    blobs: List[dict],
    max_per_scene: int = 2,
    prefer_classes: Optional[List[str]] = None,
) -> List[dict]:
    if not blobs:
        return []

    prefer_classes = prefer_classes or []

    def score(blob: dict) -> Tuple[int, int]:
        pref = 1 if blob["class_name"] in prefer_classes else 0
        return (pref, blob["n_pixels"])

    blobs_sorted = sorted(blobs, key=score, reverse=True)

    selected: List[dict] = []
    used_classes = set()

    for b in blobs_sorted:
        if len(selected) >= max_per_scene:
            break
        if b["class_name"] not in used_classes:
            selected.append(b)
            used_classes.add(b["class_name"])

    if len(selected) < max_per_scene:
        for b in blobs_sorted:
            if len(selected) >= max_per_scene:
                break
            if b not in selected:
                selected.append(b)

    return selected


def plot_single_candidate(
    scene_name: str,
    rgb: np.ndarray,
    wav: np.ndarray,
    blobs: List[dict],
    selected_blob: dict,
    out_path: Path,
    dpi: int = 300,
    show_all_ids: bool = True,
    draw_bboxes: bool = True,
    class_colored_spectrum: bool = True,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.imshow(rgb, interpolation="nearest")
    ax0.set_title(rf"Scene {scene_name}")
    ax0.axis("off")

    for b in blobs:
        r, c = b["centroid"]
        rmin, cmin, rmax, cmax = b["bbox"]
        blob_id = b["local_blob_id"]
        edge_c = b["color"]
        is_selected = blob_id == selected_blob["local_blob_id"]

        if draw_bboxes:
            rect = Rectangle(
                (cmin, rmin),
                cmax - cmin + 1,
                rmax - rmin + 1,
                fill=False,
                linewidth=2.0 if is_selected else 1.0,
                linestyle="-" if is_selected else "--",
                edgecolor=edge_c if not is_selected else DEFAULT_HIGHLIGHT,
            )
            ax0.add_patch(rect)

        if show_all_ids or is_selected:
            ax0.text(
                c,
                r,
                f"{blob_id}",
                color="white" if not is_selected else "black",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="normal",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor=edge_c if not is_selected else DEFAULT_HIGHLIGHT,
                    edgecolor="black",
                    alpha=0.72,
                    linewidth=0.6,
                ),
            )

    spectrum = selected_blob["spectrum"]
    blob_id = selected_blob["local_blob_id"]
    class_name = selected_blob["class_name"]
    n_pixels = selected_blob["n_pixels"]
    spec_color = selected_blob["color"] if class_colored_spectrum else "black"

    ax1.plot(wav, spectrum, color=spec_color)
    ax1.set_xlabel(r"$\lambda$ (nm)", fontsize=label_size)
    ax1.set_ylabel(r"$\rho (\lambda)$", fontsize=label_size)
    ax1.tick_params(axis="both", labelsize=fontsize)
    ax1.grid(True, alpha=0.3)

    text_info = (
        rf"Scene: {scene_name}"
        "\n"
        rf"Blob ID: {blob_id}"
        "\n"
        rf"Class: {display_class_name(class_name)}"
        "\n"
        rf"Pixels: {n_pixels}"
    )
    ax1.text(
        0.03,
        0.97,
        text_info,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=spec_color,
            alpha=0.92,
        ),
    )

    ax1.plot([], [], color=spec_color, label=display_class_name(class_name))
    ax1.legend(fontsize=fontsize, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(
    candidates: List[dict],
    out_path: Path,
    ncols: int = 3,
    dpi: int = 220,
) -> None:
    if not candidates:
        return

    n = len(candidates)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, item in zip(axes.flat, candidates):
        rgb = item["rgb"]
        blob = item["blob"]
        scene_name = item["scene_name"]

        ax.imshow(rgb, interpolation="nearest")
        ax.axis("off")

        rmin, cmin, rmax, cmax = blob["bbox"]
        rect = Rectangle(
            (cmin, rmin),
            cmax - cmin + 1,
            rmax - rmin + 1,
            fill=False,
            linewidth=2.0,
            edgecolor=blob["color"],
        )
        ax.add_patch(rect)

        r, c = blob["centroid"]
        ax.text(
            c,
            r,
            f"{blob['local_blob_id']}",
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="normal",
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor=blob["color"],
                edgecolor="black",
                alpha=0.72,
                linewidth=0.6,
            ),
        )

        ax.set_title(
            rf"{scene_name} | {display_class_name(blob['class_name'])} | {blob['n_pixels']} px",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-style candidate figures from HyperBlood scenes."
    )
    parser.add_argument("--outdir", type=str, default="figure_candidates")
    parser.add_argument(
        "--data-root", type=str, default="", help="Path containing data/ and anno/"
    )
    parser.add_argument("--scenes", type=str, default="")
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--max-per-scene", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=60)
    parser.add_argument("--connectivity", type=int, default=2, choices=[1, 2])
    parser.add_argument("--keep-uncertain", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--prefer-classes",
        nargs="*",
        default=["blood", "artificial blood", "ketchup"],
    )
    parser.add_argument("--hide-other-ids", action="store_true")
    args = parser.parse_args()

    set_latex_style()

    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        ds_load.PATH_DATA = str(root) + "/"

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.scenes.strip():
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = list(IMAGES)[: args.num_scenes]

    all_candidates = []
    saved_paths = []

    pdf_path = outdir / "all_candidates.pdf"
    with PdfPages(pdf_path) as pdf:
        for scene_name in scenes:
            print(f"[Process] {scene_name}")
            try:
                _, wav, rgb, _, blobs = extract_scene_blobs(
                    scene_name=scene_name,
                    remove_uncertain_blood=not args.keep_uncertain,
                    connectivity=args.connectivity,
                    min_pixels=args.min_pixels,
                )
            except Exception as e:
                print(f"  -> skipped due to error: {e}")
                continue

            if not blobs:
                print("  -> no valid blobs kept")
                continue

            selected_blobs = choose_representative_blobs(
                blobs=blobs,
                max_per_scene=args.max_per_scene,
                prefer_classes=args.prefer_classes,
            )

            for b in selected_blobs:
                fname = (
                    f"{safe_scene_name(scene_name)}"
                    f"_blob_{b['local_blob_id']:03d}"
                    f"_{safe_scene_name(b['class_name'])}.pdf"
                )
                out_path = outdir / fname

                plot_single_candidate(
                    scene_name=scene_name,
                    rgb=rgb,
                    wav=wav,
                    blobs=blobs,
                    selected_blob=b,
                    out_path=out_path,
                    dpi=args.dpi,
                    show_all_ids=not args.hide_other_ids,
                    draw_bboxes=True,
                    class_colored_spectrum=True,
                )

                fig = plt.figure(figsize=(12, 5))
                gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.18)
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 1])

                ax0.imshow(rgb, interpolation="nearest")
                ax0.set_title(rf"Scene {scene_name}")
                ax0.axis("off")

                for bb in blobs:
                    r, c = bb["centroid"]
                    rmin, cmin, rmax, cmax = bb["bbox"]
                    is_selected = bb["local_blob_id"] == b["local_blob_id"]
                    rect = Rectangle(
                        (cmin, rmin),
                        cmax - cmin + 1,
                        rmax - rmin + 1,
                        fill=False,
                        linewidth=2.0 if is_selected else 1.0,
                        linestyle="-" if is_selected else "--",
                        edgecolor=DEFAULT_HIGHLIGHT if is_selected else bb["color"],
                    )
                    ax0.add_patch(rect)

                    if not args.hide_other_ids or is_selected:
                        ax0.text(
                            c,
                            r,
                            f"{bb['local_blob_id']}",
                            color="white" if not is_selected else "black",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="normal",
                            bbox=dict(
                                boxstyle="round,pad=0.12",
                                facecolor=(
                                    bb["color"]
                                    if not is_selected
                                    else DEFAULT_HIGHLIGHT
                                ),
                                edgecolor="black",
                                alpha=0.72,
                                linewidth=0.6,
                            ),
                        )

                ax1.plot(wav, b["spectrum"], color=b["color"])
                ax1.set_xlabel(r"$\lambda$ (nm)", fontsize=label_size)
                ax1.set_ylabel(r"$\rho (\lambda)$", fontsize=label_size)
                ax1.tick_params(axis="both", labelsize=fontsize)
                ax1.grid(True, alpha=0.3)
                ax1.text(
                    0.03,
                    0.97,
                    (
                        rf"Scene: {scene_name}"
                        "\n"
                        rf"Blob ID: {b['local_blob_id']}"
                        "\n"
                        rf"Class: {display_class_name(b['class_name'])}"
                        "\n"
                        rf"Pixels: {b['n_pixels']}"
                    ),
                    transform=ax1.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=b["color"],
                        alpha=0.92,
                    ),
                )
                ax1.plot(
                    [], [], color=b["color"], label=display_class_name(b["class_name"])
                )
                ax1.legend(fontsize=fontsize, loc="upper right", frameon=True)

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append(out_path)
                all_candidates.append(
                    {
                        "scene_name": scene_name,
                        "rgb": rgb,
                        "blob": b,
                    }
                )

            print(f"  -> saved {len(selected_blobs)} candidates")

    if all_candidates:
        contact_path = outdir / "contact_sheet.pdf"
        plot_contact_sheet(all_candidates, contact_path, ncols=3, dpi=220)
        print(f"[Saved] {contact_path}")

    print(f"[Saved] {pdf_path}")
    print("\nCandidate files:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

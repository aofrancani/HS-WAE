#!/usr/bin/env python3
"""
Extract mean spectra per connected-component (blob) per class, per scene,
from the ITAI PAS HyperBlood dataset.

Outputs:
- blobs.csv (one row per blob, spectrum stored as wavelength columns)
- images_ids/<scene>_ids.png (RGB with "<class> (<id>)" at blob centroid)

Run from dataset root:
  python extract_blob_spectra.py --out blobs.csv --debug-dir images_ids
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

# dataset loader
import sys

sys.path.append(str(Path(__file__).parent / "src"))
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def label_connected_components(
    mask: np.ndarray, connectivity: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a 2D boolean mask.

    connectivity:
      1 -> 4-connected
      2 -> 8-connected
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if connectivity == 1:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    labeled, n = ndi.label(mask, structure=structure)
    return labeled, int(n)


def component_props(
    labeled: np.ndarray, comp_id: int
) -> Tuple[int, Tuple[int, int, int, int], Tuple[float, float]]:
    """
    Return:
      n_pixels,
      bbox (min_row, min_col, max_row, max_col) inclusive bounds
      centroid (row, col) in float coords
    """
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
    data: np.ndarray, labeled: np.ndarray, comp_id: int
) -> np.ndarray:
    """Mean spectrum over pixels where labeled == comp_id. Returns shape (bands,)."""
    mask = labeled == comp_id
    # data[mask] => shape (n_pix, bands)
    return data[mask].mean(axis=0)


def draw_debug_overlay(
    rgb: np.ndarray,
    anno: np.ndarray,
    per_blob_labels: List[Tuple[float, float, str]],
    out_path: Path,
    title: str,
) -> None:
    """
    Save RGB with text labels at (row, col).
    per_blob_labels: list of (row, col, text)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    # optional: lightly overlay GT boundaries to help sanity-check
    # (simple edge: pixels where label differs from neighbor)
    edges = np.zeros_like(anno, dtype=bool)
    edges[:-1, :] |= anno[:-1, :] != anno[1:, :]
    edges[:, :-1] |= anno[:, :-1] != anno[:, 1:]
    ax.contour(edges.astype(float), levels=[0.5], linewidths=0.6)

    for r, c, txt in per_blob_labels:
        ax.text(
            c,
            r,
            txt,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.6),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_scene(
    scene_name: str,
    remove_uncertain_blood: bool,
    connectivity: int,
    min_pixels: int,
    debug_dir: Path | None,
) -> List[Dict]:
    """
    Process one scene:
      - label blobs per class
      - compute mean spectra
      - return list of row dicts for DataFrame
    """
    data, wav = get_data(scene_name, remove_bands=True, clean=True)
    anno = get_anno(
        scene_name, remove_uncertain_blood=remove_uncertain_blood, clean=True
    )

    if data.shape[:2] != anno.shape:
        raise ValueError(
            f"[{scene_name}] Spatial mismatch: data {data.shape[:2]} vs anno {anno.shape}"
        )

    rgb = get_rgb(data, wav)

    rows: List[Dict] = []
    debug_labels: List[Tuple[float, float, str]] = []

    # global unique blob id inside scene
    local_blob_id = 0

    present_classes = sorted(int(x) for x in np.unique(anno).tolist())
    present_classes = [c for c in present_classes if c != 0]  # ignore background blobs

    for cls in present_classes:
        class_mask = anno == cls

        labeled, n = label_connected_components(class_mask, connectivity=connectivity)
        if n == 0:
            continue

        for comp_id in range(1, n + 1):
            n_pix, bbox, centroid = component_props(labeled, comp_id)
            if n_pix < min_pixels:
                continue

            mu = mean_spectrum_for_component(data, labeled, comp_id)

            local_blob_id += 1
            blob_uid = f"{scene_name}|{local_blob_id:04d}"

            row: Dict = {
                "scene": scene_name,
                "blob_id": blob_uid,
                "class_id": cls,
                "class_name": CLASS_NAMES.get(cls, "unknown"),
                "n_pixels": n_pix,
                "centroid_r": centroid[0],
                "centroid_c": centroid[1],
                "bbox_rmin": bbox[0],
                "bbox_cmin": bbox[1],
                "bbox_rmax": bbox[2],
                "bbox_cmax": bbox[3],
            }

            # add spectrum as columns with wavelength keys (rounded for nicer column names)
            # Example columns: wl_377.0, wl_380.1, ...
            for w, v in zip(wav.tolist(), mu.tolist()):
                row[f"wl_{float(w):.1f}"] = float(v)

            rows.append(row)

            # debug label
            txt = f"{CLASS_NAMES.get(cls,'?')} ({local_blob_id})"
            debug_labels.append((centroid[0], centroid[1], txt))

    # save debug overlay
    if debug_dir is not None:
        ensure_dir(debug_dir)
        safe_name = scene_name.replace("(", "_").replace(")", "").replace("/", "_")
        out_path = debug_dir / f"{safe_name}_ids.png"
        draw_debug_overlay(
            rgb=rgb,
            anno=anno,
            per_blob_labels=debug_labels,
            out_path=out_path,
            title=f"{scene_name} - blob ids",
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="blobs.csv", help="Output CSV path.")
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="images_ids",
        help="Folder to save debug overlays.",
    )
    parser.add_argument(
        "--keep-uncertain", action="store_true", help="Keep class 8 as a valid class."
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=2,
        choices=[1, 2],
        help="1=4-connected, 2=8-connected.",
    )
    parser.add_argument(
        "--min-pixels", type=int, default=60, help="Ignore blobs smaller than this."
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated subset, e.g. 'E(1),F(1)'. Empty=all.",
    )
    args = parser.parse_args()

    out_csv = Path(args.out)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    if args.scenes.strip():
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = list(IMAGES)

    all_rows: List[Dict] = []
    for s in scenes:
        print(f"[Process] {s}")
        rows = process_scene(
            scene_name=s,
            remove_uncertain_blood=not args.keep_uncertain,
            connectivity=args.connectivity,
            min_pixels=args.min_pixels,
            debug_dir=debug_dir,
        )
        print(f"  -> blobs kept: {len(rows)}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No blobs found. Try lowering --min-pixels or --keep-uncertain.")
        return

    # nicer column order
    meta_cols = [
        "scene",
        "blob_id",
        "class_id",
        "class_name",
        "n_pixels",
        "centroid_r",
        "centroid_c",
        "bbox_rmin",
        "bbox_cmin",
        "bbox_rmax",
        "bbox_cmax",
    ]
    spec_cols = [c for c in df.columns if c.startswith("wl_")]
    df = df[meta_cols + spec_cols]

    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}  | rows={len(df)}  | spectrum_cols={len(spec_cols)}")
    if debug_dir is not None:
        print(f"[Saved] debug images -> {debug_dir}/")


if __name__ == "__main__":
    main()

"""
Aggregate experiment metrics from files named like:
    metrics_stats_<band_region>_day_<day_number>.csv

For each file, the script:
1) reads the experiment-level metrics
2) computes summary statistics for each metric
3) saves a long-format summary CSV
4) saves an Excel-ready table with rows = (band, metric) and columns = day/date
   using the format: mean ± std

Example output table layout:
        22/08      23/08      25/08 ...
UV  loss   0.1234 ± 0.0123
    acc    0.9123 ± 0.0211
VIS loss   ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Folder containing the CSV files
model = "dwt_attn_norecon"
INPUT_DIR = Path(f"logs/stats/{model}")

# Output folder
OUTPUT_DIR = Path(f"logs/stats/{model}/aggregated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map day number -> label you want in the final table
DAY_LABELS: Dict[int, str] = {
    1: "22/08",
    2: "23/08",
    3: "25/08",
    4: "26/08",
    5: "27/08",
    6: "28/08",
    7: "29/08",
}

# Order of metrics in the final table
METRIC_ORDER: List[str] = [
    "loss",
    "ce_loss",
    "rc_loss",
    "acc",
    "f1",
    "auroc",
    "precision",
    "recall",
    "fpr",
]

# Desired order of bands in the final table
BAND_ORDER: List[str] = [
    "UV",
    "VIS",
    "RGB",
    "RE",
    "NIR",
    "SWIR-1",
    "SWIR-2a",
    "SWIR-2b",
]


# ============================================================
# HELPERS
# ============================================================

FILENAME_RE = re.compile(r"^metrics_stats_(?P<band>.+)_day_(?P<day>\d+)\.csv$")


def parse_filename(path: Path) -> Optional[Dict[str, object]]:
    """
    Extract band_region and day number from a filename.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return {
        "band_raw": m.group("band"),
        "day": int(m.group("day")),
    }


def pretty_band_name(band_raw: str) -> str:
    """
    Convert filename band names to a prettier label for tables.
    Adjust or extend this mapping as needed.
    """
    b = band_raw.strip().lower().replace("_", "-")

    mapping = {
        "uv": "UV",
        "vis": "VIS",
        "rgb": "RGB",
        "re": "RE",
        "nir": "NIR",
        "swir-1": "SWIR-1",
        "swir-2a": "SWIR-2a",
        "swir-2b": "SWIR-2b",
        # "nir-swir-1": "NIR-SWIR-1",
        # "re-nir": "RE-NIR",
        # "vis-rgb": "VIS-RGB",
    }

    if b in mapping:
        return mapping[b]

    # fallback: uppercase
    return band_raw.upper()


def find_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Find columns starting with 'test/' and convert them to plain metric names.
    Example: test/loss -> loss
    """
    metric_cols = [c for c in df.columns if c.startswith("test/")]
    return metric_cols


def compute_stats(values: pd.Series) -> Dict[str, float]:
    """
    Compute summary statistics for one metric column.
    """
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "sem": np.nan,
            "ci95": np.nan,
        }

    n = len(x)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem if n > 1 else 0.0

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "sem": sem,
        "ci95": ci95,
    }


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """
    Format as 'mean ± std' for Excel copy/paste.
    """
    if pd.isna(mean):
        return ""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def sort_bands_for_output(bands: List[str]) -> List[str]:
    """
    Sort according to BAND_ORDER, and keep unknown bands at the end.
    """
    known = [b for b in BAND_ORDER if b in bands]
    unknown = sorted([b for b in bands if b not in BAND_ORDER])
    return known + unknown


# ============================================================
# MAIN PROCESSING
# ============================================================


def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("metrics_stats_*_day_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No matching CSV files found in: {INPUT_DIR.resolve()}"
        )

    all_stats_rows = []

    for csv_path in csv_files:
        info = parse_filename(csv_path)
        if info is None:
            print(f"[Skip] Filename does not match pattern: {csv_path.name}")
            continue

        band_raw = str(info["band_raw"])
        band = pretty_band_name(band_raw)
        day = int(info["day"])
        day_label = DAY_LABELS.get(day, f"day_{day:02d}")

        df = pd.read_csv(csv_path)

        metric_cols = find_metric_columns(df)
        if not metric_cols:
            print(f"[Warning] No test/ metrics found in: {csv_path.name}")
            continue

        for col in metric_cols:
            metric_name = col.replace("test/", "")
            stats = compute_stats(df[col])

            row = {
                "file": csv_path.name,
                "band_raw": band_raw,
                "band": band,
                "day": day,
                "day_label": day_label,
                "metric": metric_name,
                **stats,
            }
            all_stats_rows.append(row)

    if not all_stats_rows:
        raise RuntimeError("No statistics could be computed from the provided files.")

    stats_df = pd.DataFrame(all_stats_rows)

    # --------------------------------------------------------
    # Save long-format summary
    # --------------------------------------------------------
    long_csv = OUTPUT_DIR / "metrics_summary_long.csv"
    stats_df.to_csv(long_csv, index=False)

    # --------------------------------------------------------
    # Excel-ready table with mean ± std
    # rows: band, metric
    # cols: date labels
    # --------------------------------------------------------
    table_df = stats_df.copy()
    table_df["value"] = table_df.apply(
        lambda r: format_mean_std(r["mean"], r["std"], decimals=4), axis=1
    )

    pivot = table_df.pivot_table(
        index=["band", "metric"],
        columns="day_label",
        values="value",
        aggfunc="first",
    )

    # Ensure desired column order
    ordered_day_labels = [
        DAY_LABELS[d] for d in sorted(DAY_LABELS) if DAY_LABELS[d] in pivot.columns
    ]
    remaining_cols = [c for c in pivot.columns if c not in ordered_day_labels]
    pivot = pivot[ordered_day_labels + remaining_cols]

    # Ensure desired row order
    present_bands = list(pivot.index.get_level_values("band").unique())
    ordered_bands = sort_bands_for_output(present_bands)

    ordered_index = []
    for band in ordered_bands:
        for metric in METRIC_ORDER:
            if (band, metric) in pivot.index:
                ordered_index.append((band, metric))

    # add anything not explicitly listed
    remaining_index = [idx for idx in pivot.index if idx not in ordered_index]
    ordered_index.extend(remaining_index)

    pivot = pivot.reindex(ordered_index)

    excel_ready_csv = OUTPUT_DIR / "metrics_table_mean_std.csv"
    pivot.to_csv(excel_ready_csv)

    # --------------------------------------------------------
    # Optional: separate tables for mean and std only
    # --------------------------------------------------------
    mean_pivot = stats_df.pivot_table(
        index=["band", "metric"],
        columns="day_label",
        values="mean",
        aggfunc="first",
    )
    std_pivot = stats_df.pivot_table(
        index=["band", "metric"],
        columns="day_label",
        values="std",
        aggfunc="first",
    )

    mean_pivot = mean_pivot.reindex(
        index=ordered_index,
        columns=ordered_day_labels
        + [c for c in mean_pivot.columns if c not in ordered_day_labels],
    )
    std_pivot = std_pivot.reindex(
        index=ordered_index,
        columns=ordered_day_labels
        + [c for c in std_pivot.columns if c not in ordered_day_labels],
    )

    mean_pivot.to_csv(OUTPUT_DIR / "metrics_table_mean.csv")
    std_pivot.to_csv(OUTPUT_DIR / "metrics_table_std.csv")

    # --------------------------------------------------------
    # Optional: save to Excel with multiple sheets
    # --------------------------------------------------------
    xlsx_path = OUTPUT_DIR / "metrics_aggregated.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="long_summary", index=False)
        pivot.to_excel(writer, sheet_name="mean_std")
        mean_pivot.to_excel(writer, sheet_name="mean")
        std_pivot.to_excel(writer, sheet_name="std")

    print(f"[Done] Long summary saved to: {long_csv}")
    print(f"[Done] Excel-ready mean±std table saved to: {excel_ready_csv}")
    print(f"[Done] Excel workbook saved to: {xlsx_path}")


if __name__ == "__main__":
    main()

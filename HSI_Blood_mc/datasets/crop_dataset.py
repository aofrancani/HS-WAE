from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
import pywt

# -----------------------------pip install PyWavelets
# Crop file mapping (filenames only)
# -----------------------------
CROP_FILES = {
    "soy": "soybean_HSI.csv",
    "cotton": "cotton_HSI.csv",
    "corn": "corn_HSI.csv",
}

HOUR_FILTERS = {
    1: 14,  # Day 1 → only 14:00 samples
    2: 11,  # Day 2 → only 11:00 samples
}


# -----------------------------
# Preprocessing configuration
# -----------------------------
@dataclass
class SpectralPreprocessConfig:
    # Wavelength filters
    input_range: Tuple[int, int] = (350, 2500)
    keep_range: Tuple[int, int] = (390, 2400)
    exclude_ranges: List[Tuple[int, int]] = None

    # Columns
    label_col: str = "Label"
    meta_cols: Sequence[str] = ("ID", "Date", "GroundTruth", "Label")

    # DWT options (disabled by default)
    dwt_enabled: bool = False
    dwt_wavelet: str = "db4"
    dwt_level: Optional[int] = None  # None = max level
    dwt_mode: str = "coeffs"  # "coeffs" or "energy"
    dwt_boundary: str = "periodization"  # stable border handling
    dwt_include_approx: bool = True  # only for "coeffs"
    dwt_log_energy: bool = False  # only for "energy": log1p

    def __post_init__(self):
        if self.exclude_ranges is None:
            self.exclude_ranges = [(1350, 1440), (1800, 1980)]


# -----------------------------
# Helpers
# -----------------------------
def _digit_string_columns(df: pd.DataFrame) -> List[str]:
    # numeric column names (e.g., "390", "391", ...)
    return [c for c in df.columns if str(c).isdigit()]


def _filter_wavelengths(
    cols: List[str], cfg: SpectralPreprocessConfig
) -> Tuple[List[str], List[int]]:
    wls = np.array(list(map(int, cols)))
    keep = (
        (wls >= cfg.input_range[0])
        & (wls <= cfg.input_range[1])
        & (wls >= cfg.keep_range[0])
        & (wls <= cfg.keep_range[1])
    )
    for lo, hi in cfg.exclude_ranges:
        keep &= ~((wls >= lo) & (wls <= hi))
    kept = np.sort(wls[keep])
    kept_cols = [str(w) for w in kept.tolist()]
    return kept_cols, kept.tolist()


def _fit_standardizer(X: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _apply_standardizer(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - stats["mean"]) / stats["std"]


def _add_dayindex_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise KeyError("Expected a 'Date' column in the CSV.")
    df = df.copy()
    day_strings = df["Date"].astype(str).str.split(" at ").str[0]
    mapping = {day: i + 1 for i, day in enumerate(sorted(day_strings.unique()))}
    df["DayIndex"] = day_strings.map(mapping).astype(int)
    return df


def _fix_year(s: str) -> str:
    """Fix weird '119' → '2019' year format."""
    parts = s.split(" at ")[0].split("/")
    if len(parts[2]) == 3:
        parts[2] = "20" + parts[2][-2:]
    return "/".join(parts) + " " + s.split(" at ")[1]


def load_crop_df(base_path: Path, crop: str, label_offset: int) -> pd.DataFrame:
    """Load one crop CSV and remap labels with offset (0–5)."""
    filename = CROP_FILES[crop.lower()]
    path = base_path / filename
    df = pd.read_csv(path).copy()
    # Apply label offset (keeps original spectral data intact)
    df["Label"] = df["Label"] + label_offset
    return df


def to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_oh[np.arange(y.shape[0]), y] = 1.0
    return y_oh


# -----------------------------
# DWT helpers
# -----------------------------
def _dwt_row(
    x: np.ndarray,
    wavelet: str,
    level: Optional[int],
    boundary: str,
    mode: str,
    include_approx: bool,
    log_energy: bool,
) -> np.ndarray:
    """
    Apply DWT to a 1D signal and return either flattened coeffs or per-level energies.
    """
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=boundary)

    if mode == "coeffs":
        parts = []
        if include_approx:
            parts.append(coeffs[0])
        parts.extend(coeffs[1:])  # details from lowest freq to highest
        return np.concatenate(parts, axis=0).astype(np.float32)

    elif mode == "energy":
        # detail energies only
        energies = np.array([np.sum(ci**2) for ci in coeffs[1:]], dtype=np.float64)
        if log_energy:
            energies = np.log1p(energies)
        return energies.astype(np.float32)

    else:
        raise ValueError("dwt_mode must be 'coeffs' or 'energy'")


def _dwt_transform_matrix(
    X: np.ndarray,
    wavelet: str,
    level: Optional[int],
    boundary: str = "periodization",
    mode: str = "coeffs",
    include_approx: bool = True,
    log_energy: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply DWT row-wise to X.

    Returns
    -------
    X_out : (N, D') array
    feat_names : list of str with names for the features
    """
    # Probe first row to infer output length and names
    c0 = pywt.wavedec(X[0], wavelet=wavelet, level=level, mode=boundary)
    if mode == "coeffs":
        parts = []
        names = []
        if include_approx:
            parts.append(("A", len(c0[0])))
        # c0[1], c0[2], ... are D_L, D_{L-1}, ..., D_1 (pywt order is [A_L, D_L, D_{L-1}, ..., D_1])
        for li, ci in enumerate(c0[1:], start=1):
            parts.append((f"D{len(c0)-li}", len(ci)))  # e.g., D6, D5, ..., D1
        for band_name, band_len in parts:
            names.extend([f"DWT_{band_name}_{k}" for k in range(band_len)])
        out_len = sum(n for _, n in parts)

    else:  # energy
        # one value per detail level
        L = len(c0) - 1
        names = [f"DWT_Energy_L{i}" for i in range(1, L + 1)]
        out_len = L

    X_out = np.empty((X.shape[0], out_len), dtype=np.float32)

    for i in range(X.shape[0]):
        X_out[i] = _dwt_row(
            X[i],
            wavelet=wavelet,
            level=level,
            boundary=boundary,
            mode=mode,
            include_approx=include_approx,
            log_energy=log_energy,
        )

    return X_out, names


# -----------------------------
# Dataset
# -----------------------------
class SpectraDataset(Dataset):
    """Simple tensor dataset for (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# DataModule
# -----------------------------
class CSVHSIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_path: str | Path,
        crops: List[str],
        preprocess_cfg: SpectralPreprocessConfig = SpectralPreprocessConfig(),
        batch_size: int = 128,
        num_workers: int = 4,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
        random_state: int = 42,
        standardize: bool = True,
        use_weighted_sampler: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        date_filters: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.base_path = Path(base_path)
        self.crops = crops
        self.cfg = preprocess_cfg
        self.label_mapping = {}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.random_state = random_state
        self.standardize = standardize
        self.use_weighted_sampler = use_weighted_sampler
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.date_filters = list(date_filters) if date_filters is not None else None

        # Attributes for introspection
        self.kept_wavelengths_nm: List[int] = []
        self.band_count: int = 0  # original band count after wavelength filters
        self.num_classes: int = 0
        self.feature_names: List[str] = (
            []
        )  # final feature names (wavelengths or DWT names)
        self.standardizer_stats: Optional[Dict[str, np.ndarray]] = None

        # Store original (concatenated) DataFrame (with label offsets) for the user
        self.df_original: Optional[pd.DataFrame] = None

        # Datasets and sampler
        self.train_ds: Optional[SpectraDataset] = None
        self.val_ds: Optional[SpectraDataset] = None
        self.test_ds: Optional[SpectraDataset] = None
        self._train_sampler = None

    @property
    def input_dim(self) -> int:
        # Return *final* input dimension (after DWT if enabled)
        return len(self.feature_names)

    def setup(self, stage: Optional[str] = None):
        # --- Load crops ---
        dfs = []
        self.label_mapping = {}

        # Create label mapping based on crop order from config
        for i, crop in enumerate(self.crops):
            offset = i * 2
            # Load data with offset
            dfc = load_crop_df(self.base_path, crop, offset)
            dfs.append(dfc)

            # Create mapping for this crop's labels
            crop_name = crop.capitalize()
            self.label_mapping[offset] = f"{crop_name} Control"  # 0 + offset
            self.label_mapping[offset + 1] = f"{crop_name} Damaged"  # 1 + offset

        df_all = pd.concat(dfs, ignore_index=True)
        # Keep an untouched copy for the user (original spectra + label offsets)
        self.df_original = df_all.copy(deep=True)

        # Spectral columns
        spectral_cols_all = _digit_string_columns(df_all)
        kept_cols, kept_wls = _filter_wavelengths(spectral_cols_all, self.cfg)
        self.kept_wavelengths_nm = kept_wls
        self.band_count = len(kept_cols)

        # Add DayIndex and Hour_only
        df_all = _add_dayindex_column(df_all)
        df_all["Datetime"] = pd.to_datetime(
            df_all["Date"].apply(_fix_year), format="%m/%d/%Y %H:%M:%S"
        )
        df_all["Hour_only"] = df_all["Datetime"].dt.hour

        # Filter by DayIndex if requested
        if self.date_filters:
            targets = set(int(x) for x in self.date_filters)
            df_all = df_all[df_all["DayIndex"].isin(targets)]
            if len(df_all) == 0:
                raise ValueError(
                    f"No rows left after filtering by DayIndex {sorted(targets)}."
                )

        # Apply per-day hour filtering
        df_filtered_list = []
        for day, group in df_all.groupby("DayIndex"):
            if day in HOUR_FILTERS:
                # Keep only rows from the specified hour for this day
                filtered = group[group["Hour_only"] == HOUR_FILTERS[day]]
                if filtered.empty:
                    print(
                        f" -- No samples found for Day {day} at {HOUR_FILTERS[day]}:00"
                    )
                df_filtered_list.append(filtered)
            else:
                # Keep all other days unchanged
                df_filtered_list.append(group)

        # Recombine all days
        df_all = pd.concat(df_filtered_list, ignore_index=True)

        # Features and labels (raw wavelength space)
        X = df_all[kept_cols].astype("float32").to_numpy()
        y_int = df_all[self.cfg.label_col].astype("int64").to_numpy()
        self.num_classes = int(np.unique(y_int).size)

        # --- Optional DWT preprocessing (before split standardization) ---
        if self.cfg.dwt_enabled:
            X, dwt_names = _dwt_transform_matrix(
                X,
                wavelet=self.cfg.dwt_wavelet,
                level=self.cfg.dwt_level,
                boundary=self.cfg.dwt_boundary,
                mode=self.cfg.dwt_mode,
                include_approx=self.cfg.dwt_include_approx,
                log_energy=self.cfg.dwt_log_energy,
            )
            self.feature_names = dwt_names
        else:
            # Keep wavelengths as feature names
            self.feature_names = [str(w) for w in self.kept_wavelengths_nm]

        # Stratified split (on labels)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X,
            y_int,
            test_size=(1.0 - self.train_pct),
            stratify=y_int,
            random_state=self.random_state,
        )
        val_ratio_of_tmp = self.val_pct / (self.val_pct + self.test_pct + 1e-12)
        X_va, X_te, y_va, y_te = train_test_split(
            X_tmp,
            y_tmp,
            test_size=(1.0 - val_ratio_of_tmp),
            stratify=y_tmp,
            random_state=self.random_state,
        )

        # Standardization (fit on training, apply to all)
        if self.standardize:
            self.standardizer_stats = _fit_standardizer(X_tr)
            X_tr = _apply_standardizer(X_tr, self.standardizer_stats)
            X_va = _apply_standardizer(X_va, self.standardizer_stats)
            X_te = _apply_standardizer(X_te, self.standardizer_stats)
        else:
            self.standardizer_stats = None

        # Build datasets
        self.train_ds = SpectraDataset(X_tr, y_tr)
        self.val_ds = SpectraDataset(X_va, y_va)
        self.test_ds = SpectraDataset(X_te, y_te)

        # Weighted sampler if needed
        if self.use_weighted_sampler:
            class_counts = np.bincount(y_tr)
            class_weights = (
                class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
            ).astype(np.float32)
            sample_weights = class_weights[y_tr]
            self._train_sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
        else:
            self._train_sampler = None

    def get_label_mapping(self) -> Dict[int, str]:
        """Return the current label mapping based on loaded crops."""
        return self.label_mapping

    # ----------------- Loaders -----------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # ----------------- Utilities -----------------
    def export_band_list(self, path: str | Path):
        """
        If DWT is disabled, exports kept wavelengths.
        If DWT is enabled, exports final feature names (e.g., DWT bands).
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {}
        if self.cfg.dwt_enabled:
            payload["feature_names"] = self.feature_names
            payload["dwt"] = {
                "wavelet": self.cfg.dwt_wavelet,
                "level": self.cfg.dwt_level,
                "mode": self.cfg.dwt_mode,
                "boundary": self.cfg.dwt_boundary,
                "include_approx": self.cfg.dwt_include_approx,
                "log_energy": self.cfg.dwt_log_energy,
            }
        else:
            payload["kept_wavelengths_nm"] = self.kept_wavelengths_nm
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def get_standardizer(self) -> Optional[Dict[str, np.ndarray]]:
        return self.standardizer_stats


# -----------------------------
# Builder
# -----------------------------
def build_datamodule(cfg: Dict[str, Any]) -> CSVHSIDataModule:
    data_cfg = cfg["data"]
    pp_cfg = data_cfg.get("preprocess", {})

    preprocess_cfg = SpectralPreprocessConfig(
        input_range=tuple(pp_cfg.get("input_range", (350, 2500))),
        keep_range=tuple(pp_cfg.get("keep_range", (390, 2400))),
        exclude_ranges=[
            tuple(t) for t in pp_cfg.get("exclude_ranges", [(1350, 1440), (1800, 1980)])
        ],
        label_col=pp_cfg.get("label_col", "Label"),
        # DWT block (optional)
        dwt_enabled=pp_cfg.get("dwt", {}).get("enabled", False),
        dwt_wavelet=pp_cfg.get("dwt", {}).get("wavelet", "db4"),
        dwt_level=pp_cfg.get("dwt", {}).get("level", None),
        dwt_mode=pp_cfg.get("dwt", {}).get("mode", "coeffs"),
        dwt_boundary=pp_cfg.get("dwt", {}).get("boundary", "periodization"),
        dwt_include_approx=pp_cfg.get("dwt", {}).get("include_approx", True),
        dwt_log_energy=pp_cfg.get("dwt", {}).get("log_energy", False),
    )
    date_filters = data_cfg.get("date_filters", None)

    dm = CSVHSIDataModule(
        base_path=Path(data_cfg["csv_path"]),  # base folder path
        crops=data_cfg.get("crops", ["soy"]),
        preprocess_cfg=preprocess_cfg,
        batch_size=data_cfg.get("batch_size", 128),
        num_workers=data_cfg.get("num_workers", 4),
        train_pct=data_cfg.get("train_pct", 0.70),
        val_pct=data_cfg.get("val_pct", 0.15),
        test_pct=data_cfg.get("test_pct", 0.15),
        standardize=data_cfg.get("standardize", True),
        use_weighted_sampler=data_cfg.get("use_weighted_sampler", False),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=data_cfg.get("drop_last", False),
        persistent_workers=True,
        date_filters=date_filters,
    )
    dm.setup()
    return dm

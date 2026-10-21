# dataloader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
import pywt

# -----------------------------
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
    dwt_log_energy: bool = False  # only for "energy": log1p (kept for compatibility)

    def __post_init__(self):
        if self.exclude_ranges is None:
            self.exclude_ranges = [(1350, 1440), (1800, 1980)]


# -----------------------------
# Helpers
# -----------------------------
def _digit_string_columns(df: pd.DataFrame) -> List[str]:
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


def _fit_standardizer_levels(
    X_levels: List[List[np.ndarray]], eps: float = 1e-8
) -> List[Dict[str, np.ndarray]]:
    """
    Fit mean/std per DWT level (token).
    Returns a list of stats dicts, one per level.
    """
    K = len(X_levels[0])
    stats: List[Dict[str, np.ndarray]] = []
    for l in range(K):
        Xl = np.stack([row[l] for row in X_levels], axis=0)  # (N, n_l)
        mean = Xl.mean(axis=0)
        std = Xl.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        stats.append({"mean": mean.astype(np.float32), "std": std.astype(np.float32)})
    return stats


def _apply_standardizer_levels(
    X_levels: List[List[np.ndarray]], stats: List[Dict[str, np.ndarray]]
) -> List[List[np.ndarray]]:
    K = len(stats)
    X_out: List[List[np.ndarray]] = []
    for row in X_levels:
        out_row: List[np.ndarray] = []
        for l in range(K):
            out_row.append(
                ((row[l] - stats[l]["mean"]) / stats[l]["std"]).astype(np.float32)
            )
        X_out.append(out_row)
    return X_out


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
    filename = CROP_FILES[crop.lower()]
    path = base_path / filename
    df = pd.read_csv(path).copy()
    df["Label"] = df["Label"] + label_offset
    return df


# -----------------------------
# DWT helpers (TOKEN MODE: return per-level vectors)
# -----------------------------
def _dwt_row_levels(
    x: np.ndarray,
    wavelet: str,
    level: Optional[int],
    boundary: str,
    include_approx: bool,
) -> List[np.ndarray]:
    """
    Returns a list of coefficient vectors in fixed PyWavelets order:
      coeffs = [A_L, D_L, D_{L-1}, ..., D_1]
    """
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=boundary)
    parts = coeffs if include_approx else coeffs[1:]
    return [p.astype(np.float32) for p in parts]


def _dwt_probe_levels(
    x0: np.ndarray,
    wavelet: str,
    level: Optional[int],
    boundary: str,
    include_approx: bool,
) -> Tuple[List[str], List[int]]:
    """
    Returns:
      names: ["A_L", "D_L", "D_{L-1}", ..., "D_1"] (or without A_L)
      lengths: [len(A_L), len(D_L), ..., len(D_1)]
    """
    c0 = pywt.wavedec(x0, wavelet=wavelet, level=level, mode=boundary)
    L = len(c0) - 1  # number of detail levels
    if include_approx:
        names = ["A_L"] + [f"D_{k}" for k in range(L, 0, -1)]  # D_L ... D_1
        parts = c0
    else:
        names = [f"D_{k}" for k in range(L, 0, -1)]
        parts = c0[1:]
    lengths = [len(p) for p in parts]
    return names, lengths


# -----------------------------
# Collate for tokenized DWT
# -----------------------------
def dwt_levels_collate_fn(batch):
    """
    batch: list of (x_raws, levels_list, y)
      levels_list: [Tensor(n_0), Tensor(n_1), ..., Tensor(n_{K-1})]
    Returns:
      x_raw: Tensor(B, bands)
      levels_batch: List[Tensor], each (B, n_l)
      y: Tensor(B,)
    """
    x_raws, levels_lists, ys = zip(*batch)
    x_raw = torch.stack(x_raws, dim=0)  # (B, bands)
    y = torch.stack(ys, dim=0)  # (B,)

    K = len(levels_lists[0])
    levels_batch = []
    for l in range(K):
        levels_batch.append(
            torch.stack([levels_lists[b][l] for b in range(len(batch))], dim=0)
        )
    return x_raw, levels_batch, y


# -----------------------------
# Dataset
# -----------------------------
class SpectraDataset(Dataset):
    """Tensor dataset for (X, y), X is (D,) vector."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SpectraDWTDataset(Dataset):
    """
    Each item: (x_raw, levels_list, y)
      x_raw: (bands,)
      levels_list: [ (n_0,), (n_1,), ..., (n_{K-1},) ]
    """

    def __init__(
        self, X_raw: np.ndarray, X_levels: List[List[np.ndarray]], y: np.ndarray
    ):
        assert X_raw.ndim == 2
        assert len(X_levels) == y.shape[0] == X_raw.shape[0]
        self.X_raw = torch.from_numpy(X_raw.astype(np.float32))
        self.X_levels = X_levels
        self.y = torch.from_numpy(y.astype(np.int64))

        K = len(X_levels[0])
        for i in range(len(X_levels)):
            if len(X_levels[i]) != K:
                raise ValueError("Inconsistent number of DWT levels across samples.")

    def __len__(self) -> int:
        return len(self.X_levels)

    def __getitem__(self, idx: int):
        x_raw = self.X_raw[idx]  # (bands,)
        levels = [torch.from_numpy(v.astype(np.float32)) for v in self.X_levels[idx]]
        return x_raw, levels, self.y[idx]


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

        # Introspection
        self.kept_wavelengths_nm: List[int] = []
        self.band_count: int = 0
        self.num_classes: int = 0

        # If DWT disabled: wavelength feature names (strings)
        # If DWT enabled (token mode): level names, e.g. ["A_L","D_7",...,"D_1"]
        self.feature_names: List[str] = []

        # DWT token metadata (only if enabled)
        self.dwt_level_lengths: Optional[List[int]] = None  # [n_0, n_1, ..., n_{K-1}]

        # Standardizer stats:
        # - if DWT disabled: Dict(mean,std)
        # - if DWT enabled: List[Dict(mean,std)] per level
        self.standardizer_stats: Optional[
            Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
        ] = None

        self.df_original: Optional[pd.DataFrame] = None

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._train_sampler = None

    @property
    def input_dim(self) -> int:
        """
        Final dimension that a downstream model should expect.
        - No DWT: number of wavelengths kept
        - DWT token mode: sum of lengths of all levels (if you later concat after attention)
        """
        if self.cfg.dwt_enabled:
            if not self.dwt_level_lengths:
                return 0
            return int(sum(self.dwt_level_lengths))
        return len(self.feature_names)

    def setup(self, stage: Optional[str] = None):
        # --- Load crops ---
        dfs = []
        self.label_mapping = {}

        for i, crop in enumerate(self.crops):
            offset = i * 2
            dfc = load_crop_df(self.base_path, crop, offset)
            dfs.append(dfc)

            crop_name = crop.capitalize()
            self.label_mapping[offset] = f"{crop_name} Control"
            self.label_mapping[offset + 1] = f"{crop_name} Damaged"

        df_all = pd.concat(dfs, ignore_index=True)
        self.df_original = df_all.copy(deep=True)

        # Spectral columns
        spectral_cols_all = _digit_string_columns(df_all)
        kept_cols, kept_wls = _filter_wavelengths(spectral_cols_all, self.cfg)
        self.kept_wavelengths_nm = kept_wls
        self.band_count = len(kept_cols)

        # Add DayIndex + hour filter
        df_all = _add_dayindex_column(df_all)
        df_all["Datetime"] = pd.to_datetime(
            df_all["Date"].apply(_fix_year), format="%m/%d/%Y %H:%M:%S"
        )
        df_all["Hour_only"] = df_all["Datetime"].dt.hour

        if self.date_filters:
            targets = set(int(x) for x in self.date_filters)
            df_all = df_all[df_all["DayIndex"].isin(targets)]
            if len(df_all) == 0:
                raise ValueError(
                    f"No rows left after filtering by DayIndex {sorted(targets)}."
                )

        df_filtered_list = []
        for day, group in df_all.groupby("DayIndex"):
            if day in HOUR_FILTERS:
                filtered = group[group["Hour_only"] == HOUR_FILTERS[day]]
                df_filtered_list.append(filtered)
            else:
                df_filtered_list.append(group)
        df_all = pd.concat(df_filtered_list, ignore_index=True)

        # Base features in wavelength space
        X = df_all[kept_cols].astype("float32").to_numpy()  # (N, bands)
        y_int = df_all[self.cfg.label_col].astype("int64").to_numpy()
        self.num_classes = int(np.unique(y_int).size)

        # ------------------------------------------------------------
        # DWT path: return tokens per level (no concatenation)
        # ------------------------------------------------------------
        if self.cfg.dwt_enabled:
            if self.cfg.dwt_mode != "coeffs":
                raise ValueError("Token-attention DWT path requires dwt_mode='coeffs'.")

            level_names, level_lengths = _dwt_probe_levels(
                X[0],
                wavelet=self.cfg.dwt_wavelet,
                level=self.cfg.dwt_level,
                boundary=self.cfg.dwt_boundary,
                include_approx=self.cfg.dwt_include_approx,
            )
            self.feature_names = level_names
            self.dwt_level_lengths = level_lengths

            X_levels: List[List[np.ndarray]] = [
                _dwt_row_levels(
                    X[i],
                    wavelet=self.cfg.dwt_wavelet,
                    level=self.cfg.dwt_level,
                    boundary=self.cfg.dwt_boundary,
                    include_approx=self.cfg.dwt_include_approx,
                )
                for i in range(X.shape[0])
            ]

            # Split by indices (since X_levels is ragged container)
            idx = np.arange(len(y_int))
            idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
                idx,
                y_int,
                test_size=(1.0 - self.train_pct),
                stratify=y_int,
                random_state=self.random_state,
            )
            val_ratio_of_tmp = self.val_pct / (self.val_pct + self.test_pct + 1e-12)
            idx_va, idx_te, y_va, y_te = train_test_split(
                idx_tmp,
                y_tmp,
                test_size=(1.0 - val_ratio_of_tmp),
                stratify=y_tmp,
                random_state=self.random_state,
            )

            X_tr_raw = X[idx_tr]
            X_va_raw = X[idx_va]
            X_te_raw = X[idx_te]

            X_tr_levels = [X_levels[i] for i in idx_tr]
            X_va_levels = [X_levels[i] for i in idx_va]
            X_te_levels = [X_levels[i] for i in idx_te]

            if self.standardize:
                self.standardizer_stats = _fit_standardizer_levels(X_tr_levels)
                X_tr_levels = _apply_standardizer_levels(X_tr_levels, self.standardizer_stats)  # type: ignore[arg-type]
                X_va_levels = _apply_standardizer_levels(X_va_levels, self.standardizer_stats)  # type: ignore[arg-type]
                X_te_levels = _apply_standardizer_levels(X_te_levels, self.standardizer_stats)  # type: ignore[arg-type]
            else:
                self.standardizer_stats = None

            self.train_ds = SpectraDWTDataset(X_tr_raw, X_tr_levels, y_tr)
            self.val_ds = SpectraDWTDataset(X_va_raw, X_va_levels, y_va)
            self.test_ds = SpectraDWTDataset(X_te_raw, X_te_levels, y_te)

            # Weighted sampler (uses y_tr)
            if self.use_weighted_sampler:
                class_counts = np.bincount(y_tr)
                class_weights = (
                    class_counts.sum()
                    / (len(class_counts) * np.maximum(class_counts, 1))
                ).astype(np.float32)
                sample_weights = class_weights[y_tr]
                self._train_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
            else:
                self._train_sampler = None

        # ------------------------------------------------------------
        # No DWT path: old behavior (single vector)
        # ------------------------------------------------------------
        else:
            self.feature_names = [str(w) for w in self.kept_wavelengths_nm]
            self.dwt_level_lengths = None

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

            if self.standardize:
                self.standardizer_stats = _fit_standardizer(X_tr)
                X_tr = _apply_standardizer(X_tr, self.standardizer_stats)
                X_va = _apply_standardizer(X_va, self.standardizer_stats)
                X_te = _apply_standardizer(X_te, self.standardizer_stats)
            else:
                self.standardizer_stats = None

            self.train_ds = SpectraDataset(X_tr, y_tr)
            self.val_ds = SpectraDataset(X_va, y_va)
            self.test_ds = SpectraDataset(X_te, y_te)

            if self.use_weighted_sampler:
                class_counts = np.bincount(y_tr)
                class_weights = (
                    class_counts.sum()
                    / (len(class_counts) * np.maximum(class_counts, 1))
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
            collate_fn=(dwt_levels_collate_fn if self.cfg.dwt_enabled else None),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=(dwt_levels_collate_fn if self.cfg.dwt_enabled else None),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=(dwt_levels_collate_fn if self.cfg.dwt_enabled else None),
        )

    # ----------------- Utilities -----------------
    def export_band_list(self, path: str | Path):
        """
        If DWT is disabled: exports kept wavelengths.
        If DWT is enabled (token mode): exports level names + lengths and DWT params.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {}
        if self.cfg.dwt_enabled:
            payload["dwt_tokens"] = {
                "level_names": self.feature_names,
                "level_lengths": self.dwt_level_lengths,
            }
            payload["dwt"] = {
                "wavelet": self.cfg.dwt_wavelet,
                "level": self.cfg.dwt_level,
                "mode": self.cfg.dwt_mode,
                "boundary": self.cfg.dwt_boundary,
                "include_approx": self.cfg.dwt_include_approx,
            }
        else:
            payload["kept_wavelengths_nm"] = self.kept_wavelengths_nm

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def get_standardizer(self):
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
        dwt_enabled=pp_cfg.get("dwt", {}).get("enabled", False),
        dwt_wavelet=pp_cfg.get("dwt", {}).get("wavelet", "db4"),
        dwt_level=pp_cfg.get("dwt", {}).get("level", None),
        dwt_mode=pp_cfg.get("dwt", {}).get("mode", "coeffs"),
        dwt_boundary=pp_cfg.get("dwt", {}).get("boundary", "periodization"),
        dwt_include_approx=pp_cfg.get("dwt", {}).get("include_approx", True),
        dwt_log_energy=pp_cfg.get("dwt", {}).get("log_energy", False),
    )

    dm = CSVHSIDataModule(
        base_path=Path(data_cfg["csv_path"]),
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
        date_filters=data_cfg.get("date_filters", None),
    )
    dm.setup()
    return dm


# -----------------------------
# Debug entry
# -----------------------------
if __name__ == "__main__":
    cfg = {
        "data": {
            "csv_path": "C:/workspace/HSI-Class/data",  # base folder path
            "crops": ["soy", "cotton", "corn"],
            "batch_size": 16,
            "num_workers": 0,
            "train_pct": 0.70,
            "val_pct": 0.15,
            "test_pct": 0.15,
            "standardize": True,
            "use_weighted_sampler": True,
            "date_filters": None,
            "preprocess": {
                "input_range": [350, 2500],
                "keep_range": [390, 2400],
                "exclude_ranges": [[1350, 1440], [1800, 1980]],
                "label_col": "Label",
                "dwt": {
                    "enabled": True,  # toggle here
                    "wavelet": "db4",
                    "level": 7,  # max level
                    "mode": "coeffs",  # or "energy"
                    "boundary": "periodization",
                    "include_approx": True,
                    "log_energy": False,  # only used if mode="energy"
                },
            },
        }
    }

    print("Building DataModule...")
    dm = build_datamodule(cfg)

    print("\n--- DataModule Info ---")
    print(f"Final input dim: {dm.input_dim} (DWT enabled: {dm.cfg.dwt_enabled})")
    print(f"Num classes: {dm.num_classes}")
    if dm.cfg.dwt_enabled:
        print(f"First 5 features: {dm.feature_names[:5]}")
    else:
        print(
            f"Wavelengths: {dm.kept_wavelengths_nm[:5]} ... {dm.kept_wavelengths_nm[-5:]}"
        )

    # --- Class distribution per split ---
    def class_stats(name: str, ds: SpectraDataset):
        labels, counts = np.unique(ds.y.numpy(), return_counts=True)
        print(f"{name} class distribution:")
        for lbl, cnt in zip(labels, counts):
            print(f"  Class {lbl}: {cnt}")
        print(f"  Total: {len(ds)}\n")

    print("\n--- Train/Val/Test sizes ---")
    class_stats("Train", dm.train_ds)
    class_stats("Val", dm.val_ds)
    class_stats("Test", dm.test_ds)
    print(f"Total:  {len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)}")

    train_loader = dm.train_dataloader()
    batch_X, batch_y = next(iter(train_loader))
    print("\n--- Example batch ---")
    print("X shape:", batch_X.shape)
    print("y shape:", batch_y.shape)

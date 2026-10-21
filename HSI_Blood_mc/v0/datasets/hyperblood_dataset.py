from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
import pywt
import re


# -----------------------------
# Preprocessing configuration
# -----------------------------
@dataclass
class SpectralPreprocessConfig:
    keep_range: Tuple[float, float] = (377.0, 1046.0)
    exclude_ranges: List[Tuple[float, float]] = None

    class_id_col: str = "class_id"
    class_name_col: str = "class_name"
    scene_col: str = "scene"
    blob_id_col: str = "blob_id"
    n_pixels_col: str = "n_pixels"

    # --- NEW (minimal) ---
    # Choose label mode:
    #   "binary" -> blood vs non-blood (old behavior)
    #   "multiclass" -> keep original classes (optionally dropping background/uncertain via existing flags),
    #                  and remap to contiguous 0..C-1 for CE loss.
    label_mode: str = "binary"  # "binary" or "multiclass"
    remap_classes_to_contiguous: bool = True  # recommended for multiclass

    blood_ids: Tuple[int, ...] = (1,)
    include_uncertain_blood: bool = False
    exclude_background: bool = True

    dwt_enabled: bool = False
    dwt_wavelet: str = "db4"
    dwt_level: Optional[int] = None
    dwt_mode: str = "coeffs"
    dwt_boundary: str = "periodization"
    dwt_include_approx: bool = True
    dwt_log_energy: bool = False

    def __post_init__(self):
        if self.exclude_ranges is None:
            self.exclude_ranges = []
        self.label_mode = str(self.label_mode).lower().strip()
        if self.label_mode not in ("binary", "multiclass"):
            raise ValueError("label_mode must be 'binary' or 'multiclass'")


WL_COL_RE = re.compile(r"^wl_(\d+(?:\.\d+)?)$")


# -----------------------------
# Helpers
# -----------------------------
def _get_wavelength_columns(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    pairs: List[Tuple[float, str]] = []
    for c in df.columns:
        m = WL_COL_RE.match(str(c))
        if m:
            pairs.append((float(m.group(1)), str(c)))
    pairs.sort(key=lambda x: x[0])
    wl_cols = [c for _, c in pairs]
    wl_vals = np.array([w for w, _ in pairs], dtype=np.float32)
    return wl_cols, wl_vals


def _filter_wavelengths(
    wl_cols: List[str],
    wl_vals: np.ndarray,
    cfg: SpectralPreprocessConfig,
) -> Tuple[List[str], np.ndarray]:
    keep = (wl_vals >= cfg.keep_range[0]) & (wl_vals <= cfg.keep_range[1])
    for lo, hi in cfg.exclude_ranges:
        keep &= ~((wl_vals >= lo) & (wl_vals <= hi))
    wl_cols_kept = [c for c, k in zip(wl_cols, keep.tolist()) if k]
    wl_vals_kept = wl_vals[keep]
    return wl_cols_kept, wl_vals_kept


def _fit_standardizer(X: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _apply_standardizer(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - stats["mean"]) / stats["std"]


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
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=boundary)

    if mode == "coeffs":
        parts = []
        if include_approx:
            parts.append(coeffs[0])
        parts.extend(coeffs[1:])
        return np.concatenate(parts, axis=0).astype(np.float32)

    if mode == "energy":
        energies = np.array([np.sum(ci**2) for ci in coeffs[1:]], dtype=np.float64)
        if log_energy:
            energies = np.log1p(energies)
        return energies.astype(np.float32)

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
    c0 = pywt.wavedec(X[0], wavelet=wavelet, level=level, mode=boundary)

    if mode == "coeffs":
        parts: List[Tuple[str, int]] = []
        names: List[str] = []
        if include_approx:
            parts.append(("A", len(c0[0])))
        for li, ci in enumerate(c0[1:], start=1):
            parts.append((f"D{len(c0) - li}", len(ci)))
        for band_name, band_len in parts:
            names.extend([f"DWT_{band_name}_{k}" for k in range(band_len)])
        out_len = sum(n for _, n in parts)
    else:
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
# Scene split helper (no leakage)
# -----------------------------
def _scene_split_indices(
    scenes: np.ndarray,
    y: np.ndarray,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by scene (group split). Returns indices for train/val/test.

    Greedy deterministic assignment:
      - shuffle scenes
      - allocate scenes to hit target sample counts (and roughly balance if binary)
    """
    assert (
        abs((train_pct + val_pct + test_pct) - 1.0) < 1e-6
    ), "train/val/test must sum to 1"

    rng = np.random.default_rng(random_state)
    unique_scenes = np.unique(scenes)

    # Detect binary vs multiclass
    uniq_y = np.unique(y)
    is_binary = len(uniq_y) <= 2

    per_scene = []
    for sc in unique_scenes:
        idx = np.where(scenes == sc)[0]
        n = len(idx)
        if is_binary:
            n_pos = int((y[idx] == 1).sum())
        else:
            # multiclass: no single "positive"; use 0 as a dummy score
            n_pos = 0
        per_scene.append((sc, idx, n, n_pos))

    rng.shuffle(per_scene)

    N = len(y)
    target_train = int(round(train_pct * N))
    target_val = int(round(val_pct * N))
    target_test = N - target_train - target_val

    global_ratio = float((y == 1).mean()) if is_binary and N > 0 else 0.0

    buckets = {
        "train": {"idx": [], "n": 0, "pos": 0},
        "val": {"idx": [], "n": 0, "pos": 0},
        "test": {"idx": [], "n": 0, "pos": 0},
    }
    targets = {"train": target_train, "val": target_val, "test": target_test}

    def score_bucket(bucket_name: str, add_n: int, add_pos: int) -> float:
        b = buckets[bucket_name]
        new_n = b["n"] + add_n
        if new_n == 0:
            return 0.0

        overshoot = max(0, new_n - targets[bucket_name]) / max(1, targets[bucket_name])
        empty_penalty = 0.2 if b["n"] == 0 else 0.0

        if is_binary:
            new_ratio = (b["pos"] + add_pos) / new_n
            ratio_dev = abs(new_ratio - global_ratio)
            return 2.0 * overshoot + 1.0 * ratio_dev + empty_penalty

        # multiclass: prioritize hitting sizes (no ratio term)
        return 2.0 * overshoot + empty_penalty

    for _, idx, n, n_pos in per_scene:
        best = min(
            {k: score_bucket(k, n, n_pos) for k in buckets.keys()}.items(),
            key=lambda kv: kv[1],
        )[0]
        buckets[best]["idx"].append(idx)
        buckets[best]["n"] += n
        buckets[best]["pos"] += n_pos

    tr_idx = (
        np.concatenate(buckets["train"]["idx"])
        if buckets["train"]["idx"]
        else np.array([], dtype=int)
    )
    va_idx = (
        np.concatenate(buckets["val"]["idx"])
        if buckets["val"]["idx"]
        else np.array([], dtype=int)
    )
    te_idx = (
        np.concatenate(buckets["test"]["idx"])
        if buckets["test"]["idx"]
        else np.array([], dtype=int)
    )

    all_idx = np.concatenate([tr_idx, va_idx, te_idx])
    if len(np.unique(all_idx)) != len(all_idx):
        raise RuntimeError(
            "Scene split produced overlapping indices (should not happen)."
        )
    if len(all_idx) != N:
        missing = set(range(N)) - set(all_idx.tolist())
        raise RuntimeError(f"Scene split missed {len(missing)} samples.")

    return tr_idx, va_idx, te_idx


# -----------------------------
# Dataset
# -----------------------------
class SpectraDataset(Dataset):
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
class HyperBloodCSVDataModule(pl.LightningDataModule):
    """
    DataModule for hsi_blood.csv (blob-level mean spectra).

    label_mode:
      - "binary": blood vs non-blood (previous behavior)
      - "multiclass": uses class_id (after filtering background), optionally remapped to 0..C-1
    """

    def __init__(
        self,
        csv_path: str | Path,
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
        scene_filters: Optional[Sequence[str]] = None,
        min_pixels: Optional[int] = None,
        split_mode: str = "random",  # "random" or "scene"
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.cfg = preprocess_cfg
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
        self.scene_filters = list(scene_filters) if scene_filters is not None else None
        self.min_pixels = min_pixels
        self.split_mode = split_mode.lower().strip()

        # introspection
        self.kept_wavelengths_nm: np.ndarray = np.array([], dtype=np.float32)
        self.feature_names: List[str] = []
        self.num_classes: int = 0
        self.standardizer_stats: Optional[Dict[str, np.ndarray]] = None
        self.label_mapping: Dict[int, str] = {}

        self.df_original: Optional[pd.DataFrame] = None

        self.train_ds: Optional[SpectraDataset] = None
        self.val_ds: Optional[SpectraDataset] = None
        self.test_ds: Optional[SpectraDataset] = None
        self._train_sampler = None

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def _build_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build y according to cfg.label_mode.
        - binary: blood vs non-blood
        - multiclass: class_id (optionally remapped to 0..C-1)
        Also sets self.label_mapping and self.num_classes.
        """
        cls_ids = df[self.cfg.class_id_col].astype(int).to_numpy()

        if self.cfg.label_mode == "binary":
            blood_ids = set(self.cfg.blood_ids)
            if self.cfg.include_uncertain_blood:
                blood_ids.add(8)
            y = np.where(np.isin(cls_ids, list(blood_ids)), 1, 0).astype(np.int64)
            self.label_mapping = {0: "non-blood", 1: "blood"}
            self.num_classes = 2
            return y

        # multiclass
        y_raw = cls_ids.astype(np.int64)

        if self.cfg.remap_classes_to_contiguous:
            uniq = sorted(np.unique(y_raw).tolist())
            id2new = {old: i for i, old in enumerate(uniq)}
            y = np.array([id2new[v] for v in y_raw], dtype=np.int64)

            if self.cfg.class_name_col in df.columns:
                name_by_old = (
                    df.groupby(self.cfg.class_id_col)[self.cfg.class_name_col]
                    .agg(lambda s: s.mode().iloc[0])
                    .to_dict()
                )
            else:
                name_by_old = {old: f"class_{old}" for old in uniq}

            self.label_mapping = {
                id2new[old]: str(name_by_old.get(old, f"class_{old}")) for old in uniq
            }
            self.num_classes = len(uniq)
            return y

        # no remap
        if self.cfg.class_name_col in df.columns:
            name_by_old = (
                df.groupby(self.cfg.class_id_col)[self.cfg.class_name_col]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )
            self.label_mapping = {int(k): str(v) for k, v in name_by_old.items()}
        else:
            self.label_mapping = {
                int(k): f"class_{int(k)}" for k in np.unique(y_raw).tolist()
            }
        self.num_classes = int(np.unique(y_raw).size)
        return y_raw

    def setup(self, stage: Optional[str] = None):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path).copy()
        self.df_original = df.copy(deep=True)

        # filters
        if self.scene_filters is not None and self.cfg.scene_col in df.columns:
            df = df[df[self.cfg.scene_col].isin(self.scene_filters)]
        if self.min_pixels is not None and self.cfg.n_pixels_col in df.columns:
            df = df[
                pd.to_numeric(df[self.cfg.n_pixels_col], errors="coerce")
                >= self.min_pixels
            ]
        if self.cfg.exclude_background:
            df = df[df[self.cfg.class_id_col] != 0]

        if df.empty:
            raise ValueError(
                "No rows left after filtering. Check scene_filters/min_pixels/exclude_background."
            )

        # wavelengths
        wl_cols_all, wl_vals_all = _get_wavelength_columns(df)
        if not wl_cols_all:
            raise ValueError("No wavelength columns found (expected wl_###.#).")

        wl_cols, wl_vals = _filter_wavelengths(wl_cols_all, wl_vals_all, self.cfg)
        if not wl_cols:
            raise ValueError(
                "No wavelength columns left after keep/exclude wavelength filtering."
            )
        self.kept_wavelengths_nm = wl_vals

        X = df[wl_cols].astype(np.float32).to_numpy()

        # labels (binary or multiclass)
        y = self._build_labels(df)

        # feature transform
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
            self.feature_names = [f"{w:.1f}" for w in self.kept_wavelengths_nm.tolist()]

        # split
        if self.split_mode == "random":
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X,
                y,
                test_size=(1.0 - self.train_pct),
                stratify=y,
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

        elif self.split_mode == "scene":
            if self.cfg.scene_col not in df.columns:
                raise KeyError(
                    f"split_mode='scene' requires column '{self.cfg.scene_col}' in the CSV."
                )
            scenes = df[self.cfg.scene_col].astype(str).to_numpy()

            tr_idx, va_idx, te_idx = _scene_split_indices(
                scenes=scenes,
                y=y,
                train_pct=self.train_pct,
                val_pct=self.val_pct,
                test_pct=self.test_pct,
                random_state=self.random_state,
            )

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            X_te, y_te = X[te_idx], y[te_idx]

        else:
            raise ValueError("split_mode must be 'random' or 'scene'")

        # standardize (fit on train only)
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

        # weighted sampler (train only)
        if self.use_weighted_sampler:
            class_counts = np.bincount(y_tr, minlength=self.num_classes)
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

    def get_label_mapping(self) -> Dict[int, str]:
        return self.label_mapping

    def get_standardizer(self) -> Optional[Dict[str, np.ndarray]]:
        return self.standardizer_stats


# -----------------------------
# Builder (yaml/json-like cfg)
# -----------------------------
def build_datamodule(cfg: Dict[str, Any]) -> HyperBloodCSVDataModule:
    data_cfg = cfg["data"]
    pp_cfg = data_cfg.get("preprocess", {}) or {}

    preprocess_cfg = SpectralPreprocessConfig(
        keep_range=tuple(pp_cfg.get("keep_range", (377.0, 1046.0))),
        exclude_ranges=[tuple(t) for t in (pp_cfg.get("exclude_ranges") or [])],
        class_id_col=pp_cfg.get("class_id_col", "class_id"),
        class_name_col=pp_cfg.get("class_name_col", "class_name"),
        scene_col=pp_cfg.get("scene_col", "scene"),
        blob_id_col=pp_cfg.get("blob_id_col", "blob_id"),
        n_pixels_col=pp_cfg.get("n_pixels_col", "n_pixels"),
        label_mode=pp_cfg.get("label_mode", "binary"),
        remap_classes_to_contiguous=bool(
            pp_cfg.get("remap_classes_to_contiguous", True)
        ),
        blood_ids=tuple(pp_cfg.get("blood_ids", [1])),
        include_uncertain_blood=bool(pp_cfg.get("include_uncertain_blood", False)),
        exclude_background=bool(pp_cfg.get("exclude_background", True)),
        dwt_enabled=pp_cfg.get("dwt", {}).get("enabled", False),
        dwt_wavelet=pp_cfg.get("dwt", {}).get("wavelet", "db4"),
        dwt_level=pp_cfg.get("dwt", {}).get("level", None),
        dwt_mode=pp_cfg.get("dwt", {}).get("mode", "coeffs"),
        dwt_boundary=pp_cfg.get("dwt", {}).get("boundary", "periodization"),
        dwt_include_approx=pp_cfg.get("dwt", {}).get("include_approx", True),
        dwt_log_energy=pp_cfg.get("dwt", {}).get("log_energy", False),
    )

    dm = HyperBloodCSVDataModule(
        csv_path=data_cfg["csv_path"],
        preprocess_cfg=preprocess_cfg,
        batch_size=data_cfg.get("batch_size", 128),
        num_workers=data_cfg.get("num_workers", 4),
        train_pct=data_cfg.get("train_pct", 0.70),
        val_pct=data_cfg.get("val_pct", 0.15),
        test_pct=data_cfg.get("test_pct", 0.15),
        random_state=data_cfg.get("random_state", data_cfg.get("seed", 42)),
        standardize=data_cfg.get("standardize", True),
        use_weighted_sampler=data_cfg.get("use_weighted_sampler", False),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=data_cfg.get("drop_last", False),
        persistent_workers=True,
        scene_filters=data_cfg.get("scene_filters", None),
        min_pixels=data_cfg.get("min_pixels", None),
        split_mode=data_cfg.get("split_mode", "random"),
    )
    dm.setup()
    return dm

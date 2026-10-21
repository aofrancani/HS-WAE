# dataloader_hyperblood.py
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
import re


# -----------------------------
# Preprocessing configuration
# -----------------------------
@dataclass
class SpectralPreprocessConfig:
    """
    HyperBlood blob CSV:
      - spectra columns: wl_###.# (float nm)
      - labels: class_id (0..8)

    label_mode:
      - "binary": blood vs non-blood
      - "multiclass": keep original classes (optionally drop background and/or uncertain),
                      optionally remap to contiguous 0..C-1
    """

    # Wavelength filters (nm)
    keep_range: Tuple[float, float] = (377.0, 1046.0)
    exclude_ranges: List[Tuple[float, float]] = None

    # Columns in hsi_blood.csv
    class_id_col: str = "class_id"
    class_name_col: str = "class_name"  # optional but recommended
    scene_col: str = "scene"
    blob_id_col: str = "blob_id"
    n_pixels_col: str = "n_pixels"

    # Label mode
    label_mode: str = "binary"  # "binary" | "multiclass"

    # Binary label definition
    blood_ids: Tuple[int, ...] = (1,)
    include_uncertain_blood: bool = (
        False  # if True, class 8 is also blood (binary mode)
    )
    exclude_background: bool = True  # if True, drop class 0 blobs

    # Multiclass options
    remove_uncertain_blood: bool = True  # if True, drop class 8 (multiclass mode)
    remap_classes_to_contiguous: bool = True  # map kept ids -> 0..C-1 (recommended)

    # DWT options (token mode supported when enabled)
    dwt_enabled: bool = False
    dwt_wavelet: str = "db4"
    dwt_level: Optional[int] = None
    dwt_mode: str = "coeffs"  # must be "coeffs" for token mode
    dwt_boundary: str = "periodization"
    dwt_include_approx: bool = True
    dwt_log_energy: bool = False  # kept for compatibility

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
    """
    Return (wl_cols_sorted, wl_vals_sorted), where wl_cols look like wl_377.0 etc.
    """
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
) -> Tuple[List[str], List[float]]:
    keep = (wl_vals >= cfg.keep_range[0]) & (wl_vals <= cfg.keep_range[1])
    for lo, hi in cfg.exclude_ranges:
        keep &= ~((wl_vals >= lo) & (wl_vals <= hi))
    wl_cols_kept = [c for c, k in zip(wl_cols, keep.tolist()) if k]
    wl_vals_kept = wl_vals[keep]
    return wl_cols_kept, wl_vals_kept.tolist()


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
    Fit mean/std per DWT level (token). Returns list of stats dicts, one per level.
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
    Returns list of coefficient vectors in fixed PyWavelets order:
      [A_L, D_L, D_{L-1}, ..., D_1]  (or without A_L if include_approx=False)
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
      names: ["A_L","D_L","D_{L-1}",...,"D_1"] (or without A_L)
      lengths: corresponding lengths
    """
    c0 = pywt.wavedec(x0, wavelet=wavelet, level=level, mode=boundary)
    L = len(c0) - 1
    if include_approx:
        names = ["A_L"] + [f"D_{k}" for k in range(L, 0, -1)]
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
    batch: list of (x_raw, levels_list, y)
      x_raw: Tensor(bands,)
      levels_list: [Tensor(n_0), ..., Tensor(n_{K-1})]
    Returns:
      x_raw: Tensor(B, bands)
      levels_batch: List[Tensor], each (B, n_l)
      y: Tensor(B,)
    """
    x_raws, levels_lists, ys = zip(*batch)
    x_raw = torch.stack(x_raws, dim=0)
    y = torch.stack(ys, dim=0)

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
        x_raw = self.X_raw[idx]
        levels = [torch.from_numpy(v.astype(np.float32)) for v in self.X_levels[idx]]
        return x_raw, levels, self.y[idx]


# -----------------------------
# Scene split indices
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
    Greedy deterministic group split by scene.
    Returns (idx_tr, idx_va, idx_te).
    """
    assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-6

    rng = np.random.default_rng(random_state)
    unique_scenes = np.unique(scenes)

    per_scene = []
    for sc in unique_scenes:
        idx = np.where(scenes == sc)[0]
        n = len(idx)
        # compute a "majority" class count proxy (to balance somewhat):
        # for multiclass this is a coarse signal; good enough for greedy.
        # we mainly try to keep proportions overall.
        n_pos = int((y[idx] == 1).sum()) if y.max() <= 1 else int((y[idx] == y[idx].mode() if hasattr(y[idx], "mode") else y[idx][0]).sum())  # type: ignore
        per_scene.append((sc, idx, n, n_pos))

    rng.shuffle(per_scene)

    N = len(y)
    target_train = int(round(train_pct * N))
    target_val = int(round(val_pct * N))
    target_test = N - target_train - target_val

    # for scoring, use binary ratio if binary; else use entropy-ish proxy via unique count
    is_binary = int(np.unique(y).size) <= 2
    global_ratio = float((y == 1).mean()) if is_binary else 0.0

    buckets = {
        "train": {"idx": [], "n": 0, "pos": 0},
        "val": {"idx": [], "n": 0, "pos": 0},
        "test": {"idx": [], "n": 0, "pos": 0},
    }
    targets = {"train": target_train, "val": target_val, "test": target_test}

    def score(bucket_name: str, add_n: int, add_pos: int) -> float:
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

        # multiclass: just try to hit target sizes (no strong ratio penalty)
        return 2.0 * overshoot + empty_penalty

    for _, idx, n, n_pos in per_scene:
        best = min(
            {k: score(k, n, n_pos) for k in buckets.keys()}.items(),
            key=lambda kv: kv[1],
        )[0]
        buckets[best]["idx"].append(idx)
        buckets[best]["n"] += n
        buckets[best]["pos"] += n_pos

    idx_tr = (
        np.concatenate(buckets["train"]["idx"])
        if buckets["train"]["idx"]
        else np.array([], dtype=int)
    )
    idx_va = (
        np.concatenate(buckets["val"]["idx"])
        if buckets["val"]["idx"]
        else np.array([], dtype=int)
    )
    idx_te = (
        np.concatenate(buckets["test"]["idx"])
        if buckets["test"]["idx"]
        else np.array([], dtype=int)
    )

    all_idx = np.concatenate([idx_tr, idx_va, idx_te])
    if len(np.unique(all_idx)) != len(all_idx):
        raise RuntimeError("Scene split produced overlapping indices.")
    if len(all_idx) != N:
        raise RuntimeError(f"Scene split missed {N - len(all_idx)} samples.")
    return idx_tr, idx_va, idx_te


# -----------------------------
# DataModule (HyperBlood)
# -----------------------------
class HyperBloodCSVDataModule(pl.LightningDataModule):
    """
    DataModule for HyperBlood blob CSV (hsi_blood.csv).

    label_mode:
      - "binary": blood vs non-blood
      - "multiclass": classes from CSV (after filtering), optionally remapped to 0..C-1

    If DWT enabled:
      returns (x_raw, levels_tokens, y) with collate_fn=dwt_levels_collate_fn
      and standardizes per-level tokens (fit on train only).

    split_mode:
      - "random": stratified by y across blobs
      - "scene":  split by whole scenes (no leakage)
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
        split_mode: str = "random",
        fixed_split_indices_path: Optional[str | Path] = None,
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
        self.fixed_split_indices_path = Path(fixed_split_indices_path) if fixed_split_indices_path else None

        # Introspection (mirrors your base)
        self.kept_wavelengths_nm: List[float] = []
        self.band_count: int = 0
        self.num_classes: int = 0

        self.feature_names: List[str] = []  # wavelengths OR DWT level names
        self.dwt_level_lengths: Optional[List[int]] = None

        self.standardizer_stats: Optional[
            Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
        ] = None

        self.df_original: Optional[pd.DataFrame] = None
        self.label_mapping: Dict[int, str] = {}

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._train_sampler = None

    @property
    def input_dim(self) -> int:
        if self.cfg.dwt_enabled:
            if not self.dwt_level_lengths:
                return 0
            return int(sum(self.dwt_level_lengths))
        return len(self.feature_names)

    def _build_labels_and_mapping(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Build labels according to cfg.label_mode.
        Returns y and label_mapping (label_int -> name).
        """
        class_ids = df[self.cfg.class_id_col].astype("int64").to_numpy()

        if self.cfg.label_mode == "binary":
            blood_ids = set(self.cfg.blood_ids)
            if self.cfg.include_uncertain_blood:
                blood_ids.add(8)
            y = np.where(np.isin(class_ids, list(blood_ids)), 1, 0).astype(np.int64)
            mapping = {0: "non-blood", 1: "blood"}
            return y, mapping

        # multiclass
        y_raw = class_ids.astype(np.int64)

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

            mapping = {
                id2new[old]: str(name_by_old.get(old, f"class_{old}")) for old in uniq
            }
            return y, mapping

        # no remap: keep original ids (not recommended unless your head expects those ids)
        if self.cfg.class_name_col in df.columns:
            name_by_old = (
                df.groupby(self.cfg.class_id_col)[self.cfg.class_name_col]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )
            mapping = {int(k): str(v) for k, v in name_by_old.items()}
        else:
            mapping = {int(k): f"class_{int(k)}" for k in np.unique(y_raw).tolist()}
        return y_raw, mapping

    def setup(self, stage: Optional[str] = None):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df_all = pd.read_csv(self.csv_path).copy()
        self.df_original = df_all.copy(deep=True)

        # Optional filters
        if self.scene_filters is not None and self.cfg.scene_col in df_all.columns:
            df_all = df_all[df_all[self.cfg.scene_col].isin(self.scene_filters)]
        if self.min_pixels is not None and self.cfg.n_pixels_col in df_all.columns:
            df_all = df_all[
                pd.to_numeric(df_all[self.cfg.n_pixels_col], errors="coerce")
                >= self.min_pixels
            ]
        if self.cfg.exclude_background:
            df_all = df_all[df_all[self.cfg.class_id_col] != 0]

        # Multiclass: optionally remove uncertain blood class 8
        if self.cfg.label_mode == "multiclass" and self.cfg.remove_uncertain_blood:
            df_all = df_all[df_all[self.cfg.class_id_col] != 8]

        if df_all.empty:
            raise ValueError(
                "No rows left after filtering. Check filters / exclude_background / remove_uncertain_blood."
            )

        # Spectral columns
        wl_cols_all, wl_vals_all = _get_wavelength_columns(df_all)
        if not wl_cols_all:
            raise ValueError("No wavelength columns found (expected wl_###.#).")

        kept_cols, kept_wls = _filter_wavelengths(wl_cols_all, wl_vals_all, self.cfg)
        self.kept_wavelengths_nm = kept_wls
        self.band_count = len(kept_cols)

        # Base features
        X = df_all[kept_cols].astype("float32").to_numpy()  # (N, bands)

        # Labels (binary or multiclass)
        y, mapping = self._build_labels_and_mapping(df_all)
        self.label_mapping = mapping
        self.num_classes = int(np.unique(y).size)

        # ------------------------------------------------------------
        # DWT path: token mode
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

            idx = np.arange(len(y))

            if self.fixed_split_indices_path is not None:
                import json
                payload = json.loads(self.fixed_split_indices_path.read_text())
                idx_tr = np.asarray(payload["train_indices"], dtype=np.int64)
                idx_va = np.asarray(payload["validation_indices"], dtype=np.int64)
                idx_te = np.asarray(payload["test_indices"], dtype=np.int64)
                combined = np.concatenate([idx_tr, idx_va, idx_te])
                if len(combined) != len(y) or len(np.unique(combined)) != len(y) or combined.min() < 0 or combined.max() >= len(y):
                    raise ValueError("Fixed split indices must be disjoint and cover every filtered row exactly once")
                y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
            elif self.split_mode == "random":
                idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
                    idx,
                    y,
                    test_size=(1.0 - self.train_pct),
                    stratify=y,
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

            elif self.split_mode == "scene":
                if self.cfg.scene_col not in df_all.columns:
                    raise KeyError(
                        f"split_mode='scene' requires column '{self.cfg.scene_col}'."
                    )
                scenes = df_all[self.cfg.scene_col].astype(str).to_numpy()
                idx_tr, idx_va, idx_te = _scene_split_indices(
                    scenes=scenes,
                    y=y,
                    train_pct=self.train_pct,
                    val_pct=self.val_pct,
                    test_pct=self.test_pct,
                    random_state=self.random_state,
                )
                y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
            else:
                raise ValueError("split_mode must be 'random' or 'scene'")

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

            if self.use_weighted_sampler:
                class_counts = np.bincount(y_tr, minlength=self.num_classes)
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
        # No DWT path
        # ------------------------------------------------------------
        else:
            self.feature_names = [f"{w:.1f}" for w in self.kept_wavelengths_nm]
            self.dwt_level_lengths = None

            if self.fixed_split_indices_path is not None:
                import json
                payload = json.loads(self.fixed_split_indices_path.read_text())
                idx_tr = np.asarray(payload["train_indices"], dtype=np.int64)
                idx_va = np.asarray(payload["validation_indices"], dtype=np.int64)
                idx_te = np.asarray(payload["test_indices"], dtype=np.int64)
                combined = np.concatenate([idx_tr, idx_va, idx_te])
                if len(combined) != len(y) or len(np.unique(combined)) != len(y) or combined.min() < 0 or combined.max() >= len(y):
                    raise ValueError("Fixed split indices must be disjoint and cover every filtered row exactly once")
                X_tr, y_tr = X[idx_tr], y[idx_tr]
                X_va, y_va = X[idx_va], y[idx_va]
                X_te, y_te = X[idx_te], y[idx_te]
            elif self.split_mode == "random":
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
                if self.cfg.scene_col not in df_all.columns:
                    raise KeyError(
                        f"split_mode='scene' requires column '{self.cfg.scene_col}'."
                    )
                scenes = df_all[self.cfg.scene_col].astype(str).to_numpy()
                idx_tr, idx_va, idx_te = _scene_split_indices(
                    scenes=scenes,
                    y=y,
                    train_pct=self.train_pct,
                    val_pct=self.val_pct,
                    test_pct=self.test_pct,
                    random_state=self.random_state,
                )
                X_tr, y_tr = X[idx_tr], y[idx_tr]
                X_va, y_va = X[idx_va], y[idx_va]
                X_te, y_te = X[idx_te], y[idx_te]

            else:
                raise ValueError("split_mode must be 'random' or 'scene'")

            if self.standardize:
                self.standardizer_stats = _fit_standardizer(X_tr)
                X_tr = _apply_standardizer(X_tr, self.standardizer_stats)  # type: ignore[arg-type]
                X_va = _apply_standardizer(X_va, self.standardizer_stats)  # type: ignore[arg-type]
                X_te = _apply_standardizer(X_te, self.standardizer_stats)  # type: ignore[arg-type]
            else:
                self.standardizer_stats = None

            self.train_ds = SpectraDataset(X_tr, y_tr)
            self.val_ds = SpectraDataset(X_va, y_va)
            self.test_ds = SpectraDataset(X_te, y_te)

            if self.use_weighted_sampler:
                class_counts = np.bincount(y_tr, minlength=self.num_classes)
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
        Also exports label mapping and label_mode.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "label_mode": self.cfg.label_mode,
            "num_classes": self.num_classes,
            "label_mapping": self.label_mapping,
        }

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
# Builder (yaml/json-like cfg)
# -----------------------------
def build_datamodule(cfg: Dict[str, Any]) -> HyperBloodCSVDataModule:
    data_cfg = cfg["data"]
    pp_cfg = data_cfg.get("preprocess", {}) or {}
    dwt_cfg = pp_cfg.get("dwt", {}) or {}

    keep_range = pp_cfg.get("keep_range") or (377.0, 1046.0)
    exclude_ranges = pp_cfg.get("exclude_ranges") or []

    preprocess_cfg = SpectralPreprocessConfig(
        keep_range=tuple(keep_range),
        exclude_ranges=[tuple(t) for t in exclude_ranges],
        class_id_col=pp_cfg.get("class_id_col", "class_id"),
        class_name_col=pp_cfg.get("class_name_col", "class_name"),
        scene_col=pp_cfg.get("scene_col", "scene"),
        blob_id_col=pp_cfg.get("blob_id_col", "blob_id"),
        n_pixels_col=pp_cfg.get("n_pixels_col", "n_pixels"),
        label_mode=pp_cfg.get("label_mode", "binary"),
        blood_ids=tuple(pp_cfg.get("blood_ids") or [1]),
        include_uncertain_blood=bool(pp_cfg.get("include_uncertain_blood", False)),
        exclude_background=bool(pp_cfg.get("exclude_background", True)),
        remove_uncertain_blood=bool(pp_cfg.get("remove_uncertain_blood", True)),
        remap_classes_to_contiguous=bool(
            pp_cfg.get("remap_classes_to_contiguous", True)
        ),
        dwt_enabled=bool(dwt_cfg.get("enabled", False)),
        dwt_wavelet=dwt_cfg.get("wavelet", "db4"),
        dwt_level=dwt_cfg.get("level", None),
        dwt_mode=dwt_cfg.get("mode", "coeffs"),
        dwt_boundary=dwt_cfg.get("boundary", "periodization"),
        dwt_include_approx=bool(dwt_cfg.get("include_approx", True)),
        dwt_log_energy=bool(dwt_cfg.get("log_energy", False)),
    )

    dm = HyperBloodCSVDataModule(
        csv_path=Path(data_cfg["csv_path"]),
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
        fixed_split_indices_path=data_cfg.get("fixed_split_indices_path", None),
    )
    dm.setup()
    return dm


# -----------------------------
# Debug entry
# -----------------------------
if __name__ == "__main__":
    cfg = {
        "data": {
            "csv_path": "hsi_blood.csv",
            "batch_size": 64,
            "num_workers": 0,
            "train_pct": 0.70,
            "val_pct": 0.15,
            "test_pct": 0.15,
            "standardize": True,
            "use_weighted_sampler": True,
            "pin_memory": False,
            "drop_last": False,
            "split_mode": "scene",
            "scene_filters": None,
            "min_pixels": 30,
            "preprocess": {
                "keep_range": [377.0, 1046.0],
                "exclude_ranges": [],
                # switch label_mode here
                "label_mode": "multiclass",  # "binary" or "multiclass"
                "exclude_background": True,
                "remove_uncertain_blood": True,
                "remap_classes_to_contiguous": True,
                # binary options (used only if label_mode=="binary")
                "blood_ids": [1],
                "include_uncertain_blood": False,
                "dwt": {
                    "enabled": True,
                    "wavelet": "db4",
                    "level": None,
                    "mode": "coeffs",
                    "boundary": "periodization",
                    "include_approx": True,
                    "log_energy": False,
                },
            },
        }
    }

    print("Building HyperBlood DataModule...")
    dm = build_datamodule(cfg)

    print("\n--- DataModule Info ---")
    print(f"label_mode: {dm.cfg.label_mode}")
    print(f"Final input dim: {dm.input_dim} (DWT enabled: {dm.cfg.dwt_enabled})")
    print(f"Num classes: {dm.num_classes}")
    print(f"Label mapping: {dm.get_label_mapping()}")

    if dm.cfg.dwt_enabled:
        print(f"Token levels: {dm.feature_names}")
        print(f"Token lengths: {dm.dwt_level_lengths}")
    else:
        print(
            f"Wavelengths: {dm.kept_wavelengths_nm[:5]} ... {dm.kept_wavelengths_nm[-5:]}"
        )

    def class_stats(name: str, ds: Dataset):
        y = ds.y.numpy()  # both datasets store y tensor as .y
        labels, counts = np.unique(y, return_counts=True)
        mapping = dm.get_label_mapping()
        print(f"{name} class distribution:")
        for lbl, cnt in zip(labels, counts):
            print(
                f"  Class {int(lbl)} ({mapping.get(int(lbl), str(int(lbl)))}): {int(cnt)}"
            )
        print(f"  Total: {len(ds)}\n")

    print("\n--- Train/Val/Test sizes ---")
    class_stats("Train", dm.train_ds)
    class_stats("Val", dm.val_ds)
    class_stats("Test", dm.test_ds)

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("\n--- Example batch ---")
    if dm.cfg.dwt_enabled:
        x_raw, levels, y = batch
        print("x_raw:", x_raw.shape)
        print("levels:", [t.shape for t in levels])
        print("y:", y.shape)
    else:
        x, y = batch
        print("x:", x.shape)
        print("y:", y.shape)

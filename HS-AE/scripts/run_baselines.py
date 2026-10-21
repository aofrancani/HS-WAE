#!/usr/bin/env python3
"""
Baseline classification experiments for HS-WAE.

Design goals
------------
* Reuse the repository's current crop DataModule so filtering, spectral presets,
  class labels, standardization, and the fixed train/val/test split match HS-WAE.
* Disable DWT for conventional/raw-spectrum baselines.
* Never use the test set for model/hyperparameter selection.
* Fit PCA / supervised band selection on the training split only.
* Repeat stochastic models with matched seeds (default 2..26).
* Save predictions and provenance for later paired tests.

Recommended first experiment:
    python -m scripts.run_baselines \
      --config configs/dwt_attn.yaml --band_region nir --days 2 \
      --feature_mode raw --models rf svm xgb mlp cnn1d --n_exp 1 --device cuda:1

Then use the supplied shell scripts for the full experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from datasets.build_dataloader import build_datamodule
from utils import load_config, override_band_region, override_date_filters

try:
    from xgboost import XGBClassifier
except Exception:  # optional dependency
    XGBClassifier = None


# -----------------------------------------------------------------------------
# Reproducibility / metrics
# -----------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(auroc),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------------------
# Data / feature transformations
# -----------------------------------------------------------------------------

def build_raw_dm(config_path: str, band_region: str, day: int, data_root: Optional[str]):
    cfg = load_config(Path(config_path))
    cfg = copy.deepcopy(cfg)
    if data_root:
        cfg.setdefault("data", {})["csv_path"] = data_root
    # Conventional baselines operate on the ordered reflectance vector.
    pp = cfg.setdefault("data", {}).setdefault("preprocess", {})
    pp.setdefault("dwt", {})["enabled"] = False
    # Keep exactly the paper's standardization and split fractions.
    override_band_region(cfg, band_region)
    override_date_filters(cfg, [day])
    dm = build_datamodule(cfg)
    return cfg, dm


def arrays_from_dm(dm):
    Xtr = dm.train_ds.X.detach().cpu().numpy().astype(np.float32)
    ytr = dm.train_ds.y.detach().cpu().numpy().astype(np.int64)
    Xva = dm.val_ds.X.detach().cpu().numpy().astype(np.float32)
    yva = dm.val_ds.y.detach().cpu().numpy().astype(np.int64)
    Xte = dm.test_ds.X.detach().cpu().numpy().astype(np.float32)
    yte = dm.test_ds.y.detach().cpu().numpy().astype(np.int64)
    return Xtr, ytr, Xva, yva, Xte, yte


@dataclass
class TransformInfo:
    mode: str
    input_dim: int
    output_dim: int
    pca_explained_variance: Optional[float] = None
    selected_indices: Optional[List[int]] = None
    selected_wavelengths_nm: Optional[List[float]] = None


def transform_features(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray,
    ytr: np.ndarray,
    mode: str,
    wavelengths: Sequence[float],
    seed: int,
    k_features: int = 64,
):
    mode = mode.lower()
    D = Xtr.shape[1]
    if mode == "raw":
        return Xtr, Xva, Xte, None, TransformInfo("raw", D, D)

    if mode == "pca95":
        # Fit ONLY on training data; validation/test are transformed afterward.
        tx = PCA(n_components=0.95, svd_solver="full")
        A = tx.fit_transform(Xtr)
        B = tx.transform(Xva)
        C = tx.transform(Xte)
        info = TransformInfo(
            mode="pca95", input_dim=D, output_dim=A.shape[1],
            pca_explained_variance=float(tx.explained_variance_ratio_.sum()),
        )
        return A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), tx, info

    if mode == "pca64":
        ncomp = max(1, min(k_features, D, Xtr.shape[0] - 1))
        tx = PCA(n_components=ncomp, svd_solver="full")
        A = tx.fit_transform(Xtr)
        B = tx.transform(Xva)
        C = tx.transform(Xte)
        info = TransformInfo(
            mode="pca64", input_dim=D, output_dim=A.shape[1],
            pca_explained_variance=float(tx.explained_variance_ratio_.sum()),
        )
        return A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), tx, info

    if mode == "selectk":
        k = max(1, min(k_features, D))
        tx = SelectKBest(score_func=f_classif, k=k)
        A = tx.fit_transform(Xtr, ytr)
        B = tx.transform(Xva)
        C = tx.transform(Xte)
        idx = np.flatnonzero(tx.get_support()).astype(int)
        # Sort by original wavelength position so the retained vector remains ordered.
        order = np.argsort(idx)
        idx = idx[order]
        A = Xtr[:, idx]
        B = Xva[:, idx]
        C = Xte[:, idx]
        wls = [float(wavelengths[i]) for i in idx]
        info = TransformInfo(
            mode="selectk", input_dim=D, output_dim=len(idx),
            selected_indices=idx.tolist(), selected_wavelengths_nm=wls,
        )
        return A.astype(np.float32), B.astype(np.float32), C.astype(np.float32), tx, info

    raise ValueError(f"Unknown feature mode: {mode}")


# -----------------------------------------------------------------------------
# Classical model tuning: validation only
# -----------------------------------------------------------------------------

def make_rf(params: Dict[str, Any], seed: int):
    return RandomForestClassifier(
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
        **params,
    )


def make_svm(params: Dict[str, Any], seed: int):
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=seed,
        **params,
    )


def make_xgb(params: Dict[str, Any], seed: int, ytr: np.ndarray):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed in this environment.")
    n0 = max(int((ytr == 0).sum()), 1)
    n1 = max(int((ytr == 1).sum()), 1)
    spw = n0 / n1
    return XGBClassifier(
        random_state=seed,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        **params,
    )


def score_estimator(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2:
            return p[:, 1]
        return p
    # Preserve the estimator's exact continuous decision score. AUROC only
    # depends on its ordering, so probability calibration is unnecessary.
    return np.asarray(model.decision_function(X))


def validation_objective(model, Xva, yva) -> Tuple[float, float]:
    pred = model.predict(Xva)
    score = score_estimator(model, Xva)
    try:
        auc = float(roc_auc_score(yva, score))
    except ValueError:
        auc = -np.inf
    f1 = float(f1_score(yva, pred, zero_division=0))
    return auc, f1


def param_grid(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "rf":
        return [
            {"n_estimators": 500, "max_depth": d, "min_samples_leaf": leaf, "max_features": "sqrt"}
            for d in (None, 12, 24) for leaf in (1, 2)
        ]
    if model_name == "svm":
        return [{"C": c, "gamma": "scale"} for c in (0.1, 1.0, 10.0, 100.0)]
    if model_name == "xgb":
        return [
            {"n_estimators": 300, "max_depth": d, "learning_rate": lr,
             "min_child_weight": 1, "reg_lambda": 1.0}
            for d in (3, 6) for lr in (0.05, 0.1)
        ]
    raise ValueError(model_name)


def tune_classical(model_name, Xtr, ytr, Xva, yva, seed: int):
    best = None
    records = []
    for params in param_grid(model_name):
        if model_name == "rf":
            model = make_rf(params, seed)
        elif model_name == "svm":
            model = make_svm(params, seed)
        elif model_name == "xgb":
            model = make_xgb(params, seed, ytr)
        else:
            raise ValueError(model_name)
        model.fit(Xtr, ytr)
        auc, f1 = validation_objective(model, Xva, yva)
        rec = {**params, "val_auroc": auc, "val_f1": f1}
        records.append(rec)
        key = (auc, f1)
        if best is None or key > best[0]:
            best = (key, params)
    assert best is not None
    return best[1], records


# -----------------------------------------------------------------------------
# Neural baselines
# -----------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """Architecture-matched classification-only baseline: no AE, DWT, or WTAM."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            # Match the HS-WAE encoder bottleneck and classifier head exactly:
            # encoder 128->64, then head 64->64->32->2.
            nn.Linear(128, 64),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)


class CNN1DBaseline(nn.Module):
    """Compact spectral 1D CNN operating on ordered wavelength samples."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


def make_weighted_loader(X, y, batch_size: int, seed: int, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    counts = np.bincount(y, minlength=2).astype(np.float64)
    cw = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    weights = torch.from_numpy(cw[y].astype(np.float64))
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True, generator=g)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)


def plain_loader(X, y, batch_size: int):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def train_torch_model(
    model_name: str,
    Xtr, ytr, Xva, yva, Xte, yte,
    seed: int,
    device: torch.device,
    batch_size: int = 64,
    max_epochs: int = 300,
    patience: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
):
    seed_all(seed)
    if model_name == "mlp":
        model = MLPBaseline(Xtr.shape[1])
    elif model_name == "cnn1d":
        model = CNN1DBaseline()
    else:
        raise ValueError(model_name)
    model.to(device)

    tr_loader = make_weighted_loader(Xtr, ytr, batch_size, seed)
    va_loader = plain_loader(Xva, yva, batch_size)
    te_loader = plain_loader(Xte, yte, batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_loss = math.inf
    best_state = None
    best_epoch = -1
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                losses.append(float(criterion(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else math.inf
        sched.step(val_loss)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()
    probs, preds, ys = [], [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            preds.append((p >= 0.5).astype(np.int64))
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    return model, y_true, y_pred, y_prob, best_epoch, best_loss


# -----------------------------------------------------------------------------
# Main condition runner
# -----------------------------------------------------------------------------

def model_dir_name(model: str) -> str:
    return {"cnn1d": "cnn1d", "mlp": "mlp", "rf": "rf", "svm": "svm", "xgb": "xgb"}[model]


def run_condition(args):
    cfg, dm = build_raw_dm(args.config, args.band_region, args.day, args.data_root)
    Xtr, ytr, Xva, yva, Xte, yte = arrays_from_dm(dm)
    base_seed = int(args.seed)

    Xtr2, Xva2, Xte2, tx, tx_info = transform_features(
        Xtr, Xva, Xte, ytr, args.feature_mode, dm.kept_wavelengths_nm,
        base_seed, args.k_features,
    )

    out_root = Path(args.outdir) / args.feature_mode / args.band_region / f"day_{args.day:02d}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "feature_transform.json").write_text(json.dumps(asdict(tx_info), indent=2), encoding="utf-8")
    provenance = {
        "config": str(Path(args.config)),
        "data_root": str(cfg["data"]["csv_path"]),
        "band_region": args.band_region,
        "day": args.day,
        "feature_mode": args.feature_mode,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "input_dim_raw": int(Xtr.shape[1]), "input_dim_model": int(Xtr2.shape[1]),
        "wavelength_min": float(min(dm.kept_wavelengths_nm)),
        "wavelength_max": float(max(dm.kept_wavelengths_nm)),
        "split_note": "Repository DataModule; fixed split. Feature transforms fit on train only.",
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    requested = list(args.models)
    if args.feature_mode.startswith("pca") and "cnn1d" in requested:
        raise SystemExit("cnn1d is intentionally disallowed for PCA components because component index has no wavelength locality.")

    for model_name in requested:
        mdir = out_root / model_dir_name(model_name)
        mdir.mkdir(parents=True, exist_ok=True)
        metrics_path = mdir / "metrics_runs.csv"
        rows: List[Dict[str, Any]] = []
        if metrics_path.exists() and not args.overwrite:
            try:
                rows = pd.read_csv(metrics_path).to_dict("records")
            except Exception:
                rows = []
        done = {(int(r["exp_idx"]), int(r["seed"])) for r in rows if "exp_idx" in r and "seed" in r}

        selected_params: Dict[str, Any] = {}
        if model_name in {"rf", "svm", "xgb"}:
            params_file = mdir / "selected_params.json"
            if params_file.exists() and not args.retune:
                selected_params = json.loads(params_file.read_text())
            else:
                selected_params, tuning_records = tune_classical(
                    model_name, Xtr2, ytr, Xva2, yva, base_seed
                )
                params_file.write_text(json.dumps(selected_params, indent=2), encoding="utf-8")
                pd.DataFrame(tuning_records).to_csv(mdir / "validation_tuning.csv", index=False)

        for exp_idx in range(args.n_exp):
            seed = base_seed + exp_idx
            if (exp_idx, seed) in done and not args.overwrite:
                continue
            print(f"[{args.feature_mode}/{args.band_region}/day{args.day:02d}] {model_name} exp={exp_idx} seed={seed}")
            seed_all(seed)

            if model_name == "rf":
                model = make_rf(selected_params, seed)
                model.fit(Xtr2, ytr)
                y_pred = model.predict(Xte2)
                y_prob = score_estimator(model, Xte2)
                score_kind = "positive_class_probability"
                best_epoch = np.nan
                best_monitor = np.nan
            elif model_name == "svm":
                model = make_svm(selected_params, seed)
                model.fit(Xtr2, ytr)
                y_pred = model.predict(Xte2)
                y_prob = score_estimator(model, Xte2)
                score_kind = "decision_function"
                best_epoch = np.nan
                best_monitor = np.nan
            elif model_name == "xgb":
                model = make_xgb(selected_params, seed, ytr)
                model.fit(Xtr2, ytr)
                y_pred = model.predict(Xte2).astype(np.int64)
                y_prob = score_estimator(model, Xte2)
                score_kind = "positive_class_probability"
                best_epoch = np.nan
                best_monitor = np.nan
            elif model_name in {"mlp", "cnn1d"}:
                device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
                model, yt, y_pred, y_prob, best_epoch, best_monitor = train_torch_model(
                    model_name, Xtr2, ytr, Xva2, yva, Xte2, yte,
                    seed=seed, device=device, batch_size=args.batch_size,
                    max_epochs=args.max_epochs, patience=args.patience,
                )
                if not np.array_equal(yt, yte):
                    raise RuntimeError("Unexpected test-label ordering mismatch.")
                score_kind = "positive_class_probability"
            else:
                raise ValueError(model_name)

            met = binary_metrics(yte, np.asarray(y_pred), np.asarray(y_prob))
            row = {
                "model": model_name, "feature_mode": args.feature_mode,
                "band_region": args.band_region, "day": args.day,
                "exp_idx": exp_idx, "seed": seed,
                "best_epoch": best_epoch, "best_val_loss": best_monitor,
                **met,
            }
            rows = [r for r in rows if not (int(r.get("exp_idx", -1)) == exp_idx and int(r.get("seed", -999)) == seed)]
            rows.append(row)
            rows = sorted(rows, key=lambda r: int(r["exp_idx"]))
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

            pred_dir = mdir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                pred_dir / f"run_{exp_idx:02d}_seed_{seed}.npz",
                y_true=yte.astype(np.int64), y_pred=np.asarray(y_pred).astype(np.int64),
                y_score=np.asarray(y_prob).astype(np.float32),
                score_kind=np.array(score_kind),
                exp_idx=np.array(exp_idx), seed=np.array(seed),
            )

        print(f"Saved: {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dwt_attn.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--band_region", required=True)
    p.add_argument("--days", nargs="+", type=int, required=True)
    p.add_argument("--feature_mode", choices=["raw", "pca95", "pca64", "selectk"], default="raw")
    p.add_argument("--k_features", type=int, default=64)
    p.add_argument("--models", nargs="+", choices=["rf", "svm", "xgb", "mlp", "cnn1d"], required=True)
    p.add_argument("--n_exp", type=int, default=25)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--outdir", default="results/baselines")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retune", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for d in args.days:
        args.day = int(d)
        run_condition(args)

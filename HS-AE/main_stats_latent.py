"""
Repeated training and full latent-space export for soybean experiments.

This is intentionally separate from main_stats.py so the original experiments
and result files are never overwritten.

Example:
python -m main_stats_latent --mode train --config configs/dwt_attn.yaml \
    --band_region nir --days 2 --devices 1 --n_exp 25 \
    --latent_out_root results/latent_25
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import math

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor

from datasets.build_dataloader import build_datamodule
from models.build_model import build_model_and_lit
from utils import (
    load_config,
    resolve_run_name,
    make_parser,
    get_device,
    override_band_region,
    override_date_filters,
    parse_int_list,
    _format_day_folder,
)
from trainer import build_trainer, build_logger_and_callbacks


class InMemoryBestModelCheckpoint(Callback):
    """Keep the best model state in memory without writing a checkpoint."""

    def __init__(self, monitor: str = "val/loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_state_dict = None
        self.best_epoch = None
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")

    def _is_better(self, current: float) -> bool:
        if self.best_score is None:
            return True
        return current < self.best_score if self.mode == "min" else current > self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor, None)
        if current is None:
            return
        current = current.detach().cpu().item() if isinstance(current, torch.Tensor) else float(current)
        if self._is_better(current):
            self.best_score = current
            self.best_epoch = int(trainer.current_epoch)
            self.best_state_dict = copy.deepcopy(pl_module.state_dict())

    def restore_best_weights(self, pl_module):
        if self.best_state_dict is None:
            print(f"[Warning] No best state stored for {self.monitor}; using current weights.")
            return
        pl_module.load_state_dict(self.best_state_dict)


def _bind_callbacks(cb_list):
    build_logger_and_callbacks.__wrapped_callbacks = cb_list


def _to_plain_dict(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().item()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _move_input(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [xi.to(device) for xi in x]
    raise TypeError(f"Unsupported model input type: {type(x)}")


def _safe_metric(fn, *args):
    try:
        return float(fn(*args))
    except Exception:
        return float("nan")


def _latent_metrics(z: np.ndarray, y: np.ndarray) -> dict:
    """Metrics in original 64-D latent space + z-scored sensitivity analysis."""
    y = np.asarray(y)
    z = np.asarray(z, dtype=np.float64)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels for latent analysis; got {classes.tolist()}")

    z0, z1 = z[y == classes[0]], z[y == classes[1]]
    mu0, mu1 = z0.mean(axis=0), z1.mean(axis=0)
    centroid_dist = float(np.linalg.norm(mu1 - mu0))

    rms0 = float(np.sqrt(np.mean(np.sum((z0 - mu0) ** 2, axis=1))))
    rms1 = float(np.sqrt(np.mean(np.sum((z1 - mu1) ** 2, axis=1))))
    within_mean = 0.5 * (rms0 + rms1)
    sep_ratio = centroid_dist / within_mean if within_mean > 0 else float("nan")

    pooled_ss = float(np.sum((z0 - mu0) ** 2) + np.sum((z1 - mu1) ** 2))
    pooled_var = pooled_ss / max((len(z0) + len(z1) - 2), 1)
    fisher = (centroid_dist ** 2) / pooled_var if pooled_var > 0 else float("nan")

    result = {
        "silhouette_64d": _safe_metric(silhouette_score, z, y),
        "davies_bouldin_64d": _safe_metric(davies_bouldin_score, z, y),
        "centroid_distance_64d": centroid_dist,
        "within_rms_class0_64d": rms0,
        "within_rms_class1_64d": rms1,
        "within_rms_control_64d": rms0,
        "within_rms_damaged_64d": rms1,
        "within_rms_mean_64d": within_mean,
        "separation_ratio_64d": sep_ratio,
        "fisher_ratio_64d": fisher,
    }

    # Sensitivity analysis only; never substitute this for the primary raw-latent metrics.
    z_std = StandardScaler().fit_transform(z)
    zs0, zs1 = z_std[y == classes[0]], z_std[y == classes[1]]
    m0, m1 = zs0.mean(axis=0), zs1.mean(axis=0)
    cd = float(np.linalg.norm(m1 - m0))
    r0 = float(np.sqrt(np.mean(np.sum((zs0 - m0) ** 2, axis=1))))
    r1 = float(np.sqrt(np.mean(np.sum((zs1 - m1) ** 2, axis=1))))
    wm = 0.5 * (r0 + r1)
    pss = float(np.sum((zs0 - m0) ** 2) + np.sum((zs1 - m1) ** 2))
    pvar = pss / max((len(zs0) + len(zs1) - 2), 1)
    result.update({
        "silhouette_zscore64d": _safe_metric(silhouette_score, z_std, y),
        "davies_bouldin_zscore64d": _safe_metric(davies_bouldin_score, z_std, y),
        "centroid_distance_zscore64d": cd,
        "within_rms_control_zscore64d": r0,
        "within_rms_damaged_zscore64d": r1,
        "within_rms_mean_zscore64d": wm,
        "separation_ratio_zscore64d": cd / wm if wm > 0 else float("nan"),
        "fisher_ratio_zscore64d": (cd ** 2) / pvar if pvar > 0 else float("nan"),
    })
    return result


@torch.no_grad()
def extract_test_outputs(lit, dm, device: torch.device):
    """Extract full test-set outputs from the restored best model."""
    lit.eval()
    model = lit.model
    model.eval()

    latent_all, logits_all, y_all, alpha_all = [], [], [], []
    have_alpha = True

    for batch in dm.test_dataloader():
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            _x_raw, x, y = batch
        else:
            raise ValueError(f"Unexpected batch format len={len(batch)}")

        x = _move_input(x, device)
        y = y.to(device).long()

        # compute_loss=False prevents any dependence on reconstruction targets here.
        try:
            out = model(x, compute_loss=False)
        except TypeError:
            out = model(x)

        if not isinstance(out, dict) or "latent" not in out or "logits" not in out:
            raise RuntimeError("Expected model output dict containing 'latent' and 'logits'.")

        latent_all.append(out["latent"].detach().cpu())
        logits_all.append(out["logits"].detach().cpu())
        y_all.append(y.detach().cpu())

        alpha = out.get("level_alpha", None)
        if alpha is None:
            have_alpha = False
        else:
            alpha_all.append(alpha.detach().cpu())

    latent = torch.cat(latent_all).numpy().astype(np.float32)
    logits = torch.cat(logits_all).numpy().astype(np.float32)
    y_true = torch.cat(y_all).numpy().astype(np.int64)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)
    y_pred = np.argmax(probs, axis=1).astype(np.int64)

    alpha = None
    if have_alpha and alpha_all:
        alpha = torch.cat(alpha_all).numpy().astype(np.float32)

    return {
        "latent": latent,
        "logits": logits,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "alpha": alpha,
    }


def save_run_npz(path: Path, payload: dict, dm, *, exp_idx: int, seed: int,
                 monitor: str, mode: str, best_epoch, best_score, config_path: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = payload["alpha"]
    alpha_arr = alpha if alpha is not None else np.empty((len(payload["y_true"]), 0), dtype=np.float32)
    np.savez_compressed(
        path,
        latent=payload["latent"],
        logits=payload["logits"],
        y_true=payload["y_true"],
        y_pred=payload["y_pred"],
        probs=payload["probs"],
        alpha=alpha_arr,
        test_order=np.arange(len(payload["y_true"]), dtype=np.int64),
        exp_idx=np.array(exp_idx, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        level_names=np.asarray(getattr(dm, "feature_names", []) or [], dtype=str),
        level_lengths=np.asarray(getattr(dm, "dwt_level_lengths", []) or [], dtype=np.int64),
        kept_wavelengths_nm=np.asarray(getattr(dm, "kept_wavelengths_nm", []) or [], dtype=np.int64),
        monitor=np.array(str(monitor)),
        monitor_mode=np.array(str(mode)),
        best_epoch=np.array(-1 if best_epoch is None else int(best_epoch), dtype=np.int64),
        best_monitor=np.array(np.nan if best_score is None else float(best_score), dtype=np.float64),
        config_path=np.array(str(config_path)),
    )


def save_alpha_long_csv(path: Path, payload: dict, dm, exp_idx: int, seed: int):
    alpha = payload["alpha"]
    if alpha is None or alpha.size == 0:
        return
    names = list(getattr(dm, "feature_names", []) or [])
    if len(names) != alpha.shape[1]:
        names = [f"level_{i}" for i in range(alpha.shape[1])]
    rows = []
    y = payload["y_true"]
    for level_idx, level_name in enumerate(names):
        for cls in np.unique(y):
            vals = alpha[y == cls, level_idx]
            rows.append({
                "exp_idx": exp_idx,
                "seed": seed,
                "class": int(cls),
                "level_idx": level_idx,
                "level_name": level_name,
                "n_samples": int(len(vals)),
                "alpha_mean": float(np.mean(vals)),
                "alpha_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "alpha_median": float(np.median(vals)),
                "alpha_min": float(np.min(vals)),
                "alpha_max": float(np.max(vals)),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = make_parser()
    parser.add_argument("--n_exp", type=int, default=25, help="Number of independent training runs.")
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--latent_out_root", type=str, default="results/latent_25")
    parser.add_argument("--no_save_npz", action="store_true", help="Save metrics only; omit latent arrays needed for downstream analysis.")
    args = parser.parse_args()

    if args.mode != "train":
        raise SystemExit("main_stats_latent.py is intended for --mode train only.")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    run_name = resolve_run_name(cfg, cfg_path)
    devices = get_device(args.devices, cfg.get("trainer", {}))

    base_seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(base_seed, workers=True)

    override_band_region(cfg, args.band_region)
    override_date_filters(cfg, args.days)

    days_list = parse_int_list(args.days) or cfg.get("data", {}).get("date_filters", None)
    if isinstance(days_list, int):
        days_list = [days_list]
    day_folder = _format_day_folder(days_list)

    out_root = Path(args.latent_out_root) / cfg_path.stem / str(args.band_region) / day_folder
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[LATENT 25-RUN EXPERIMENT]")
    print(f"Config             : {cfg_path}")
    print(f"Band               : {args.band_region}")
    print(f"Days               : {days_list}")
    print(f"Base seed          : {base_seed}")
    print(f"N experiments      : {args.n_exp}")
    print(f"Output             : {out_root}")
    print("=" * 80 + "\n")

    # Fixed split for all 25 repetitions, matching main_stats.py behavior.
    dm = build_datamodule(cfg)
    print(f"[Data] input_dim={dm.input_dim}, bands={dm.band_count}, classes={dm.num_classes}")
    print(f"[Data] train/val/test={len(dm.train_ds)}/{len(dm.val_ds)}/{len(dm.test_ds)}")
    print(f"[Data] DWT levels={getattr(dm, 'feature_names', None)}")

    rows = []
    for exp_idx in tqdm(range(args.n_exp), desc=f"{cfg_path.stem} {args.band_region} {day_folder}"):
        exp_seed = int(base_seed + exp_idx * args.seed_stride)
        seed_everything(exp_seed, workers=True)

        model, lit = build_model_and_lit(cfg, input_dim=dm.input_dim, num_classes=dm.num_classes, dm=dm)
        _, callbacks = build_logger_and_callbacks(cfg, run_name, out_root)
        callbacks = [cb for cb in callbacks if not isinstance(cb, (ModelCheckpoint, LearningRateMonitor))]
        # Presence of this disabled callback prevents Lightning from silently
        # installing its default on-disk checkpoint writer. Best weights remain
        # managed exclusively by InMemoryBestModelCheckpoint.
        callbacks.append(ModelCheckpoint(save_top_k=0, save_last=False))
        early_stop_cb = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)
        monitor = early_stop_cb.monitor if early_stop_cb is not None else "val/loss"
        mode = early_stop_cb.mode if early_stop_cb is not None else "min"
        mem_ckpt = InMemoryBestModelCheckpoint(monitor=monitor, mode=mode)
        callbacks.append(mem_ckpt)
        _bind_callbacks(callbacks)

        trainer = build_trainer(
            cfg,
            logger=False,
            devices=devices,
            precision_override=args.precision,
            max_epochs_override=args.max_epochs,
        )
        trainer.fit(lit, datamodule=dm)
        mem_ckpt.restore_best_weights(lit)

        # Existing test metrics, retained for cross-checking with the normal experiment pipeline.
        test_results = trainer.test(lit, datamodule=dm, ckpt_path=None)
        test_metrics = _to_plain_dict(test_results[0])

        # Extract from the SAME restored best model before it is discarded.
        device = lit.device
        payload = extract_test_outputs(lit, dm, device)
        latent_metrics = _latent_metrics(payload["latent"], payload["y_true"])
        extracted_acc = float(np.mean(payload["y_pred"] == payload["y_true"]))

        run_dir = out_root / f"run_{exp_idx:02d}_seed_{exp_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_save_npz:
            save_run_npz(
                run_dir / "test_latent_outputs.npz",
                payload,
                dm,
                exp_idx=exp_idx,
                seed=exp_seed,
                monitor=monitor,
                mode=mode,
                best_epoch=mem_ckpt.best_epoch,
                best_score=mem_ckpt.best_score,
                config_path=str(cfg_path),
            )
        save_alpha_long_csv(run_dir / "wtam_alpha_by_class.csv", payload, dm, exp_idx, exp_seed)

        row = {
            "method": cfg_path.stem,
            "band": args.band_region,
            "days": ",".join(map(str, days_list)) if days_list else "",
            "exp_idx": exp_idx,
            "seed": exp_seed,
            "monitor": monitor,
            "monitor_mode": mode,
            "best_monitor": mem_ckpt.best_score,
            "best_epoch": mem_ckpt.best_epoch,
            "n_test": int(len(payload["y_true"])),
            "latent_dim": int(payload["latent"].shape[1]),
            "extracted_accuracy": extracted_acc,
            "has_alpha": bool(payload["alpha"] is not None),
            **test_metrics,
            **latent_metrics,
        }
        rows.append(row)

        # Incremental save protects the campaign if a later run is interrupted.
        pd.DataFrame(rows).to_csv(out_root / "metrics_latent_runs.csv", index=False)
        print(
            f"[Run {exp_idx:02d}] seed={exp_seed} epoch={mem_ckpt.best_epoch} "
            f"acc={extracted_acc:.4f} silhouette={latent_metrics['silhouette_64d']:.4f} "
            f"DB={latent_metrics['davies_bouldin_64d']:.4f}"
        )

    # Reproducibility snapshot of the resolved runtime configuration.
    with open(out_root / "resolved_runtime_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    df = pd.DataFrame(rows)
    print("\n[FINAL SUMMARY]")
    cols = [c for c in ["test/acc", "test/f1", "test/auroc", "silhouette_64d", "davies_bouldin_64d", "separation_ratio_64d"] if c in df.columns]
    print(df[cols].agg(["mean", "std"]).T)
    print(f"\nSaved: {out_root / 'metrics_latent_runs.csv'}")


if __name__ == "__main__":
    main()

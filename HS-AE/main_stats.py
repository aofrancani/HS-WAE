"""
USAGE

-------- Train / Repeated train-test stats -------
python -m main --mode train --config configs/run_sae.yaml

-------- Test -------
# Test using a checkpoint name (looks up under logs/<run_name>/version_*/checkpoints)
python -m main --mode test --config configs/run_sae.yaml --ckpt best

# Test using a full checkpoint path
python -m main --mode test --config configs/run_sae.yaml --ckpt logs/run_sae/version_3/checkpoints/epoch=12-step=999.ckpt

# Optional overrides
python -m main --mode train --config configs/run_sae.yaml --devices 0 --precision 32-true --max_epochs 50
"""

from __future__ import annotations
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from datasets.build_dataloader import build_datamodule
from models.build_model import build_model_and_lit
from utils import (
    load_config,
    save_resolved_config,
    find_checkpoint,
    resolve_run_name,
    make_parser,
    get_device,
    save_test_metrics,
    override_band_region,
    override_date_filters,
    resolve_save_dir,
    parse_int_list,
    _format_day_folder,
)
from trainer import build_trainer, build_logger_and_callbacks


class InMemoryBestModelCheckpoint(Callback):
    """
    Keeps the best model weights in memory, without writing checkpoints to disk.
    """

    def __init__(self, monitor: str = "val/loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_state_dict = None
        self.best_epoch = None

        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got: {self.mode}")

    def _is_better(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        current = trainer.callback_metrics.get(self.monitor, None)
        if current is None:
            return

        if isinstance(current, torch.Tensor):
            current = current.detach().cpu().item()
        else:
            current = float(current)

        if self._is_better(current):
            self.best_score = current
            self.best_epoch = trainer.current_epoch
            self.best_state_dict = copy.deepcopy(pl_module.state_dict())

    def restore_best_weights(self, pl_module):
        if self.best_state_dict is None:
            print(
                f"[Warning] No best in-memory weights were stored for monitor='{self.monitor}'. "
                f"Using current model weights."
            )
            return
        pl_module.load_state_dict(self.best_state_dict)


def _bind_callbacks(cb_list):
    # closure trick already used by your project
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


if __name__ == "__main__":

    n_exp = 25
    seed_stride = 1

    parser = make_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    run_name = resolve_run_name(cfg, cfg_path)
    model_dir = (cfg.get("model") or {}).get("name")

    devices = get_device(args.devices, cfg.get("trainer", {}))

    # base seed: used ONLY to build fixed datamodule/splits once
    base_seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(base_seed, workers=True)

    # apply CLI overrides into cfg
    override_band_region(cfg, args.band_region)
    override_date_filters(cfg, args.days)

    # parsed days for folder naming
    days_list = parse_int_list(args.days) or cfg.get("data", {}).get(
        "date_filters", None
    )
    if isinstance(days_list, int):
        days_list = [days_list]

    # resolve save_dir
    save_dir = Path(cfg.get("logging", {}).get("save_dir", "logs"))
    save_dir = save_dir / cfg_path.stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    day_folder = _format_day_folder(days_list)
    csv_name = f"metrics_stats_{args.band_region}_{day_folder}.csv"
    # --------------------------------------------------------
    # Confirm selected spectral band + days
    # --------------------------------------------------------
    pp = cfg.get("data", {}).get("preprocess", {})
    print("\n" + "=" * 80)
    print("[CONFIG SUMMARY]")
    print(f"  Band region      : {args.band_region or 'from_config'}")
    print(f"  Keep range (nm)  : {pp.get('keep_range')}")
    print(f"  Exclude ranges   : {pp.get('exclude_ranges')}")
    print(f"  Selected days    : {cfg.get('data', {}).get('date_filters', [])}")
    print(f"  Save directory   : {save_dir}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------
    # Build DataModule ONCE (fixed split / fixed datamodule seed)
    # --------------------------------------------------------
    dm = build_datamodule(cfg)
    print(
        f"[Data] Kept Wavelengths ={len(dm.kept_wavelengths_nm)}  Input dim ={dm.input_dim} classes={dm.num_classes}"
    )

    # Save resolved config once
    # save_resolved_config(cfg, save_dir)

    # --- Class distribution per split ---
    def class_stats(name: str, ds):
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

    if args.mode == "train":
        all_results = []

        for exp_idx in tqdm(range(n_exp), desc="Experiments"):
            exp_seed = base_seed + exp_idx * seed_stride
            print("\n" + "-" * 80)
            print(f"[Experiment {exp_idx + 1}/{n_exp}] model seed = {exp_seed}")
            print("-" * 80)

            # IMPORTANT:
            # Reseed here so ONLY model/training randomness changes.
            # DataModule was already built and remains fixed.
            seed_everything(exp_seed, workers=True)

            # Fresh model/lit for this repetition
            model, lit = build_model_and_lit(
                cfg,
                input_dim=dm.input_dim,
                num_classes=dm.num_classes,
                dm=dm,
            )

            # Get logger + callbacks from your existing helper
            _, callbacks = build_logger_and_callbacks(cfg, run_name, save_dir)

            callbacks = [
                cb
                for cb in callbacks
                if not isinstance(cb, (ModelCheckpoint, LearningRateMonitor))
            ]

            tb = False

            early_stop_cb = next(
                (cb for cb in callbacks if isinstance(cb, EarlyStopping)), None
            )

            monitor = "val/loss"
            mode = "min"
            if early_stop_cb is not None:
                monitor = early_stop_cb.monitor
                mode = early_stop_cb.mode

            mem_ckpt = InMemoryBestModelCheckpoint(monitor=monitor, mode=mode)
            callbacks.append(mem_ckpt)

            _bind_callbacks(callbacks)

            trainer = build_trainer(
                cfg,
                logger=tb,
                devices=devices,
                precision_override=args.precision,
                max_epochs_override=args.max_epochs,
            )

            # Resume is not meaningful for repeated independent experiments
            if args.resume and n_exp == 1:
                try:
                    resume_ckpt = find_checkpoint(args.resume, run_dir=save_dir)
                    print(f"[Train] Resuming from checkpoint: {resume_ckpt}")
                    trainer.fit(lit, datamodule=dm, ckpt_path=str(resume_ckpt))
                except FileNotFoundError as e:
                    raise SystemExit(f"[Train] --resume '{args.resume}' not found: {e}")
            else:
                trainer.fit(lit, datamodule=dm)

            # Restore best weights from memory before test
            mem_ckpt.restore_best_weights(lit)

            results = trainer.test(lit, datamodule=dm, ckpt_path=None)
            metrics = _to_plain_dict(results[0])

            row = {
                "exp_idx": exp_idx,
                "seed": exp_seed,
                "best_monitor": mem_ckpt.best_score,
                "best_epoch": mem_ckpt.best_epoch,
                **metrics,
            }
            all_results.append(row)

            print(f"[Experiment {exp_idx + 1}] Test metrics: {metrics}")

        # Final dataframe
        df = pd.DataFrame(all_results)

        out_csv = save_dir / csv_name
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 80)
        print("[FINAL RESULTS]")
        print(df)
        print("\n[SUMMARY]")
        print(df.describe(include="all"))
        print(f"\nSaved final metrics dataframe to: {out_csv}")
        print("=" * 80)

    else:  # args.mode == "test"
        # Original single-checkpoint test path preserved
        # Rebuild model/lit once
        model, lit = build_model_and_lit(
            cfg,
            input_dim=dm.input_dim,
            num_classes=dm.num_classes,
            dm=dm,
        )

        tb, callbacks = build_logger_and_callbacks(cfg, run_name, save_dir)
        _bind_callbacks(callbacks)

        trainer = build_trainer(
            cfg,
            logger=tb,
            devices=devices,
            precision_override=args.precision,
            max_epochs_override=None,
        )

        if not args.ckpt:
            raise SystemExit("You must provide --ckpt when --mode test.")

        ckpt_path = find_checkpoint(args.ckpt, run_dir=save_dir)
        print(f"[Test] Using checkpoint: {ckpt_path}")
        results = trainer.test(lit, datamodule=dm, ckpt_path=str(ckpt_path))
        print("[Test] Metrics:", results)

        metrics = results[0]
        model_name = Path(ckpt_path).name.replace(".ckpt", "")
        save_test_metrics(
            metrics, cfg_path, Path(tb.log_dir), f"test_metrics_{model_name}.txt"
        )

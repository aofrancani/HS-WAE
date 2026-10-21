"""USAGE

-------- Train-------
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

import numpy as np
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

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
)
from trainer import build_trainer, build_logger_and_callbacks


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    run_name = resolve_run_name(cfg, cfg_path)
    model_dir = (cfg.get("model") or {}).get("name")

    devices = get_device(args.devices, cfg.get("trainer", {}))

    # seed
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed, workers=True)

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
    save_dir = resolve_save_dir(cfg, cfg_path, args.band_region, days_list)

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

    # DataModule
    dm = build_datamodule(cfg)
    print(
        f"[Data] Kept Wavelengths ={len(dm.kept_wavelengths_nm)}  Input dim ={dm.input_dim} classes={dm.num_classes}"
    )

    # Model + LitModule (shared construction)
    model, lit = build_model_and_lit(
        cfg,
        input_dim=dm.input_dim,
        num_classes=dm.num_classes,
        dm=dm,  # pass datamodule to model builder for wavelet attention config if needed
    )

    # Logger & callbacks
    tb, callbacks = build_logger_and_callbacks(cfg, run_name, save_dir)

    # Feed callbacks into trainer builder via a small trick
    def _bind_callbacks(cb_list):  # closure to hand callbacks to build_trainer
        build_logger_and_callbacks.__wrapped_callbacks = cb_list

    _bind_callbacks(callbacks)

    # Trainer
    trainer = build_trainer(
        cfg,
        logger=tb,
        devices=devices,
        precision_override=args.precision,
        max_epochs_override=args.max_epochs if args.mode == "train" else None,
    )

    # Save resolved config near logs
    save_resolved_config(cfg, Path(tb.log_dir))

    if args.mode == "train":
        resume_ckpt = None
        if args.resume:
            try:
                # accept either a full path or a checkpoint name
                resume_ckpt = find_checkpoint(args.resume, run_dir=save_dir)
                print(f"[Train] Resuming from checkpoint: {resume_ckpt}")
            except FileNotFoundError as e:
                raise SystemExit(f"[Train] --resume '{args.resume}' not found: {e}")

        trainer.fit(
            lit, datamodule=dm, ckpt_path=str(resume_ckpt) if resume_ckpt else None
        )

        # Optional test-after-fit from config
        if cfg.get("trainer", {}).get("test_after_fit", False):
            # use best checkpoint if available
            ckpt_cb = next(
                (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None
            )
            best = (
                ckpt_cb.best_model_path if ckpt_cb and ckpt_cb.best_model_path else None
            )
            results = trainer.test(lit, datamodule=dm, ckpt_path=best)

            # # Save a copy of test metrics
            # out_json = Path(tb.log_dir) / "test_metrics.json"
            # out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

            # save metrics
            metrics = results[0]
            save_test_metrics(metrics, cfg_path, Path(tb.log_dir), "test_metrics.txt")

    else:  # args.mode == "test"
        if not args.ckpt:
            raise SystemExit("You must provide --ckpt when --mode test.")

        # Reuse same logger namespace; resolve checkpoint by name or path
        ckpt_path = find_checkpoint(args.ckpt, run_dir=save_dir)
        print(f"[Test] Using checkpoint: {ckpt_path}")
        results = trainer.test(lit, datamodule=dm, ckpt_path=str(ckpt_path))
        print("[Test] Metrics:", results)

        # save metrics
        metrics = results[0]
        model_name = Path(ckpt_path).name.replace(".ckpt", "")
        save_test_metrics(
            metrics, cfg_path, Path(tb.log_dir), f"test_metrics_{model_name}.txt"
        )

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

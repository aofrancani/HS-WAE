from __future__ import annotations

"""
Extract full-dimensional test latent representations from existing HS-AE/HS-WAE runs.

This script DOES NOT train anything. It scans one or more existing log roots for
`config_resolved.yaml`, rebuilds the corresponding DataModule, loads the saved
checkpoint, runs deterministic test inference, and saves a compact per-run package
containing:
  - full latent vectors (e.g. 64-D)
  - ground-truth labels
  - predictions and class probabilities
  - WTAM level-importance weights alpha (when available)
  - test sample metadata / stable split indices
  - DWT level names and lengths
  - provenance (config path, checkpoint path, run directory, method, band, day)

Intended use:
    python -m scripts.extract_latent_runs ...

If you place this file under scripts/, execute it from the project root.
"""

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

# Allow execution both as `python scripts/...py` and `python -m scripts...`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import (  # noqa: E402
    CROP_FILES,
    HOUR_FILTERS,
    _add_dayindex_column,
    _fix_year,
    build_datamodule,
    load_crop_df,
)
from models.build_model import build_model_and_lit  # noqa: E402


BAND_RANGES: Dict[str, Tuple[int, int]] = {
    "uv": (340, 380),
    "vis": (380, 680),
    "rgb": (430, 690),
    "re": (680, 750),
    "nir": (750, 1000),
    "swir-1": (1000, 1350),
    "swir-2a": (1440, 1800),
    "swir-2b": (1980, 2350),
    "all": (350, 2460),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract full latent vectors from existing trained runs without retraining."
    )
    p.add_argument(
        "--search-root",
        action="append",
        required=True,
        help=(
            "Directory to scan recursively for config_resolved.yaml. "
            "Repeat this argument to scan multiple roots."
        ),
    )
    p.add_argument(
        "--outdir",
        default="results/latent",
        help="Output directory for extracted latent packages.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help=(
            "Optional replacement for data.csv_path stored in old configs. "
            "Useful when analyzing old server runs on another machine."
        ),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "dwt", "hswae"],
        choices=["raw", "dwt", "hswae"],
        help="Methods to extract.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Optional band filter, e.g. --bands nir or --bands nir re.",
    )
    p.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=None,
        help="Optional day filter, e.g. --days 1 2 3 4 5 6 7.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto", "cpu", "cuda", "cuda:0", etc.',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted run packages.",
    )
    p.add_argument(
        "--checkpoint-policy",
        choices=["auto", "latest"],
        default="auto",
        help=(
            "auto: prefer a single non-last checkpoint / best.ckpt; latest: latest mtime. "
            "With save_top_k=1, auto normally selects the unique checkpoint."
        ),
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for smoke testing the extractor.",
    )
    return p.parse_args()


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device(value)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is not a dictionary: {path}")
    return cfg


def infer_method(cfg: Dict[str, Any]) -> str:
    dwt_enabled = bool(
        cfg.get("data", {}).get("preprocess", {}).get("dwt", {}).get("enabled", False)
    )
    wtam_enabled = bool(
        cfg.get("model", {}).get("wavelet_attn", {}).get("enabled", False)
    )
    if not dwt_enabled:
        return "raw"
    if wtam_enabled:
        return "hswae"
    return "dwt"


def infer_band(cfg: Dict[str, Any], run_dir: Path) -> str:
    # First use path tokens, which is usually the least ambiguous source.
    parts = [p.lower() for p in run_dir.parts]
    for band in ["swir-2a", "swir-2b", "swir-1", "nir", "re", "rgb", "vis", "uv"]:
        if band in parts:
            return band

    keep = cfg.get("data", {}).get("preprocess", {}).get("keep_range")
    if keep is not None and len(keep) == 2:
        pair = (int(keep[0]), int(keep[1]))
        exact = [name for name, rng in BAND_RANGES.items() if pair == rng]
        if len(exact) == 1:
            return exact[0]
    return "unknown"


def infer_days(cfg: Dict[str, Any], run_dir: Path) -> List[int]:
    days = cfg.get("data", {}).get("date_filters")
    if days is not None:
        if isinstance(days, int):
            return [int(days)]
        return [int(x) for x in days]

    for part in reversed(run_dir.parts):
        m = re.fullmatch(r"day_(\d+)", part.lower())
        if m:
            return [int(m.group(1))]
    return []


def version_number(run_dir: Path) -> Optional[int]:
    for part in reversed(run_dir.parts):
        m = re.fullmatch(r"version_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def choose_checkpoint(run_dir: Path, policy: str = "auto") -> Path:
    ckpts = sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found under {run_dir}")

    if policy == "latest":
        return max(ckpts, key=lambda p: p.stat().st_mtime)

    # Common Lightning conventions.
    for exact_name in ["best.ckpt"]:
        matches = [p for p in ckpts if p.name == exact_name]
        if len(matches) == 1:
            return matches[0]

    non_last = [p for p in ckpts if p.name != "last.ckpt"]
    if len(non_last) == 1:
        return non_last[0]
    if len(ckpts) == 1:
        return ckpts[0]

    # save_top_k=1 plus last.ckpt commonly leaves two files; prefer non-last.
    if non_last:
        chosen = max(non_last, key=lambda p: p.stat().st_mtime)
        warnings.warn(
            f"Multiple checkpoints found under {run_dir}; selecting latest non-last: {chosen}"
        )
        return chosen

    return max(ckpts, key=lambda p: p.stat().st_mtime)


def discover_runs(search_roots: Sequence[Path]) -> List[Tuple[Path, Path, Path]]:
    """Return (source_root, run_dir, config_path) without collapsing roots."""
    found: List[Tuple[Path, Path, Path]] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            warnings.warn(f"Search root does not exist: {root}")
            continue
        for cfg_path in root.rglob("config_resolved.yaml"):
            run_dir = cfg_path.parent
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((root, run_dir, cfg_path))
    return sorted(found, key=lambda x: str(x[1]))


def make_run_id(source_root: Path, run_dir: Path, vnum: Optional[int]) -> str:
    root_name = re.sub(r"[^a-z0-9]+", "_", source_root.name.lower()).strip("_")
    version = f"version_{vnum}" if vnum is not None else "version_unknown"
    digest = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{root_name}__{version}__{digest}"


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return state


def restore_checkpoint_decoder_shape(model: Any, state: Dict[str, torch.Tensor]) -> Optional[int]:
    """Rebuild only the SAE output layer when an old checkpoint records its exact size."""
    candidates = []
    prefix = "model.ae.decoder."
    for key, value in state.items():
        if key.startswith(prefix) and key.endswith(".weight") and value.ndim == 2:
            try:
                layer_idx = int(key[len(prefix):].split(".", 1)[0])
            except ValueError:
                continue
            candidates.append((layer_idx, value))
    if not candidates or not hasattr(model, "ae"):
        return None

    layer_idx, weight = max(candidates, key=lambda item: item[0])
    layer = model.ae.decoder[layer_idx]
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError(f"Checkpoint decoder layer {layer_idx} is not Linear in current model")
    checkpoint_out_dim, checkpoint_in_dim = map(int, weight.shape)
    if layer.in_features != checkpoint_in_dim:
        raise RuntimeError(
            f"Decoder input mismatch: checkpoint={checkpoint_in_dim}, current={layer.in_features}"
        )
    if layer.out_features != checkpoint_out_dim:
        model.ae.decoder[layer_idx] = torch.nn.Linear(checkpoint_in_dim, checkpoint_out_dim)
        model.ae.cfg.dec_out_dim = checkpoint_out_dim
    return checkpoint_out_dim


def read_saved_test_accuracy(run_dir: Path) -> Optional[float]:
    metrics_path = run_dir / "test_metrics.txt"
    if not metrics_path.exists():
        return None
    text = metrics_path.read_text(encoding="utf-8")
    names = re.findall(r"^  (test/\w+)$", text, flags=re.MULTILINE)
    values = [float(x) for x in re.findall(r"^  ([-+\d.eE]+)$", text, flags=re.MULTILINE)]
    return dict(zip(names, values)).get("test/acc")


def rebuild_filtered_dataframe(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Reproduce the metadata row ordering used by CSVHSIDataModule.setup()."""
    data_cfg = cfg["data"]
    base_path = Path(data_cfg["csv_path"])
    crops = data_cfg.get("crops", ["soy"])

    dfs = []
    for i, crop in enumerate(crops):
        dfs.append(load_crop_df(base_path, crop, i * 2))
    df_all = pd.concat(dfs, ignore_index=True)

    df_all = _add_dayindex_column(df_all)
    df_all["Datetime"] = pd.to_datetime(
        df_all["Date"].apply(_fix_year), format="%m/%d/%Y %H:%M:%S"
    )
    df_all["Hour_only"] = df_all["Datetime"].dt.hour

    date_filters = data_cfg.get("date_filters")
    if date_filters:
        targets = set(int(x) for x in date_filters)
        df_all = df_all[df_all["DayIndex"].isin(targets)]

    filtered = []
    for day, group in df_all.groupby("DayIndex"):
        if day in HOUR_FILTERS:
            filtered.append(group[group["Hour_only"] == HOUR_FILTERS[day]])
        else:
            filtered.append(group)
    df_all = pd.concat(filtered, ignore_index=True)
    return df_all


def recover_test_metadata(cfg: Dict[str, Any], dm: Any) -> pd.DataFrame:
    """Recover test rows using the same two stratified splits as the DataModule."""
    df = rebuild_filtered_dataframe(cfg)
    label_col = cfg.get("data", {}).get("preprocess", {}).get("label_col", "Label")
    y = df[label_col].astype("int64").to_numpy()
    idx = np.arange(len(y))

    train_pct = float(cfg.get("data", {}).get("train_pct", 0.70))
    val_pct = float(cfg.get("data", {}).get("val_pct", 0.15))
    test_pct = float(cfg.get("data", {}).get("test_pct", 0.15))
    random_state = int(getattr(dm, "random_state", 42))

    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx,
        y,
        test_size=(1.0 - train_pct),
        stratify=y,
        random_state=random_state,
    )
    del idx_tr, y_tr

    val_ratio_of_tmp = val_pct / (val_pct + test_pct + 1e-12)
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp,
        y_tmp,
        test_size=(1.0 - val_ratio_of_tmp),
        stratify=y_tmp,
        random_state=random_state,
    )
    del idx_va, y_va

    meta = df.iloc[idx_te].copy().reset_index(drop=False).rename(
        columns={"index": "filtered_row_index"}
    )
    meta["test_order"] = np.arange(len(meta), dtype=int)
    meta["split_random_state"] = random_state

    if not np.array_equal(meta[label_col].astype(int).to_numpy(), y_te.astype(int)):
        raise RuntimeError("Recovered metadata labels do not match recovered test labels.")
    return meta


def method_forward_features(model: Any, x: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return latent, logits, alpha without relying on the DWT-only forward path."""
    alpha: Optional[torch.Tensor] = None
    if isinstance(x, (list, tuple)):
        levels = list(x)
        if getattr(model, "wavelet_attn", None) is not None:
            attn = model.wavelet_attn(levels)
            x_vec = attn["x_weighted"]
            alpha = attn["alpha"]
        else:
            # DWT-only baseline: concatenate levels exactly in DataModule order.
            x_vec = torch.cat(levels, dim=1)
    else:
        x_vec = x

    z = model.ae.encode(x_vec)
    logits = model.head(z)
    return z, logits, alpha


def move_batch_x(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [xi.to(device) for xi in x]
    raise TypeError(f"Unsupported batch input type: {type(x)}")


def safe_json_value(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def extract_one_run(
    source_root: Path,
    run_dir: Path,
    cfg_path: Path,
    outdir: Path,
    device: torch.device,
    data_root: Optional[str],
    checkpoint_policy: str,
    overwrite: bool,
) -> Dict[str, Any]:
    cfg = load_yaml(cfg_path)
    if data_root is not None:
        cfg.setdefault("data", {})["csv_path"] = str(Path(data_root))

    method = infer_method(cfg)
    band = infer_band(cfg, run_dir)
    days = infer_days(cfg, run_dir)
    day_tag = "-".join(f"{d:02d}" for d in days) if days else "unknown"
    vnum = version_number(run_dir)
    run_id = make_run_id(source_root, run_dir, vnum)

    run_out = outdir / method / band / f"day_{day_tag}" / run_id
    npz_path = run_out / "test_latent_outputs.npz"
    csv_path = run_out / "test_samples.csv"
    info_path = run_out / "run_info.json"

    if npz_path.exists() and csv_path.exists() and info_path.exists() and not overwrite:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["status"] = "skipped_existing"
        return info

    checkpoint = choose_checkpoint(run_dir, checkpoint_policy)
    state = load_checkpoint_state(checkpoint)

    dm = build_datamodule(cfg)
    model, lit = build_model_and_lit(
        cfg, input_dim=dm.input_dim, num_classes=dm.num_classes, dm=dm
    )

    checkpoint_recon_dim = restore_checkpoint_decoder_shape(model, state)
    try:
        lit.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Strict checkpoint loading failed for {checkpoint}. "
            "This usually means the source code does not match the code used to train the run.\n"
            f"Original error: {e}"
        ) from e

    model = lit.model.to(device)
    model.eval()

    latents: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    alphas: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in dm.test_dataloader():
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _x_raw, x, y = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                raise TypeError(
                    "Expected a (x, y) or (x_raw, levels, y) test batch, "
                    f"got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}"
                )
            x = move_batch_x(x, device)
            y = y.to(device).long()
            z, logits, alpha = method_forward_features(model, x)
            p = torch.softmax(logits, dim=1)

            latents.append(z.detach().cpu().numpy().astype(np.float32))
            labels.append(y.detach().cpu().numpy().astype(np.int64))
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64))
            probs.append(p.detach().cpu().numpy().astype(np.float32))
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            if alpha is not None:
                alphas.append(alpha.detach().cpu().numpy().astype(np.float32))

    latent = np.concatenate(latents, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    logits_arr = np.concatenate(logits_all, axis=0)
    alpha_arr = np.concatenate(alphas, axis=0) if alphas else np.empty((len(y_true), 0), dtype=np.float32)

    meta = recover_test_metadata(cfg, dm)
    if len(meta) != len(y_true):
        raise RuntimeError(
            f"Metadata/test-loader length mismatch for {run_dir}: {len(meta)} vs {len(y_true)}"
        )

    label_col = cfg.get("data", {}).get("preprocess", {}).get("label_col", "Label")
    meta_labels = meta[label_col].astype(int).to_numpy()
    if not np.array_equal(meta_labels, y_true):
        raise RuntimeError(
            f"Metadata labels do not match DataLoader labels for {run_dir}; refusing to save misaligned outputs."
        )

    run_out.mkdir(parents=True, exist_ok=True)

    # Canonical compact numeric package.
    np.savez_compressed(
        npz_path,
        latent=latent,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        logits=logits_arr,
        alpha=alpha_arr,
        test_order=np.arange(len(y_true), dtype=np.int64),
        filtered_row_index=meta["filtered_row_index"].to_numpy(dtype=np.int64),
    )

    # Human-readable per-sample table. Latent dimensions are deliberately saved as
    # columns so future analyses do not depend on NPZ-specific code.
    samples = meta.copy()
    samples["y_true"] = y_true
    samples["y_pred"] = y_pred
    samples["correct"] = (y_true == y_pred).astype(int)
    for c in range(y_prob.shape[1]):
        samples[f"prob_class_{c}"] = y_prob[:, c]
    for j in range(latent.shape[1]):
        samples[f"latent_{j:03d}"] = latent[:, j]

    level_names = list(getattr(dm, "feature_names", [])) if getattr(dm.cfg, "dwt_enabled", False) else []
    if alpha_arr.shape[1] > 0:
        if len(level_names) != alpha_arr.shape[1]:
            level_names = [f"level_{i}" for i in range(alpha_arr.shape[1])]
        for j, name in enumerate(level_names):
            samples[f"alpha_{name}"] = alpha_arr[:, j]

    samples.to_csv(csv_path, index=False)

    saved_test_accuracy = read_saved_test_accuracy(run_dir)
    extracted_test_accuracy = float(np.mean(y_true == y_pred))
    accuracy_matches = (
        None if saved_test_accuracy is None
        else bool(np.isclose(extracted_test_accuracy, saved_test_accuracy, atol=1e-7))
    )

    info: Dict[str, Any] = {
        "status": "extracted",
        "method": method,
        "band": band,
        "days": days,
        "run_dir": str(run_dir.resolve()),
        "source_root": str(source_root.resolve()),
        "replicate_id": run_id,
        "config_path": str(cfg_path.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_mtime": checkpoint.stat().st_mtime,
        "version_number": vnum,
        "config_seed": cfg.get("seed"),
        "split_random_state": int(getattr(dm, "random_state", 42)),
        "n_test": int(len(y_true)),
        "latent_dim": int(latent.shape[1]),
        "num_classes": int(y_prob.shape[1]),
        "saved_test_accuracy": saved_test_accuracy,
        "extracted_test_accuracy": extracted_test_accuracy,
        "saved_accuracy_matches": accuracy_matches,
        "class_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(y_true, return_counts=True))
        },
        "dwt_enabled": bool(getattr(dm.cfg, "dwt_enabled", False)),
        "current_raw_band_count": int(dm.band_count),
        "checkpoint_reconstruction_dim": checkpoint_recon_dim,
        "dwt_level_names": level_names,
        "dwt_level_lengths": getattr(dm, "dwt_level_lengths", None),
        "wtam_alpha_saved": bool(alpha_arr.shape[1] > 0),
        "npz_path": str(npz_path.resolve()),
        "samples_csv": str(csv_path.resolve()),
    }
    info_path.write_text(
        json.dumps(info, indent=2, default=safe_json_value), encoding="utf-8"
    )
    return info


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    roots = [Path(p) for p in args.search_root]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    discovered = discover_runs(roots)
    print(f"[Info] Found {len(discovered)} directories containing config_resolved.yaml")
    print(f"[Info] Inference device: {device}")

    candidates: List[Tuple[Path, Path, Path]] = []
    for source_root, run_dir, cfg_path in discovered:
        try:
            cfg = load_yaml(cfg_path)
            method = infer_method(cfg)
            band = infer_band(cfg, run_dir)
            days = infer_days(cfg, run_dir)
        except Exception as e:
            warnings.warn(f"Skipping unreadable config {cfg_path}: {e}")
            continue

        if method not in args.methods:
            continue
        if args.bands is not None and band not in set(args.bands):
            continue
        if args.days is not None and not set(days).intersection(args.days):
            continue
        candidates.append((source_root, run_dir, cfg_path))

    if args.max_runs is not None:
        candidates = candidates[: args.max_runs]

    print(f"[Info] Selected {len(candidates)} runs after filtering")

    manifest_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, (source_root, run_dir, cfg_path) in enumerate(candidates, start=1):
        print(f"\n[{i}/{len(candidates)}] {run_dir}")
        try:
            row = extract_one_run(
                source_root=source_root,
                run_dir=run_dir,
                cfg_path=cfg_path,
                outdir=outdir,
                device=device,
                data_root=args.data_root,
                checkpoint_policy=args.checkpoint_policy,
                overwrite=args.overwrite,
            )
            manifest_rows.append(row)
            print(
                f"  -> {row.get('method')} {row.get('band')} days={row.get('days')} "
                f"n={row.get('n_test')} latent={row.get('latent_dim')} [{row.get('status')}]"
            )
        except Exception as e:
            fail = {
                "run_dir": str(run_dir),
                "config_path": str(cfg_path),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            failures.append(fail)
            print(f"  !! FAILED: {type(e).__name__}: {e}")

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        manifest.to_csv(outdir / "manifest.csv", index=False)

    failures_df = pd.DataFrame(
        failures, columns=["run_dir", "config_path", "error_type", "error"]
    )
    failures_df.to_csv(outdir / "extraction_failures.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Extracted/skipped successfully: {len(manifest_rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {outdir.resolve()}")
    if failures:
        print(f"Review: {(outdir / 'extraction_failures.csv').resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

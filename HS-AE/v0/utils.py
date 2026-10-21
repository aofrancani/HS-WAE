from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Sequence
import yaml
import argparse

default_exclude_ranges = [
    [1350, 1440],
    [1800, 1980],
]
_BAND_PRESETS: Dict[str, Dict[str, Any]] = {
    "nir": {"keep_range": [750, 1000], "exclude_ranges": default_exclude_ranges},
    "re": {"keep_range": [680, 750], "exclude_ranges": default_exclude_ranges},
    "rgb": {
        "keep_range": [430, 690],
        "exclude_ranges": [
            [470, 530],  # B (430-470) G(530-570) R(650-690)
            [570, 650],
        ],
    },
    "swir-1": {"keep_range": [1000, 1350], "exclude_ranges": default_exclude_ranges},
    "swir-2a": {"keep_range": [1440, 1800], "exclude_ranges": default_exclude_ranges},
    "swir-2b": {
        "keep_range": [1980, 2350],
        "exclude_ranges": default_exclude_ranges,
    },  # fixed typo siw->swir
    "uv": {"keep_range": [340, 380], "exclude_ranges": default_exclude_ranges},
    "vis": {"keep_range": [380, 680], "exclude_ranges": default_exclude_ranges},
    "all": {"keep_range": [350, 2460], "exclude_ranges": default_exclude_ranges},
    "nir-swir-1": {"keep_range": [750, 1350], "exclude_ranges": default_exclude_ranges},
    "re-nir": {"keep_range": [680, 1000], "exclude_ranges": default_exclude_ranges},
    "re-swir-1": {
        "keep_range": [680, 1350],
        "exclude_ranges": [
            [750, 1000],
            [1350, 1440],
            [1800, 1980],
        ],
    },
    "vis-nir": {
        "keep_range": [380, 1000],
        "exclude_ranges": [
            [680, 750],
        ],
    },
    "vis-re": {
        "keep_range": [380, 750],
        "exclude_ranges": [
            [680, 750],
        ],
    },
    "vis-swir-1": {
        "keep_range": [380, 1350],
        "exclude_ranges": [
            [680, 1000],
            [1350, 1440],
        ],
    },
}


def override_band_region(cfg: Dict[str, Any], band_region: Optional[str]) -> None:
    """
    Override cfg['data']['preprocess']['keep_range'] (and exclude_ranges for rgb)
    based on a CLI band_region.

    This mutates cfg in-place.
    """
    if not band_region:
        return

    key = band_region.strip().lower()
    if key not in _BAND_PRESETS:
        valid = ", ".join(sorted(_BAND_PRESETS.keys()))
        raise ValueError(f"Unknown band_region={band_region!r}. Valid: {valid}")

    data = cfg.setdefault("data", {})
    pp = data.setdefault("preprocess", {})

    preset = _BAND_PRESETS[key]
    pp["keep_range"] = preset["keep_range"]

    # Only override exclude_ranges when preset provides it; otherwise remove if you want
    if preset.get("exclude_ranges") is not None:
        pp["exclude_ranges"] = preset["exclude_ranges"]
    else:
        # If you prefer to keep whatever is in YAML, comment this out.
        pp["exclude_ranges"] = None


def parse_int_list(
    value: Optional[Union[str, int, Sequence[int]]],
) -> Optional[List[int]]:
    """
    Parse an int list from:
      - None
      - int (e.g., 5)
      - str: "5" or "1,2,3" or "1 2 3"
      - sequence of ints
    Returns None if value is None.
    """
    if value is None:
        return None

    if isinstance(value, int):
        return [value]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # allow comma or whitespace separated
        parts = [p for p in s.replace(",", " ").split() if p.strip()]
        return [int(p) for p in parts]

    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    raise TypeError(f"Unsupported days format: {value!r}")


# ----------------------------
# Save dir resolver
# ----------------------------
def _format_day_folder(days: Optional[List[int]]) -> str:
    """
    If single day: day_05
    If multiple days: days_01-02-03
    If None/empty: day_unknown
    """
    if not days:
        return "day_unknown"
    if len(days) == 1:
        return f"day_{days[0]:02d}"
    return "days_" + "-".join(f"{d:02d}" for d in sorted(days))


def resolve_save_dir(
    cfg: Dict[str, Any],
    cfg_path: Path,
    band_region: Optional[str],
    days: Optional[List[int]],
) -> Path:
    """
    Build a save_dir that mimics your old structure like:
      logs/test/<cfg_path.stem>/<band_region>/<day_XX>
    Example:
      logging.save_dir = "logs/test"
      cfg_path.stem = "run_sae"   (if configs/dwt/run_sae.yaml)
      band_region = "nir"
      days = [5]
    => logs/test/run_sae/nir/day_05
    """
    base = Path(cfg.get("logging", {}).get("save_dir", "logs"))
    group = cfg_path.stem
    band = (band_region or "band_unknown").strip().lower()
    day_folder = _format_day_folder(days)

    return base / group / band / day_folder


def override_date_filters(
    cfg: Dict[str, Any], days: Optional[Union[str, int, Sequence[int]]]
) -> None:
    """
    Override cfg['data']['date_filters'] from CLI.
    Accepts days like: 5, "5", "1,2,3", [1,2,3].
    Mutates cfg in-place.
    """
    parsed = parse_int_list(days)
    if parsed is None:
        return

    data = cfg.setdefault("data", {})
    data["date_filters"] = parsed


def load_config(cfg_path: Path) -> Dict[str, Any]:
    """Load YAML or JSON config file."""
    if cfg_path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg_path.suffix.lower() == ".json":
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {cfg_path.suffix}")


def save_resolved_config(cfg: Dict[str, Any], run_dir: Path) -> None:
    """Write a copy of resolved config near logs for reproducibility."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )


def resolve_run_name(cfg: Dict[str, Any], cfg_path: Path) -> str:
    """Prefer logging.run_name, else config filename stem."""
    return cfg.get("logging", {}).get("run_name", cfg_path.stem)


def find_checkpoint(run_name: str, ckpt_arg: str, save_dir: str = "logs") -> Path:
    """
    If `ckpt_arg` is a file → return it.
    Else search under logs/<run_name>/version_*/checkpoints/<ckpt_arg>[.ckpt]
    picking the newest version_* first.
    """
    p = Path(ckpt_arg)
    if p.is_file():
        return p.resolve()

    ckpt_name = ckpt_arg if ckpt_arg.endswith(".ckpt") else f"{ckpt_arg}.ckpt"
    root = Path(save_dir) / run_name
    versions = sorted(
        (d for d in root.glob("version_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
        reverse=True,
    )
    for v in versions:
        cand = v / "checkpoints" / ckpt_name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Checkpoint '{ckpt_arg}' not found under {root.as_posix()}/version_*/checkpoints/."
    )


def get_device(
    devices_override: Any, tr_cfg: Dict[str, Any]
) -> Union[int, List[int], str]:
    """
    Normalize device input for Lightning Trainer.

    Priority:
      1. devices_override (CLI or function arg)
      2. tr_cfg["devices"] from config
      3. "auto" (default)

    Returns:
      list[int] | "auto"
    """
    devices = (
        devices_override
        if devices_override is not None
        else tr_cfg.get("devices", "auto")
    )

    # Case: int → treat as GPU index
    if isinstance(devices, int):
        return [devices]

    # Case: str
    if isinstance(devices, str):
        d = devices.strip().lower()
        if d == "auto":
            return d
        if "," in d:
            return [int(x) for x in d.split(",") if x.strip()]
        return [int(d)]  # single number string → GPU index

    # Case: list/tuple
    if isinstance(devices, (list, tuple)):
        return [int(x) for x in devices]

    raise TypeError(
        f"Unsupported devices format {devices!r}. Must be int, str, list[int], tuple[int], or 'auto'."
    )


def make_parser():
    p = argparse.ArgumentParser(description="HSI pipeline (train/test)")
    p.add_argument(
        "--mode", required=True, choices=["train", "test"], help="Pipeline to run"
    )
    p.add_argument("--config", required=True, type=str, help="Path to YAML/JSON config")
    p.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint file or name (test mode)"
    )
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help='Override devices, e.g. "auto", "0", "0,1"',
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint file or name to resume training from",
    )
    p.add_argument(
        "--precision",
        type=str,
        default=None,
        help='Override precision, e.g. "16-mixed", "32-true"',
    )
    p.add_argument(
        "--max_epochs", type=int, default=None, help="Override max_epochs (train mode)"
    )
    p.add_argument(
        "--band_region",
        type=str,
        default="all",
        choices=sorted(_BAND_PRESETS.keys()),
        help="Override spectral band region preset (updates keep_range / exclude_ranges).",
    )
    p.add_argument(
        "--days",
        type=str,
        default="1,2,3,4,5,6",
        help='Override data.date_filters. Examples: "5" or "1,2,3".',
    )
    return p


def save_test_metrics(
    metrics: Dict[str, Any],
    cfg_path: Path,
    save_dir: Path,
    fname: str = "test_metrics.txt",
) -> None:
    """
    Save and print test metrics to a text file inside the TensorBoard log directory.

    Args:
        metrics (Dict[str, Any]): Dictionary of metric names and their values.
        cfg_path (Path): Path to the configuration file used for the experiment.
        tb_log_dir (Path): Path to the TensorBoard log directory (tb.log_dir).
    """
    # Target file
    txt_path = Path(save_dir) / fname

    # Build formatted content
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Experiment: {cfg_path.stem}")
    lines.append(f"Config: {cfg_path}")
    lines.append("-" * 80)

    # Append metric names (print + save)
    lines.append("Metric Names:")
    for k in metrics.keys():
        print(k)
        lines.append(f"  {k}")

    lines.append("\nMetric Values:")
    for v in metrics.values():
        print(v)
        lines.append(f"  {v}")

    lines.append("=" * 80)
    lines.append("")

    # Save to file
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Saved metrics to] {txt_path}\n")

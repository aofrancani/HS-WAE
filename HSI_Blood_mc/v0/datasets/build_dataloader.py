from typing import Dict, Any
from v0.datasets.hyperblood_dataset import (
    build_datamodule as build_hyperblood_datamodule,
)
from v0.datasets.crop_dataset import build_datamodule as build_crop_datamodule


def build_datamodule(cfg: Dict[str, Any]):
    """
    Build the appropriate DataModule based on the dataset name in config.

    Args:
        cfg: Configuration dictionary with 'data' key containing 'dataset' field.

    Returns:
        DataModule: Configured data module (HyperBloodCSVDataModule or CSVHSIDataModule).
    """
    dataset_name = cfg.get("data", {}).get("dataset", "crop").lower()
    if dataset_name == "hyperblood":
        return build_hyperblood_datamodule(cfg)
    elif dataset_name == "crop":
        return build_crop_datamodule(cfg)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: 'hyperblood', 'crop'"
        )

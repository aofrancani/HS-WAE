from __future__ import annotations
from typing import Any, Dict, Tuple
import torch.nn as nn

from v0.models.sae import SpectralAEConfig, SpectralAEClassifier
from v0.models.simple_nn import MLPConfig, MLPClassifier
from v0.trainer import LitClassifier


def build_model(
    name: str, model_cfg: Dict[str, Any], *, input_dim: int, num_classes: int
) -> Tuple[nn.Module, type]:
    """
    Returns:
      model (nn.Module), LitModuleClass (class, not instance)
    """
    name = name.lower()
    if name == "sae":
        cfg = model_cfg.get("cfg", {})
        head = model_cfg.get("head", {})
        full = SpectralAEConfig(
            input_dim=input_dim,
            hidden_dims=cfg.get("hidden_dims", [512, 256]),
            latent_dim=cfg.get("latent_dim", 128),
            activation=cfg.get("activation", "gelu"),
            dropout=cfg.get("dropout", 0.0),
            norm=cfg.get("norm", "layer"),
            final_activation=cfg.get("final_activation", None),
        )
        model = SpectralAEClassifier(
            cfg=full,
            num_classes=num_classes,
            head_hidden=head.get("head_hidden", [128]),
            head_dropout=head.get("head_dropout", 0.0),
            recon_weight=head.get("recon_weight", 1.0),
            recon_kind=head.get("recon_kind", "mse"),
            ce_weight=head.get("ce_weight", 1.0),
        )
    elif name == "mlp":
        cfg = model_cfg.get("cfg", {})
        head = model_cfg.get("head", {})
        full = MLPConfig(
            input_dim=input_dim,
            hidden_dims=cfg.get("hidden_dims", [512, 256]),
            activation=cfg.get("activation", "gelu"),
            dropout=cfg.get("dropout", 0.0),
            norm=cfg.get("norm", "layer"),
        )
        model = MLPClassifier(
            cfg=full,
            num_classes=num_classes,
            head_hidden=head.get("head_hidden", []),
            head_dropout=head.get("head_dropout", 0.0),
        )

    else:
        raise ValueError(f"Unknown model name: {name}")
    return model


def build_model_and_lit(cfg: Dict[str, Any], input_dim: int, num_classes: int):
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    model = build_model(
        model_name, model_cfg, input_dim=input_dim, num_classes=num_classes
    )

    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    lit = LitClassifier(
        model,
        lr=opt_cfg.get("lr", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-2),
        scheduler_name=sched_cfg.get("name", "onecycle"),
        scheduler_kwargs=sched_cfg.get("kwargs", {}),
        monitor_metric=sched_cfg.get("monitor", "val/acc"),
    )
    return model, lit

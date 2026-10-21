from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import torch.nn as nn

from models.sae import (
    SpectralAEConfig,
    SpectralAEClassifier,
    WaveletTokenAttentionConfig,
    WaveletTokenAttention,
)
from models.simple_nn import MLPConfig, MLPClassifier
from trainer import LitClassifier


def build_model(
    name: str,
    model_cfg: Dict[str, Any],
    *,
    input_dim: int,
    num_classes: int,
    dm: Optional[Any] = None,  # pass your datamodule to access dwt_level_lengths
) -> nn.Module:
    """
    Returns:
      model (nn.Module)
    """
    name = name.lower()

    if name == "sae":
        cfg = model_cfg.get("cfg", {})
        head = model_cfg.get("head", {})
        wavelet_cfg = model_cfg.get("wavelet_attn", {})  # NEW BLOCK (optional)

        # ---- Optional Wavelet Token Attention ----
        wavelet_attn = None
        if wavelet_cfg.get("enabled", False):
            if dm is None or getattr(dm, "dwt_level_lengths", None) is None:
                raise ValueError(
                    "wavelet_attn.enabled=True, but DataModule does not expose dwt_level_lengths. "
                    "Make sure DWT token mode is enabled in the dataloader and pass dm into build_model."
                )

            wcfg = WaveletTokenAttentionConfig(
                level_lengths=dm.dwt_level_lengths,  # critical
                embed_dim=wavelet_cfg.get("embed_dim", 128),
                num_heads=wavelet_cfg.get("num_heads", 4),
                num_layers=wavelet_cfg.get("num_layers", 1),
                attn_dropout=wavelet_cfg.get("attn_dropout", 0.0),
                ff_dropout=wavelet_cfg.get("ff_dropout", 0.0),
                use_level_embedding=wavelet_cfg.get("use_level_embedding", True),
                importance_softmax=wavelet_cfg.get("importance_softmax", True),
            )
            wavelet_attn = WaveletTokenAttention(wcfg)

        # ---- AE ----
        full = SpectralAEConfig(
            enc_in_dim=input_dim,  # dm.input_dim
            dec_out_dim=dm.band_count,  # raw bands
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
            wavelet_attn=wavelet_attn,  # NEW
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


def build_model_and_lit(
    cfg: Dict[str, Any], input_dim: int, num_classes: int, dm: Optional[Any] = None
):
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    model = build_model(
        model_name,
        model_cfg,
        input_dim=input_dim,
        num_classes=num_classes,
        dm=dm,  # NEW
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

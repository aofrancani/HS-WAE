from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpectralAEConfig:
    input_dim: int
    hidden_dims: Sequence[int] = (512, 256)
    latent_dim: int = 128
    activation: str = "gelu"
    dropout: float = 0.0
    norm: str = "layer"
    final_activation: Optional[str] = None


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(norm: str, dim: int) -> nn.Module:
    norm = norm.lower()
    if norm == "none":
        return nn.Identity()
    if norm == "layer":
        return nn.LayerNorm(dim)
    if norm == "batch":
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unsupported norm: {norm}")


class SpectralAutoencoder(nn.Module):
    def __init__(self, cfg: SpectralAEConfig):
        super().__init__()
        self.cfg = cfg

        act = _make_activation(cfg.activation)

        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), _make_norm(cfg.norm, h), act]
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), _make_norm(cfg.norm, h), act]
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        if cfg.final_activation:
            dec_layers.append(_make_activation(cfg.final_activation))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    @staticmethod
    def recon_loss(
        x: torch.Tensor, x_hat: torch.Tensor, kind: str = "mse"
    ) -> torch.Tensor:
        if kind == "mse":
            return F.mse_loss(x_hat, x)
        if kind == "l1":
            return F.l1_loss(x_hat, x)
        if kind == "smoothl1":
            return F.smooth_l1_loss(x_hat, x, beta=0.1)
        raise ValueError(f"Unknown loss kind: {kind}")


class SpectralAEClassifier(nn.Module):
    """
    Autoencoder + classifier head on latent.
    - Binary default (normal vs damaged), but supports multi-class by num_classes.
    - Combined loss: CE (classification) + lambda * MSE (reconstruction).
    """

    def __init__(
        self,
        cfg: SpectralAEConfig,
        num_classes: int = 2,
        head_hidden: Sequence[int] = (128,),
        head_dropout: float = 0.0,
        recon_weight: float = 1.0,  # lambda for recon loss
        ce_weight: float = 1.0,
        recon_kind: str = "mse",
    ):
        super().__init__()
        self.ae = SpectralAutoencoder(cfg)
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.ce_weight = ce_weight
        self.recon_kind = recon_kind

        # Small MLP head on the latent
        layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Dict[str, Any]:
        z = self.ae.encode(x)  # (B, latent)
        logits = self.head(z)  # (B, C)
        x_hat = self.ae.decode(z)  # (B, L)

        out: Dict[str, Any] = {"logits": logits, "recon": x_hat, "latent": z}

        if compute_loss and y is not None:
            ce = F.cross_entropy(logits, y)
            recon = self.ae.recon_loss(x, x_hat, self.recon_kind)
            loss = self.ce_weight * ce + self.recon_weight * recon
            out.update({"loss": loss, "ce_loss": ce, "recon_loss": recon})

        return out

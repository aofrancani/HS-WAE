# models/simple_nn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- small helpers ---
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


# -----------------------------
# Config
# -----------------------------
@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: Sequence[int] = (512, 256)
    activation: str = "gelu"
    dropout: float = 0.0
    norm: str = "layer"  # "layer" | "batch" | "none"


# -----------------------------
# Model (encoder + classifier)
# -----------------------------
class MLPClassifier(nn.Module):
    """
    Plain MLP classifier for 1D spectra.

    Architecture:
      input_dim -> [Linear -> Norm -> Act -> Dropout]* -> Linear(num_classes)

    Returns a dict compatible with LitClassifier:
      {"logits": logits, "loss": ..., "ce_loss": ...}  (loss only when y provided and compute_loss=True)
    """

    def __init__(
        self,
        cfg: MLPConfig,
        num_classes: int = 2,
        head_hidden: Sequence[int] = (),  # optionally add extra hidden for the head
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        act = _make_activation(cfg.activation)

        # Encoder stack
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), _make_norm(cfg.norm, h), act]
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else nn.Identity()
        feat_dim = in_dim  # last hidden or input_dim if no hidden layers

        # Classification head
        head_layers: List[nn.Module] = []
        h_in = feat_dim
        for h in head_hidden:
            head_layers += [nn.Linear(h_in, h), nn.ReLU(inplace=True)]
            if head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            h_in = h
        head_layers.append(nn.Linear(h_in, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Dict[str, Any]:
        feats = self.encoder(x)  # (B, feat_dim)
        logits = self.head(feats)  # (B, C)
        out: Dict[str, Any] = {"logits": logits, "feat": feats}

        if compute_loss and y is not None:
            ce = F.cross_entropy(logits, y)
            out.update({"loss": ce, "ce_loss": ce})
        return out

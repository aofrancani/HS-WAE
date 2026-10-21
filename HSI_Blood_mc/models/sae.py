# model_wavelet_attention.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Your existing AE config
# -----------------------------
@dataclass
class SpectralAEConfig:
    enc_in_dim: int
    dec_out_dim: int
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

        enc_layers: List[nn.Module] = []
        in_dim = cfg.enc_in_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), _make_norm(cfg.norm, h), act]
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), _make_norm(cfg.norm, h), act]
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.dec_out_dim))
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


# -----------------------------
# NEW: Wavelet Token Attention
# -----------------------------
@dataclass
class WaveletTokenAttentionConfig:
    level_lengths: Sequence[int]  # [n_0, n_1, ..., n_{K-1}] in dataloader order
    embed_dim: int = 128  # token embedding dim E
    num_heads: int = 4
    num_layers: int = 1
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    use_level_embedding: bool = True  # like positional embedding but for levels
    importance_softmax: bool = True  # True => weights sum to 1 across levels


class WaveletTokenAttention(nn.Module):
    """
    Input:
      levels: List[Tensor] length K, each (B, n_l)
    Output:
      x_weighted: (B, sum(n_l))  -> ready for AE
      alpha: (B, K)             -> level importance
    """

    def __init__(self, cfg: WaveletTokenAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.level_lengths = list(cfg.level_lengths)
        self.K = len(self.level_lengths)

        # Per-level projection: R^{n_l} -> R^{E}
        self.proj = nn.ModuleList(
            [nn.Linear(nl, cfg.embed_dim) for nl in self.level_lengths]
        )

        self.level_embed = (
            nn.Embedding(self.K, cfg.embed_dim) if cfg.use_level_embedding else None
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dropout=cfg.attn_dropout,
            dim_feedforward=4 * cfg.embed_dim,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        # Importance head: per token -> scalar score
        self.score = nn.Linear(cfg.embed_dim, 1)

        self.ff_dropout = (
            nn.Dropout(cfg.ff_dropout) if cfg.ff_dropout > 0 else nn.Identity()
        )

    def forward(self, levels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(levels) != self.K:
            raise ValueError(f"Expected {self.K} levels, got {len(levels)}.")

        # Build token embeddings: (B, K, E)
        toks = []
        for l in range(self.K):
            x_l = levels[l]  # (B, n_l)
            toks.append(self.proj[l](x_l))  # (B, E)
        E = torch.stack(toks, dim=1)  # (B, K, E)

        if self.level_embed is not None:
            idx = torch.arange(self.K, device=E.device)
            E = E + self.level_embed(idx)[None, :, :]

        E = self.ff_dropout(E)
        H = self.encoder(E)  # (B, K, E)

        scores = self.score(H).squeeze(-1)  # (B, K)

        if self.cfg.importance_softmax:
            alpha = torch.softmax(scores, dim=-1)  # (B, K), sum=1
        else:
            alpha = torch.sigmoid(scores)  # (B, K), independent gates

        # Reweight original coefficient vectors and concatenate: (B, sum(n_l))
        weighted = []
        for l in range(self.K):
            w = alpha[:, l].unsqueeze(1)  # (B, 1)
            weighted.append(levels[l] * w)  # (B, n_l)
        x_weighted = torch.cat(weighted, dim=1)

        return {"x_weighted": x_weighted, "alpha": alpha, "token_out": H}


# -----------------------------
# AE + Classifier, now supports either:
#   x: Tensor(B,D)    (no DWT tokenization)
#   x: List[Tensor]   (DWT tokens per level)
# -----------------------------
class SpectralAEClassifier(nn.Module):
    """
    Autoencoder + classifier head on latent.
    Combined loss: CE (classification) + lambda * recon loss.

    If wavelet_attn is provided, forward can accept x as List[Tensor] tokens
    and will produce a weighted concatenated vector before the AE.
    """

    def __init__(
        self,
        cfg: SpectralAEConfig,
        num_classes: int = 2,
        head_hidden: Sequence[int] = (128,),
        head_dropout: float = 0.0,
        recon_weight: float = 1.0,
        ce_weight: float = 1.0,
        recon_kind: str = "mse",
        wavelet_attn: Optional[WaveletTokenAttention] = None,
    ):
        super().__init__()
        self.ae = SpectralAutoencoder(cfg)
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.ce_weight = ce_weight
        self.recon_kind = recon_kind
        self.wavelet_attn = wavelet_attn

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
        x: Union[torch.Tensor, List[torch.Tensor]],
        y: Optional[torch.Tensor] = None,
        x_recon_target: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Dict[str, Any]:

        attn_out: Optional[Dict[str, torch.Tensor]] = None

        if isinstance(x, list):
            if self.wavelet_attn is None:
                raise ValueError("x is a list of DWT tokens, but wavelet_attn is None.")
            attn_out = self.wavelet_attn(x)
            x_vec = attn_out["x_weighted"]  # (B, D')
        else:
            x_vec = x  # (B, D)

        z = self.ae.encode(x_vec)
        logits = self.head(z)
        x_hat = self.ae.decode(z)

        out: Dict[str, Any] = {"logits": logits, "recon": x_hat, "latent": z}

        # Expose importance weights if present
        if attn_out is not None:
            out["level_alpha"] = attn_out["alpha"]  # (B, K)
            out["token_out"] = attn_out["token_out"]  # (B, K, E)
            out["x_weighted"] = attn_out["x_weighted"]  # (B, D')

        if compute_loss and y is not None:
            ce = F.cross_entropy(logits, y)

            # recon target:
            # - if x_recon_target is provided (DWT token path), use it (raw spectrum)
            # - else fall back to x (non-DWT path)
            if x_recon_target is None:
                x_target = x  # must be Tensor(B, D)
                if isinstance(x_target, list):
                    raise ValueError(
                        "x_recon_target is required when x is a list of tokens."
                    )
            else:
                x_target = x_recon_target  # Tensor(B, bands)

            recon = self.ae.recon_loss(x_target, x_hat, self.recon_kind)
            loss = self.ce_weight * ce + self.recon_weight * recon
            out.update({"loss": loss, "ce_loss": ce, "recon_loss": recon})

        return out

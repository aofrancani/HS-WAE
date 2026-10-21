"""Checkpoint-retaining controlled WTAM architecture runner.

Run from the root of HS-AE, HSI_Blood, or HSI_Blood_mc.  The script imports
that repository's DataModule and historical SAE/WTAM implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, davies_bouldin_score, f1_score, log_loss,
    roc_auc_score, silhouette_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset

from datasets.build_dataloader import build_datamodule
from models.sae import (
    SpectralAEClassifier, SpectralAEConfig, SpectralAutoencoder,
    WaveletTokenAttention, WaveletTokenAttentionConfig,
)
from trainer import LitClassifier
from utils import load_config, override_band_region


VARIANTS = ("raw_ae", "dwt_concat", "dwt_static_gate", "dwt_dynamic_gate", "wtam_full")


class RawTargetDataset(Dataset):
    """Expose (unstandardized raw target, standardized input, label)."""
    def __init__(self, base: Dataset, mean: np.ndarray, std: np.ndarray):
        self.base = base
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.y = base.y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x * self.std + self.mean, x, y


class StaticGate(nn.Module):
    def __init__(self, level_lengths: Sequence[int]):
        super().__init__()
        self.level_lengths = list(level_lengths)
        self.logits = nn.Parameter(torch.zeros(len(level_lengths)))

    def alpha(self, levels: List[torch.Tensor]) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0).unsqueeze(0).expand(levels[0].shape[0], -1)


class DynamicGate(nn.Module):
    """Per-level sample gate with WTAM-sized embeddings and no cross-level attention."""
    def __init__(self, level_lengths: Sequence[int], embed_dim: int, use_level_embedding: bool):
        super().__init__()
        self.level_lengths = list(level_lengths)
        self.proj = nn.ModuleList([nn.Linear(n, embed_dim) for n in level_lengths])
        self.level_embed = nn.Embedding(len(level_lengths), embed_dim) if use_level_embedding else None
        self.score = nn.Linear(embed_dim, 1)

    def alpha(self, levels: List[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([proj(x) for proj, x in zip(self.proj, levels)], dim=1)
        if self.level_embed is not None:
            idx = torch.arange(len(levels), device=tokens.device)
            tokens = tokens + self.level_embed(idx)[None, :, :]
        return torch.softmax(self.score(tokens).squeeze(-1), dim=-1)


class DefinitiveAEClassifier(nn.Module):
    """Historical SAE/head with one explicitly selected DWT front end."""
    def __init__(self, ae_cfg: SpectralAEConfig, num_classes: int, head_hidden: Sequence[int],
                 head_dropout: float, recon_weight: float, ce_weight: float,
                 recon_kind: str, variant: str, level_lengths: Optional[Sequence[int]],
                 wat_cfg: Dict[str, Any]):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.ce_weight = ce_weight
        self.recon_kind = recon_kind
        # Match the historical builder for M4: construct WTAM before AE/head.
        self.wavelet_attn: Optional[WaveletTokenAttention] = None
        self.static_gate: Optional[StaticGate] = None
        self.dynamic_gate: Optional[DynamicGate] = None
        if variant != "raw_ae" and not level_lengths:
            raise ValueError("DWT variants require level lengths")
        if variant == "wtam_full":
            self.wavelet_attn = WaveletTokenAttention(WaveletTokenAttentionConfig(
                    level_lengths=level_lengths, embed_dim=int(wat_cfg.get("embed_dim", 64)),
                    num_heads=int(wat_cfg.get("num_heads", 4)), num_layers=int(wat_cfg.get("num_layers", 1)),
                    attn_dropout=float(wat_cfg.get("attn_dropout", 0.0)),
                    ff_dropout=float(wat_cfg.get("ff_dropout", 0.0)),
                    use_level_embedding=bool(wat_cfg.get("use_level_embedding", True)),
                    importance_softmax=bool(wat_cfg.get("importance_softmax", True))))
        self.ae = SpectralAutoencoder(ae_cfg)
        layers: List[nn.Module] = []
        in_dim = ae_cfg.latent_dim
        for h in head_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)
        # M2/M3 retain the construction order validated in their smoke tests.
        if variant == "dwt_static_gate":
            self.static_gate = StaticGate(level_lengths)
        elif variant == "dwt_dynamic_gate":
            self.dynamic_gate = DynamicGate(level_lengths, int(wat_cfg.get("embed_dim", 64)),
                                            bool(wat_cfg.get("use_level_embedding", True)))

    def _front(self, x: Union[torch.Tensor, List[torch.Tensor]],
               alpha_override: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(x, list):
            return x, None
        if self.variant == "dwt_concat":
            return torch.cat(x, dim=1), None
        if self.variant == "dwt_static_gate":
            assert self.static_gate is not None
            alpha = self.static_gate.alpha(x)
        elif self.variant == "dwt_dynamic_gate":
            assert self.dynamic_gate is not None
            alpha = self.dynamic_gate.alpha(x)
        elif self.variant == "wtam_full":
            assert self.wavelet_attn is not None
            alpha = self.wavelet_attn(x)["alpha"]
        else:
            raise ValueError(self.variant)
        if alpha_override is not None:
            override = alpha_override.to(x[0].device, x[0].dtype)
            if override.ndim == 1:
                override = override.unsqueeze(0).expand(x[0].shape[0], -1)
            alpha = override
        return torch.cat([level * alpha[:, i:i+1] for i, level in enumerate(x)], dim=1), alpha

    def forward(self, x, y=None, x_recon_target=None, compute_loss=True,
                alpha_override: Optional[torch.Tensor] = None):
        x_vec, alpha = self._front(x, alpha_override)
        z = self.ae.encode(x_vec)
        logits = self.head(z)
        recon = self.ae.decode(z)
        out = {"logits": logits, "recon": recon, "latent": z}
        if alpha is not None:
            out["level_alpha"] = alpha
            out["x_weighted"] = x_vec
        if compute_loss and y is not None:
            target = x_recon_target if x_recon_target is not None else x
            if isinstance(target, list):
                raise ValueError("Raw reconstruction target is required for DWT")
            ce = F.cross_entropy(logits, y)
            rc = self.ae.recon_loss(target, recon, self.recon_kind)
            out.update(loss=self.ce_weight * ce + self.recon_weight * rc,
                       ce_loss=ce, recon_loss=rc)
        return out


def make_model(cfg: Dict[str, Any], dm, variant: str) -> DefinitiveAEClassifier:
    mc = cfg["model"]
    ac = mc.get("cfg", {})
    head = mc.get("head", {})
    input_dim = dm.band_count if variant == "raw_ae" else dm.input_dim
    ae_cfg = SpectralAEConfig(
        enc_in_dim=int(input_dim), dec_out_dim=int(dm.band_count),
        hidden_dims=tuple(ac.get("hidden_dims", [256, 128])),
        latent_dim=int(ac.get("latent_dim", 64)), activation=ac.get("activation", "gelu"),
        dropout=float(ac.get("dropout", 0.0)), norm=ac.get("norm", "layer"),
        final_activation=ac.get("final_activation"),
    )
    return DefinitiveAEClassifier(
        ae_cfg, dm.num_classes, tuple(head.get("head_hidden", [64, 32])),
        float(head.get("head_dropout", 0.0)), float(head.get("recon_weight", 1.0)),
        float(head.get("ce_weight", 1.0)), head.get("recon_kind", "mse"), variant,
        dm.dwt_level_lengths, mc.get("wavelet_attn", {}),
    )


def prepare_batch(batch, device: torch.device):
    if len(batch) == 3:
        raw, x, y = batch
        if isinstance(x, (list, tuple)):
            x = [v.to(device) for v in x]
        else:
            x = x.to(device)
        return raw.to(device), x, y.to(device).long()
    x, y = batch
    x = x.to(device)
    return x, x, y.to(device).long()


def latent_metrics(z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    classes = np.unique(y)
    means = [z[y == c].mean(0) for c in classes]
    global_mean = z.mean(0)
    between = sum((y == c).sum() * float(np.sum((m-global_mean)**2)) for c, m in zip(classes, means))
    within = sum(float(np.sum((z[y == c]-m)**2)) for c, m in zip(classes, means))
    dists = [float(np.linalg.norm(means[i]-means[j])) for i in range(len(means)) for j in range(i+1, len(means))]
    centroid = float(np.mean(dists)) if dists else float("nan")
    within_rms = math.sqrt(within / max(len(y), 1))
    return {
        "latent_silhouette": float(silhouette_score(z, y)) if len(classes) > 1 else float("nan"),
        "latent_davies_bouldin": float(davies_bouldin_score(z, y)) if len(classes) > 1 else float("nan"),
        "latent_fisher": float(between / max(within, 1e-12)),
        "latent_centroid_distance": centroid,
        "latent_normalized_separation": float(centroid / max(within_rms, 1e-12)),
    }


def endpoint_metrics(y: np.ndarray, probs: np.ndarray, recon: np.ndarray,
                     raw: np.ndarray, latent: np.ndarray) -> Dict[str, float]:
    pred = probs.argmax(1)
    nclass = probs.shape[1]
    avg = "binary" if nclass == 2 else "macro"
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average=avg, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "nll": float(log_loss(y, probs, labels=list(range(nclass)))),
        "recon_mse": float(np.mean((recon-raw)**2)),
        "recon_mae": float(np.mean(np.abs(recon-raw))),
        "recon_derivative_rmse": float(np.sqrt(np.mean((np.diff(recon, axis=1)-np.diff(raw, axis=1))**2))),
    }
    dot = np.sum(recon * raw, axis=1)
    den = np.linalg.norm(recon, axis=1) * np.linalg.norm(raw, axis=1)
    out["recon_sam"] = float(np.mean(np.arccos(np.clip(dot / np.maximum(den, 1e-12), -1, 1))))
    if nclass == 2:
        out["auroc"] = float(roc_auc_score(y, probs[:, 1]))
        out["auprc"] = float(average_precision_score(y, probs[:, 1]))
        out["brier"] = float(brier_score_loss(y, probs[:, 1]))
    else:
        onehot = label_binarize(y, classes=np.arange(nclass))
        out["auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        out["auprc"] = float(average_precision_score(onehot, probs, average="macro"))
        out["brier"] = float(np.mean(np.sum((probs-onehot)**2, axis=1)))
    out.update(latent_metrics(latent, y))
    return out


@torch.no_grad()
def evaluate(model, loader, device, alpha_override=None, corrupt_level=None,
             corruption=None, severity=0.0):
    model.eval()
    ys=[]; probs=[]; logits=[]; raws=[]; recons=[]; latents=[]; alphas=[]
    override_cursor = 0
    for batch in loader:
        raw, x, y = prepare_batch(batch, device)
        if isinstance(x, list) and corrupt_level is not None:
            x = [v.clone() for v in x]
            if corruption == "occlusion":
                x[corrupt_level].zero_()
            elif corruption == "gaussian":
                x[corrupt_level].add_(torch.randn_like(x[corrupt_level]) * severity)
            elif corruption == "coefficient_dropout":
                keep = torch.rand_like(x[corrupt_level]) >= severity
                x[corrupt_level].mul_(keep)
        batch_override = alpha_override
        if torch.is_tensor(alpha_override) and alpha_override.ndim == 2:
            batch_override = alpha_override[override_cursor:override_cursor + y.shape[0]]
            override_cursor += y.shape[0]
        out = model(x, y=y, x_recon_target=raw, compute_loss=False, alpha_override=batch_override)
        p = torch.softmax(out["logits"], 1)
        ys.append(y.cpu()); probs.append(p.cpu()); logits.append(out["logits"].cpu())
        raws.append(raw.cpu()); recons.append(out["recon"].cpu()); latents.append(out["latent"].cpu())
        if "level_alpha" in out: alphas.append(out["level_alpha"].cpu())
    arrays = {
        "y_true": torch.cat(ys).numpy(), "probabilities": torch.cat(probs).numpy(),
        "logits": torch.cat(logits).numpy(), "raw_target": torch.cat(raws).numpy(),
        "reconstruction": torch.cat(recons).numpy(), "latent": torch.cat(latents).numpy(),
    }
    arrays["y_pred"] = arrays["probabilities"].argmax(1)
    if alphas: arrays["alpha"] = torch.cat(alphas).numpy()
    return endpoint_metrics(arrays["y_true"], arrays["probabilities"], arrays["reconstruction"],
                            arrays["raw_target"], arrays["latent"]), arrays


@torch.no_grad()
def historical_forward_check(model, batch, cfg, dm, device) -> float:
    if model.variant != "wtam_full": return float("nan")
    head = cfg["model"]["head"]
    ref = SpectralAEClassifier(
        model.ae.cfg, num_classes=dm.num_classes,
        head_hidden=tuple(head.get("head_hidden", [64,32])),
        head_dropout=float(head.get("head_dropout", 0.0)),
        recon_weight=float(head.get("recon_weight", 1.0)), ce_weight=float(head.get("ce_weight", 1.0)),
        recon_kind=head.get("recon_kind", "mse"), wavelet_attn=copy.deepcopy(model.wavelet_attn),
    ).to(device)
    ref.ae.load_state_dict(model.ae.state_dict()); ref.head.load_state_dict(model.head.state_dict())
    _, x, _ = prepare_batch(batch, device)
    a = model(x, compute_loss=False)
    b = ref(x, compute_loss=False)
    return max(float((a[k]-b[k]).abs().max().cpu()) for k in ("logits","recon","latent","level_alpha"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dwt_attn.yaml")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--band-region", required=True)
    ap.add_argument("--days", nargs="+", type=int)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--exp-idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="results/wtam_definitive/smoke")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--smoke-interventions", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    override_band_region(cfg, args.band_region)
    if args.days is not None: cfg.setdefault("data", {})["date_filters"] = args.days
    cfg["data"].setdefault("preprocess", {}).setdefault("dwt", {})["enabled"] = args.variant != "raw_ae"
    seed_everything(args.seed, workers=True)
    dm = build_datamodule(cfg)
    if args.variant == "raw_ae":
        stats = dm.standardizer_stats
        if not isinstance(stats, dict): raise RuntimeError("Raw mode requires training-fitted standardizer")
        for name in ("train_ds", "val_ds", "test_ds"):
            setattr(dm, name, RawTargetDataset(getattr(dm, name), stats["mean"], stats["std"]))

    model = make_model(cfg, dm, args.variant)
    param_count = sum(p.numel() for p in model.parameters())
    opt = cfg.get("optimizer", {}); sched = cfg.get("scheduler", {}); log = cfg.get("logging", {})
    lit = LitClassifier(model, lr=float(opt.get("lr", .001)), weight_decay=float(opt.get("weight_decay", .01)),
                        scheduler_name=sched.get("name", "reduce_on_plateau"),
                        scheduler_kwargs=sched.get("kwargs", {}), monitor_metric=sched.get("monitor", "val/loss"))
    lit.best_monitor_mode = log.get("mode", "min")
    condition = args.band_region + (("_days_" + "-".join(map(str,args.days))) if args.days else "")
    run_dir = Path(args.outdir) / args.variant / condition / f"run_{args.exp_idx:02d}_seed_{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True)
    monitor=log.get("monitor", "val/loss"); mode=log.get("mode", "min")
    checkpoint = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor=monitor, mode=mode,
                                 save_top_k=1, save_last=False, auto_insert_metric_name=False)
    early = EarlyStopping(monitor=monitor, mode=mode, patience=int(log.get("patience",50)),
                          min_delta=float(log.get("min_delta",1e-4)))
    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = Trainer(max_epochs=args.max_epochs or int(cfg.get("trainer",{}).get("max_epochs",300)),
                      accelerator="gpu" if requested_device.type=="cuda" else "cpu",
                      devices=[requested_device.index or 0] if requested_device.type=="cuda" else 1,
                      precision=cfg.get("trainer",{}).get("precision","32-true"),
                      gradient_clip_val=float(cfg.get("trainer",{}).get("grad_clip",1.0)),
                      callbacks=[checkpoint, early], logger=False, enable_progress_bar=True,
                      log_every_n_steps=int(cfg.get("trainer",{}).get("log_every_n_steps",10)))
    trainer.fit(lit, datamodule=dm)
    best_path = Path(checkpoint.best_model_path)
    state = torch.load(best_path, map_location="cpu")
    lit.load_state_dict(state["state_dict"])
    device = requested_device
    model.to(device)
    metrics, arrays = evaluate(model, dm.test_dataloader(), device)
    first_batch = next(iter(dm.test_dataloader()))
    parity = historical_forward_check(model, first_batch, cfg, dm, device)
    metrics.update({
        "parameter_count": int(param_count), "best_epoch": int(state.get("epoch", -1)),
        "best_validation_metric": float(checkpoint.best_model_score.cpu()),
        "historical_forward_max_abs_diff": parity,
        "alpha_sum_max_abs_error": float(np.max(np.abs(arrays["alpha"].sum(1)-1))) if "alpha" in arrays else float("nan"),
        "alpha_sample_std_max": float(np.max(arrays["alpha"].std(0))) if "alpha" in arrays else float("nan"),
    })
    np.savez_compressed(run_dir / "test_outputs.npz", **arrays)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True)+"\n")
    provenance = {
        "repository": str(Path.cwd()), "variant": args.variant, "band_region": args.band_region,
        "days": args.days, "exp_idx": args.exp_idx, "seed": args.seed,
        "split_seed": cfg["data"].get("random_state", cfg["data"].get("seed",42)),
        "fixed_split_indices_path": cfg["data"].get("fixed_split_indices_path"),
        "split_sizes": {"train":len(dm.train_ds),"validation":len(dm.val_ds),"test":len(dm.test_ds)},
        "class_mapping": getattr(dm,"label_mapping",None), "raw_bands":dm.band_count,
        "dwt_level_names":getattr(dm,"feature_names",None) if args.variant != "raw_ae" else None,
        "dwt_level_lengths":dm.dwt_level_lengths, "latent_dim":cfg["model"]["cfg"].get("latent_dim"),
        "parameter_count":param_count, "checkpoint_monitor":monitor, "checkpoint_mode":mode,
        "best_checkpoint":str(best_path.resolve()), "config":str(cfg_path.resolve()),
        "reconstruction_target":"unstandardized retained raw spectrum",
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if args.smoke_interventions and args.variant == "wtam_full":
        uniform = torch.full((len(dm.dwt_level_lengths),), 1/len(dm.dwt_level_lengths))
        intervention = {}
        intervention["uniform"], _ = evaluate(model, dm.test_dataloader(), device, alpha_override=uniform)
        intervention["level0_occlusion"], _ = evaluate(model, dm.test_dataloader(), device,
                                                        corrupt_level=0, corruption="occlusion")
        torch.manual_seed(20260911)
        intervention["level0_gaussian_sigma_0.25"], corrupted = evaluate(
            model, dm.test_dataloader(), device, corrupt_level=0, corruption="gaussian", severity=.25)
        if "alpha" in arrays and "alpha" in corrupted:
            intervention["level0_gaussian_sigma_0.25"]["mean_alpha_shift"] = (corrupted["alpha"]-arrays["alpha"]).mean(0).tolist()
        (run_dir / "smoke_interventions.json").write_text(json.dumps(intervention, indent=2, allow_nan=True)+"\n")
    print(json.dumps({"run_dir":str(run_dir), "metrics":metrics, "provenance":provenance}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

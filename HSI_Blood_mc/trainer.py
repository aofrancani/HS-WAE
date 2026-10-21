from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torchmetrics as tm


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        scheduler_name: str = "onecycle",
        scheduler_kwargs: Optional[dict] = None,
        monitor_metric: str = "val/acc",  # used by ReduceLROnPlateau
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()
        self._scheduler_name = scheduler_name.lower()
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._monitor_metric = monitor_metric

        # Get num_classes from model's head
        self._num_classes = getattr(model, "num_classes", None)
        if self._num_classes is None:
            try:
                self._num_classes = model.head[-1].out_features
            except:
                self._num_classes = None  # Will be set from first batch

        # Initialize metrics immediately if num_classes is known
        self.acc = None
        self.f1 = None
        self.au = None
        self.prec = None
        self.recall = None
        self.fpr = None
        self.cm = None
        if self._num_classes is not None:
            self._init_metrics(self._num_classes, self.device)

    # -------- utilities --------
    def _init_metrics(self, num_classes: int, device: torch.device):
        """Initialize metrics with proper device placement"""
        if num_classes == 2:
            self.acc = tm.classification.BinaryAccuracy().to(device)
            self.f1 = tm.classification.BinaryF1Score().to(device)
            self.au = tm.classification.BinaryAUROC().to(device)
            self.prec = tm.classification.BinaryPrecision().to(device)
            self.recall = tm.classification.BinaryRecall().to(device)
            self.cm = tm.classification.ConfusionMatrix(
                task="binary", num_classes=2
            ).to(device)
        else:
            self.acc = tm.classification.MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.f1 = tm.classification.MulticlassF1Score(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.au = tm.classification.MulticlassAUROC(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.prec = tm.classification.MulticlassPrecision(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.recall = tm.classification.MulticlassRecall(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.cm = tm.classification.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ).to(device)

    def _ensure_num_classes_and_metrics(self, logits: torch.Tensor):
        if self._num_classes is None:
            self._num_classes = int(logits.shape[1])
        if self.acc is None:
            self._init_metrics(self._num_classes, logits.device)
        else:
            for m in [self.acc, self.f1, self.au, self.prec, self.recall, self.cm]:
                if m is not None:
                    m.to(logits.device)

    # -------- forward --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out  # assume it's already logits

    # -------- shared step --------
    def _step(self, batch, stage: str):
        # batch can be:
        #  - (x, y)                when DWT disabled
        #  - (x_raw, levels, y)    when DWT enabled
        if len(batch) == 2:
            x, y = batch
            x_recon_target = None
        elif len(batch) == 3:
            x_raw, x, y = batch  # x is the DWT token list (levels)
            x_recon_target = x_raw  # raw spectrum target for recon
        else:
            raise ValueError(f"Unexpected batch format (len={len(batch)}).")

        # Ensure batch is on the same device as the module
        device = self.device
        if torch.is_tensor(x):
            x = x.to(device)
        elif isinstance(x, (list, tuple)):
            x = [xi.to(device) for xi in x]
        y = y.to(device).long()
        if x_recon_target is not None:
            x_recon_target = x_recon_target.to(device)

        # Call model: prefer (x, y, compute_loss=True), fall back to (x,)
        try:
            out = self.model(x, y=y, x_recon_target=x_recon_target, compute_loss=True)
        except TypeError:
            # fallback for models that don't support that signature
            out = self.model(x)

        # Unify outputs
        if isinstance(out, dict):
            logits = out["logits"]
            loss = out.get("loss")
            ce = out.get("ce_loss")
            recon = out.get("recon_loss")
        else:
            logits = out
            loss = None
            ce = None
            recon = None

        # ALWAYS ensure metrics exist
        self._ensure_num_classes_and_metrics(logits)

        # If the model didn't return a loss, compute CE here
        if loss is None:
            loss = self.criterion(logits, y)
            ce = ce if ce is not None else loss

        # Metrics - ensure they're on the correct device and properly updated
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Move metrics to correct device if needed
        device = logits.device
        self.acc = self.acc.to(device)
        self.f1 = self.f1.to(device)
        self.au = self.au.to(device)
        self.prec = self.prec.to(device)
        self.recall = self.recall.to(device)
        if self.cm is not None:
            self.cm = self.cm.to(device)

        # Update metrics
        self.acc(preds, y)
        self.f1(preds, y)
        self.prec(preds, y)
        self.recall(preds, y)

        if self._num_classes == 2:
            self.au(probs[:, 1], y)
        else:
            self.au(probs, y)

        if self.cm is not None:
            self.cm(preds, y)

        # Logs
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if ce is not None:
            self.log(f"{stage}/ce_loss", ce, on_step=False, on_epoch=True)
        if recon is not None:
            self.log(f"{stage}/rc_loss", recon, on_step=False, on_epoch=True)

        return loss

    @staticmethod
    def _macro_fpr_from_confmat(cm: torch.Tensor) -> torch.Tensor:
        """
        Compute FPR from confusion matrix.
        - Binary: cm shape [2,2] → FP/(FP+TN)
        - Multiclass: macro-average over one-vs-rest
        """
        eps = 1e-8
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError("Confusion matrix must be square [K,K].")

        K = cm.shape[0]
        if K == 2:
            tn = cm[0, 0]
            fp = cm[0, 1]
            return fp / (fp + tn + eps)

        # Multiclass macro-FPR
        total = cm.sum()
        fprs = []
        for i in range(K):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            fprs.append(fp / (fp + tn + eps))
        return torch.stack(fprs).mean()

    def _reset_metrics(self):
        """Helper to reset all metrics"""
        if self.acc is not None:
            self.acc.reset()
            self.f1.reset()
            self.au.reset()
            self.prec.reset()
            self.recall.reset()
            if self.cm is not None:
                self.cm.reset()

    def on_train_start(self):
        """Reset metrics at start of training"""
        self._reset_metrics()

    def on_validation_start(self):
        """Reset metrics at start of validation"""
        self._reset_metrics()

    def on_test_start(self):
        """Reset metrics at start of test"""
        self._reset_metrics()

    def training_step(self, batch, batch_idx):
        """Ensure metrics are on correct device before training step."""
        self.model.train()

        # support both (x,y) and (x_raw, x, y)
        x = batch[1] if len(batch) == 3 else batch[0]
        device = self._batch_device(x)
        device = self._batch_device(x)

        if self.acc is not None:
            self.acc = self.acc.to(device)
            self.f1 = self.f1.to(device)
            self.au = self.au.to(device)
            self.prec = self.prec.to(device)
            self.recall = self.recall.to(device)
            if self.cm is not None:
                self.cm = self.cm.to(device)

        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            return self._step(batch, "test")

    def on_validation_epoch_end(self):
        if self.acc is None:
            return

        # Only compute metrics if they have been updated
        metrics = {}

        try:
            metrics["acc"] = self.acc.compute()
            metrics["f1"] = self.f1.compute()
            metrics["auroc"] = self.au.compute()
            metrics["precision"] = self.prec.compute()
            metrics["recall"] = self.recall.compute()

            # Log metrics
            for name, value in metrics.items():
                self.log(f"val/{name}", value, prog_bar=(name in ["acc", "f1"]))

        finally:
            # Always reset metrics at the end
            self._reset_metrics()

    def on_test_epoch_end(self):
        if self.acc is None:
            return

        acc = self.acc.compute()
        self.acc.reset()

        f1 = self.f1.compute()
        self.f1.reset()

        au = self.au.compute()
        self.au.reset()

        prec = self.prec.compute()
        self.prec.reset()

        rec = self.recall.compute()
        self.recall.reset()

        fpr = torch.tensor(float("nan"), device=acc.device)

        # check whether a logger exists
        has_logger = self.logger is not None and hasattr(self.logger, "experiment")

        # Confusion matrix
        if self.cm is not None:
            cm = self.cm.compute().detach().cpu()
            self.cm.reset()
            fpr = self._macro_fpr_from_confmat(cm.float()).to(acc.device)

            if has_logger:
                fig, ax = plt.subplots(figsize=(5, 5))
                sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Test Confusion Matrix")
                self.logger.experiment.add_figure(
                    "test/confusion_matrix", fig, self.current_epoch
                )
                plt.close(fig)

        self.log("test/acc", acc, prog_bar=True)
        self.log("test/f1", f1, prog_bar=True)
        self.log("test/auroc", au, prog_bar=False)
        self.log("test/precision", prec, prog_bar=False)
        self.log("test/recall", rec, prog_bar=False)
        self.log("test/fpr", fpr, prog_bar=False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        name = self._scheduler_name
        kw = dict(self._scheduler_kwargs)  # copy so we can mutate

        # Convenience values
        max_epochs = self.trainer.max_epochs if self.trainer else kw.pop("epochs", 100)
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // max_epochs
            if self.trainer and self.trainer.max_epochs > 0
            else kw.pop("steps_per_epoch", 1000)
        )

        # Build scheduler by name
        if name == "onecycle":
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.hparams.lr,
                epochs=max_epochs,
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=kw.get("pct_start", 0.1),
                anneal_strategy=kw.get("anneal_strategy", "cos"),
                div_factor=kw.get("div_factor", 25.0),
                final_div_factor=kw.get("final_div_factor", 1e4),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step"},
            }

        elif name == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max_epochs,
                eta_min=kw.get("eta_min", 0.0),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
            }

        elif name == "cosine_warm_restarts":
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=kw.get("T_0", 10),
                T_mult=kw.get("T_mult", 2),
                eta_min=kw.get("eta_min", 0.0),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
            }

        elif name == "reduce_on_plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=kw.get("mode", "max"),
                factor=kw.get("factor", 0.5),
                patience=kw.get("patience", 10),
                threshold=kw.get("threshold", 1e-4),
                min_lr=kw.get("min_lr", 0.0),
                cooldown=kw.get("cooldown", 0),
            )
            # Lightning needs 'monitor' for plateau schedulers:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": self._monitor_metric,
                    "interval": "epoch",
                },
            }

        elif name == "step":
            sch = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=kw.get("step_size", 30),
                gamma=kw.get("gamma", 0.1),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
            }

        elif name == "multistep":
            sch = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=kw.get("milestones", [60, 120, 180]),
                gamma=kw.get("gamma", 0.1),
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
            }

        elif name == "warmup_cosine":
            # Linear warmup for N epochs, then cosine
            warmup_epochs = kw.get("warmup_epochs", 5)
            eta_min = kw.get("eta_min", 0.0)

            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=kw.get("start_factor", 0.1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_epochs - warmup_epochs, eta_min=eta_min
            )
            sch = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
            }

        else:
            # No scheduler
            return {"optimizer": opt}

    @staticmethod
    def _batch_device(x) -> torch.device:
        """
        Returns the device for a batch input that may be:
          - a Tensor
          - a list/tuple of Tensors (DWT token mode)
        """
        if torch.is_tensor(x):
            return x.device
        if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
            return x[0].device
        # fallback (shouldn't happen in normal training)
        return torch.device("cpu")


def build_trainer(
    cfg: Dict[str, Any],
    logger: TensorBoardLogger,
    *,
    devices=None,
    precision_override=None,
    max_epochs_override=None,
):
    tr_cfg = cfg.get("trainer", {})
    trainer = Trainer(
        max_epochs=(
            max_epochs_override
            if max_epochs_override is not None
            else tr_cfg.get("max_epochs", 100)
        ),
        accelerator=tr_cfg.get("accelerator", "auto"),
        devices=devices,
        precision=(
            precision_override
            if precision_override is not None
            else tr_cfg.get("precision", "16-mixed")
        ),
        gradient_clip_val=tr_cfg.get("grad_clip", 1.0),
        log_every_n_steps=tr_cfg.get("log_every_n_steps", 10),
        logger=logger,
        callbacks=(
            build_logger_and_callbacks.__wrapped_callbacks
            if hasattr(build_logger_and_callbacks, "__wrapped_callbacks")
            else None
        ),
    )
    return trainer


def build_logger_and_callbacks(cfg: Dict[str, Any], run_name: str, save_dir: Path):
    log_cfg = cfg.get("logging", {})
    tb = TensorBoardLogger(save_dir=save_dir, name="", default_hp_metric=False)

    monitor = log_cfg.get("monitor", "val/acc")
    mode = log_cfg.get("mode", "max")

    ckpt = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=log_cfg.get("save_top_k", 1),
        save_last=log_cfg.get("save_last", True),
        filename=log_cfg.get("filename", f"{run_name}" + "-{epoch:02d}-{val_acc:.3f}"),
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=log_cfg.get("patience", 15),
        min_delta=log_cfg.get("min_delta", 1e-4),
    )
    lrmon = LearningRateMonitor(logging_interval=log_cfg.get("lr_log_interval", "step"))
    rich = RichProgressBar()
    return tb, [ckpt, early, lrmon, rich]

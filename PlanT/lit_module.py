import logging
import math
import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torchmetrics import Accuracy

from model import HFLM

from plant_variables import PlanTVariables

logger = logging.getLogger(__name__)

# Samples with target_speed below this (m/s) get STOP_SPEED_LOSS_WEIGHT on CE.
_STOP_SPEED_THR_MPS = 0.5


def _cosine_warmup_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.plant_variables = PlanTVariables()

        self.wp_rep = self.cfg.model.waypoints.representation

        # self.last_epoch = 0
        self.cfg_train = self.cfg.model.training
        self.model = HFLM(self.cfg.model.network, self.cfg)

        # Loss functions
        self.criterion_speed = nn.CrossEntropyLoss() # weight & label smoothing
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)

        # Metrics
        self.metrics_forecasting_acc = nn.ModuleList(
            [Accuracy(task="multiclass", num_classes=classes) for classes in self.model.vocab_size]
        )

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.model.training)
        sched_name = str(self.cfg.get("lr_scheduler", "multistep")).lower()
        if sched_name == "cosine_warmup":
            total_steps = int(self.cfg.get("total_training_steps", 0) or 0)
            warmup_steps = int(self.cfg.get("warmup_steps", 0) or 0)
            if total_steps <= 0:
                raise ValueError(
                    "lr_scheduler=cosine_warmup requires cfg.total_training_steps > 0"
                )
            warmup_steps = max(1, min(warmup_steps, total_steps - 1))
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: _cosine_warmup_lambda(
                    step, warmup_steps, total_steps
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        waypoints_batch = batch["waypoints"]
        path_batch = batch["route"]
        path_batch = path_batch[..., :self.cfg.model.waypoints.path_len, :]

        targetspeed_batch = batch["target_speed"]

        logits, targets, pred_plan, _ = self(batch)

        losses = {}

        (pred_path, pred_wps, pred_speed) = pred_plan

        if pred_wps is not None:
            losses["loss_wp"] = F.l1_loss(pred_wps, waypoints_batch)

        if pred_path is not None:
            losses["loss_path"] = self._path_loss(pred_path, path_batch)

        if pred_speed is not None:
            target_speeds = torch.tensor(self.plant_variables.target_speeds, device=targetspeed_batch.device)
            brake = torch.zeros_like(targetspeed_batch, dtype=torch.bool, device=targetspeed_batch.device)
            twohot_targs = self.get_two_hot_encoding(targetspeed_batch, target_speeds, brake)

            # Soft two-hot targets: CE with class probabilities (float N×C).
            # Manual form is robust across torch builds that reject multi-dim CE targets.
            # Optional per-bin class weights (H2) scale each bin's contribution.
            log_probs = F.log_softmax(pred_speed, dim=-1)
            cw = self._speed_class_weights_tensor(
                device=log_probs.device, dtype=log_probs.dtype, n_classes=log_probs.shape[-1]
            )
            if cw is not None:
                per_sample_ce = -(twohot_targs * log_probs * cw).sum(dim=-1)
            else:
                per_sample_ce = -(twohot_targs * log_probs).sum(dim=-1)
            losses["loss_egospeed"] = self._weighted_egospeed_ce(
                per_sample_ce, targetspeed_batch
            )

        losses_forecast = []
        for i in range(len(logits)):
            t = targets[i].squeeze()
            if (t != -999).any():
                losses_forecast.append(
                    torch.mean(self.criterion_forecast(logits[i], t))
                )
            else:
                losses_forecast.append(logits[i].new_zeros(()))
        losses["loss_forecast"] = torch.mean(torch.stack(losses_forecast))

        aux_loss = self._aux_sign_presence_loss(batch)
        if aux_loss is not None:
            losses["loss_sign_presence"] = aux_loss

        for name, loss in losses.items():
            self.log(
                f"train/{name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

        weights = {
            "loss_wp": self.cfg.model.waypoints.get("wp_weight", 1),
            "loss_forecast": self.cfg.model.pre_training.get("forecastLoss_weight", 0),
            "loss_path": self.cfg.model.waypoints.get("path_weight", 1),
            "loss_egospeed": self.cfg.model.waypoints.get("speed_weight", 1),
            "loss_sign_presence": self.cfg.model.training.get("aux_sign_presence_weight", 0.0),
        }

        loss_all = sum([loss*weights[name] for name, loss in losses.items()])

        self.log(
            "train/loss_all",
            loss_all,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.cfg.gpus > 1,
            batch_size=self.cfg.model.training.batch_size,
        )

        for i, name in enumerate(
                ["x", "y", "yaw", "speed"]
            ):
                if i > self.model.num_attributes:
                    break
                self.log(
                    f"train/loss_{name}",
                    losses_forecast[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

                mask = targets[i].squeeze() != -999
                # Skip Accuracy on empty selection (e.g. sign-only tokens with no
                # forecast targets) — torchmetrics cannot reshape a 0-length tensor.
                if mask.any():
                    self.metrics_forecasting_acc[i](
                        logits[i][mask], targets[i][mask].squeeze()
                    )
                    self.log(
                        f"train/acc_{name}",
                        self.metrics_forecasting_acc[i],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=self.cfg.gpus > 1,
                        batch_size=self.cfg.model.training.batch_size,
                    )

        return loss_all

    def validation_step(self, batch, batch_idx):
        # Same losses as training_step, logged under val/*.
        waypoints_batch = batch["waypoints"]
        path_batch = batch["route"]
        path_batch = path_batch[..., : self.cfg.model.waypoints.path_len, :]
        targetspeed_batch = batch["target_speed"]

        logits, targets, pred_plan, _ = self(batch)
        (pred_path, pred_wps, pred_speed) = pred_plan
        losses = {}
        if pred_wps is not None:
            losses["loss_wp"] = F.l1_loss(pred_wps, waypoints_batch)
        if pred_path is not None:
            losses["loss_path"] = self._path_loss(pred_path, path_batch)
        if pred_speed is not None:
            target_speeds = torch.tensor(
                self.plant_variables.target_speeds, device=targetspeed_batch.device
            )
            brake = torch.zeros_like(
                targetspeed_batch, dtype=torch.bool, device=targetspeed_batch.device
            )
            twohot_targs = self.get_two_hot_encoding(
                targetspeed_batch, target_speeds, brake
            )
            log_probs = F.log_softmax(pred_speed, dim=-1)
            cw = self._speed_class_weights_tensor(
                device=log_probs.device, dtype=log_probs.dtype, n_classes=log_probs.shape[-1]
            )
            if cw is not None:
                per_sample_ce = -(twohot_targs * log_probs * cw).sum(dim=-1)
            else:
                per_sample_ce = -(twohot_targs * log_probs).sum(dim=-1)
            losses["loss_egospeed"] = self._weighted_egospeed_ce(
                per_sample_ce, targetspeed_batch
            )

        losses_forecast = []
        for i in range(len(logits)):
            t = targets[i].squeeze()
            if (t != -999).any():
                losses_forecast.append(
                    torch.mean(self.criterion_forecast(logits[i], t))
                )
            else:
                losses_forecast.append(logits[i].new_zeros(()))
        losses["loss_forecast"] = torch.mean(torch.stack(losses_forecast))

        aux_loss = self._aux_sign_presence_loss(batch)
        if aux_loss is not None:
            losses["loss_sign_presence"] = aux_loss

        weights = {
            "loss_wp": self.cfg.model.waypoints.get("wp_weight", 1),
            "loss_forecast": self.cfg.model.pre_training.get("forecastLoss_weight", 0),
            "loss_path": self.cfg.model.waypoints.get("path_weight", 1),
            "loss_egospeed": self.cfg.model.waypoints.get("speed_weight", 1),
            "loss_sign_presence": self.cfg.model.training.get("aux_sign_presence_weight", 0.0),
        }
        loss_all = sum(loss * weights[name] for name, loss in losses.items())

        for name, loss in losses.items():
            self.log(
                f"val/{name}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )
        self.log(
            "val/loss_all",
            loss_all,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.cfg.gpus > 1,
            batch_size=self.cfg.model.training.batch_size,
        )
        return loss_all

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )

    def _path_loss(self, pred_path, path_batch):
        """L1 path loss, optionally up-weighted per-sample by how much lateral
        deviation the ground-truth path itself requires (H2-detour: obstacle
        avoidance is a handful of frames per route with a real 2-3m swerve,
        drowned out by the ~96% of frames that are near-straight -- a uniform
        global path_weight scales both equally and can't fix that imbalance).

        ``model.waypoints.lateral_reweight_alpha`` (default 0 = disabled,
        exact prior behaviour): per-sample weight is
        ``1 + alpha * max(|path_y|)`` over the sample's path points, so a
        route-segment with no lateral need keeps weight 1 and a segment
        needing e.g. 3m of swerve gets ``1 + 3*alpha``.
        """
        alpha = float(self.cfg.model.waypoints.get("lateral_reweight_alpha", 0.0))
        if alpha <= 0.0:
            return F.l1_loss(pred_path, path_batch)
        per_elem = F.l1_loss(pred_path, path_batch, reduction="none")  # (B, path_len, 2)
        per_sample = per_elem.mean(dim=tuple(range(1, per_elem.dim())))  # (B,)
        lateral_dev = path_batch[..., 1].abs().amax(dim=-1)  # (B,) max |y| over path points
        sample_w = 1.0 + alpha * lateral_dev
        return (per_sample * sample_w).sum() / sample_w.sum()

    def _aux_sign_presence_loss(self, batch):
        """H4: BCE for the auxiliary sign-presence probe (see model.py
        aux_sign_presence / dataset.py sign_present_now). None when disabled."""
        logit = getattr(self.model, "_last_aux_sign_logit", None)
        if logit is None or "sign_present_now" not in batch:
            return None
        target = batch["sign_present_now"].reshape(-1).to(dtype=logit.dtype)
        return F.binary_cross_entropy_with_logits(logit, target)

    def _stop_speed_loss_weight(self) -> float:
        """Per-sample multiplier for stop targets (target_speed≈0).

        Prefer Hydra ``model.training.stop_speed_loss_weight``; fall back to env
        ``STOP_SPEED_LOSS_WEIGHT``. Default 1.0 preserves existing FT behaviour.
        """
        cfg_w = self.cfg.model.training.get("stop_speed_loss_weight", None)
        if cfg_w is not None:
            return float(cfg_w)
        return float(os.environ.get("STOP_SPEED_LOSS_WEIGHT", "1"))

    def _speed_class_weights_tensor(self, device, dtype, n_classes: int):
        """Optional per-bin weights for soft two-hot ego-speed CE (H2).

        Prefer Hydra ``model.training.speed_class_weights`` (list of length C);
        fall back to env ``SPEED_CLASS_WEIGHTS`` as comma-separated floats.
        ``None`` / empty → uniform (legacy behaviour).
        """
        cfg_w = self.cfg.model.training.get("speed_class_weights", None)
        if cfg_w is None or cfg_w == "" or cfg_w is False:
            env = os.environ.get("SPEED_CLASS_WEIGHTS", "").strip()
            if not env:
                return None
            weights = [float(x) for x in env.split(",") if x.strip() != ""]
        else:
            weights = [float(x) for x in list(cfg_w)]
        if len(weights) != n_classes:
            raise ValueError(
                f"speed_class_weights length {len(weights)} != n_classes {n_classes}"
            )
        return torch.tensor(weights, device=device, dtype=dtype)

    def _weighted_egospeed_ce(self, per_sample_ce, targetspeed_batch):
        """Mean CE; upweight samples with target_speed < 0.5 m/s when weight≠1."""
        w = self._stop_speed_loss_weight()
        if abs(w - 1.0) < 1e-12:
            return per_sample_ce.mean()
        ts = targetspeed_batch.reshape(-1).to(dtype=per_sample_ce.dtype)
        sample_w = torch.where(
            ts < _STOP_SPEED_THR_MPS,
            per_sample_ce.new_full((), w),
            per_sample_ce.new_ones(()),
        )
        return (per_sample_ce * sample_w).mean()

    # # Torch version of get_two_hot_encoding in data.py which also works with batches
    # # target_speed Bx1, config_target_speeds C, brake Bx1
    def get_two_hot_encoding(self, target_speed, config_target_speeds, brake):
        if torch.any(target_speed < 0):
            raise ValueError('Target speed value must be non-negative for two-hot encoding.')
        
        # Calculate two-hot labes as described in https://arxiv.org/pdf/2403.03950.pdf
        labels = torch.zeros(target_speed.shape[0], len(config_target_speeds), device=target_speed.device)

        # Compare each target speed with the config speeds
        diffs = (config_target_speeds > target_speed[:, None]).float()
        vals, idxs = diffs.max(dim=1)

        # Doing this calculation for all rows and fixing the exceptions later
        upper_ind = idxs
        lower_ind = idxs - 1
        upper_val = config_target_speeds[upper_ind]
        lower_val = config_target_speeds[lower_ind]

        lower_weight = (upper_val-target_speed) / (upper_val - lower_val)
        upper_weight = (target_speed-lower_val) / (upper_val - lower_val)

        labels[torch.arange(target_speed.shape[0]), lower_ind] = lower_weight
        labels[torch.arange(target_speed.shape[0]), upper_ind] = upper_weight

        # Clear rows where brake or no config value greater than target speed
        labels[torch.logical_or(brake, vals==0)] = 0

        # Set brake rows to 0
        labels[brake, 0] = 1.0

        # Set rows with max speed and without brake pressed to last bin
        labels[torch.logical_and(vals==0, ~brake), -1] = 1.0

        return labels

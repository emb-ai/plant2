#!/usr/bin/env python3
"""Fine-tune PlanT2 from a pretrained .ckpt with train/val loaders.

Differences vs lit_train.py:
- loads pretrained weights only (does not resume optimizer/epoch)
- builds a val DataLoader from $DS_VAL/data
- validates every epoch and saves a checkpoint every N epochs (default 5)
- wandb offline by default (WANDB_MODE)
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from dataset import PlanTDataset, generate_batch
from lit_module import LitHFLM
from util.logging import setup_logging


def _prepare_fresh_parameters(model, ckpt_path) -> None:
    """Give the parameters the checkpoint does not carry a fighting chance.

    They start from noise inside a pretrained trunk and, with one learning rate
    for everything, an AdamW step moves a weight by about `lr`; over a 12-epoch
    finetune that is roughly the initialisation scale itself, and a sign class
    present in a tenth of the frames gets a tenth of that. Two env knobs:

      INIT_SIGN_FROM_STOP=1   seed every PDD sign tok_emb from the trained
                              stop_sign one (class 4) instead of random init
      NEW_PARAM_LR_MULT=<x>   put the checkpoint-missing parameters in their own
                              optimiser group at lr*x

    Both default to off; the fresh-parameter set is computed either way so the
    log always states what the checkpoint failed to provide.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    fresh = {k for k in model.state_dict() if k not in sd}
    # HFLM.configure_optimizers works in its own namespace, without the prefix.
    model.model.fresh_params = {k[len("model."):] for k in fresh if k.startswith("model.")}
    print(f"fresh parameters (absent from the checkpoint): {len(fresh)} tensors")

    if os.environ.get("INIT_SIGN_FROM_STOP", "0").strip() in ("1", "true", "True"):
        from plant_variables import PDD_OBJECT_CLASS_START

        src = model.model.tok_emb[4]  # stop_sign: a small static object, like every PDD plate
        seeded = 0
        with torch.no_grad():
            for i in range(PDD_OBJECT_CLASS_START, len(model.model.tok_emb)):
                if f"tok_emb.{i}.weight" not in model.model.fresh_params:
                    continue
                dst = model.model.tok_emb[i]
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
                # Identical copies would still diverge (each class sees its own
                # frames), but a little jitter removes the tie from step one.
                dst.weight.add_(torch.randn_like(dst.weight) * 0.02 * src.weight.std())
                seeded += 1
        print(f"seeded {seeded} PDD sign tok_emb from the trained stop_sign layer")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    os.environ.setdefault("WANDB_MODE", "offline")

    shared_dict = None
    if cfg.use_caching:
        from diskcache import Cache

        tmp_folder = os.environ.get("DS_LOCAL")
        print("Tmp folder for dataset cache: ", tmp_folder)
        if tmp_folder:
            Path(tmp_folder).mkdir(parents=True, exist_ok=True)
            cache_gb = float(os.environ.get("CACHE_SIZE_GB", "700"))
            size_limit = int(cache_gb * 1024**3)
            print(f"diskcache size_limit={cache_gb:g} GiB ({size_limit} bytes)")
            shared_dict = Cache(directory=tmp_folder, size_limit=size_limit)

    seed = int(os.environ.get("SEED", "1"))
    print("The current seed is", seed)
    pl.seed_everything(seed)
    setup_logging(cfg)

    log_path = cfg.model.training.log_path
    Path(log_path).mkdir(parents=True, exist_ok=True)
    csvlogger = CSVLogger(log_path, "CSVLogger")
    wandb.init(
        project=cfg.exp_folder_name,
        name="PlanT_2_ft_" + os.environ.get("CHECKPOINT_ADDON", "ft") + "_" + str(seed),
        mode=os.environ.get("WANDB_MODE", "offline"),
    )
    wandblogger = WandbLogger(
        project=cfg.exp_folder_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=None,
    )
    Path(f"{log_path}/TBLogger").mkdir(parents=True, exist_ok=True)
    tblogger = TensorBoardLogger(log_path, name="TBLogger")

    ckpt_in = cfg.resume_path
    if not ckpt_in or not Path(ckpt_in).is_file():
        raise FileNotFoundError(f"pretrained checkpoint missing: {ckpt_in!r}")

    addon = os.environ.get("CHECKPOINT_ADDON", "ft")
    # Per-run subdirectory so LR sweeps do not clobber each other.
    ckpt_dir = Path(cfg.user.working_dir) / "PlanT" / "checkpoints_ft" / addon
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = "{epoch:03d}_" + addon + "_" + str(seed)
    print("Checkpoint dir:", ckpt_dir)

    ckpt_every = int(os.environ.get("CKPT_EVERY_N_EPOCHS", "5"))
    print(
        f"ModelCheckpoint periodic every_n_epochs={ckpt_every} "
        f"(save_last=True) + best by val/loss_all (save_top_k=1)"
    )
    # Periodic snapshots every N epochs (and last).
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        dirpath=str(ckpt_dir),
        filename=out_path,
        save_last=True,
        every_n_epochs=ckpt_every,
        save_on_train_epoch_end=True,
    )
    last_name = f"last_ft_{addon}_{seed}"
    checkpoint_callback.CHECKPOINT_NAME_LAST = last_name

    # Keep the single best checkpoint by validation loss.
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_all",
        mode="min",
        save_top_k=1,
        dirpath=str(ckpt_dir),
        filename=f"best_{{epoch:03d}}_{addon}_{seed}",
        auto_insert_metric_name=False,
    )

    ds_train = os.environ["DS"] + "/data"
    ds_val = os.environ.get("DS_VAL", "")
    if not ds_val:
        raise RuntimeError("DS_VAL must point to the val dataset root (contains data/)")
    ds_val = ds_val.rstrip("/") + "/data"

    print("Train data:", ds_train)
    print("Val data:  ", ds_val)
    print("Pretrained:", ckpt_in)

    train_dataset = PlanTDataset(ds_train, cfg, shared_dict=shared_dict)
    val_dataset = PlanTDataset(ds_val, cfg, shared_dict=shared_dict)

    n_workers = int(cfg.model.training.num_workers)
    # Fork-after-CUDA breaks workers (cudaErrorInitializationError). Use spawn.
    loader_kw = dict(
        pin_memory=True,
        batch_size=cfg.model.training.batch_size,
        collate_fn=generate_batch,
        num_workers=n_workers,
    )
    if n_workers > 0:
        loader_kw["multiprocessing_context"] = torch.multiprocessing.get_context("spawn")
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

    # Cosine+warmup needs total/warmup step counts before configure_optimizers.
    sched_name = str(cfg.get("lr_scheduler", "multistep")).lower()
    if sched_name == "cosine_warmup":
        # DistributedSampler is attached by the trainer, after this point: the
        # loader still has its full length here, while each rank will really run
        # 1/world_size of it. Sizing the schedule off the full length stretched
        # it by the number of GPUs -- on 8 GPUs the 10% warmup covered 80% of the
        # run and the cosine decay never started.
        world_size = max(1, int(cfg.gpus))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / world_size))
        total_steps = steps_per_epoch * int(cfg.model.training.max_epochs)
        warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_steps))
        with open_dict(cfg):
            cfg.total_training_steps = total_steps
            cfg.warmup_steps = warmup_steps
        print(
            f"Scheduler cosine_warmup: steps/epoch={steps_per_epoch} "
            f"total={total_steps} warmup={warmup_steps} ({warmup_ratio:.0%}) "
            f"lr={cfg.model.training.learning_rate}"
        )

    # Fine-tune: load weights only, do not resume Lightning training state.
    # Load on CPU first so CUDA is not initialized before we configure DataLoader.
    print("Loading pretrained weights from", ckpt_in)
    model = LitHFLM.load_from_checkpoint(
        ckpt_in, cfg=cfg, strict=False, map_location="cpu"
    )
    model.cfg = cfg

    # The pretrained checkpoint holds tok_emb.0-6 only (car..emergency). Every
    # PDD sign class, sign_emb, speed_token and the ego-speed head are created
    # from scratch on each finetune, yet they share the trunk's learning rate.
    # Both knobs below default to off, so runs made before them stay comparable.
    _prepare_fresh_parameters(model, ckpt_in)

    callbacks = [
        checkpoint_callback,
        best_checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]

    n_gpus = int(cfg.gpus)
    trainer_kw = dict(
        callbacks=callbacks,
        accelerator="gpu",
        devices=n_gpus if n_gpus > 0 else 1,
        logger=[csvlogger, wandblogger, tblogger],
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        max_epochs=cfg.model.training.max_epochs,
        enable_progress_bar=True,
    )
    if n_gpus > 1:
        # A single-sign finetune leaves most of tok_emb (one embedding per
        # object type) out of the loss, and plain DDP refuses to run when any
        # parameter goes unused. DDP_STRATEGY=ddp_find_unused_parameters_true
        # allows it; the default stays plain ddp for full-mixture runs.
        trainer_kw["strategy"] = os.environ.get("DDP_STRATEGY", "ddp")

    trainer = Trainer(**trainer_kw)
    torch.set_float32_matmul_precision("high")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import sys
from itertools import product

import hydra
import matplotlib
import numpy as np
import omegaconf
import torch
import wandb
from loguru import logger

from sa_generalization.dinosaur.data import MOViFrameDataSet
from sa_generalization.dinosaur.model import Dinosaur

matplotlib.use("Agg")
import os
import time
from pathlib import Path

from sa_generalization.dinosaur.utils import (
    adjusted_rand_index,
    compute_total_loss,
    compute_upscaling_segmentations,
    evaluate_scheduled_weights,
    load_last_checkpoint,
    nearest_neighbor_scaling,
    plot_segmentations,
    save_checkpoint,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_from_checkpoint(cfg):
    # Set up logging and checkpointing
    torch.manual_seed(cfg["seed"])
    run_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_directory = Path(run_directory).joinpath("checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    _, last_checkpoint = load_last_checkpoint(checkpoint_directory)

    wandb.init(project="sa-dinosaur", dir=run_directory)

    model = Dinosaur(
        encoder_name=cfg["encoder_name"],
        n_input_features=cfg["n_features"],
        feature_size=cfg["feature_dim"],
        num_slots=cfg["n_slots"],
        sa_kwargs=cfg["sa_kwargs"],
        iters=cfg["iters"],
    )
    device = cfg["device"]
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-8, end_factor=1.0, total_iters=cfg["n_warmup_steps"]
    )
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=(0.5) ** (1 / cfg["decay_halflife"])
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[cfg["n_warmup_steps"]],
    )

    if last_checkpoint is None:
        rng = np.random.RandomState(cfg["seed"])
        i_restart = 0
        step = 0
    else:
        rng = last_checkpoint["rng"]
        model.load_state_dict(last_checkpoint["model"])
        optim.load_state_dict(last_checkpoint["optim"])
        lr_scheduler.load_state_dict(last_checkpoint["lr_scheduler"])
        i_restart, step = last_checkpoint["i_restart"], last_checkpoint["step"]

    device_count = torch.cuda.device_count()
    if device_count > 1 and cfg["use_multi_gpu"]:
        print(f"Using {device_count} GPUS")
        model = torch.nn.DataParallel(model)

    torch.manual_seed(rng.randint(0, 1e8))
    np.random.seed(rng.randint(0, 1e8))

    # Set up train and validation data
    print("Setting up datasets...")

    dataset = cfg["dataset"]
    encoder_name = cfg["encoder_name"]
    train_dset = MOViFrameDataSet(
        dataset,
        split="train",
        feature_names=(encoder_name, 0),
        load_to_memory=False,
        copy_to_temp=not cfg["debugging"],
        copy_timeout_h=cfg["copy_timeout_h"],
        seed=rng.randint(0, 1e8),
        n_whole_sequences=cfg["n_training_sequences"],
        max_objects=cfg["max_objects"],
    )
    val_dset = MOViFrameDataSet(
        dataset,
        dset_dir=None if cfg["debugging"] else train_dset.tmp_dir.name,
        split="validation",
        feature_names=(encoder_name, "video", "segmentations"),
        load_to_memory=False,
        copy_to_temp=False,
        seed=rng.randint(0, 1e8),
        n_whole_sequences=cfg["n_val_sequences"],
    )

    batch_size = cfg["batch_size"]
    train_loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    val_iter = iter(val_loader)

    t_begin = time.time()

    t_last_log, step_last_log = None, None
    last_ckpt_path = None
    last_ckpt_time = 0

    while True:
        for x, _ in tqdm(train_loader, total=len(train_loader)):
            x = x[:, 1:]  # Remove class token

            x = x.to(device)
            pred = model(x, is_features=True)
            loss_weights = evaluate_scheduled_weights(cfg["loss_schedule"], step)
            losses, loss = compute_total_loss(x, pred, loss_weights)
            loss = loss.mean(dim=0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clipping_norm"])
            optim.step()
            lr_scheduler.step()

            if step % (10 if cfg["debugging"] else 100) == 0:
                lrs = [group["lr"] for group in optim.param_groups]
                logging_info = {"lr": lrs[0]}
                logging_info.update(
                    {f"loss/{k}": v.mean(dim=0).item() for k, v in losses.items()}
                )
                logging_info.update(
                    {f"loss_weights/{k}": v for k, v in loss_weights.items()}
                )
                logging_info["time/i_restart"] = i_restart

                if t_last_log is not None:
                    time_per_iter = (time.time() - t_last_log) / (step - step_last_log)
                    logging_info["time/ETA[H]"] = (
                        (cfg["n_steps"] - step) * time_per_iter / 3600
                    )
                    logging_info["time/second_per_iter"] = time_per_iter
                t_last_log, step_last_log = time.time(), step
                wandb.log(logging_info, step=step)

            if step % (20 if cfg["debugging"] else 1000) == 0:
                try:
                    val_x, true_images, true_segmentations = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_x, true_images, true_segmentations = next(val_iter)

                val_x = val_x[:, 1:]

                val_x = val_x.to(device)
                model.eval()
                prediction = model(
                    val_x,
                    is_features=True,
                    num_iters=cfg["val_iters"],
                    num_slots=cfg["val_slots"],
                )
                model.train()

                true_images = np.transpose(true_images.cpu().numpy(), (0, 2, 3, 1))
                true_segmentations = nearest_neighbor_scaling(
                    true_segmentations.cpu().numpy()[..., 0], (-1, 224, 224)
                )
                for source, refine in product(
                    ["masks", "attention"], [True, False]
                ):
                    label = f"{source}{'_refined' if refine else ''}"
                    upscaled, segmentation = compute_upscaling_segmentations(
                        prediction[source], refine=refine
                    )
                    fig = plot_segmentations(
                        true_images,
                        upscaled,
                        segmentation,
                        true_segmentations,
                        cfg["n_plotting_instances"],
                    )
                    ari = adjusted_rand_index(
                        true_segmentations, segmentation, with_background=True
                    )
                    logging_info = {f"ari_{label}": np.mean(ari)}
                    ari = adjusted_rand_index(
                        true_segmentations, segmentation, with_background=False
                    )
                    logging_info[f"ari_{label}_no_bg"] = np.mean(ari)
                    logging_info[f"fig_{label}"] = wandb.Image(fig)
                    wandb.log(logging_info, step=step)

            step += 1

            t = time.time()
            if (t - last_ckpt_time) / (60 * 60) > cfg["checkpoint_every_h"]:
                last_ckpt_time = time.time()
                if last_ckpt_path is not None:
                    os.remove(last_ckpt_path)  # Remove the periodic checkpoints
                last_ckpt_path = save_checkpoint(
                    checkpoint_directory,
                    step,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    rng=rng,
                    i_restart=i_restart,
                )

            if step % 10_000 == 0:
                last_ckpt_time = time.time()
                save_checkpoint(
                    checkpoint_directory,
                    step,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    rng=rng,
                    i_restart=i_restart,
                )

            if (t - t_begin) / (60 * 60) > cfg["leap_time_h"]:
                save_checkpoint(
                    checkpoint_directory,
                    step,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    rng=rng,
                    i_restart=i_restart + 1,
                )
                train_dset.close()
                val_dset.close()
                return 124

            if step >= cfg["n_steps"]:
                return 0

        train_dset.permute()


@hydra.main(config_path="../../configs/dinosaur", config_name="config", version_base=None)
def main(cfg):
    logger.info(f"Starting training with config: {cfg}")
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    logger.info(f"Updated config to: {cfg_dict}")
    exit_code = train_from_checkpoint(cfg)
    logger.info(f"Exiting with code {exit_code}...")
    time.sleep(2)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

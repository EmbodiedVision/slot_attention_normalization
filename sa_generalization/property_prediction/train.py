"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import os
import sys
import time
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from sa_generalization.property_prediction import DATA_DIR
from sa_generalization.property_prediction.data.dataset import PropertyDataset
from sa_generalization.property_prediction.metrics import (
    adjusted_rand_index,
    average_precision_clevr,
)
from sa_generalization.property_prediction.set_predictor import SASetPredictor
from sa_generalization.property_prediction.visualization import (
    as_numpy_image,
    segmentation_image,
)

matplotlib.use("Agg")


def load_last_checkpoint(checkpoint_dir, map_location=None):
    steps = []
    for file_ in checkpoint_dir.iterdir():
        steps.append(int(file_.stem.split("_")[1]))
    if len(steps) > 0:
        max_step = max(steps)
        last_checkpoint_step = max_step
        last_checkpoint = torch.load(
            checkpoint_dir.joinpath(f"step_{max_step}.pkl"), map_location=map_location
        )
    else:
        last_checkpoint_step = last_checkpoint = None
    return last_checkpoint_step, last_checkpoint


# TODO: Boilerplate
def save_checkpoint(
    checkpoint_directory,
    step,
    **stuff,
):
    def to_serializable(obj):
        if isinstance(obj, torch.nn.DataParallel) or isinstance(
            obj, torch.nn.parallel.DistributedDataParallel
        ):
            return obj.module.state_dict()
        elif isinstance(obj, torch.Generator):
            return obj.get_state()
        elif hasattr(obj, "state_dict"):
            return obj.state_dict()
        elif isinstance(obj, dict):
            return {key: to_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(val) for val in obj]
        return obj

    checkpoint_data = {key: to_serializable(val) for key, val in stuff.items()}
    checkpoint_data.update({"step": step})
    checkpoint_path = checkpoint_directory.joinpath(f"step_{step}.pkl")
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


def train_from_checkpoint(cfg):
    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(
        project=cfg.wandb.project,
        config=cfg_dict,
        name=cfg.wandb.name,
        dir=outdir,
        resume="allow",
        id=outdir.replace("/", "_"),
        mode=cfg.wandb.mode,
    )

    print("Loading dataset")
    train_ds = PropertyDataset(DATA_DIR, features=("image", "property"), preload=True)
    val_ds = PropertyDataset(
        DATA_DIR, features=("image", "property", "mask"), subset="val", preload=True
    )
    print("Done")

    set_predictor = SASetPredictor(
        cfg.slot_dim, train_ds.property_size, sa_kwargs=cfg.sa_kwargs
    )
    set_predictor = set_predictor.to(cfg.device)

    optim = Adam(set_predictor.parameters(), cfg.lr)
    warmup_scheduler = LinearLR(
        optim, start_factor=0.001, end_factor=1, total_iters=cfg.warmup_iters
    )
    decay_scheduler = ExponentialLR(
        optim, gamma=cfg.decay_rate ** (1 / cfg.decay_steps)
    )
    lr_scheduler = SequentialLR(
        optim,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[cfg.warmup_iters],
    )

    checkpoint_directory = Path(outdir).joinpath("checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    last_checkpoint_step, ckpt = load_last_checkpoint(checkpoint_directory)

    if last_checkpoint_step is None:
        print("No previous checkpoint found")
        i_restart = 0
        step = 0
        rng = np.random.RandomState(cfg.seed)
        g = torch.Generator()
        g.manual_seed(rng.randint(0, 1e8))
    else:
        print(f"Continuing from step {last_checkpoint_step}")
        i_restart = ckpt["i_restart"] + 1
        step = ckpt["step"]
        set_predictor.load_state_dict(ckpt["set_predictor"])
        optim.load_state_dict(ckpt["optim"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        rng = ckpt["rng"]
        g = torch.Generator()
        g.set_state(ckpt["generator_state"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batchsize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    set_predictor.train()
    torch.manual_seed(rng.randint(0, 1e8))
    np.random.seed(rng.randint(0, 1e8))

    training_start = time.time()

    while True:
        try:
            image_batch, property_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            image_batch, property_batch = next(train_iter)
        image_batch = image_batch.to(cfg.device)
        property_batch = property_batch.to(cfg.device)
        loss = set_predictor.loss(image_batch, property_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()
        step += 1

        # Do some logging
        if (step == 1) or step % 500 == 0:
            logging_info = {"loss": loss.item()}
            del loss

            try:
                val_image, val_property, val_mask = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_image, val_property, val_mask = next(val_iter)

            val_image = val_image.to(cfg.device)
            set_predictor.eval()
            for n_slots in [10, 21]:
                val_pred = set_predictor.forward(val_image, num_slots=n_slots)

                ari_all, fg_ari = adjusted_rand_index(val_pred["attention"], val_mask)

                seg_visual = segmentation_image(
                    val_pred["attention"].reshape(cfg.val_batchsize, n_slots, 32, 32)
                )
                fig, ax = plt.subplots(ncols=2)
                ax[0].imshow(as_numpy_image((val_pred["image"][0] + 1) / 2))
                ax[1].imshow(seg_visual[0])
                last_lr = [group["lr"] for group in optim.param_groups]
                fig.tight_layout()

                logging_info.update(
                    {
                        f"f-ari {n_slots} slots": fg_ari.mean(),
                        f"ari {n_slots} slots": ari_all.mean(),
                        f"segmentation {n_slots} slots": fig,
                        "lr": last_lr[0],
                    }
                )

                average_prec = {
                    f"mAP-{threshold if threshold != -1 else 'inf'} {n_slots} slots": average_precision_clevr(
                        val_pred["predicted_properties"].detach().cpu().numpy(),
                        val_property.cpu().numpy(),
                        threshold,
                    )
                    for threshold in [-1.0, 1.0, 0.5, 0.25, 0.125]
                }

                logging_info.update(average_prec)
                del val_pred

            wandb.log(logging_info, step=step)
            print(f"Step {step}, time {time.time()}, info: {logging_info}")
            plt.close()
            set_predictor.train()

        finished = step > cfg.max_steps
        leap_done = time.time() - training_start > 60 * 60 * cfg.leap_timelimit_h

        if step % 2_000 == 0 or finished or leap_done:
            save_checkpoint(
                checkpoint_directory,
                step,
                set_predictor=set_predictor,
                optim=optim,
                lr_scheduler=lr_scheduler,
                rng=rng,
                generator_state=g,
                i_restart=i_restart,
            )

        if finished or leap_done:
            train_ds.close()
            val_ds.close()
            break

    wandb.finish()
    time.sleep(30)

    if finished:
        return 0
    return 124


@hydra.main(
    config_path="../../configs/property_prediction",
    config_name="config",
    version_base=None,
)
def main(cfg):
    print(f"Starting training with config {cfg}")
    exit_code = train_from_checkpoint(cfg)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

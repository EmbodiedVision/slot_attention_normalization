"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import collections.abc
import copy
import os
import sys
import time
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import omegaconf
import torch
import torchvision.transforms as T
import wandb
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from sa_generalization.object_discovery.autoencoder import SlotAttentionAutoencoder
from sa_generalization.object_discovery.data.dataset import EndlessDataLoader, SADataset
from sa_generalization.object_discovery.data.transforms import DiscreteRandomRotation
from sa_generalization.object_discovery.losses import (
    compute_total_loss,
    evaluate_scheduled_weights,
)
from sa_generalization.object_discovery.tf_compatibility import AdamTF
from sa_generalization.object_discovery.visualization import visualization_and_ari

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_optim(optim_type, parameters, lr, **kwargs):
    if optim_type == "AdamTF":
        optim = AdamTF(parameters, lr=lr, eps=1e-08, **kwargs)
    elif optim_type == "RMSprop":
        optim = torch.optim.RMSprop(parameters, lr=lr, **kwargs)
    elif optim_type == "Adam":
        optim = torch.optim.Adam(parameters, lr=lr, **kwargs)
    elif optim_type == "SGD":
        optim = torch.optim.SGD(parameters, lr=lr, **kwargs)
    else:
        raise NotImplementedError
    return optim


def config_hook(config):
    autoencoder_kwargs = {
        "num_slots": 7,
        "cnn_feature_dim": 64,
        "common_dim": 64,
        "slot_dim": 64,
        "encoder_kernel_size": 5,
        "num_iterations": 3,
        "im_size": 128,
    }

    config["autoencoder_kwargs"] = recursive_update(
        autoencoder_kwargs, config["autoencoder_kwargs"]
    )

    validation_arguments = [
        "num_slots",
        "num_iterations",
    ]
    config["validation_slots"]["val_like_train"] = {
        k: config["autoencoder_kwargs"][k] for k in validation_arguments
    }
    return config


def log_visualizations_ari(figures, scalars, dset_name, step):
    logging_info = {
        f"{dset_name}/{name}": wandb.Image(fig) for name, fig in figures.items()
    }
    logging_info.update({f"{dset_name}/{name}": val for name, val in scalars.items()})
    wandb.log(logging_info, step=step)

    for fig in figures.values():
        plt.close(fig)


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


def unpack_batch(maybe_packed):
    """
    Depending on whether we use masks in validation, the data loader will give a list or just one Tensor.
    This functions returns the original tuple if we use masks in validation, otherwise the tuple Tensor, None
    :param maybe_packed:
    :return:
    """
    if type(maybe_packed) == list:
        return maybe_packed
    return maybe_packed, None


def step_lr_scheduler(lr_scheduler):
    if lr_scheduler is None:
        return
    if isinstance(lr_scheduler, dict):
        return {key: step_lr_scheduler(val) for key, val in lr_scheduler.items()}
    return lr_scheduler.step()


def recursive_load_state_dict(obj, state_dict):
    if obj is None:
        return
    if isinstance(obj, dict):
        assert isinstance(state_dict, dict)
        for key in obj:
            obj[key].load_state_dict(state_dict[key])
    else:
        obj.load_state_dict(state_dict)


def clip_norm(tensor_list, norm):
    l2 = torch.sqrt(sum(torch.sum(t**2) for t in tensor_list if t is not None))
    normalizer = min(norm / l2, 1)
    return [t * normalizer if t is not None else t for t in tensor_list]


def grad_l2_norm(params):
    with torch.no_grad():
        grad_l2 = 0
        for param in params:
            if param.grad is not None:
                grad_l2 += torch.sum(param.grad**2).item()
    return np.sqrt(grad_l2)


def train_from_checkpoint(cfg):
    loss_kwargs = copy.deepcopy(cfg["loss_kwargs"])
    torch.manual_seed(cfg["seed"])

    if cfg["anomaly_detection"]:
        torch.autograd.set_detect_anomaly(True)

    run_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_directory = Path(run_directory).joinpath("checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    last_checkpoint_step, last_checkpoint = load_last_checkpoint(checkpoint_directory)

    wandb.init(project="sa-object-discovery", dir=run_directory)

    auto_encoder = SlotAttentionAutoencoder(
        encoder_pos_embedding_kwargs=cfg["encoder_pos_embedding_kwargs"],
        decoder_pos_embedding_kwargs=cfg["decoder_pos_embedding_kwargs"],
        **cfg["autoencoder_kwargs"],
    )

    auto_encoder = auto_encoder.to(cfg["device"])
    optim = get_optim(cfg["optim_type"], auto_encoder.parameters(), cfg["lr"])

    if cfg["lr_scheduling"]:
        warmup_scheduler = LinearLR(
            optim, start_factor=0.001, end_factor=1, total_iters=cfg["warmup_iters"]
        )
        decay_scheduler = ExponentialLR(
            optim, gamma=cfg["decay_rate"] ** (1 / cfg["decay_steps"])
        )
        lr_scheduler = SequentialLR(
            optim,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[cfg["warmup_iters"]],
        )
    else:
        lr_scheduler = None

    if last_checkpoint_step == None:
        i_restart = 0
        step = 0
        rng = np.random.RandomState(cfg["seed"])
        g = torch.Generator()
        g.manual_seed(rng.randint(0, 1e8))
    else:
        logger.info(f"Continuing from step {last_checkpoint_step}")
        i_restart = last_checkpoint["i_restart"] + 1
        step = last_checkpoint["step"]
        auto_encoder.load_state_dict(last_checkpoint["auto_encoder"])
        recursive_load_state_dict(optim, last_checkpoint["optim"])
        recursive_load_state_dict(lr_scheduler, last_checkpoint["lr_scheduler"])
        rng = last_checkpoint["rng"]
        g = torch.Generator()
        g.set_state(last_checkpoint["generator_state"])

    auto_encoder = auto_encoder.to(cfg["device"])
    auto_encoder.train()
    torch.manual_seed(rng.randint(0, 1e8))
    np.random.seed(rng.randint(0, 1e8))

    dataset_options = cfg["dataset_options"]
    if cfg["augment_data"]:
        if dataset_options["dset"] in ["tetrominoes", "bouncing_balls"]:
            augmentation_transforms = T.Compose(
                [DiscreteRandomRotation([0, 90, 180, 270]), T.RandomHorizontalFlip()]
            )
        elif dataset_options["dset"] in ["original", "multi_mnist", "iodine"]:
            augmentation_transforms = T.RandomHorizontalFlip()
        else:
            raise Exception(
                f"Data augmentatino not possible for {dataset_options['dset']}"
            )
    else:
        augmentation_transforms = None

    train_dataset = SADataset(
        dset=dataset_options["dset"],
        transform=augmentation_transforms,
        max_objects=cfg["dataset_options"]["max_objects"],
        min_objects=dataset_options.get("min_objects", 0),
        preload=True,
        seed=rng.randint(0, 1e8),
        num_images=dataset_options["n_images"],
        driver=cfg["driver"],
        copy_to_tmp=cfg["copy_to_tmp"],
    )

    val_dataset = SADataset(
        dset=dataset_options["dset"],
        subset="val",
        max_objects=cfg["max_val_objects"],
        preload=True,
        seed=rng.randint(0, 1e8),
        num_images=cfg["n_val_images"],
        with_masks=cfg["masks_in_val"],
        driver=cfg["driver"],
        copy_to_tmp=False,
        parent_dir=train_dataset.tmp_dir.name if cfg["copy_to_tmp"] else None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batchsize"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        generator=g,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["val_batchsize"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        generator=g,
    )
    val_iter = EndlessDataLoader(val_dataloader)
    train_iter = EndlessDataLoader(train_dataloader)

    training_start = time.time()
    t_last_log = None

    wandb.log({"time/i_restart": i_restart}, step=step)
    logger.info(f"Starting training loop at restart {i_restart}")

    while True:
        loss_weights = evaluate_scheduled_weights(cfg["loss_schedule"], step)
        logging_info = {}

        image_batch = next(train_iter)
        image_batch = image_batch.to(cfg["device"])
        result = auto_encoder(image_batch)
        all_losses, loss = compute_total_loss(
            image_batch,
            result,
            loss_weights,
            irrelevant_losses=cfg["log_irrelevant_losses"],
            loss_kwargs=loss_kwargs,
        )
        loss = loss.mean(dim=0)
        optim.zero_grad()
        loss.backward()
        if cfg["clipping_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                auto_encoder.parameters(), cfg["clipping_norm"], error_if_nonfinite=True
            )
        if cfg["log_irrelevant_losses"]:
            logging_info["grads/magnitude"] = grad_l2_norm(auto_encoder.parameters())
        optim.step()

        step_lr_scheduler(lr_scheduler)
        step += 1

        if (step == 1) or step % cfg["log_scalar_every"] == 0:
            logging_info["lr"] = [group["lr"] for group in optim.param_groups][0]
            logging_info["loss"] = loss.item()
            logger.info(
                f"Last loss at step {step - 1} / {cfg['max_steps']}: {loss.item()}"
            )

            if t_last_log is not None:
                logging_info["time/ETA[H]"] = (
                    (time.time() - t_last_log)
                    / cfg["log_scalar_every"]
                    * (cfg["max_steps"] - step)
                    / 3600
                )
                logging_info["time/sec_per_iter"] = (
                    (time.time() - t_last_log) / cfg["log_scalar_every"],
                )

            t_last_log = time.time()
            for k, v in all_losses.items():
                logging_info[f"loss/{k}"] = v.mean(dim=0).item()
            for k, v in loss_weights.items():
                logging_info[f"loss_weights/{k}"] = v

            wandb.log(logging_info, step=step)

        if (
            (step == 1)
            or ((step < 5000) and step % cfg["log_val_every"] == 0)
            or ((step >= 5000) and step % (4 * cfg["log_val_every"]) == 0)
        ):
            figures, scalars = visualization_and_ari(result, None)
            log_visualizations_ari(figures, scalars, "train", step)

            auto_encoder.eval()
            # Evaluate on validation instance
            torch.set_grad_enabled(False)

            val_batch, val_masks = unpack_batch(next(val_iter))
            val_batch = val_batch.to(cfg["device"])
            for name, setting in cfg["validation_slots"].items():
                kwargs = {
                    "num_slots": setting["num_slots"],
                    "num_iterations": setting["num_iterations"],
                }

                result = auto_encoder(val_batch, **kwargs)
                all_losses, loss = compute_total_loss(
                    val_batch, result, loss_weights, loss_kwargs=loss_kwargs
                )
                val_logging_info = {
                    f"{name}/loss/{k}": v.mean(dim=0).item()
                    for k, v in all_losses.items()
                }
                val_logging_info[f"{name}/loss/total"] = loss.mean(dim=0).item()
                wandb.log(val_logging_info, step=step)

                figures, scalars = visualization_and_ari(result, val_masks)
                log_visualizations_ari(figures, scalars, name, step)

            torch.set_grad_enabled(True)

            auto_encoder.train()

        finished = step > cfg["max_steps"]
        leap_done = time.time() - training_start > 60 * 60 * cfg["leap_timelimit_h"]

        if step % 950 == 0 or finished or leap_done:
            save_checkpoint(
                checkpoint_directory,
                step,
                auto_encoder=auto_encoder,
                optim=optim,
                lr_scheduler=lr_scheduler,
                rng=rng,
                generator_state=g,
                i_restart=i_restart,
            )

        if finished or leap_done:
            break

    if finished:
        return 0
    if leap_done:
        training_end = time.time()
        wandb.log({"time/training_duration": training_end - training_start}, step=step)
        if cfg["restart"]:
            return 124
        else:
            return 0


@hydra.main(
    config_path="../../configs/object_discovery", config_name="config", version_base=None
)
def main(cfg):
    logger.info(f"Starting training with config: {cfg}")
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    cfg_dict = config_hook(cfg_dict)
    logger.info(f"Updated config to: {cfg_dict}")
    exit_code = train_from_checkpoint(cfg)
    logger.info(f"Exiting with code {exit_code}...")
    time.sleep(2)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

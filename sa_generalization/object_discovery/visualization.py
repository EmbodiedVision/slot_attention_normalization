"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sa_generalization.object_discovery.losses import adjusted_rand_index


def as_numpy_image(tensor):
    return np.clip(tensor.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)


def visualization_and_ari(info_dict, maybe_true_masks):
    n_slots = info_dict["slot_wise_reconstructions"].shape[1]
    im_size = info_dict["slot_wise_reconstructions"].shape[3:]
    result_figures = {}
    result_scalars = {}
    fig, ax = plt.subplots(nrows=3, ncols=n_slots, figsize=(2 * (n_slots), 6), dpi=200)
    for slot_idx in range(n_slots):
        alpha_mask = (
            info_dict["slot_wise_masks"][0][slot_idx]
            .detach()
            .cpu()
            .numpy()
            .reshape(*im_size, 1)
        )
        ax[0, slot_idx].imshow(
            (1 - alpha_mask)
            + alpha_mask
            * as_numpy_image(
                (info_dict["slot_wise_reconstructions"][0][slot_idx] + 1) / 2
            ),
        )
        ax[1, slot_idx].imshow(
            as_numpy_image(info_dict["slot_wise_masks"][0][slot_idx]), vmin=0, vmax=1
        )
        ax[2, slot_idx].imshow(
            as_numpy_image(info_dict["attention"][0][slot_idx].reshape((1, *im_size))),
            vmin=0,
            vmax=1,
        )
        for i in range(3):
            for spine in ["top", "bottom", "left", "right"]:
                ax[i, slot_idx].spines[spine].set_visible(False)
            ax[i, slot_idx].set_xticks([])
            ax[i, slot_idx].set_yticks([])

    ax[0, 0].set_ylabel("Reconstruction", fontsize=13)
    ax[1, 0].set_ylabel("Mask", fontsize=13)
    ax[2, 0].set_ylabel("Attention", fontsize=13)
    fig.tight_layout()
    result_figures[f"reconstruction"] = fig

    np_masks = (
        info_dict["slot_wise_masks"][:, :, 0, ...]
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    np_attention = (
        info_dict["attention"]
        .detach()
        .cpu()
        .reshape(*info_dict["attention"].shape[:2], *im_size)
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    pre_attention_labels = np.argmax(np_attention, axis=-1)
    mask_labels = np.argmax(np_masks, axis=-1)

    if maybe_true_masks is not None:
        np_true_masks = maybe_true_masks.detach().cpu().numpy().transpose(0, 2, 3, 1)
        true_labels = np.argmax(np_true_masks, axis=-1)
        for name, segmentation in zip(["masks", "attention"], [np_masks, np_attention]):
            ari_all, ari_no_background = adjusted_rand_index(
                np_true_masks, segmentation
            )
            result_scalars[f"ARI/{name}"] = ari_all.mean()
            result_scalars[f"ARI/{name}_no_background"] = ari_no_background.mean()

        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 2), dpi=100)
        ax[0].imshow(as_numpy_image((info_dict["image"][0] + 1) / 2))
        ax[1].imshow(as_numpy_image((info_dict["reconstruction"][0] + 1) / 2))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].set_title("Image")
        ax[1].set_title("Reconstruction")
        ax[2].imshow(true_labels[0], cmap="jet", interpolation="nearest")
        ax[2].set_title("True Seg.")
        ax[2].axis("off")
        ax[3].imshow(mask_labels[0], cmap="jet", interpolation="nearest")
        ax[3].set_title("Mask Seg.")
        ax[3].axis("off")
        ax[4].imshow(pre_attention_labels[0], cmap="jet", interpolation="nearest")
        ax[4].set_title("Attn Seg.")
        ax[4].axis("off")
        fig.tight_layout()
        result_figures["masks"] = fig

    else:
        fig, ax = plt.subplots(ncols=3, figsize=(6, 2))
        ax[0].imshow(as_numpy_image((info_dict["image"][0] + 1) / 2))
        ax[1].imshow(as_numpy_image((info_dict["reconstruction"][0] + 1) / 2))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].set_title("Image")
        ax[1].set_title("Reconstruction")
        ax[2].imshow(mask_labels[0], cmap="jet", interpolation="nearest")
        ax[2].set_title("Predicted Masks")
        ax[2].axis("off")
        fig.tight_layout()
        result_figures["masks"] = fig

    return result_figures, result_scalars

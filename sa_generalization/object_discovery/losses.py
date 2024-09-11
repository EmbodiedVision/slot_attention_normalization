"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score


def adjusted_rand_index(true_masks, predicted_masks):
    """
    We use the convention that true_masks[..., 0] are the background masks. We compute the ARIs both
    for the entire image and for the foreground.

    :param true_masks:e  tensor or np.array of shape [batch_size, width, height, n_objects_true]
    :param predicted_masks: tensor or np.array of shape [batch_size, width, height, n_objects_pred]
    :return: Pair of np.arrays of shape [batch_size,] with adjusted rand score for (i) the entire image (ii) the image without background.
    """
    if torch.is_tensor(true_masks):
        true_masks = true_masks.detach.cpu().numpy()

    if torch.is_tensor(predicted_masks):
        predicted_masks = predicted_masks.detach.cpu().numpy()

    batch_size, width, height, n_objects_true = true_masks.shape
    b, w, h, n_objects_pred = predicted_masks.shape
    assert b == batch_size and w == width and h == height

    true_labels = np.argmax(true_masks, axis=-1)
    predicted_labels = np.argmax(predicted_masks, axis=-1)
    true_labels_flat = true_labels.reshape(batch_size, width * height)
    predicted_labels_flat = predicted_labels.reshape(batch_size, width * height)

    predicted_labels_no_background = []
    true_labels_no_background = []
    # We go through each scene individually
    for image_index in range(batch_size):
        # We extract exactly those pixels that correspond to objects in this image (i.e. no background)
        object_positions = np.logical_not(true_masks[image_index, ..., 0]).nonzero()
        predicted_labels_no_background.append(
            predicted_labels[image_index][object_positions]
        )
        true_labels_no_background.append(true_labels[image_index][object_positions])

    ari_all = [
        adjusted_rand_score(true_labeling, predicted_labeling)
        for true_labeling, predicted_labeling in zip(
            true_labels_flat, predicted_labels_flat
        )
    ]
    ari_no_background = [
        adjusted_rand_score(true_labeling, predicted_labeling)
        for true_labeling, predicted_labeling in zip(
            true_labels_no_background, predicted_labels_no_background
        )
    ]
    return np.array(ari_all), np.array(ari_no_background)


def evaluate_scheduled_weights(weight_schedules, step):
    result = {}
    for name, schedule in weight_schedules.items():
        if isinstance(schedule, (int, float)):
            result[name] = schedule
        else:
            assert isinstance(schedule, list)
            upper_idx = min(i for i, (t, _) in enumerate(schedule) if t >= step)
            assert upper_idx > 0
            upper_t, upper_val = schedule[upper_idx]
            lower_t, lower_val = schedule[upper_idx - 1]
            assert lower_t <= step <= upper_t
            result[name] = (
                upper_val * (step - lower_t) + lower_val * (upper_t - step)
            ) / (upper_t - lower_t)
    return result


def compute_total_loss(
    original,
    fwd_result,
    loss_weights,
    model=None,
    loss_kwargs=None,
    irrelevant_losses=True,
):
    all_losses = {}
    total_loss = 0
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    for key, value in loss_weights.items():
        if value == 0 and not irrelevant_losses:
            continue
        if key == "reconstruction_loss":
            partial_loss = reconstruction_loss(original, fwd_result["reconstruction"])
        else:
            raise NotImplementedError
        all_losses[key] = partial_loss
        if value != 0:
            total_loss = total_loss + value * partial_loss

    all_losses["total"] = total_loss
    return all_losses, total_loss


def reconstruction_loss(original, reconstruction):
    assert original.dim() == 4
    return F.mse_loss(original, reconstruction, reduction="none").mean(dim=(1, 2, 3))

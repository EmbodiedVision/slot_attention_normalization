"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import adjusted_rand_score


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
        elif hasattr(obj, "state_dict"):
            return obj.state_dict()
        return obj

    checkpoint_data = {key: to_serializable(val) for key, val in stuff.items()}
    checkpoint_data.update({"step": step})
    checkpoint_path = checkpoint_directory.joinpath(f"step_{step}.pkl")
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


def nearest_neighbor_scaling(arr, new_shape):
    new_shape = [new if new != -1 else old for (old, new) in zip(arr.shape, new_shape)]
    assert isinstance(arr, np.ndarray) and len(new_shape) == arr.ndim
    new_indices = np.indices(new_shape).transpose((*range(1, arr.ndim + 1), 0))
    take_from = np.round(
        np.array(arr.shape) / np.array(new_shape) * new_indices
    ).astype(int)
    return arr[tuple([take_from[..., i] for i in range(arr.ndim)])]


def block_refine(arr, dims, block_size):
    """Bilinear interpolation factor block_size"""
    result = arr
    for dim in dims:
        result = block_refine_1d(result, dim, block_size)
    return result


def block_refine_1d(arr, dim, block_size):
    assert block_size % 2 == 0
    repeated = np.repeat(arr, block_size, axis=dim)
    right_neighbor_slice = [slice(None) for _ in arr.shape]
    left_neighbor_slice = [slice(None) for _ in arr.shape]
    right_neighbor_slice[dim] = slice(block_size, None)
    left_neighbor_slice[dim] = slice(None, -block_size)
    right_neighbors = repeated[tuple(right_neighbor_slice)]
    left_neighbors = repeated[tuple(left_neighbor_slice)]
    mixture_coef = np.linspace(
        1 / (2 * block_size), 1 - 1 / (2 * block_size), block_size
    )
    mixture_coef = np.expand_dims(
        mixture_coef, tuple(i for i in range(arr.ndim) if i != dim)
    )
    reps = [1 if i != dim else arr.shape[dim] - 1 for i in range(arr.ndim)]
    mixture_coef = np.tile(mixture_coef, reps)
    interior_refined = (
        mixture_coef * right_neighbors + (1 - mixture_coef) * left_neighbors
    )
    left_slice = [slice(None) for _ in arr.shape]
    right_slice = [slice(None) for _ in arr.shape]
    left_slice[dim] = slice(None, block_size // 2)
    right_slice[dim] = slice(-block_size // 2, None)
    return np.concatenate(
        [repeated[tuple(left_slice)], interior_refined, repeated[tuple(right_slice)]],
        axis=dim,
    )


def adjusted_rand_index(
    true_segmentations, predicted_segmentations, with_background=True
):
    """Compute adjusted rand index for a batch of segmentations

    :param true_segmentation: Numpy array of shape (B, W, H) and arbitrary singleton dimensions and integer entries
    :param predicted_segmentation: Numpy array of shape (B, W, H) and arbitrary singleton dimensions and integer entries
    :return: ARI
    """
    assert isinstance(true_segmentations, np.ndarray) and isinstance(
        predicted_segmentations, np.ndarray
    )
    true_segmentations = np.squeeze(true_segmentations)
    predicted_segmentations = np.squeeze(predicted_segmentations)
    assert true_segmentations.ndim == 3 and predicted_segmentations.ndim == 3
    ari_list = []
    for true_instance, predicted_instance in zip(
        true_segmentations, predicted_segmentations
    ):
        flat_truth = np.ravel(true_instance)
        flat_prediction = np.ravel(predicted_instance)
        if not with_background:
            foreground_indices = flat_truth.nonzero()
            flat_truth = flat_truth[foreground_indices]
            flat_prediction = flat_prediction[foreground_indices]
        ari_list.append(adjusted_rand_score(flat_truth, flat_prediction))
    return np.array(ari_list)


def compute_upscaling_segmentations(attention, refine=True):
    """Compute upscaled attention and argmax segmentation of batch of images.

    :param attention: Torch tensor of shape (B, #slots, 28*28)
    :param refine: Whether to use bi-linear interpolation to upscale. Otherwise repeat.
    :return: (upscaled_attention, segmentations)
    """
    upscalings = []
    segmentations = []
    for instance in attention:
        np_instance = np.reshape(instance.detach().cpu().numpy(), (-1, 28, 28))
        if refine:
            instance_attention = block_refine(np_instance, (1, 2), 8)
        else:
            instance_attention = np_instance.repeat(8, axis=-1).repeat(8, axis=-2)
        upscalings.append(instance_attention)
        segmentation = np.argmax(instance_attention, axis=0)
        segmentations.append(segmentation)
    return np.stack(upscalings, axis=0), np.stack(segmentations, axis=0)


def plot_segmentations(
    images, attention, segmentations, true_segmentations, n_instances
):
    n_slots = attention.shape[1]
    fig, ax = plt.subplots(
        ncols=n_instances,
        nrows=3 + n_slots,
        figsize=(2 * n_instances, 2 * (3 + n_slots)),
    )
    for i in range(n_instances):
        segmentation = segmentations[i]
        true_segmentation = true_segmentations[i]
        real_image = images[i]
        ax[0, i].imshow(real_image)
        ax[0, i].axis("off")
        for k, seg in enumerate([true_segmentation, segmentation]):
            ax[1 + k, i].imshow(real_image, extent=(-1, 1, -1, 1))
            ax[1 + k, i].matshow(
                seg,
                alpha=0.6,
                cmap=matplotlib.colormaps["tab20"],
                extent=(-1, 1, -1, 1),
            )
            ax[1 + k, i].axis("off")

        for slot_idx in range(n_slots):
            ax[3 + slot_idx, i].matshow(attention[i][slot_idx], vmin=0, vmax=1)
            ax[3 + slot_idx, i].axis("off")
    fig.tight_layout()
    return fig


def render_video_sequence(image_sequence, attention_sequence, refine=True, fps=4):
    frames = []
    for image, attention in zip(image_sequence, attention_sequence):
        fig, ax = plt.subplots(nrows=2, figsize=(2, 4))
        canvas = FigureCanvas(fig)

        reshaped_attention = np.reshape(attention.detach().cpu().numpy(), (-1, 28, 28))
        if refine:
            reshaped_attention = block_refine(reshaped_attention, (1, 2), 8)
        segmentation = np.argmax(reshaped_attention, axis=0)
        ax[0].imshow(image, extent=(-1, 1, -1, 1))
        ax[1].imshow(image, extent=(-1, 1, -1, 1))
        ax[1].matshow(
            segmentation,
            alpha=0.6,
            cmap=matplotlib.colormaps["tab20"],
            extent=(-1, 1, -1, 1),
        )
        ax[1].axis("off")
        ax[0].axis("off")
        fig.tight_layout()

        canvas.draw()  # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi()
        frame = np.transpose(
            np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
                (int(height), int(width), 3)
            ),
            (2, 0, 1),
        )
        frames.append(frame)
        plt.close(fig)

    return np.stack(frames, axis=0)


def render_batch_video_sequence(
    batch_image_sequence, batch_attention_sequence, n_instances=4, refine=True, fps=4
):
    videos = []
    for i in range(n_instances):
        video = render_video_sequence(
            batch_image_sequence[i], batch_attention_sequence[i], refine=refine, fps=fps
        )
        videos.append(video)
    return np.stack(videos, axis=0)


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


def compute_total_loss(original, fwd_result, loss_weights):
    all_losses = {}
    total_loss = 0
    for key, value in loss_weights.items():
        if key == "reconstruction_loss":
            partial_loss = reconstruction_loss(original, fwd_result["reconstruction"])
        else:
            raise NotImplementedError
        all_losses[key] = partial_loss
        total_loss = total_loss + value * partial_loss

    all_losses["total"] = total_loss
    return all_losses, total_loss


def marginal_entropy(posterior_matrix):
    """Compute entropy of slot marginals.

    :param prob_matrix: Tensor of shape (B, #slots, #inputs) such that the column sums are 1
    """
    assert posterior_matrix.dim() == 3 and torch.allclose(
        torch.sum(posterior_matrix, dim=1), torch.tensor(1.0)
    )
    marginal = torch.sum(posterior_matrix, dim=-1)
    marginal = marginal / torch.sum(marginal, dim=1, keepdim=True)
    assert torch.allclose(torch.sum(marginal, dim=1), torch.tensor(1.0))
    return -torch.sum(torch.special.xlogy(marginal, marginal), dim=1)


def reconstruction_loss(original, reconstruction):
    assert original.dim() == 3
    return F.mse_loss(original, reconstruction, reduction="none").mean(dim=(1, 2))

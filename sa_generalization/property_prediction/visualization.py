"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import numpy as np


def as_numpy_image(tensor):
    return np.clip(tensor.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)


def segmentation_image(masks):
    colors = np.array(
        [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Maroon
            [0, 128, 0],  # Green (dark)
            [0, 0, 128],  # Navy
            [128, 128, 0],  # Olive
            [128, 0, 128],  # Purple
            [0, 128, 128],  # Teal
            [255, 128, 0],  # Orange
            [255, 0, 128],  # Pink
            [128, 255, 0],  # Lime
            [0, 255, 128],  # Spring Green
            [128, 0, 255],  # Violet
            [0, 128, 255],  # Sky Blue
            [128, 255, 255],  # Light Blue
            [255, 128, 128],  # Salmon
            [64, 64, 128],  # Gray
        ]
    )  # Gray
    n_slots = masks.shape[1]
    colors = np.expand_dims(colors, (0, 3, 4))[:, :n_slots]
    segmentation_image = colors * np.expand_dims(masks.detach().cpu().numpy(), 2)
    segmentation_image = np.sum(segmentation_image, axis=1).astype(np.uint8)
    return np.transpose(segmentation_image, (0, 2, 3, 1))

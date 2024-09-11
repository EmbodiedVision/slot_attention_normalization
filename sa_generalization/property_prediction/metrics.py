# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This file contains code originally licensed under the Apache License,
# Version 2.0, with modifications by Markus Krimmel.
#
# The functions 'average_precision_clevr` and `compute_average_precision`
# are modified from https://github.com/google-research/google-research/blob/bfa1a6eaaac2bbde8ab6a376de6974233b7456c1/slot_attention/utils.py

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from sa_generalization.property_prediction.data.dataset import PropertyDataset


def adjusted_rand_index(attention, true_masks):
    if torch.is_tensor(true_masks):
        true_masks = true_masks.detach().cpu().numpy()
    if torch.is_tensor(attention):
        attention = attention.detach().cpu().numpy()

    assert attention.shape[2] == 32 * 32
    attention = attention.reshape(
        attention.shape[0], attention.shape[1], 32, 32
    ).transpose((0, 2, 3, 1))

    assert true_masks.ndim == 4  # (B, 11, 128, 128)
    true_masks = true_masks.transpose(0, 2, 3, 1)
    true_masks = true_masks[:, ::4, ::4, :]  # downsample to 32 x 32

    batch_size, width, height, n_objects_true = true_masks.shape
    b, w, h, n_objects_pred = attention.shape
    assert b == batch_size and w == width and h == height, (
        (b, w, h),
        (batch_size, width, height),
    )

    true_labels = np.argmax(true_masks, axis=-1)
    predicted_labels = np.argmax(attention, axis=-1)
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


"""
The following section of the file contains code originally licensed under the 
Apache License, Version 2.0.
See: https://github.com/google-research/google-research/blob/bfa1a6eaaac2bbde8ab6a376de6974233b7456c1/slot_attention/utils.py
"""


def average_precision_clevr(pred, attributes, distance_threshold):
    """Computes the average precision for CLEVR.

    This function computes the average precision of the predictions specifically
    for the CLEVR dataset. First, we sort the predictions of the model by
    confidence (highest confidence first). Then, for each prediction we check
    whether there was a corresponding object in the input image. A prediction is
    considered a true positive if the discrete features are predicted correctly
    and the predicted position is within a certain distance from the ground truth
    object.

    Args:
      pred: Tensor of shape [batch_size, num_elements, dimension] containing
        predictions. The last dimension is expected to be the confidence of the
        prediction.
      attributes: Tensor of shape [batch_size, num_elements, dimension] containing
        ground-truth object properties.
      distance_threshold: Threshold to accept match. -1 indicates no threshold.

    Returns:
      Average precision of the predictions.
    """

    [batch_size, _, element_size] = attributes.shape
    [_, predicted_elements, _] = pred.shape

    def unsorted_id_to_image(detection_id, predicted_elements):
        """Find the index of the image from the unsorted detection index."""
        return int(detection_id // predicted_elements)

    flat_size = batch_size * predicted_elements
    flat_pred = np.reshape(pred, [flat_size, element_size])
    sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

    sorted_predictions = np.take_along_axis(
        flat_pred, np.expand_dims(sort_idx, axis=1), axis=0
    )
    idx_sorted_to_unsorted = np.take_along_axis(np.arange(flat_size), sort_idx, axis=0)

    process_targets = PropertyDataset.decode_property
    true_positives = np.zeros(sorted_predictions.shape[0])
    false_positives = np.zeros(sorted_predictions.shape[0])

    detection_set = set()

    for detection_id in range(sorted_predictions.shape[0]):
        # Extract the current prediction.
        current_pred = sorted_predictions[detection_id, :]
        # Find which image the prediction belongs to. Get the unsorted index from
        # the sorted one and then apply to unsorted_id_to_image function that undoes
        # the reshape.
        original_image_idx = unsorted_id_to_image(
            idx_sorted_to_unsorted[detection_id], predicted_elements
        )
        # Get the ground truth image.
        gt_image = attributes[original_image_idx, :, :]

        # Initialize the maximum distance and the id of the groud-truth object that
        # was found.
        best_distance = 10000
        best_id = None

        # Unpack the prediction by taking the argmax on the discrete attributes.
        (
            pred_coords,
            pred_object_size,
            pred_material,
            pred_shape,
            pred_color,
            _,
        ) = process_targets(current_pred)

        # Loop through all objects in the ground-truth image to check for hits.
        for target_object_id in range(gt_image.shape[0]):
            target_object = gt_image[target_object_id, :]
            # Unpack the targets taking the argmax on the discrete attributes.
            (
                target_coords,
                target_object_size,
                target_material,
                target_shape,
                target_color,
                target_real_obj,
            ) = process_targets(target_object)
            # Only consider real objects as matches.
            if target_real_obj:
                # For the match to be valid all attributes need to be correctly
                # predicted.
                pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
                target_attr = [
                    target_object_size,
                    target_material,
                    target_shape,
                    target_color,
                ]
                match = pred_attr == target_attr
                if match:
                    # If a match was found, we check if the distance is below the
                    # specified threshold. Recall that we have rescaled the coordinates
                    # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
                    # `pred_coords`. To compare in the original scale, we thus need to
                    # multiply the distance values by 6 before applying the norm.
                    distance = np.linalg.norm((target_coords - pred_coords) * 6.0)

                    # If this is the best match we've found so far we remember it.
                    if distance < best_distance:
                        best_distance = distance
                        best_id = target_object_id
        if best_distance < distance_threshold or distance_threshold == -1:
            # We have detected an object correctly within the distance confidence.
            # If this object was not detected before it's a true positive.
            if best_id is not None:
                if (original_image_idx, best_id) not in detection_set:
                    true_positives[detection_id] = 1
                    detection_set.add((original_image_idx, best_id))
                else:
                    false_positives[detection_id] = 1
            else:
                false_positives[detection_id] = 1
        else:
            false_positives[detection_id] = 1
    accumulated_fp = np.cumsum(false_positives)
    accumulated_tp = np.cumsum(true_positives)
    recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
    precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

    return compute_average_precision(
        np.array(precision_array, dtype=np.float32),
        np.array(recall_array, dtype=np.float32),
    )


def compute_average_precision(precision, recall):
    """Computation of the average precision from precision and recall arrays."""
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.0
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision

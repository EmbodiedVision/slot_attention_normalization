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
from scipy.optimize import linear_sum_assignment
from torch import nn

from sa_generalization.slot_attention import SlotAttention


def tf_initialize_sequential(sequential):
    for layer in sequential:
        if hasattr(layer, "bias"):
            layer.bias.data.fill_(0)
        if hasattr(layer, "weight"):
            nn.init.xavier_uniform_(layer.weight)


def sequential_hungarian(losses):
    row_indices = []
    col_indices = []
    for cost_matrix in losses:
        r, c = linear_sum_assignment(cost_matrix)
        row_indices.append(r)
        col_indices.append(c)
    return row_indices, col_indices


class SoftPositionEmbedding(nn.Module):
    def __init__(
        self,
        n_channels,
        resolution,
    ):
        super(SoftPositionEmbedding, self).__init__()
        self._n_channels = n_channels
        h_inc = torch.linspace(0, 1, resolution)
        w_inc = torch.linspace(0, 1, resolution)
        h_grid, w_grid = torch.meshgrid(h_inc, w_inc)
        positional_embedding = [h_grid, w_grid, 1 - h_grid, 1 - w_grid]
        pos_embedding = torch.stack(positional_embedding, dim=-1)
        self.register_buffer("pos_embedding", pos_embedding)
        self.dense = nn.Linear(pos_embedding.shape[-1], n_channels)
        self.dense.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense.weight)

    @property
    def output_dim(self):
        return self._n_channels

    def forward(self, features):
        # features: B x C x H x W
        assert len(features.shape) == 4
        # self.pos_embedding: H x W x D
        pos_features = self.dense(self.pos_embedding)
        return features + pos_features.permute(2, 0, 1).unsqueeze(0)


class SASetPredictor(nn.Module):
    def __init__(self, slot_dim, output_size, cnn_feature_dim=64, sa_kwargs=None):
        super(SASetPredictor, self).__init__()
        sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        # image size is assumed to be 128x128
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, cnn_feature_dim, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(
                cnn_feature_dim,
                cnn_feature_dim,
                5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                cnn_feature_dim,
                cnn_feature_dim,
                5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(cnn_feature_dim, cnn_feature_dim, 5, padding="same"),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(cnn_feature_dim, eps=0.001)
        self.feature_mlp = nn.Sequential(
            nn.Linear(cnn_feature_dim, cnn_feature_dim),
            nn.ReLU(),
            nn.Linear(cnn_feature_dim, cnn_feature_dim),
        )
        self.pos_embed = SoftPositionEmbedding(cnn_feature_dim, 32)
        self.slot_attention = SlotAttention(
            input_dim=cnn_feature_dim, slot_dim=slot_dim, **sa_kwargs
        )
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 64), nn.ReLU(), nn.Linear(64, output_size), nn.Sigmoid()
        )

        tf_initialize_sequential(self.feature_mlp)
        tf_initialize_sequential(self.cnn_encoder)
        tf_initialize_sequential(self.mlp)

    def encode(self, image, num_slots=None, num_iterations=None):
        features = self.cnn_encoder(image)
        features = self.pos_embed(features)
        features = features.permute(0, 2, 3, 1)  # push channel dim to back
        features = features.view(
            features.shape[0], features.shape[1] * features.shape[2], features.shape[3]
        )
        features = self.feature_mlp(self.layer_norm(features))
        return self.slot_attention(
            features, num_slots=num_slots, num_iters=num_iterations
        )

    def predict(self, latents):
        return self.mlp(latents)

    def forward(self, images, num_slots=None, num_iterations=None):
        info = self.encode(images, num_slots, num_iterations)
        latents = info["slots"]
        result = self.predict(latents)
        info["predicted_properties"] = result
        info["image"] = images
        return info

    def loss(self, images, truth, num_slots=None, num_iterations=None):
        info = self.forward(images, num_slots=num_slots, num_iterations=num_iterations)
        losses = F.huber_loss(
            info["predicted_properties"].unsqueeze(-2),
            truth.unsqueeze(-3),
            reduction="none",
        )  # (B, n_slots, n_objects, n_properties)
        losses = losses.mean(dim=-1)  # (B, n_slots, n_objects)
        row_indices, col_indices = sequential_hungarian(losses.cpu().detach().numpy())
        all_row_indices = np.concatenate(row_indices)
        all_col_indices = np.concatenate(col_indices)
        all_batch_indices = np.repeat(
            np.arange(losses.shape[0]).astype(int), losses.shape[-1]
        )
        total_loss = losses[all_batch_indices, all_row_indices, all_col_indices].mean()
        return total_loss

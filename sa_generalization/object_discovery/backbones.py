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
from torch import nn
from torch.nn import init


class SoftPositionEmbedding(nn.Module):
    def __init__(
        self,
        n_channels,
        resolution,
        concat=False,
        normalize=False,
        learnable_dim=0,
        embedding_weight=None,
    ):
        """Module to add positional embedding to image-like input.

        Module that adds (default) or concatenates positional embedding to input.
        Either fixed (linear ramps, default) or learnable.

        Args:
            n_channels: The feature dimensionality of input
            resolution: The image shape of the input. Input is assumed to be of shape B x resolution x resolution x n_channels
            concat: Whether to concatenate the embedding to the input. Otherwise learn linear layer and add
            normalize: Apply layer norm to input and positional embedding before combination
            learnable_dim: If this is 0, the embedding is learnable of this feature dimension, instead of the linear ramps
            embedding_weight: If normalize is true, this is the weight that the embedding is given
        """
        super(SoftPositionEmbedding, self).__init__()
        self._n_channels = n_channels
        self._concat = concat
        self._normalize = normalize
        if learnable_dim == 0:
            h_inc = torch.linspace(0, 1, resolution)
            w_inc = torch.linspace(0, 1, resolution)
            h_grid, w_grid = torch.meshgrid(h_inc, w_inc)
            positional_embedding = [h_grid, w_grid, 1 - h_grid, 1 - w_grid]
            pos_embedding = torch.stack(positional_embedding, dim=-1)
        else:
            pos_embedding = torch.zeros((resolution, resolution, learnable_dim))
            init.xavier_uniform_(pos_embedding)

        if self._normalize:
            self.normalize_input = nn.LayerNorm(
                (n_channels, resolution, resolution), elementwise_affine=False
            )
            embedding_weight = (
                embedding_weight * torch.ones_like(pos_embedding)
                if embedding_weight != None
                else None
            )
            pos_embedding = F.layer_norm(
                pos_embedding,
                pos_embedding.shape,
                weight=embedding_weight,
            )
        elif embedding_weight != None:
            pos_embedding = embedding_weight * pos_embedding

        if learnable_dim == 0:
            self.register_buffer("pos_embedding", pos_embedding)
        else:
            self.pos_embedding = nn.Parameter(pos_embedding)

        if not self._concat:
            self.dense = nn.Linear(pos_embedding.shape[-1], n_channels)
            self.dense.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.dense.weight)

    @property
    def output_dim(self):
        if not self._concat:
            return self._n_channels
        return self._n_channels + self.pos_embedding.shape[-1]

    @staticmethod
    def output_dim_from_kwargs(**kwargs):
        if not kwargs.get("concat", False):
            return kwargs["n_channels"]

        return kwargs["n_channels"] + 4

    def forward(self, features):
        # features: B x C x H x W
        assert len(features.shape) == 4
        if self._normalize:
            features = self.normalize_input(features)

        # self.pos_embedding: H x W x D
        if not self._concat:
            pos_features = self.dense(self.pos_embedding)
            # Now pos_features is of shape H x W x C

            features = features + pos_features.permute(2, 0, 1).unsqueeze(0)
        else:
            expanded_pos = (
                self.pos_embedding.permute(2, 0, 1)
                .unsqueeze(0)
                .expand((features.shape[0], -1, -1, -1))
            )
            features = torch.cat([features, expanded_pos], dim=1)
        return features


class Cropping2D(nn.Module):
    def __init__(self, d):
        super(Cropping2D, self).__init__()
        self.d = d

    def forward(self, img):
        return img[..., self.d : -self.d, self.d : -self.d]


# From https://github.com/dontLoveBugs/FCRN_pytorch
class UnPool2D(nn.Module):
    def __init__(self, num_channels, stride=2):
        super(UnPool2D, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.weights = nn.Parameter(
            data=torch.zeros(self.num_channels, 1, self.stride, self.stride),
            requires_grad=False,
        )
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(
            x, self.weights, stride=self.stride, groups=self.num_channels
        )


# As described in https://arxiv.org/abs/1606.00373
class UpConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unpool = UnPool2D(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding="same")

    def forward(self, x):
        return self.conv(self.unpool(x))


def get_encoder(cnn_feature_dim, encoder_kernel_size):
    encoder_cnn = nn.Sequential(
        nn.Conv2d(3, cnn_feature_dim, encoder_kernel_size, padding="same"),
        nn.ReLU(),
        nn.Conv2d(
            cnn_feature_dim, cnn_feature_dim, encoder_kernel_size, padding="same"
        ),
        nn.ReLU(),
        nn.Conv2d(
            cnn_feature_dim, cnn_feature_dim, encoder_kernel_size, padding="same"
        ),
        nn.ReLU(),
        nn.Conv2d(
            cnn_feature_dim, cnn_feature_dim, encoder_kernel_size, padding="same"
        ),
        nn.ReLU(),
    )
    return encoder_cnn


def get_decoder(
    im_size,
    slot_dim,
    deconv_architecture,
    decoder_pos_embedding_kwargs,
    cnn_feature_dim,
):
    if deconv_architecture == "expanding" or deconv_architecture == "up_convolve":
        initial_size = 8
    elif deconv_architecture == "stationary":
        initial_size = im_size
    else:
        raise NotImplementedError

    decoder_pos = SoftPositionEmbedding(
        n_channels=slot_dim,
        resolution=initial_size,
        **decoder_pos_embedding_kwargs,
    )
    conv_input_size = cnn_feature_dim

    if deconv_architecture == "expanding":
        activations = [
            [
                nn.ConvTranspose2d(
                    decoder_pos.output_dim,
                    cnn_feature_dim,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    cnn_feature_dim,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    cnn_feature_dim,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    cnn_feature_dim,
                    5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size, cnn_feature_dim, 5, stride=1, padding=2
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    4,
                    3,
                    stride=1,
                    padding=1,
                )
            ],
        ]
    elif deconv_architecture == "stationary":
        # In this case, the initial feature map size matches the output size
        activations = [
            [
                nn.ConvTranspose2d(
                    decoder_pos.output_dim,
                    cnn_feature_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    cnn_feature_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    cnn_feature_dim,
                    5,
                    stride=1,
                    padding=2,
                ),
                nn.ReLU(),
            ],
            [
                nn.ConvTranspose2d(
                    conv_input_size,
                    4,
                    3,
                    stride=1,
                    padding=1,
                ),
            ],
        ]
    elif deconv_architecture == "up_convolve":
        # Uses up-convolution blocks from https://arxiv.org/abs/1606.00373
        activations = [
            [UpConv2D(decoder_pos.output_dim, cnn_feature_dim), nn.ReLU()],
            [UpConv2D(conv_input_size, cnn_feature_dim), nn.ReLU()],
            [UpConv2D(conv_input_size, cnn_feature_dim), nn.ReLU()],
            [UpConv2D(conv_input_size, cnn_feature_dim), nn.ReLU()],
            [
                nn.Conv2d(
                    conv_input_size,
                    4,
                    3,
                    padding="same",
                ),
            ],
        ]
    else:
        raise NotImplementedError

    all_layers = []
    for i, activation in enumerate(activations):
        all_layers += activation

    return initial_size, decoder_pos, nn.Sequential(*all_layers)

"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""


import torch
import torch.nn.functional as F
from numpy import pi
from torch import nn

from sa_generalization.object_discovery.backbones import (
    SoftPositionEmbedding,
    get_decoder,
    get_encoder,
)
from sa_generalization.object_discovery.tf_compatibility import tf_initialize_sequential
from sa_generalization.slot_attention import SlotAttention


class SlotAttentionAutoencoder(nn.Module):
    def __init__(
        self,
        num_slots,
        im_size,
        cnn_feature_dim,
        slot_dim,
        common_dim,
        encoder_kernel_size,
        encoder_pos_embedding_kwargs=None,
        decoder_pos_embedding_kwargs=None,
        num_iterations=3,
        deconv_architecture="expanding",
        sa_kwargs=None,
    ):
        super(SlotAttentionAutoencoder, self).__init__()

        encoder_pos_embedding_kwargs = (
            {} if encoder_pos_embedding_kwargs is None else encoder_pos_embedding_kwargs
        )
        decoder_pos_embedding_kwargs = (
            {} if decoder_pos_embedding_kwargs is None else decoder_pos_embedding_kwargs
        )

        sa_kwargs = sa_kwargs if sa_kwargs is not None else {}

        self.num_slots = num_slots
        self.im_size = im_size

        ###################### ENCODER ######################

        self.encoder_cnn = get_encoder(cnn_feature_dim, encoder_kernel_size)
        self.encoder_pos = SoftPositionEmbedding(
            n_channels=cnn_feature_dim,
            resolution=im_size,
            **encoder_pos_embedding_kwargs,
        )
        feature_map_dim = self.encoder_pos.output_dim
        self.layer_norm = nn.LayerNorm(feature_map_dim, eps=0.001)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_map_dim, cnn_feature_dim),
            nn.ReLU(),
            nn.Linear(cnn_feature_dim, cnn_feature_dim),
        )
        tf_initialize_sequential(self.feature_mlp)
        tf_initialize_sequential(self.encoder_cnn)

        ###################### SLOT ATTENTION ######################

        self.num_iterations = num_iterations

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            input_dim=cnn_feature_dim,
            slot_dim=slot_dim,
            common_dim=common_dim,
            hidden_dim=128,
            iters=num_iterations,  # iterations of attention, defaults to 3
            **sa_kwargs,
        )

        ###################### DECODER ######################

        self.initial_size, self.decoder_pos, self.decoder_cnn = get_decoder(
            im_size,
            slot_dim,
            deconv_architecture,
            decoder_pos_embedding_kwargs,
            cnn_feature_dim,
        )
        tf_initialize_sequential(self.decoder_cnn)

    @property
    def scale(self):
        return self.slot_attention.scale

    @scale.setter
    def scale(self, val):
        self.slot_attention.scale = val

    def decode(self, latent):
        # latent : B x slots x latent_dim
        original_latent_shape = latent.shape
        # We flatten the latent to process everything in one batch
        latent = latent.view(-1, latent.shape[-1])
        # We now broadcast the input
        z = latent.view(latent.shape + (1, 1))
        # Tile across to match image size
        # Shape: NxDx<initial_size>x<initial_size>
        z = z.expand(-1, -1, self.initial_size, self.initial_size)
        z = self.decoder_pos(z)
        image_mask = self.decoder_cnn(z)
        images = image_mask[..., :3, :, :]

        mask_logits = image_mask[..., 3:, :, :]  # Removed softplus
        mask_logits = mask_logits.view(
            original_latent_shape[:-1] + mask_logits.shape[1:]
        )

        masks = torch.softmax(mask_logits, dim=1)

        # images : B x slots x <image dim>>
        images = images.view(original_latent_shape[:-1] + images.shape[1:])
        weighted_images = images * masks
        image = torch.sum(weighted_images, dim=1)
        return {
            "reconstruction": image,
            "slot_wise_reconstructions": images,
            "slot_wise_masks": masks,
            "slot_wise_mask_logits": mask_logits,
        }

    def encode(self, image, num_slots=None, num_iterations=None):
        num_slots = num_slots if num_slots != None else self.num_slots
        num_iterations = (
            num_iterations if num_iterations != None else self.num_iterations
        )
        features = self.encoder_cnn(image)
        features = self.encoder_pos(features)
        # To do things like in tensorflow, we must push the channel dimension to the back before flattening
        features = features.permute(0, 2, 3, 1)

        # features: B x H x W x D
        features = features.view(
            features.shape[0], features.shape[1] * features.shape[2], features.shape[3]
        )
        features = self.feature_mlp(self.layer_norm(features))
        info = self.slot_attention(
            features,
            num_slots=num_slots,
            num_iters=num_iterations,
        )
        return info

    def forward(self, image, num_slots=None, num_iterations=None):
        info = self.encode(image, num_slots=num_slots, num_iterations=num_iterations)
        latent = info["slots"]
        result = self.decode(latent)
        result.update(info)
        result["image"] = image
        return result

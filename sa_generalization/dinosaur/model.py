"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import timm
import torch
from torch import nn

from sa_generalization.slot_attention import SlotAttention


class MLPDecoder(nn.Module):
    def __init__(
        self, slot_size, num_features, hidden_size, output_size, concat_position=False
    ):
        super(MLPDecoder, self).__init__()
        self.num_features = num_features
        self.concat_position = concat_position
        self.decoder_pos_encoding = nn.Parameter(
            torch.randn((1, 1, num_features, slot_size))
        )

        # Define the MLP (First output dimension is mask)
        self.mlp_decoder = nn.Sequential(
            nn.Linear(slot_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, slots):
        batch_size, num_slots = slots.shape[:2]
        broadcasted_slots = slots.unsqueeze(-2).expand((-1, -1, self.num_features, -1))
        expanded_encoding = self.decoder_pos_encoding.expand(
            (batch_size, num_slots, -1, -1)
        )
        if self.concat_position:
            slots_with_encoding = torch.cat(
                [broadcasted_slots, expanded_encoding], dim=-1
            )
        else:
            slots_with_encoding = broadcasted_slots + expanded_encoding
        return self.mlp_decoder(slots_with_encoding)


def parse_decoding(decoded_features_and_masks):
    decoded_features = decoded_features_and_masks[..., 1:]
    masks = decoded_features_and_masks[..., :1]
    masks = torch.softmax(masks, dim=1)
    reconstructed_features = torch.sum(masks * decoded_features, dim=1)
    return {
        "decoded_features": decoded_features,
        "masks": masks,
        "reconstruction": reconstructed_features,
    }


class Dinosaur(nn.Module):
    def __init__(
        self,
        n_input_features,
        feature_size,
        slot_size=128,
        num_slots=11,
        iters=3,
        encoder_name="vit_base_patch8_224_dino",
        decoder_hidden_size=1024,
        sa_kwargs=None,
    ):
        super(Dinosaur, self).__init__()
        sa_kwargs = sa_kwargs if sa_kwargs is not None else {}
        self.num_slots = num_slots
        self.num_features = n_input_features
        self.feature_size = feature_size
        self.slot_size = slot_size

        # Load the encoder and freeze its weights
        self.encoder = timm.create_model(encoder_name, pretrained=True)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Define the slot-attention module
        self.feature_layer_norm = nn.LayerNorm(feature_size)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, slot_size),
        )

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_size,
            input_dim=slot_size,
            iters=iters,
            hidden_dim=4 * slot_size,
            **sa_kwargs
        )
        self.mlp_decoder = MLPDecoder(
            slot_size=slot_size,
            num_features=n_input_features,
            hidden_size=decoder_hidden_size,
            output_size=feature_size + 1,
        )

    def forward(self, x, is_features=False, num_iters=None, num_slots=None):
        # If x does not already represent features, transform it
        if not is_features:
            x = self.encoder.forward_features(x)
            x = x[:, 1:, :]

        result = {"features": x}
        x = self.feature_layer_norm(x)
        x = self.feature_mlp(x)

        info = self.slot_attention(x, num_iters=num_iters, num_slots=num_slots)
        slots = info["slots"]
        decoded_features_and_masks = self.mlp_decoder(slots)
        result.update(parse_decoding(decoded_features_and_masks))
        result.update(info)
        return result

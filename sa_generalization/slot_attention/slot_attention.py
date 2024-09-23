"""
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
This file contains code originally from https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py

Originally licensed under the MIT license, with substantial modifications by Markus Krimmel.
"""

import torch
from torch import nn
from torch.nn import init


def tf_initialize_sequential(sequential):
    for layer in sequential:
        if hasattr(layer, "bias"):
            print(f"Initializing bias of {layer} to 0...")
            layer.bias.data.fill_(0)
        if hasattr(layer, "weight"):
            print(f"Initializing weight of {layer} via Glorot... ")
            nn.init.xavier_uniform_(layer.weight)


class GRUCellTF(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCellTF, self).__init__()
        assert input_dim == hidden_dim
        self.dim = input_dim
        self.kernel = nn.Parameter(torch.zeros(self.dim, 3 * self.dim))
        self.recurrent_kernel = nn.Parameter(torch.zeros(self.dim, 3 * self.dim))
        self.bias = nn.Parameter(torch.zeros(2, 3 * self.dim))
        init.xavier_uniform_(self.kernel)
        init.orthogonal_(self.recurrent_kernel)

    def forward(self, input, hidden):
        input_mapped = torch.matmul(input, self.kernel)
        hidden_mapped = torch.matmul(hidden, self.recurrent_kernel)

        z = torch.sigmoid(
            input_mapped[:, : self.dim]
            + self.bias[0, : self.dim]
            + hidden_mapped[:, : self.dim]
            + self.bias[1, : self.dim]
        )
        r = torch.sigmoid(
            input_mapped[:, self.dim : 2 * self.dim]
            + self.bias[0, self.dim : 2 * self.dim]
            + hidden_mapped[:, self.dim : 2 * self.dim]
            + self.bias[1, self.dim : 2 * self.dim]
        )
        n = torch.tanh(
            input_mapped[:, 2 * self.dim :]
            + self.bias[0, 2 * self.dim :]
            + r * (hidden_mapped[:, 2 * self.dim :] + self.bias[1, 2 * self.dim :])
        )
        return (1 - z) * n + z * hidden


class CustomBatchNorm(nn.Module):
    def __init__(
        self,
        num_iters,
        num_axes,
        channel_axes=(),
        channel_dim=(),
        learnable=True,
        epsilon=1e-5,
        momentum=0.1,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.channel_axes = channel_axes
        self.channel_dim = channel_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.learnable = learnable
        self.reduce_axes = [i for i in range(num_axes) if i not in channel_axes]

        self.register_buffer("means", torch.zeros((num_iters, *channel_dim)))
        self.register_buffer("variances", torch.ones((num_iters, *channel_dim)))

        self.last_statistics = (None, None)

        if learnable:
            param_shape = [num_iters] + [
                1,
            ] * num_axes
            for ax, dim in zip(channel_axes, channel_dim):
                param_shape[ax + 1] = dim
            param_shape = tuple(param_shape)
            self.gamma = nn.Parameter(data=torch.ones(param_shape))
            self.beta = nn.Parameter(data=torch.zeros(param_shape))

    def forward(self, x, idx=0):
        if self.training:
            if idx < self.num_iters:
                var, mean = torch.var_mean(
                    x, dim=self.reduce_axes, keepdim=True, correction=0
                )
                self.last_statistics = (var, mean)

                # We update the buffer!
                with torch.no_grad():
                    buffer_mean = torch.squeeze(mean)
                    buffer_var = torch.squeeze(var)

                    self.means[idx] = (
                        self.momentum * buffer_mean
                        + (1 - self.momentum) * self.means[idx]
                    )
                    self.variances[idx] = (
                        self.momentum * buffer_var
                        + (1 - self.momentum) * self.variances[idx]
                    )
            else:
                idx = self.num_iters - 1
                var, mean = self.last_statistics
        else:
            idx = min(idx, len(self.means) - 1)
            var, mean = self.variances[idx], self.means[idx]

        x = x - mean
        x = x / torch.sqrt(var + self.epsilon)

        if self.learnable:
            x = x * self.gamma[idx] + self.beta[idx]
        return x


class ScalarLayerNorm(nn.Module):
    """
    Differs from standard layer norm in that we can specify the axes to reduce and we only learn scalar parameters for the affine transformation.
    """

    def __init__(self, layer_axes, eps=1e-5):
        super().__init__()
        self.layer_axes = layer_axes
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((1,)))
        self.beta = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=self.layer_axes, keepdim=True, correction=0)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        slot_dim,
        common_dim=None,
        num_slots=None,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
        use_imlicit=False,
        update_normalization="mean",
        tf_gru=True,
    ):
        super().__init__()
        self.common_dim = slot_dim if common_dim is None else common_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.use_implicit = use_imlicit

        self.update_normalization = update_normalization
        if update_normalization == "layer_norm":
            self.update_ln = nn.LayerNorm(slot_dim)
            self.update_normalizer = lambda updates, idx: self.update_ln(updates)
        elif update_normalization == "scalar_batch_norm_single":
            self.update_normalizer = CustomBatchNorm(num_iters=1, num_axes=3)
        elif update_normalization == "image_level":
            self.update_ln = ScalarLayerNorm(layer_axes=(-1, -2))
            self.update_normalizer = lambda updates, idx: self.update_ln(updates)
        elif update_normalization in ["constant", "mean", "none", "scaled_mean"]:
            self.update_normalizer = None
        else:
            raise NotImplementedError

        self.scale = self.common_dim**-0.5

        self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_mu)
        init.xavier_uniform_(self.slots_logsigma)

        self.to_k = nn.Linear(input_dim, self.common_dim, bias=False)
        init.xavier_uniform_(self.to_k.weight)
        self.to_q = nn.Linear(slot_dim, self.common_dim, bias=False)
        init.xavier_uniform_(self.to_q.weight)
        self.to_v = nn.Linear(input_dim, self.common_dim, bias=False)
        init.xavier_uniform_(self.to_v.weight)

        hidden_dim = max(self.common_dim, hidden_dim)

        if tf_gru:
            self.gru = GRUCellTF(self.common_dim, slot_dim)
        else:
            self.gru = nn.GRUCell(self.common_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )
        tf_initialize_sequential(self.mlp)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

    def forward(self, inputs, num_slots=None, num_iters=None, slots=None):
        b, n, _, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        n_i = num_iters if num_iters is not None else self.iters

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # If we use implicit differentiation, we propagate gradients only through the last iteration
        # In particular, slots_mu and slots_logsigma will obtain no gradients at all
        if self.use_implicit:
            prev_mode = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        if slots is None:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

            slots = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            assert num_slots is None

        for idx in range(n_i):
            if idx == (n_i - 1) and self.use_implicit:
                torch.set_grad_enabled(prev_mode)
                slots = slots.detach()

            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1)

            if self.update_normalization in ["mean", "scaled_mean"]:
                _attn = (attn + self.eps) / (attn + self.eps).sum(dim=-1, keepdim=True)
            elif self.update_normalization == "constant":
                _attn = attn / n
            else:
                assert self.update_normalization in [
                    "layer_norm",
                    "scalar_batch_norm_single",
                    "none",
                    "image_level",
                ]
                _attn = attn

            updates = torch.einsum("bjd,bij->bid", v, _attn)
            if self.update_normalizer is not None:
                updates = self.update_normalizer(updates, idx=idx)

            if self.update_normalization == "scaled_mean":
                # Scale updates by the factor num_train_slots / num_inference slots
                updates = updates * (self.num_slots / n_s)

            slots = self.gru(
                updates.reshape(-1, self.common_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )

            slots = slots.reshape(b, -1, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return {"slots": slots, "attention": attn}

from typing import Type

import torch
import torch.nn as nn
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models.models import _iter_maybe_over_single, create_on_device


class ExpandNewDimension(nn.Module):
    def __init__(self, *args, dim_size: int, dim: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_size = dim_size
        self.dim = dim

    def _expand(self, val: torch.Tensor):
        val = val.unsqueeze(self.dim)
        next_dim_idx = self.dim % val.ndim + 1 - val.ndim
        dims_before, dims_after = val.shape[: self.dim], val.shape[next_dim_idx:] if next_dim_idx != 0 else tuple()
        val = val.expand(*dims_before, self.dim_size, *dims_after)
        return val

    def forward(self, *vals: torch.Tensor):
        expanded_vals = tuple(self._expand(val) for val in vals)
        return expanded_vals


class Split(nn.Module):
    def __init__(self, *args, split_size_or_sections: int | list[int], dim: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, val: torch.Tensor):
        split_val = torch.split(val, self.split_size_or_sections, self.dim)
        return split_val


class NegConcat(nn.Module):
    def __init__(self, *args, dim: int = -1, neg_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.neg_first = neg_first

    def forward(self, val: torch.Tensor):
        first, second = val, -val
        if self.neg_first:
            first, second = second, first
        return torch.cat([first, second], dim=self.dim)


class ResMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_cells: list[int] = None,
        activation_class: Type[nn.Module] = nn.PReLU,
        activation_kwargs: dict | None = None,
        norm_class: Type[nn.Module] | None = None,
        norm_kwargs: dict | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        layer_class: Type[nn.Module] = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.dropout = dropout
        self.bias_last_layer = bias_last_layer
        self.layer_class = layer_class

        self.activation_kwargs = activation_kwargs
        self.norm_kwargs = norm_kwargs
        self.layer_kwargs = layer_kwargs

        self.activate_last_layer = activate_last_layer

        self.num_cells = num_cells
        self.depth = len(self.num_cells)

        self._activation_kwargs_iter = _iter_maybe_over_single(
            activation_kwargs, n=self.depth + self.activate_last_layer
        )
        self._norm_kwargs_iter = _iter_maybe_over_single(norm_kwargs, n=self.depth + self.activate_last_layer)
        self._layer_kwargs_iter = _iter_maybe_over_single(layer_kwargs, n=self.depth + 1)
        self.layers = self._make_net(device)

    def _make_net(self, device: DEVICE_TYPING | None):
        layers: list[nn.Sequential] = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self.out_features]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            layer = []
            layer_kwargs = next(self._layer_kwargs_iter)
            _bias = layer_kwargs.pop("bias", self.bias_last_layer if i == self.depth else True)
            if i > 0:
                _in += in_features[i - 1]
            layer.append(create_on_device(self.layer_class, device, _in, _out, bias=_bias, **layer_kwargs))

            if i < self.depth or self.activate_last_layer:
                norm_kwargs = next(self._norm_kwargs_iter)
                activation_kwargs = next(self._activation_kwargs_iter)
                if self.dropout is not None:
                    layer.append(create_on_device(nn.Dropout, device, p=self.dropout))
                if self.norm_class is not None:
                    layer.append(create_on_device(self.norm_class, device, **norm_kwargs))
                layer.append(create_on_device(self.activation_class, device, **activation_kwargs))

            layers.append(nn.Sequential(*layer))

        return nn.ModuleList(layers)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        val = torch.cat([*inputs], -1)

        prev_val, val = val, self.layers[0](val)
        for layer in self.layers[1:]:
            prev_val, val = val, layer(torch.cat([val, prev_val], dim=-1))

        return val

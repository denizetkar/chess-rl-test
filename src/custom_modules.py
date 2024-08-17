import torch
import torch.nn as nn


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

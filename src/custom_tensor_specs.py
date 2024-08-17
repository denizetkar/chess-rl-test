from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch

from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, CompositeSpec
from torchrl.data.tensor_specs import (
    Box,
    invertible_dict,
    DEVICE_TYPING,
    _remove_neg_shapes,
    _CHECK_SPEC_ENCODE,
    _squeezed_shape,
    _unsqueezed_shape,
    indent,
)


@dataclass(repr=False)
class DependentDiscreteBoxes(Box):
    """A sequence of dependent box of discrete values."""

    n_vec: tuple[int]
    register = invertible_dict()

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> "DependentDiscreteBoxes":
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_vec={self.n_vec})"


class DependentDiscreteTensorsSpec(CompositeSpec):
    shape: torch.Size
    dtype: torch.dtype = torch.long
    domain: str = "composite"

    def __init__(
        self,
        n_vec: tuple[int],
        shape: torch.Size | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype = torch.long,
        mask: torch.Tensor | None = None,
    ):
        super().__init__(
            {str(i): DiscreteTensorSpec(n, shape, device, dtype) for i, n in enumerate(n_vec)},
            shape=shape,
            device=device,
        )
        self.n_vec = n_vec
        self.update_mask(mask)

    def update_mask(self, mask: torch.Tensor | None = None):
        if mask is not None:
            try:
                mask = mask.expand(_remove_neg_shapes(*self.shape, *self.n_vec))
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def _get_next_mask(self, actions: list[torch.Tensor]) -> torch.Tensor:
        batch_dims = self.shape
        decision_dims = self.n_vec
        num_actions_taken = len(actions)
        num_actions_not_taken = len(decision_dims) - num_actions_taken

        grid_ranges = [torch.arange(s) for s in batch_dims]
        batch_indices = torch.meshgrid(*grid_ranges, indexing="ij") if len(grid_ranges) > 0 else ()
        action_indices = tuple(actions)
        indices = batch_indices + action_indices

        # self.mask cannot be None here
        filtered_mask = self.mask[indices]

        legal_action_mask = filtered_mask.any(dim=list(range(-num_actions_not_taken + 1, 0)))
        return legal_action_mask

    def _update_next_mask(self, actions: list[torch.Tensor]):
        i = len(actions)
        dependent_mask = self._get_next_mask(actions) if self.mask is not None else None
        self[str(i)].update_mask(dependent_mask)

    def rand(self, shape=None) -> TensorDict:
        # We cannot offer flexible shapes because our mask is only for `self.shape`
        actions: list[torch.Tensor] = []
        for i, dependent_var in enumerate(self.values()):
            self._update_next_mask(actions)
            action_i = dependent_var.rand()
            actions.append(action_i)
        return TensorDict({str(i): v for i, v in enumerate(actions)})

    def _project(self, val: TensorDict) -> TensorDict:
        if len(val.keys()) != len(self.n_vec):
            raise ValueError(f"Cannot project value: n_vec doesn't match {self.n_vec}")
        actions: list[torch.Tensor] = []
        for i, dependent_var in enumerate(self.values()):
            self._update_next_mask(actions)
            action_i = dependent_var._project(val[str(i)])
            actions.append(action_i)
        return TensorDict({str(i): v for i, v in enumerate(actions)})

    def is_in(self, val: TensorDict) -> bool:
        if len(val.keys()) != len(self.n_vec):
            return False
        actions = []
        for i, dependent_var in enumerate(self.values()):
            self._update_next_mask(actions)
            action_i = val[str(i)]
            is_in_i = dependent_var.is_in(action_i)
            if not is_in_i:
                return False
            actions.append(action_i)
        return True

    def to_one_hot(self, val: TensorDict, safe: bool = None) -> TensorDict:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        td_out = TensorDict()
        for i_str, action_i in val.items():
            one_hot_i = self[i_str].to_one_hot(action_i, safe)
            td_out[i_str] = one_hot_i
        return td_out

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        mask = self.mask.expand(shape) if self.mask is not None else None
        return self.__class__(n_vec=self.n_vec, shape=shape, device=self.device, dtype=self.dtype, mask=mask)

    def _reshape(self, shape):
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return self.__class__(n_vec=self.n_vec, shape=shape, device=self.device, dtype=self.dtype, mask=mask)

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        mask = self.mask.unflatten(shape) if self.mask is not None else None
        return self.__class__(n_vec=self.n_vec, shape=shape, device=self.device, dtype=self.dtype, mask=mask)

    def squeeze(self, dim=None):
        shape = _squeezed_shape(self.shape, dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape, *mask.shape[-len(self.n_vec) :])

        if shape is None:
            return self
        return self.__class__(
            n_vec=self.n_vec,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape, *mask.shape[-len(self.n_vec) :])
        return self.__class__(
            n_vec=self.n_vec,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unbind(self, dim: int = 0):
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(f"Cannot unbind along dim {orig_dim} with shape {self.shape}.")
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        mask = self.mask
        if mask is None:
            mask = (None,) * self.shape[dim]
        else:
            mask = mask.unbind(dim)
        return tuple(
            self.__class__(
                n_vec=self.n_vec,
                shape=shape,
                device=self.device,
                dtype=self.dtype,
                mask=mask[i],
            )
            for i in range(self.shape[dim])
        )

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]):
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n_vec=self.n_vec,
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
            mask=self.mask.to(device=dest_device) if self.mask is not None else None,
        )

    def clone(self):
        return self.__class__(
            n_vec=self.n_vec,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def empty(self):
        raise NotImplementedError()

    def __repr__(self) -> str:
        sub_str = [indent(f"{k}: {str(item)}", 4 * " ") for k, item in self._specs.items()]
        sub_str = ",\n".join(sub_str)
        return f"DependentDiscreteTensorsSpec(\n{sub_str},\n    device={self._device},\n    shape={self.shape})"

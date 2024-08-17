from typing import Sequence
import torch
from tensordict.utils import NestedKey
from torch import dtype
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms import DTypeCastTransform


class MyDTypeCastTransform(DTypeCastTransform):
    def __init__(
        self,
        dtype_in: dtype,
        dtype_out: dtype,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__(dtype_in, dtype_out, in_keys, out_keys, in_keys_inv, out_keys_inv)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        full_action_spec = input_spec["full_action_spec"]
        full_state_spec = input_spec["full_state_spec"]
        # if this method is called, then it must have a parent and in_keys_inv will be defined
        if self.in_keys_inv is None:
            raise NotImplementedError(
                f"Calling transform_input_spec without a parent environment isn't supported yet for {type(self)}."
            )
        for in_key_inv, out_key_inv in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key_inv in full_action_spec.keys(True):
                _spec = full_action_spec[in_key_inv]
                target = "action"
            elif in_key_inv in full_state_spec.keys(True):
                _spec = full_state_spec[in_key_inv]
                target = "state"
            else:
                # It may be that we are modifying an output spec. But `DTypeCastTransform` throws an error here. WTF?
                return input_spec
            if _spec.dtype != self.dtype_in:
                raise TypeError(f"input_spec[{in_key_inv}].dtype is not {self.dtype_in}: {in_key_inv.dtype}")
            _spec = self._transform_spec(_spec)
            if target == "action":
                full_action_spec[out_key_inv] = _spec
            elif target == "state":
                full_state_spec[out_key_inv] = _spec
            else:
                # unreachable
                raise RuntimeError
        return input_spec


class DiscreteToContinuousTransform(MyDTypeCastTransform):
    def __init__(
        self,
        dtype_in: dtype,
        dtype_out: dtype,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        if dtype_in.is_floating_point or not dtype_out.is_floating_point:
            raise ValueError("Input type must be discrete and output type continuous")
        super().__init__(dtype_in, dtype_out, in_keys, out_keys, in_keys_inv, out_keys_inv)

    def _transform_spec(self, spec: TensorSpec) -> None:
        if not isinstance(spec, DiscreteTensorSpec):
            return super()._transform_spec(spec)

        cont_spec = UnboundedContinuousTensorSpec(
            shape=spec.shape, device=spec.device, dtype=self.dtype_out, domain="continuous"
        )
        return cont_spec

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return state.round().to(self.dtype_in)


class DoubleToFloat(MyDTypeCastTransform):
    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        super().__init__(
            dtype_in=torch.double,
            dtype_out=torch.float,
            in_keys=in_keys,
            in_keys_inv=in_keys_inv,
            out_keys=out_keys,
            out_keys_inv=out_keys_inv,
        )

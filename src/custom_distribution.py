from typing import Callable

import torch

from tensordict.nn import CompositeDistribution
from torchrl.modules.distributions import MaskedCategorical

from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.utils import _unravel_key_to_tuple, unravel_key

from custom_modules import ExpandNewDimension
from utils import _get_next_mask


class DependentCategoricalsDistribution(CompositeDistribution):
    def __init__(
        self,
        n_actions: int,
        observations: TensorDict | torch.Tensor,
        action_key: NestedKey,
        log_prob_key: NestedKey,
        logits_fn: Callable[[TensorDict | torch.Tensor, list[torch.Tensor]], torch.Tensor],
        n_agents: int | None = None,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            logits_fn: takes in observations as well as all previous actions as input
                and gives the logits of the next action.
        """
        super().__init__(TensorDict(), {}, name_map=None, extra_kwargs=None)
        self.dists: dict[str, MaskedCategorical]

        if unravel_key(log_prob_key) != "sample_log_prob":
            raise ValueError("CompositeDistribution uses `sample_log_prob` as the hardcoded key. WTF?")

        self.n_actions = n_actions
        self.observations = observations
        self.action_key = _unravel_key_to_tuple(action_key)
        self.log_prob_key = _unravel_key_to_tuple(log_prob_key)
        self.logits_fn = logits_fn
        self.agent_dim_expander = ExpandNewDimension(dim_size=n_agents, dim=-1) if n_agents is not None else None
        self.mask = mask

    def _get_next_mask(self, actions: list[torch.Tensor]) -> torch.Tensor:
        return _get_next_mask(
            self.mask.shape[: -self.n_actions], self.mask.shape[-self.n_actions :], self.mask, actions
        )

    def sample(self, shape=None) -> TensorDictBase:
        actions: list[torch.Tensor] = []
        for i in range(self.n_actions):
            logits_i = self.logits_fn(self.observations, actions)
            dependent_mask = (
                self._get_next_mask(actions) if self.mask is not None else torch.ones_like(logits_i, dtype=torch.bool)
            )
            self.dists[str(i)] = MaskedCategorical(logits=logits_i, mask=dependent_mask)
            action_i = self.dists[str(i)].sample(shape)
            actions.append(action_i)
        return TensorDict({(*self.action_key, str(i)): v for i, v in enumerate(actions)})

    def _fill_dists(self, actions_td: TensorDictBase):
        actions: list[torch.Tensor] = []
        for i in range(self.n_actions):
            logits_i = self.logits_fn(self.observations, actions)
            dependent_mask = (
                self._get_next_mask(actions) if self.mask is not None else torch.ones_like(logits_i, dtype=torch.bool)
            )
            self.dists[str(i)] = MaskedCategorical(logits=logits_i, mask=dependent_mask)
            action_i = actions_td[str(i)]
            actions.append(action_i)

    def log_prob(self, actions_td: TensorDictBase) -> TensorDictBase | torch.Tensor:
        # `PPOLoss._log_weight` sends the inner td, `ProbabilisticTensorDictModule.forward` sends the whole td. WTF?
        inner_actions_td = actions_td.get(self.action_key, actions_td)

        self._fill_dists(inner_actions_td)
        log_p = torch.stack(
            [self.dists[str(i)].log_prob(inner_actions_td[str(i)]) for i in range(self.n_actions)], dim=-1
        ).sum(dim=-1)
        # `ClipPPOLoss.forward` expects the same shape from `sample_log_probs` and `advantage`. WTF?
        if self.agent_dim_expander is not None:
            remaining_shape = log_p.shape[len(inner_actions_td.shape) :]
            self.agent_dim_expander.dim = -1 if len(remaining_shape) == 0 else -2
            expanded_log_p: tuple[torch.Tensor] = self.agent_dim_expander(log_p)
            log_p = expanded_log_p[0]

        # `PPOLoss._log_weight` expects a torch.Tensor unlike `ProbabilisticTensorDictModule.forward`. WTF?
        if self.action_key not in actions_td:
            return log_p

        actions_td.set(self.log_prob_key, log_p)
        return actions_td

    def entropy(self) -> torch.Tensor:
        return torch.stack([self.dists[str(i)].entropy() for i in range(self.n_actions)], dim=-1).sum(dim=-1)

    @property
    def mode(self) -> TensorDictBase:
        raise NotImplementedError()

    @property
    def mean(self) -> TensorDictBase:
        raise NotImplementedError()

    @property
    def deterministic_sample(self) -> TensorDictBase:
        raise NotImplementedError()

    def rsample(self, shape=None) -> TensorDictBase:
        raise NotImplementedError()

    def cdf(self, sample: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError()

    def icdf(self, sample: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError()

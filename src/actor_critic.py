from typing import Any, Callable
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from chess_env import ChessEnv
from torchrl.envs import EnvBase

# from torchrl.envs.utils import check_env_specs

from torchrl.modules import MLP, ProbabilisticActor
from custom_modules import ExpandNewDimension, NegConcat
from tensordict.nn import InteractionType
from custom_distribution import DependentCategoricalsDistribution


def create_action_nets(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device):
    action_dims = [a.n for a in final_env.full_action_spec[base_env.action_key].values()]
    obs_total_dims = sum([final_env.full_observation_spec[key].shape[-1] for key in base_env.observation_keys])
    # Only thing to persist to disk is `action_nets`
    action_nets = nn.ModuleDict(
        {
            str(i): MLP(
                in_features=obs_total_dims + i,
                out_features=action_dim,
                depth=2,
                num_cells=512,
                activation_class=torch.nn.PReLU,
            ).to(device=default_device)
            for i, action_dim in enumerate(action_dims)
        }
    )
    return action_nets


def save_action_nets(action_nets: nn.ModuleDict, save_path: str):
    with open(save_path, "wb") as f:
        params = action_nets.state_dict()
        torch.save(params, f)


def load_action_nets(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device, save_path: str):
    action_nets = create_action_nets(base_env, final_env, default_device)
    with open(save_path, "rb") as f:
        params: dict[str, Any] = torch.load(f)
    action_nets.load_state_dict(params)
    return action_nets


def create_logits_fn(base_env: ChessEnv, action_nets: nn.ModuleDict):

    def logits_fn(observations: TensorDict, actions: list[torch.Tensor]) -> torch.Tensor:
        i = len(actions)
        obs_tensors = [observations[key[1:]] for key in base_env.observation_keys]
        concatable_actions = [a.unsqueeze(-1) for a in actions]
        logit_i: torch.Tensor = action_nets[str(i)](*obs_tensors, *concatable_actions)
        return logit_i

    return logits_fn


def create_actor(
    base_env: ChessEnv,
    final_env: EnvBase,
    default_device: torch.device,
    logits_fn: Callable[[TensorDict, list[torch.Tensor]], torch.Tensor],
):
    # Nothing to persist to disk here
    action_dims = [a.n for a in final_env.full_action_spec[base_env.action_key].values()]
    identity_module = TensorDictModule(module=lambda *args: args, in_keys=[], out_keys=[])
    actor = ProbabilisticActor(
        module=identity_module,
        spec=final_env.full_action_spec,
        in_keys={"observations": base_env.OBSERVATION_KEY, "mask": "action_mask"},
        # We shouldn't have to specify the `out_keys` below but otherwise `ProbabilisticActor.__init__` complains about
        # not having `distribution_map` in `distribution_kwargs`. It doesn't allow to customize `CompositeDistribution`
        # that doesn't take `distribution_map` arg. WTF?
        out_keys=[*final_env.full_action_spec.keys(True, True)],
        distribution_class=DependentCategoricalsDistribution,
        distribution_kwargs={
            "n_actions": len(action_dims),
            "action_key": base_env.action_key,
            "log_prob_key": "sample_log_prob",
            "logits_fn": logits_fn,
            "n_agents": base_env.n_agents,
        },
        # For some reason, the default is deterministic sample. Why wouldn't they randomly sample by default? WTF?
        default_interaction_type=InteractionType.RANDOM,
        # The default is to cache and therefore cannot get the latest action mask params. WTF?
        cache_dist=False,
        return_log_prob=True,  # PPO loss needs log probs
        log_prob_key="sample_log_prob",
    )

    return actor


def create_critic(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device):
    obs_without_turn_keys = [key for key in base_env.observation_keys if key != (base_env.OBSERVATION_KEY, "turn")]
    obs_without_turn_total_dims = sum([final_env.full_observation_spec[key].shape[-1] for key in obs_without_turn_keys])
    critic_modules = (
        TensorDictModule(
            MLP(
                in_features=obs_without_turn_total_dims,
                out_features=1,
                depth=2,
                num_cells=512,
                activation_class=torch.nn.PReLU,
            ),
            in_keys=[*obs_without_turn_keys],
            out_keys=["state_value"],
        ),
        # Single state value estimation. Invert for the white player
        TensorDictModule(
            NegConcat(dim=-1, neg_first=False), in_keys=[("state_value")], out_keys=[("agents", "state_value")]
        ),
        # Unsqueeze state_value from the last dimension
        TensorDictModule(
            ExpandNewDimension(dim_size=1, dim=-1),
            in_keys=[("agents", "state_value")],
            out_keys=[("agents", "state_value")],
        ),
    )
    # Only thing to persist to disk is `critic.state_dict()`
    critic = TensorDictSequential(*critic_modules).to(device=default_device)

    return critic


def save_critic(critic: TensorDictSequential, save_path: str):
    with open(save_path, "wb") as f:
        torch.save(critic.state_dict(), f)


def load_critic(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device, save_path: str):
    critic = create_critic(base_env, final_env, default_device)
    with open(save_path, "rb") as f:
        params: dict[str, Any] = torch.load(f)
    critic.load_state_dict(params)
    return critic

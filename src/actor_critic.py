import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from chess_env import ChessEnv
from torchrl.envs import EnvBase

# from torchrl.envs.utils import check_env_specs

from torchrl.modules import MLP, ProbabilisticActor
from custom_modules import ExpandNewDimension
from tensordict.nn import InteractionType
from custom_distribution import DependentCategoricalsDistribution


def get_actor(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device):

    action_dims = [a.n for a in final_env.full_action_spec[base_env.action_key].values()]
    obs_total_dims = sum([final_env.full_observation_spec[key].shape[-1] for key in base_env.observation_keys])
    action_nets = {
        i: MLP(
            in_features=obs_total_dims + i,
            out_features=action_dim,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.PReLU,
        ).to(device=default_device)
        for i, action_dim in enumerate(action_dims)
    }

    def logits_fn(observations: TensorDict, actions: list[torch.Tensor]) -> torch.Tensor:
        i = len(actions)
        obs_tensors = [observations[key[1:]] for key in base_env.observation_keys]
        concatable_actions = [a.unsqueeze(-1) for a in actions]
        logit_i: torch.Tensor = action_nets[i](*obs_tensors, *concatable_actions)
        return logit_i

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


def get_critic(base_env: ChessEnv, final_env: EnvBase, default_device: torch.device):
    obs_without_turn_keys = [key for key in base_env.observation_keys if key != (base_env.OBSERVATION_KEY, "turn")]
    obs_without_turn_total_dims = sum([final_env.full_observation_spec[key].shape[-1] for key in obs_without_turn_keys])
    critic_modules = (
        TensorDictModule(
            MLP(
                in_features=obs_without_turn_total_dims,
                out_features=base_env.n_agents,  # One state value estimation for each agent
                depth=2,
                num_cells=256,
                activation_class=torch.nn.PReLU,
            ),
            in_keys=[*obs_without_turn_keys],
            out_keys=[("agents", "state_value")],
        ),
        # Unsqueeze state_value from the last dimension
        TensorDictModule(
            ExpandNewDimension(dim_size=1, dim=-1),
            in_keys=[("agents", "state_value")],
            out_keys=[("agents", "state_value")],
        ),
    )
    critic = TensorDictSequential(*critic_modules).to(device=default_device)

    return critic

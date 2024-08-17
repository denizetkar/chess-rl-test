import torch
import logging

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from chess_env import ChessEnv
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs import ParallelEnv, RewardSum

# from torchrl.envs.utils import check_env_specs

from torchrl.modules import MLP, ProbabilisticActor
from custom_modules import ExpandNewDimension
from tensordict.nn import InteractionType
from custom_distribution import DependentCategoricalsDistribution

from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torchrl.record.loggers import TensorboardLogger


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
        level=logging.INFO,
    )

    # HYPERPARAMS
    is_fork = multiprocessing.get_start_method() == "fork"
    # default_device = torch.device("cuda") if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    default_device = torch.device("cpu")

    frames_per_worker_batch = 500
    n_envs = 4
    frames_per_batch = frames_per_worker_batch * n_envs

    outer_epochs = 1000
    total_frames = frames_per_batch * outer_epochs
    inner_epochs = 10
    training_iters = 1
    minibatch_size = frames_per_batch // training_iters
    max_grad_norm = 1.0
    lr = 1e-4

    clip_epsilon = 0.1
    gamma = 0.999
    lmbda = 0.9
    entropy_eps = 1e-4

    env = ChessEnv()
    transforms = Compose(
        *env.create_obs_transforms(), RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    )

    def create_tenv():
        env = ChessEnv()
        tenv = TransformedEnv(env, transforms.clone(), cache_specs=False, device=default_device)
        # check_env_specs(tenv)
        return tenv

    penv = ParallelEnv(n_envs, create_tenv)
    # penv = create_tenv()

    action_dims = [a.n for a in penv.full_action_spec[env.action_key].values()]
    obs_total_dims = sum([penv.full_observation_spec[key].shape[-1] for key in env.observation_keys])
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
        obs_tensors = [observations[key[1:]] for key in env.observation_keys]
        concatable_actions = [a.unsqueeze(-1) for a in actions]
        logit_i: torch.Tensor = action_nets[i](*obs_tensors, *concatable_actions)
        return logit_i

    identity_module = TensorDictModule(module=lambda *args: args, in_keys=[], out_keys=[])
    actor = ProbabilisticActor(
        module=identity_module,
        spec=penv.full_action_spec,
        in_keys={"observations": env.OBSERVATION_KEY, "mask": "action_mask"},
        # We shouldn't have to specify the `out_keys` below but otherwise `ProbabilisticActor.__init__` complains about
        # not having `distribution_map` in `distribution_kwargs`. It doesn't allow to customize `CompositeDistribution`
        # that doesn't take `distribution_map` arg. WTF?
        out_keys=[*penv.full_action_spec.keys(True, True)],
        distribution_class=DependentCategoricalsDistribution,
        distribution_kwargs={
            "n_actions": len(action_dims),
            "action_key": env.action_key,
            "log_prob_key": "sample_log_prob",
            "logits_fn": logits_fn,
            "n_agents": env.n_agents,
        },
        # For some reason, the default is deterministic sample. Why wouldn't they randomly sample by default? WTF?
        default_interaction_type=InteractionType.RANDOM,
        # The default is to cache and therefore cannot get the latest action mask params. WTF?
        cache_dist=False,
        return_log_prob=True,  # PPO loss needs log probs
        log_prob_key="sample_log_prob",
    )

    obs_without_turn_keys = [key for key in env.observation_keys if key != (env.OBSERVATION_KEY, "turn")]
    obs_without_turn_total_dims = sum([penv.full_observation_spec[key].shape[-1] for key in obs_without_turn_keys])
    critic_modules = (
        TensorDictModule(
            MLP(
                in_features=obs_without_turn_total_dims,
                out_features=env.n_agents,  # One state value estimation for each agent
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

    collector = SyncDataCollector(
        penv,
        actor,
        device=default_device,
        storing_device=default_device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=default_device),
        sampler=SamplerWithoutReplacement(shuffle=False),
        batch_size=minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # avoid normalizing across the agent dimension
        reduction="none",  # Otherwise, at each step all agent losses are (attempted to be) minimized
    )
    loss_module.set_keys(
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
        value=("agents", "state_value"),
        sample_log_prob="sample_log_prob",
        action=env.action_key,
        reward=env.reward_key,
        done=env.done_key,
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, outer_epochs, 0.0)

    logger = TensorboardLogger(exp_name="Chess PPO", log_dir="tb_logs")
    for outer_i, td_data in enumerate(collector):

        with torch.no_grad():
            GAE(
                td_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        # Flatten (n_envs, frames_per_worker_batch) to (frames_per_batch,)
        data_view = td_data.reshape(-1)
        replay_buffer.extend(data_view)

        logging.info(f"Outer epoch {outer_i}")
        for _ in range(inner_epochs):
            for _ in range(training_iters):
                subdata: TensorDict = replay_buffer.sample()
                loss_vals: TensorDict = loss_module(subdata)

                turn_data = transforms.inv(subdata)[env.OBSERVATION_KEY, "turn"]
                losses: TensorDict = (
                    loss_vals.select("loss_objective").auto_batch_size_().gather(index=turn_data, dim=-1).mean()
                )
                losses.update(loss_vals.select("loss_critic", "loss_entropy").mean())
                loss_value = torch.stack(list(losses.values()), dim=0).sum(dim=0)

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        turn_data = transforms.inv(td_data)[env.OBSERVATION_KEY, "turn"].unsqueeze(-2)
        selected_rewards = td_data.get(("next", "agents", "episode_reward")).gather(index=turn_data, dim=-2)
        episode_reward_min = selected_rewards.min().item()
        episode_reward_mean = selected_rewards.mean().item()
        episode_reward_max = selected_rewards.max().item()
        episode_games_done = td_data["next", "done"].sum().item()
        logger.log_scalar("episode_reward_min", episode_reward_min, step=outer_i)
        logger.log_scalar("episode_reward_mean", episode_reward_mean, step=outer_i)
        logger.log_scalar("episode_reward_max", episode_reward_max, step=outer_i)
        logger.log_scalar("episode_games_done", episode_games_done, step=outer_i)
        logging.info(
            "Min/Average/Max episode reward: %f/%f/%f", episode_reward_min, episode_reward_mean, episode_reward_max
        )
        logging.info("Episode games done: %d", episode_games_done)

        collector.update_policy_weights_()
        scheduler.step()

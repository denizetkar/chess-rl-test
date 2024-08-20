import torch
import logging
import os

from tensordict import TensorDict
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from chess_env import ChessEnv
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.envs import ParallelEnv, RewardSum

# from torchrl.envs.utils import check_env_specs

from actor_critic import load_action_nets, save_action_nets, create_actor, load_critic, save_critic, create_logits_fn

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
    n_envs = 6
    frames_per_batch = frames_per_worker_batch * n_envs

    outer_epochs = 1000
    total_frames = frames_per_batch * outer_epochs
    inner_epochs = 10
    training_iters = 1
    minibatch_size = frames_per_batch // training_iters
    max_grad_norm = 10.0
    lr = 1e-4

    clip_epsilon = 0.1
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    obs_transforms_save_path = "./lightning_logs/version_0/checkpoints/epoch=74-step=37575-obs_transforms.pt"
    action_nets_save_path = "./lightning_logs/version_0/checkpoints/epoch=74-step=37575-action_nets.pt"
    critic_save_path = "./lightning_logs/version_0/checkpoints/epoch=74-step=37575-critic.pt"

    env = ChessEnv()
    transforms = Compose(
        *env.load_obs_transforms(obs_transforms_save_path),
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    def create_tenv():
        env = ChessEnv()
        tenv = TransformedEnv(env, transforms.clone(), cache_specs=False, device=default_device)
        # check_env_specs(tenv)
        return tenv

    penv = ParallelEnv(n_envs, create_tenv)
    # penv = create_tenv()

    action_nets = load_action_nets(env, penv, default_device, action_nets_save_path)
    actor = create_actor(env, penv, default_device, create_logits_fn(env, action_nets))
    critic = load_critic(env, penv, default_device, critic_save_path)

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
                losses.update(loss_vals.select("loss_critic", "loss_entropy").auto_batch_size_().mean(dim=0).sum())
                loss_value = torch.stack(list(losses.values()), dim=0).sum(dim=0)

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        turn_data = transforms.inv(td_data)[env.OBSERVATION_KEY, "turn"].unsqueeze(-2)
        selected_rewards = td_data.get(("next", "agents", "episode_reward")).gather(index=turn_data, dim=-2)
        episode_reward_mean = selected_rewards.mean().item()
        episode_reward_abs_sum = selected_rewards.type(torch.long).abs().sum().item()
        episode_games_done = td_data["next", "done"].sum().item()
        logger.log_scalar("episode_reward_mean", episode_reward_mean, step=outer_i)
        logger.log_scalar("episode_reward_abs_sum", episode_reward_abs_sum, step=outer_i)
        logger.log_scalar("episode_games_done", episode_games_done, step=outer_i)
        logging.info("Average/AbsSum episode reward: %f/%f", episode_reward_mean, episode_reward_abs_sum)
        logging.info("Episode games done: %d", episode_games_done)

        collector.update_policy_weights_()
        scheduler.step()

    save_action_nets(action_nets, os.path.splitext(action_nets_save_path)[0] + "-PPO.pt")
    save_critic(critic, os.path.splitext(critic_save_path)[0] + "-PPO.pt")

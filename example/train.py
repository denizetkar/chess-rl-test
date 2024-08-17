import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from torchrl.objectives import ClipPPOLoss, ValueEstimators

torch.manual_seed(0)


if __name__ == "__main__":
    # HYPERPARAMS
    is_fork = multiprocessing.get_start_method() == "fork"
    default_device = torch.device("cuda") if torch.cuda.is_available() and not is_fork else torch.device("cpu")

    frames_per_worker_batch = 100
    n_envs = 60
    frames_per_batch = frames_per_worker_batch * n_envs

    outer_epochs = 5
    total_frames = frames_per_batch * outer_epochs
    inner_epochs = 10
    training_iters = 1
    minibatch_size = frames_per_batch // training_iters
    max_grad_norm = 1.0
    lr = 1e-4

    clip_epsilon = 0.1
    gamma = 0.99
    lmbda = 0.9
    entropy_eps = 1e-4

    max_steps = frames_per_worker_batch
    scenario_name = "navigation"
    n_agents = 3

    env = VmasEnv(
        scenario=scenario_name,
        num_envs=n_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=default_device,
        n_agents=n_agents,
    )
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]))
    check_env_specs(env)
    r = env.rollout(5)
    print(r)

    share_parameters_policy = True
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=n_agents,
            centralized=True,
            share_params=share_parameters_policy,
            device=default_device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        policy_net, in_keys=[("agents", "observation")], out_keys=[("agents", "loc"), ("agents", "scale")]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,  # PPO loss needs log probs
        log_prob_key=("agents", "sample_log_prob"),
    )

    share_parameters_critic = True
    mappo = True
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=n_agents,
        centralized=mappo,
        share_params=share_parameters_critic,
        device=default_device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    critic = TensorDictModule(
        module=critic_net, in_keys=[("agents", "observation")], out_keys=[("agents", "state_value")]
    )

    td = policy(env.reset())
    print(td)

    collector = SyncDataCollector(
        env,
        policy,
        device=default_device,
        storing_device=default_device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=default_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # avoid normalizing across the agent dimension
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, outer_epochs, 0.0)

    episode_reward_mean_list = []
    for tensordict_data in collector:
        # Expand to match the reward shape (expected by the value estimator ???)
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-2)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-2)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(inner_epochs):
            for _ in range(training_iters):
                subdata = replay_buffer.sample()
                loss_vals: TensorDict = loss_module(subdata)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()
        scheduler.step()

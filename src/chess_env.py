from typing import Any
import chess
import logging
import os

import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.envs.transforms import TransformedEnv, ObservationNorm, Compose, Transform
from torchrl.data import CompositeSpec, DiscreteTensorSpec, MultiOneHotDiscreteTensorSpec, BoundedTensorSpec

from custom_tensor_specs import DependentDiscreteTensorsSpec
from custom_transforms import DiscreteToContinuousTransform, DoubleToFloat
from utils import _get_move_external


class ChessEnv(EnvBase):
    OBSERVATION_KEY = "observation"

    def __init__(
        self,
        *,
        rand_player_idx: int | None = None,
        device: torch.device | str | int = None,
        batch_size: torch.Size | None = None,
        run_type_checks: bool = False,
        allow_done_after_reset: bool = False,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
            allow_done_after_reset=allow_done_after_reset,
        )
        self.rand_player_idx = rand_player_idx % 2 == 1 if rand_player_idx is not None else None
        self.n_agents = 2
        self.total_piece_cnt = 2 * len(chess.PIECE_TYPES)
        self.observation_spec = CompositeSpec(
            {
                ChessEnv.OBSERVATION_KEY: CompositeSpec(
                    # 0 for no piece at that position, 1-6 for black, 7-12 for white pieces
                    piece_at_pos=MultiOneHotDiscreteTensorSpec([self.total_piece_cnt] * 64, dtype=torch.bool),
                    turn=DiscreteTensorSpec(self.n_agents, shape=(1,), dtype=torch.bool),
                )
            },
            action_mask=DiscreteTensorSpec(2, shape=(64, 64), dtype=torch.bool),
        )
        self.action_spec = CompositeSpec(action=DependentDiscreteTensorsSpec((64, 64)))
        self._action_keys = ["action"]

        # GAE value estimator expects all shapes to match: state_value, reward, done etc.
        reward_specs, done_specs = self._build_agent_specs()
        self.reward_spec = CompositeSpec(
            agents=CompositeSpec({"reward": torch.stack(reward_specs, dim=0)}, shape=(self.n_agents,))
        )
        self.done_spec = CompositeSpec(
            agents=CompositeSpec({"done": torch.stack(done_specs, dim=0)}, shape=(self.n_agents,)),
            # Needed in SyncDataCollector. Otherwise, the env reset td_out doesn't get used in the next step.
            # It updates the parent object of the `done` spec, and since we need the whole td_out to be used the
            # next time, we need a root level `done` spec as well. WTF?
            # Also, the shape must be provided. Otherwise the `done_spec` setter cries about empty leaf shape. WTF?
            done=DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool),
        )
        self._done_keys = [("agents", "done")]

        self.board = chess.Board()

    def _build_agent_specs(self):
        reward_specs: list[BoundedTensorSpec] = []
        done_specs: list[DiscreteTensorSpec] = []
        for i in range(self.n_agents):
            reward_specs.append(BoundedTensorSpec(-1, 1, shape=(1,)))
            done_specs.append(DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool))
        return reward_specs, done_specs

    @property
    def observation_keys(self):
        obs_keys: list[tuple[str]] = []
        for obs_key in self.full_observation_spec[ChessEnv.OBSERVATION_KEY].keys():
            key = (ChessEnv.OBSERVATION_KEY, obs_key)
            obs_keys.append(key)
        return obs_keys

    @property
    def observation_td(self):
        obs_td = self.full_observation_spec.zero()

        piece_positions, piece_types = [], []
        for piece_pos, piece in self.board.piece_map().items():
            piece_positions.append(piece_pos)
            piece_types.append(int(piece.color) * len(chess.PIECE_TYPES) + piece.piece_type - 1)
        piece_positions = torch.tensor(piece_positions, device=obs_td.device)
        piece_types = torch.tensor(piece_types, device=obs_td.device)
        obs_td[ChessEnv.OBSERVATION_KEY, "piece_at_pos"][piece_positions * self.total_piece_cnt + piece_types] = True
        obs_td[ChessEnv.OBSERVATION_KEY, "turn"][...] = self.board.turn

        legal_moves = []
        for move in self.board.legal_moves:
            legal_moves.append((move.from_square, move.to_square))
        legal_moves = torch.tensor(legal_moves, device=obs_td.device)
        obs_td["action_mask"][*zip(*legal_moves)] = True
        self.full_action_spec[self.action_key].update_mask(obs_td["action_mask"])

        return obs_td

    @property
    def done_td(self):
        d_td = self.full_done_spec.zero()
        is_done = self.board.is_game_over()
        d_td["done"][...] = is_done
        d_td[self.done_key][...] = is_done
        return d_td

    @property
    def reward_td(self):
        r_td = self.full_reward_spec.zero()
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return r_td

        indexes = [int(outcome.winner), int(not outcome.winner)]
        r_td[self.reward_key][indexes] = torch.tensor([[1.0], [-1.0]], device=r_td.device)
        return r_td

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.board.reset()

        # Return observations and done
        td = self.observation_td
        td.update(self.done_td)
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.rand_player_idx is None or self.rand_player_idx != self.board.turn:
            action: TensorDict = tensordict[self.action_key]
        else:
            tensordict = _get_move_external(tensordict)
            action = tensordict[self.action_key]
        from_square, to_square = action["0"].item(), action["1"].item()
        move = chess.Move(from_square, to_square)
        try:
            self.board.push(move)
        except AssertionError:
            logging.error("Current board state:\n%s", self.board)
            raise

        # Return observations, rewards and done
        td = self.observation_td
        td.update(self.reward_td)
        td.update(self.done_td)
        return td

    def _set_seed(self, seed: int | None):
        if seed is None:
            seed: int = torch.empty((), dtype=torch.int64).random_().item()
        torch.manual_seed(seed)

    def create_obs_transforms(self, sample_population: int | None = 10000):
        if sample_population is not None:
            obs_samples: TensorDict = self.full_observation_spec.rand((sample_population,)).type(torch.float64)
        else:
            obs_samples: TensorDict = self.full_observation_spec.zero((2,)).type(torch.float64)
        mu = obs_samples.mean(dim=0)
        std = obs_samples.std(dim=0)
        obs_transforms: list[Transform] = []
        for obs_key in self.observation_keys:
            obs_transforms.append(
                DiscreteToContinuousTransform(
                    dtype_in=self.full_observation_spec[obs_key].dtype,
                    dtype_out=torch.double,
                    in_keys=[obs_key],
                    out_keys=[obs_key],
                    in_keys_inv=[obs_key],
                )
            )
            obs_transforms.append(
                ObservationNorm(
                    loc=mu[obs_key], scale=std[obs_key], in_keys=[obs_key], in_keys_inv=[obs_key], standard_normal=True
                )
            )
            obs_transforms.append(DoubleToFloat(in_keys=[obs_key], out_keys=[obs_key], in_keys_inv=[obs_key]))

        return obs_transforms

    @staticmethod
    def save_obs_transforms(obs_transforms_list: list[Transform], save_path: str):
        # To be consistent with how we save it in `pretrain.py`:
        obs_transforms = Compose(*obs_transforms_list)
        torch.save(obs_transforms.state_dict(), save_path)

    def load_obs_transforms(self, save_path: str | None = None):
        obs_transforms_list = self.create_obs_transforms(sample_population=None)
        if save_path is None or not os.path.exists(save_path):
            return obs_transforms_list

        obs_transforms = Compose(*obs_transforms_list)
        params: list[dict[str, Any]] = torch.load(save_path)
        obs_transforms.load_state_dict(params)
        return obs_transforms_list


if __name__ == "__main__":
    env = ChessEnv()
    transforms = Compose(*env.create_obs_transforms())

    def create_tenv():
        env = ChessEnv()
        tenv = TransformedEnv(env, transforms.clone(), cache_specs=False)
        return tenv

    penv = ParallelEnv(2, create_tenv)
    # penv = create_tenv()
    pass

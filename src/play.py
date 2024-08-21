import torch
import chess
from tensordict import TensorDict
from actor_critic import create_actor, create_logits_fn, load_action_nets
from chess_env import ChessEnv
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.data.tensor_specs import DiscreteTensorSpec
from utils import _get_next_mask


def get_move(td: TensorDict):
    # move_str = input("Your turn. Write your move (UCI):")
    # move = chess.Move.from_uci(move_str)
    actions: list[torch.Tensor] = []
    for i in range(2):
        dependent_mask = _get_next_mask([], [64, 64], td["action_mask"], actions)
        action_i = DiscreteTensorSpec(64, device=td.device, mask=dependent_mask).sample()
        actions.append(action_i)
    move = chess.Move(actions[0].item(), actions[1].item())
    return move


if __name__ == "__main__":
    obs_transforms_save_path = "./lightning_logs/version_1/checkpoints/epoch=34-step=17535-obs_transforms.pt"
    action_nets_save_path = "./lightning_logs/version_1/checkpoints/epoch=34-step=17535-action_nets-PPO.pt"
    default_device = torch.device("cpu")

    env = ChessEnv()
    transforms = Compose(*env.load_obs_transforms(obs_transforms_save_path))
    tenv = TransformedEnv(env, transforms, cache_specs=False, device=default_device)

    action_nets = load_action_nets(env, tenv, default_device, action_nets_save_path)
    actor = create_actor(env, tenv, default_device, create_logits_fn(env, action_nets))

    turn_names = ["black", "white"]
    winners = {"black": 0, "white": 0, "draw": 0}
    for game_i in range(1000):
        td = tenv.reset()
        # print(env.board)
        with torch.no_grad():
            while not env.board.is_game_over():
                # print(f"Turn: {turn_names[int(env.board.turn)]}")
                # input("Click to continue...")

                if env.board.turn:
                    move = get_move(td)
                    td_actor = td.set(
                        "action", {"0": torch.tensor(move.from_square), "1": torch.tensor(move.to_square)}
                    )
                else:
                    td_actor = actor(td)
                    move = chess.Move(td_actor["action", "0"].item(), td_actor["action", "1"].item())
                # print(f"Move: {move}")

                td_step = tenv.step(td_actor)
                td = td_step["next"]

                # print(env.board)

        outcome = env.board.outcome()
        if outcome.winner is None:
            winners["draw"] += 1
        else:
            winners[turn_names[int(outcome.winner)]] += 1
        print(f"Play {game_i}:", winners)

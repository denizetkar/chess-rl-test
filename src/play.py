import torch
import chess
from actor_critic import create_actor, create_logits_fn, load_action_nets
from chess_env import ChessEnv
from torchrl.envs.transforms import TransformedEnv, Compose


if __name__ == "__main__":
    obs_transforms_save_path = "./lightning_logs/version_0/checkpoints/epoch=74-step=37575-obs_transforms.pt"
    action_nets_save_path = "./lightning_logs/version_0/checkpoints/epoch=74-step=37575-action_nets.pt"
    default_device = torch.device("cpu")

    env = ChessEnv()
    transforms = Compose(*env.load_obs_transforms(obs_transforms_save_path))
    tenv = TransformedEnv(env, transforms, cache_specs=False, device=default_device)

    action_nets = load_action_nets(env, tenv, default_device, action_nets_save_path)
    actor = create_actor(env, tenv, default_device, create_logits_fn(env, action_nets))

    td = tenv.reset()
    print(env.board)
    with torch.no_grad():
        while not env.board.is_game_over():
            input(f"Turn: {int(env.board.turn)}. Click to continue...")
            td_actor = actor(td)
            td_step = tenv.step(td_actor)
            td = td_step["next"]
            last_move = chess.Move(td_actor["action", "0"].item(), td_actor["action", "1"].item())
            print(f"Move: {last_move}")
            print(env.board)
    print(env.board.outcome())

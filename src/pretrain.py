import logging
import os
import json
from typing import Any

import chess
import pandas as pd
import dask.dataframe as dd

import dask.dataframe.core as ddcore
import dask.dataframe.utils as ddutils
from dask.distributed import Client

import torch
import torch.nn as nn
from tensordict import TensorDict

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from chess_env import ChessEnv
from torchrl.envs.transforms import TransformedEnv, Compose
from actor_critic import create_action_nets, create_logits_fn


def load_winner_moves_df():
    games_data_path = "./data/games.csv"
    winner_moves_path = "./data/winner_moves.csv"
    winner_moves_df: pd.DataFrame
    if os.path.exists(winner_moves_path):
        winner_moves_df = pd.read_csv(winner_moves_path, sep="|")
        parse_loaded_winner_moves(winner_moves_df)
    else:
        games_data_df = pd.read_csv(games_data_path)[["id", "winner", "moves"]]
        winner_moves_df = get_winner_moves(games_data_df)
        winner_moves_df.to_csv(winner_moves_path, sep="|", index=False)

    return winner_moves_df.reset_index(drop=True)


def map_game_to_winner_moves(row: pd.Series):
    game = chess.Board()
    moves: list[str] = row["moves"].split()
    winner: bool = row["winner"] == "white"  # True for white, False for black

    game_data = []
    for move_san in moves:
        move = game.parse_san(move_san)
        # Check if it's the winner's turn
        if game.turn == winner:
            piece_at_pos = [0] * 64
            owner_at_pos = [0] * 64
            for square, piece in game.piece_map().items():
                index = square  # Use the square index directly
                piece_at_pos[index] = piece.piece_type
                owner_at_pos[index] = int(piece.color) + 1  # 1 for black, 2 for white

            game_data.append(
                {
                    "game_id": row["id"],
                    "piece_at_pos": piece_at_pos,
                    "owner_at_pos": owner_at_pos,
                    "turn": [int(winner)],
                    "move": [move.from_square, move.to_square],
                }
            )

        # Make the move (even if it's not the winner's turn)
        try:
            game.push(move)
        except AssertionError as e:
            logging.error(f"Invalid move: {move} in game {row['id']}")
            logging.error(e, exc_info=True)
            break

    return pd.DataFrame(game_data)


def get_winner_moves(games_data_df: pd.DataFrame) -> pd.DataFrame:
    # Filter out draw games before applying the mapping
    games_with_winner_df = games_data_df[games_data_df["winner"] != "draw"]

    # Create a Dask client with the "processes" scheduler
    n_workers = 8
    client = Client(processes=True, n_workers=n_workers, memory_limit="4GB")

    # Convert to Dask DataFrame (no need to specify npartitions)
    ddf: ddcore.DataFrame = dd.from_pandas(games_with_winner_df, npartitions=n_workers)

    # Apply the mapping function in parallel using the "processes" scheduler
    with client:
        winner_moves_df = (
            ddf.apply(
                map_game_to_winner_moves,
                axis=1,
                meta=ddutils.make_meta(
                    pd.DataFrame(
                        [["game id", [0], [0], [0], [0]]],
                        columns=["game_id", "piece_at_pos", "owner_at_pos", "turn", "move"],
                    )
                ),
            )
            .compute()
            .to_list()
        )
    winner_moves_df = pd.concat(winner_moves_df)

    return winner_moves_df


def parse_loaded_winner_moves(winner_moves_df: pd.DataFrame):
    def apply_row(row: pd.Series):
        return row.apply(json.loads)

    winner_moves_df[["piece_at_pos", "owner_at_pos", "turn", "move"]] = winner_moves_df[
        ["piece_at_pos", "owner_at_pos", "turn", "move"]
    ].apply(apply_row)


def winner_moves_df_to_np_arrays(winner_moves_df: pd.DataFrame):
    # The output is respectively: piece_at_pos, owner_at_pos, turn, move
    np_arrays = [np.array(winner_moves_df.iloc[..., i].to_list()) for i in range(1, 5)]
    for i, arr in enumerate(np_arrays):
        if not np.issubdtype(arr.dtype, np.integer):
            continue
        np_arrays[i] = np.array(arr, dtype=np.int64)

    return tuple(np_arrays)


class ChessDataset(Dataset):
    def __init__(self, piece_at_pos: np.ndarray, owner_at_pos: np.ndarray, turn: np.ndarray, move: np.ndarray):
        if not len(piece_at_pos) == len(owner_at_pos) == len(turn) == len(move):
            raise ValueError("All np arrays must have the same number of data points")

        self.piece_at_pos, self.owner_at_pos, self.turn, self.move = piece_at_pos, owner_at_pos, turn, move

    def __getitem__(self, index):
        return self.piece_at_pos[index], self.owner_at_pos[index], self.turn[index], self.move[index]

    def __len__(self):
        return len(self.piece_at_pos)


class ChessPretrainingModule(L.LightningModule):
    def __init__(self, *args: Any, max_epochs: int, lr: float = 1e-4, **kwargs: Any):
        super().__init__(*args, **kwargs)
        base_env = ChessEnv()
        self.obs_transforms = Compose(*base_env.create_obs_transforms())
        final_env = TransformedEnv(base_env, self.obs_transforms)
        self.action_nets = create_action_nets(base_env, final_env, self.device)
        self.logits_fn = create_logits_fn(base_env, self.action_nets)

        self.max_epochs = max_epochs
        self.lr = lr
        self.n_actions = 2
        self.ce_loss = nn.CrossEntropyLoss()

    def get_ce_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        piece_at_pos, owner_at_pos, turn, move = batch
        observations = TensorDict(
            {
                ChessEnv.OBSERVATION_KEY: {
                    "piece_at_pos": piece_at_pos,
                    "owner_at_pos": owner_at_pos,
                    "turn": turn,
                }
            }
        ).auto_batch_size_()
        observations = self.obs_transforms(observations)[ChessEnv.OBSERVATION_KEY]
        actions: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for i in range(self.n_actions):
            logits_i = self.logits_fn(observations, actions)
            logits.append(logits_i)
            action_i = move[..., i]
            actions.append(action_i)
        actions = torch.stack(actions, dim=1).flatten(start_dim=0, end_dim=1)
        logits = torch.stack(logits, dim=1).flatten(start_dim=0, end_dim=1)
        return self.ce_loss(logits, actions)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        train_loss = self.get_ce_loss(batch, batch_idx)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        val_loss = self.get_ce_loss(batch, batch_idx)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.action_nets.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
        level=logging.INFO,
    )

    # Hyperparameters
    max_epochs = 75
    batch_size = 1024
    lr = 5e-4
    gradient_clip_val = 10.0

    piece_at_pos, owner_at_pos, turn, move = winner_moves_df_to_np_arrays(load_winner_moves_df())
    chess_dataset = ChessDataset(piece_at_pos, owner_at_pos, turn, move)
    train_dataset, val_dataset = random_split(chess_dataset, [0.9, 0.1])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2, prefetch_factor=4, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=2, prefetch_factor=4, persistent_workers=True
    )

    model = ChessPretrainingModule(max_epochs=max_epochs, lr=lr)
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_cb],
        accelerator="gpu",
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model = ChessPretrainingModule.load_from_checkpoint(checkpoint_cb.best_model_path)
    path_before_ext, _ = os.path.splitext(checkpoint_cb.best_model_path)
    torch.save(best_model.obs_transforms.state_dict(), f"{path_before_ext}-obs_transforms.pt")
    torch.save(best_model.action_nets.state_dict(), f"{path_before_ext}-action_nets.pt")

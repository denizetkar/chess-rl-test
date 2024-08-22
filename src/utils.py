import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import DiscreteTensorSpec


def _get_next_mask(
    batch_dims: list[int], decision_dims: list[int], mask: torch.Tensor, actions: list[torch.Tensor]
) -> torch.Tensor:
    num_actions_taken = len(actions)
    num_actions_not_taken = len(decision_dims) - num_actions_taken

    grid_ranges = [torch.arange(s) for s in batch_dims]
    batch_indices = torch.meshgrid(*grid_ranges, indexing="ij") if len(grid_ranges) > 0 else ()
    action_indices = tuple(actions)
    indices = batch_indices + action_indices

    # self.mask cannot be None here
    filtered_mask = mask[indices]

    legal_action_mask = filtered_mask.any(dim=list(range(-num_actions_not_taken + 1, 0)))
    return legal_action_mask


def _get_move_external(td: TensorDict):
    # move_str = input("Your turn. Write your move (UCI):")
    # move = chess.Move.from_uci(move_str)

    # Choose a random legal action pair (from_square, to_square)
    actions: list[torch.Tensor] = []
    actions_td = {}
    for i in range(2):
        dependent_mask = _get_next_mask([], [64, 64], td["action_mask"], actions)
        action_i = DiscreteTensorSpec(64, device=td.device, mask=dependent_mask).sample()
        actions.append(action_i)
        actions_td[str(i)] = action_i

    td.set("action", actions_td)
    return td

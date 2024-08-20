import torch


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

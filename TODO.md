- Does `ClipPPOLoss` optimize for all agents at each step ?

  - Yes -> Change the reduction to `"none"` and calculate mean on your own

- Write your own `MaskedDependentDiscreteTensorsSpec(TensorSpec)` to sample from UCI moves given `action_mask`

  - Use that as the action specs
  - Update the action mask in `_reset()` and `_step()` methods

- Write your own `MaskedDependentCategorical(D.Categorical)`

  - This is the last (probabilistic) module in `ProbabilisticActor`
  - Takes the logits from the MLP
  - `from_square_raw_logits` from `("agents", "from_logits")`
  - `to_square_raw_logits` from `("agents", "to_logits")`
  - Use `from_square_raw_logits` and `action_mask.any(dim=-1)` in `MaskedCategorical`
    - Draw the first action `a1`
  - Use `to_square_raw_logits` and `action_mask[..., a1, :]` in `MaskedCategorical`
    - Draw the second action `a2`
  - `dist.log_prob(action)` should return `a1_masked_categorical.log_prob(a1) + a2_masked_categorical.log_prob(a2)`
    - Because `P(a1, a2) = P(a1) * P(a2 | a1)`

- Parallelize traj collection: `ParallelEnv` or `MultiSyncDataCollector` ???

---

- Maybe turn categorical vars to one-hot vectors (?)

  ```python
  x = F.one_hot(torch.arange(6).reshape(3, 2) % 4, num_classes=4).flatten(start_dim=-2, end_dim=-1)

  tensor([[1, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 1],
          [1, 0, 0, 0, 0, 1, 0, 0]])
  ```

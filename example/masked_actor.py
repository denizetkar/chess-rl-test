from tensordict.nn import (
    TensorDictModule,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    InteractionType,
)
import torch.nn as nn
import torch
from torchrl.modules import MaskedCategorical

from masked_env import MyMaskedEnv

module = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])

dist = ProbabilisticTensorDictModule(
    in_keys={"logits": "logits", "mask": "action_mask"},
    out_keys=["action"],
    distribution_class=MaskedCategorical,
    default_interaction_type=InteractionType.RANDOM,
)

actor = ProbabilisticTensorDictSequential(module, dist)

torch.manual_seed(0)
env = MyMaskedEnv()
r = env.rollout(10, policy=actor)
print(r["action_mask"])

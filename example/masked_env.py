from tensordict import TensorDict
import torch
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec,
)
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.envs.common import EnvBase


class MyMaskedEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_spec = DiscreteTensorSpec(4)
        self.state_spec = CompositeSpec(action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool))
        self.observation_spec = CompositeSpec(obs=UnboundedContinuousTensorSpec(3))
        self.reward_spec = UnboundedContinuousTensorSpec(1)

    def _reset(self, tensordict: TensorDict = None):
        td = self.observation_spec.rand()
        td.update(torch.ones_like(self.state_spec.rand()))
        return td

    def _step(self, data: TensorDict):
        td = self.observation_spec.rand()
        mask = data.get("action_mask")
        action = data.get("action")
        mask = mask.scatter(-1, action.unsqueeze(-1), 0)

        td.set("action_mask", mask)
        td.set("reward", self.reward_spec.rand())
        td.set("done", ~mask.any().view(1))
        return td

    def _set_seed(self, seed):
        return seed


if __name__ == "__main__":
    torch.manual_seed(0)
    base_env = MyMaskedEnv()
    env = TransformedEnv(base_env, ActionMask())
    r = env.rollout(10)
    print(r["action_mask"])

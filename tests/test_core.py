import torch
from torch import nn

from trajflow_kv.objective import normalized_advantages, trajectory_policy_loss
from trajflow_kv.projector import attach_kv_projectors


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.k_proj = nn.Linear(8, 8); self.v_proj = nn.Linear(8, 8)
    def forward(self, x): return self.k_proj(x) + self.v_proj(x)


def test_hooks_change_output_and_receive_gradients():
    model = Tiny()
    baseline = model(torch.ones(2, 8)).detach()
    for p in model.parameters(): p.requires_grad_(False)
    bundle = attach_kv_projectors(model, rank=2, alpha=1.0)
    bundle.modules[0].up.weight.data.fill_(0.1)
    output = model(torch.ones(2, 8))
    assert not torch.allclose(output, baseline)
    output.sum().backward()
    assert bundle.modules[0].up.weight.grad is not None


def test_return_objective():
    returns = torch.tensor([1., 0., 0., 1.])
    adv = normalized_advantages(returns, ["a", "a", "b", "b"])
    assert torch.allclose(adv.mean(), torch.tensor(0.))
    loss = trajectory_policy_loss(torch.tensor([-1., -2., -3., -1.]), adv)
    assert torch.isfinite(loss)

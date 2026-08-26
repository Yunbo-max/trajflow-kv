from __future__ import annotations

import torch


def normalized_advantages(returns: torch.Tensor, task_ids: list[str]) -> torch.Tensor:
    """Leave-one-group-style task baseline followed by stable normalization."""
    advantages = torch.empty_like(returns)
    for task in sorted(set(task_ids)):
        idx = torch.tensor([x == task for x in task_ids], device=returns.device)
        group = returns[idx]
        advantages[idx] = group - group.mean()
    std = advantages.std(unbiased=False)
    return advantages / std.clamp_min(1e-6)


def trajectory_policy_loss(logprob_sums: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    if logprob_sums.shape != advantages.shape:
        raise ValueError("one log-probability sum and advantage required per trajectory")
    return -(advantages.detach() * logprob_sums).mean()


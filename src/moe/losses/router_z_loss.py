import torch


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE router z-loss. Penalizes large router normalizer (stabilizes training).

    logits: (BT, n_experts)
    Returns scalar; multiply by beta (e.g. 0.001) before summing into the graph.
    """
    log_z = torch.logsumexp(logits, dim=-1)  # (BT,)
    return (log_z**2).mean()

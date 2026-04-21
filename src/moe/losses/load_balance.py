import torch
import torch.nn.functional as F


def load_balance_loss(
    full_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Switch-style auxiliary load-balance loss (extends to top-k).

    ``full_probs``: (BT, n_experts) — full softmax over experts.
    ``topk_idx``: (BT, k) — selected expert ids per token.

    Let ``f_i`` be (number of top-k slots assigning expert i) / BT, and ``P_i`` the
    batch-mean router prob for expert i. Then ``sum_i f_i = k`` (each token adds
    ``k`` assignments). Under perfectly balanced routing and uniform ``P``,
    ``f_i = k/n_experts`` and ``P_i = 1/n_experts``, so this returns ``k`` (not 1).
    For ``k=1`` (Switch) the minimum is 1.0.

    Multiply by a small weight (e.g. 0.01) before adding to the task loss.
    """
    BT = full_probs.size(0)
    # f is built from topk_idx (ints from argmax) so it carries no gradient;
    # only P propagates grads into the router — matches Switch-Transformer intent.
    one_hot = F.one_hot(topk_idx, n_experts).float()  # (BT, k, n_experts)
    f = one_hot.sum(dim=(0, 1)) / BT  # (n_experts,)
    P = full_probs.mean(dim=0)  # (n_experts,)
    return n_experts * (f * P).sum()

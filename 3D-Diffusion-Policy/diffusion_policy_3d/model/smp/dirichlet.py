from typing import Dict, Tuple

import torch

# KL散度的实现

def dirichlet_kl(concentration_q: torch.Tensor,
                 concentration_p: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(Dir(q) || Dir(p)) from SMP Eq. 39, evaluated in FP32."""
    q = concentration_q.float()
    p = concentration_p.float()
    q, p = torch.broadcast_tensors(q, p)
    if torch.any(q <= 0) or torch.any(p <= 0):
        raise ValueError("Dirichlet concentrations must be strictly positive")

    # 论文 Eq. (39)：Dirichlet KL 的闭式解，最后一维为专家维度。
    q_sum = q.sum(dim=-1)
    p_sum = p.sum(dim=-1)
    log_normalizer_ratio = (
        torch.lgamma(q_sum) - torch.lgamma(p_sum)
        - torch.lgamma(q).sum(dim=-1) + torch.lgamma(p).sum(dim=-1)
    )
    expectation = ((q - p) * (
        torch.digamma(q) - torch.digamma(q_sum).unsqueeze(-1)
    )).sum(dim=-1)
    return log_normalizer_ratio + expectation


def sticky_gate_kl(
        global_concentration: torch.Tensor,
        gate_concentration: torch.Tensor,
        alpha: float,
        alpha0: float,
        kappa: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """SMP Eq. 38 with posterior means substituted into sticky priors.

    Returns a batch mean. The temporal terms are summed, matching Eq. 38.
    """
    if gate_concentration.ndim != 3:
        raise ValueError("gate_concentration must have shape [B, T, K]")
    if alpha <= 0 or alpha0 <= 0 or kappa < 0:
        raise ValueError("alpha and alpha0 must be positive; kappa must be nonnegative")

    gates = gate_concentration.float()
    global_q = global_concentration.float()
    if global_q.ndim == 1:
        global_q = global_q.unsqueeze(0).expand(gates.shape[0], -1)
    elif global_q.ndim != 2:
        raise ValueError("global_concentration must have shape [K] or [B, K]")
    if global_q.shape != (gates.shape[0], gates.shape[-1]):
        raise ValueError("global and gate concentration shapes are incompatible")

    # 论文 Eq. (37)-(38)：以 posterior mean 代入全局与前一时刻 gate。
    theta_mean = global_q / global_q.sum(dim=-1, keepdim=True)
    gate_mean = gates / gates.sum(dim=-1, keepdim=True)
    global_prior = torch.full_like(global_q, alpha)
    global_term = dirichlet_kl(global_q, global_prior)
    initial_prior = alpha0 * theta_mean
    initial_term = dirichlet_kl(gates[:, 0], initial_prior)

    if gates.shape[1] > 1:
        sticky_prior = (
            kappa * gate_mean[:, :-1]
            + alpha0 * theta_mean.unsqueeze(1)
        )
        # 论文 Eq. (38)：时间 KL 按轨迹求和，再在下方对 batch 取均值。
        sticky_term = dirichlet_kl(gates[:, 1:], sticky_prior).sum(dim=1)
    else:
        sticky_term = torch.zeros_like(initial_term)

    terms = {
        "global_usage_kl": global_term.mean(),
        "initial_gate_kl": initial_term.mean(),
        "sticky_gate_kl": sticky_term.mean(),
    }
    return sum(terms.values()), terms


def router_alignment_kl(
        posterior_concentration: torch.Tensor,
        prior_concentration: torch.Tensor) -> torch.Tensor:
    """Trajectory KL(q(g | s, a) || p_phi(g | s)) from SMP Eq. 40."""
    if posterior_concentration.shape != prior_concentration.shape:
        raise ValueError("posterior and prior concentration shapes must match")
    if posterior_concentration.ndim != 3:
        raise ValueError("posterior and prior must have shape [B, T, K]")
    # 论文 Eq. (40)：逐时刻 KL 求和；仅对 batch 取均值。
    return dirichlet_kl(
        posterior_concentration, prior_concentration).sum(dim=1).mean()

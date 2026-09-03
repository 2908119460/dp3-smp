from typing import Tuple

import torch

# 动作at的反推出zi

def _validate_shapes(basis: torch.Tensor, gate: torch.Tensor,
                     value: torch.Tensor) -> None:
    if basis.ndim != 3 or gate.ndim != 3 or value.ndim != 3:
        raise ValueError("basis, gate, and value must all be rank-3 tensors")
    if basis.shape[0] != gate.shape[0] or gate.shape[:2] != value.shape[:2]:
        raise ValueError("basis, gate, and value batch/horizon shapes do not match")
    if basis.shape[-1] != gate.shape[-1]:
        raise ValueError("basis and gate expert dimensions do not match")


def build_coefficient_targets(
        basis: torch.Tensor,
        normalized_action: torch.Tensor,
        gate_mean: torch.Tensor,
        eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build z_hat_sg and z_hat_rec from SMP Eq. 34.

    The basis is current-state conditioned and shared across the action chunk.
    Only the basis is detached in z_hat_sg; gate gradients are retained as in Eq. 36.
    """
    _validate_shapes(basis, gate_mean, normalized_action)
    if basis.shape[-2] != normalized_action.shape[-1]:
        raise ValueError("basis action dimension does not match normalized_action")
    if eps <= 0:
        raise ValueError("eps must be positive")

    # 论文 Eq. (34)：仅截断 B 的梯度，L_coeff 仍可更新 gate posterior。
    denominator = gate_mean + eps
    stopped_projection = torch.einsum(
        "bdk,btd->btk", basis.detach(), normalized_action)
    # 论文 Eq. (34)：重建分支保留 B 的梯度，使 L_recon 学习技能子空间。
    reconstruction_projection = torch.einsum(
        "bdk,btd->btk", basis, normalized_action)
    return (
        stopped_projection / denominator,
        reconstruction_projection / denominator,
    )


def decode_action(basis: torch.Tensor, gate: torch.Tensor,
                  coefficient: torch.Tensor) -> torch.Tensor:
    """Decode a = B(s) (g elementwise-multiplied by z), SMP Eq. 1."""
    _validate_shapes(basis, gate, coefficient)
    # 论文 Eq. (1)/(23)：每个专家只贡献正交方向 b_i(g_i z_i)。
    return torch.einsum("bdk,btk->btd", basis, gate * coefficient)

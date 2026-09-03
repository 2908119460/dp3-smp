import torch
import torch.nn as nn

# 生成矩阵B(s)的文件

class SkillBasisGenerator(nn.Module):
    """Generate the state-conditioned orthonormal basis from SMP Eq. 2."""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            num_experts: int,
            hidden_dim: int = 256,
            qr_fp32: bool = True,
            sign_stabilization: bool = True):
        super().__init__()
        if num_experts > action_dim:
            raise ValueError(
                f"num_experts ({num_experts}) must not exceed action_dim "
                f"({action_dim}) for a thin QR basis")

        self.action_dim = action_dim
        self.num_experts = num_experts
        self.qr_fp32 = qr_fp32
        self.sign_stabilization = sign_stabilization
        self.generator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim * num_experts),
        )

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Return B(s) with shape [batch, action_dim, num_experts]."""
        if state_feature.ndim != 2:
            raise ValueError(
                "state_feature must have shape [batch, state_dim], got "
                f"{tuple(state_feature.shape)}")

        unconstrained = self.generator(state_feature).reshape(
            state_feature.shape[0], self.action_dim, self.num_experts)
        qr_input = unconstrained.float() if self.qr_fp32 else unconstrained
        # 论文 Eq. (2)/(14)：thin QR 将 W(s) 投影到正交 Stiefel 流形。
        basis, upper = torch.linalg.qr(qr_input, mode="reduced")

        if self.sign_stabilization:
            # 论文 Eq. (15)-(16)：固定 R 对角线符号，消除 basis 列方向歧义。
            diagonal = torch.diagonal(upper, dim1=-2, dim2=-1)
            signs = torch.where(diagonal < 0, -torch.ones_like(diagonal),
                                torch.ones_like(diagonal))
            basis = basis * signs.unsqueeze(-2)

        return basis.to(unconstrained.dtype)

    @staticmethod
    def orthogonality_error(basis: torch.Tensor) -> torch.Tensor:
        """Per-sample Frobenius norm ||B^T B - I||_F."""
        gram = torch.matmul(basis.transpose(-2, -1), basis)
        identity = torch.eye(
            gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return torch.linalg.matrix_norm(gram - identity, ord="fro")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 生成门控gating的文件

class _PositiveConcentrationHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 min_concentration: float):
        super().__init__()
        if min_concentration <= 0:
            raise ValueError("min_concentration must be positive")
        self.min_concentration = min_concentration
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Dirichlet 浓度必须为正；softplus 不改变论文中的分布定义。
        return F.softplus(self.network(inputs)) + self.min_concentration


class PosteriorGate(nn.Module):
    """Training-only q(g_t | s, a_t) Dirichlet amortizer."""

    def __init__(self, state_dim: int, action_dim: int, num_experts: int,
                 hidden_dim: int = 256, min_concentration: float = 1e-4):
        super().__init__()
        self.head = _PositiveConcentrationHead(
            state_dim + action_dim, num_experts, hidden_dim,
            min_concentration)

    def forward(self, state_feature: torch.Tensor,
                normalized_action: torch.Tensor) -> torch.Tensor:
        """Return beta_hat with shape [batch, action_horizon, num_experts]."""
        if state_feature.ndim != 2 or normalized_action.ndim != 3:
            raise ValueError(
                "expected state_feature [B, Ds] and normalized_action "
                f"[B, Ta, Da], got {tuple(state_feature.shape)} and "
                f"{tuple(normalized_action.shape)}")
        if state_feature.shape[0] != normalized_action.shape[0]:
            raise ValueError("state and action batch sizes must match")

        horizon = normalized_action.shape[1]
        # DP3 chunk 近似：当前状态与每个动作步组成 q(g_h|s_current,a_h)。
        expanded_state = state_feature.unsqueeze(1).expand(-1, horizon, -1)
        return self.head(torch.cat([expanded_state, normalized_action], dim=-1))


class PriorRouter(nn.Module):
    """Deployment p_phi(g_t | s) Dirichlet router; never consumes actions."""

    def __init__(self, state_dim: int, action_horizon: int, num_experts: int,
                 hidden_dim: int = 256, min_concentration: float = 1e-4):
        super().__init__()
        self.action_horizon = action_horizon
        self.num_experts = num_experts
        self.head = _PositiveConcentrationHead(
            state_dim, action_horizon * num_experts, hidden_dim,
            min_concentration)

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Return beta_tilde with shape [batch, action_horizon, num_experts]."""
        # DP3 chunk 近似：由当前状态一次预测整段 gate，并非论文逐时刻 p(g_t|s_t)。
        concentration = self.head(state_feature)
        return concentration.reshape(
            state_feature.shape[0], self.action_horizon, self.num_experts)


class GlobalUsagePosterior(nn.Module):
    """Learnable q(theta) = Dir(alpha_hat) shared across trajectories."""

    def __init__(self, num_experts: int, initial_concentration: float = 2.0,
                 min_concentration: float = 1e-4):
        super().__init__()
        if initial_concentration <= min_concentration:
            raise ValueError(
                "initial_concentration must exceed min_concentration")
        initial_raw = torch.log(torch.expm1(torch.tensor(
            initial_concentration - min_concentration)))
        self.raw_concentration = nn.Parameter(
            initial_raw.repeat(num_experts))
        self.min_concentration = min_concentration

    def forward(self) -> torch.Tensor:
        # 论文 Eq. (38) 未规定 q(theta) 的网络输入，此处采用共享可学习浓度。
        return F.softplus(self.raw_concentration) + self.min_concentration


def dirichlet_mean(concentration: torch.Tensor) -> torch.Tensor:
    # Dirichlet 均值 E[g_i] = beta_i / sum_j beta_j。
    return concentration / concentration.sum(dim=-1, keepdim=True)

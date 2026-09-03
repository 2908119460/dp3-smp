from typing import Sequence, Union

import torch
import torch.nn as nn

from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D

# 专家生成Zi的实现

class CoefficientExpertBank(nn.Module):
    """K compact diffusion experts, each denoising one scalar skill coefficient."""

    def __init__(
            self,
            num_experts: int,
            global_cond_dim: int,
            diffusion_step_embed_dim: int = 128,
            down_dims: Sequence[int] = (64, 128, 256),
            kernel_size: int = 3,
            n_groups: int = 8,
            condition_type: str = "film"):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ConditionalUnet1D(
                input_dim=1,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
            ) for _ in range(num_experts)
        ])

    def forward(
            self,
            noisy_coefficient: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond: torch.Tensor) -> torch.Tensor:
        """Denoise [B, Ta, K] without ever predicting a full action vector."""
        if noisy_coefficient.ndim != 3:
            raise ValueError("noisy_coefficient must have shape [B, Ta, K]")
        if noisy_coefficient.shape[-1] != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} coefficient channels, got "
                f"{noisy_coefficient.shape[-1]}")

        # 论文 Eq. (36)：K 个扩散专家分别预测一个系数通道，而非完整动作。
        predictions = [
            expert(
                sample=noisy_coefficient[..., index:index + 1],
                timestep=timestep,
                global_cond=global_cond,
            )
            for index, expert in enumerate(self.experts)
        ]
        return torch.cat(predictions, dim=-1)

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from diffusion_policy_3d.model.smp.coefficients import (
    build_coefficient_targets,
    decode_action,
)
from diffusion_policy_3d.model.smp.expert_bank import CoefficientExpertBank
from diffusion_policy_3d.model.smp.gating import (
    GlobalUsagePosterior,
    PosteriorGate,
    PriorRouter,
    dirichlet_mean,
)
from diffusion_policy_3d.model.smp.skill_basis import SkillBasisGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder


class TaskConditionedStateFusion(nn.Module):
    """Fuse DP3 observation features with a learned task embedding."""

    def __init__(self, obs_feature_dim: int, obs_horizon: int, num_tasks: int,
                 task_embed_dim: int, state_dim: int):
        super().__init__()
        self.obs_feature_dim = obs_feature_dim
        self.obs_horizon = obs_horizon
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        input_dim = obs_feature_dim * obs_horizon + task_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Mish(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, obs_feature: torch.Tensor,
                task_id: torch.Tensor) -> torch.Tensor:
        if obs_feature.ndim != 3:
            raise ValueError("obs_feature must have shape [B, To, Do]")
        expected = (self.obs_horizon, self.obs_feature_dim)
        if tuple(obs_feature.shape[1:]) != expected:
            raise ValueError(
                f"expected observation shape [B, {expected[0]}, {expected[1]}], "
                f"got {tuple(obs_feature.shape)}")
        if task_id.ndim != 1 or task_id.shape[0] != obs_feature.shape[0]:
            raise ValueError("task_id must have shape [B]")
        task_feature = self.task_embedding(task_id.long())
        flattened_obs = obs_feature.reshape(obs_feature.shape[0], -1)
        return self.fusion(torch.cat([flattened_obs, task_feature], dim=-1))


class SMPModel(nn.Module):
    """SMP architecture head for DP3-encoded observations.

    This module deliberately excludes normalization, diffusion scheduling, datasets,
    and policy sampling. Those remain policy/workspace concerns in the next phase.
    """

    def __init__(
            self,
            obs_feature_dim: int,
            obs_horizon: int,
            action_dim: int,
            action_horizon: int,
            num_tasks: int,
            num_experts: int,
            task_embed_dim: int = 16,
            state_dim: int = 256,
            basis_hidden_dim: int = 256,
            basis_qr_fp32: bool = True,
            basis_sign_stabilization: bool = True,
            gate_hidden_dim: int = 256,
            min_concentration: float = 1e-4,
            global_initial_concentration: float = 2.0,
            expert_down_dims: Sequence[int] = (64, 128, 256),
            expert_diffusion_step_embed_dim: int = 128,
            expert_kernel_size: int = 3,
            expert_n_groups: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_experts = num_experts
        self.state_fusion = TaskConditionedStateFusion(
            obs_feature_dim=obs_feature_dim,
            obs_horizon=obs_horizon,
            num_tasks=num_tasks,
            task_embed_dim=task_embed_dim,
            state_dim=state_dim,
        )
        self.basis_generator = SkillBasisGenerator(
            state_dim=state_dim,
            action_dim=action_dim,
            num_experts=num_experts,
            hidden_dim=basis_hidden_dim,
            qr_fp32=basis_qr_fp32,
            sign_stabilization=basis_sign_stabilization,
        )
        self.posterior_gate = PosteriorGate(
            state_dim=state_dim,
            action_dim=action_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            min_concentration=min_concentration,
        )
        self.prior_router = PriorRouter(
            state_dim=state_dim,
            action_horizon=action_horizon,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            min_concentration=min_concentration,
        )
        self.global_usage_posterior = GlobalUsagePosterior(
            num_experts=num_experts,
            initial_concentration=global_initial_concentration,
            min_concentration=min_concentration,
        )
        self.expert_bank = CoefficientExpertBank(
            num_experts=num_experts,
            global_cond_dim=state_dim,
            diffusion_step_embed_dim=expert_diffusion_step_embed_dim,
            down_dims=expert_down_dims,
            kernel_size=expert_kernel_size,
            n_groups=expert_n_groups,
        )

    def encode_state(self, obs_feature: torch.Tensor,
                     task_id: torch.Tensor) -> torch.Tensor:
        return self.state_fusion(obs_feature, task_id)

    def route(self, state_feature: torch.Tensor,
              normalized_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        prior_concentration = self.prior_router(state_feature)
        result = {
            "prior_concentration": prior_concentration,
            "prior_gate": dirichlet_mean(prior_concentration),
            "global_concentration": self.global_usage_posterior(),
        }
        if normalized_action is not None:
            posterior_concentration = self.posterior_gate(
                state_feature, normalized_action)
            result.update({
                "posterior_concentration": posterior_concentration,
                "posterior_gate": dirichlet_mean(posterior_concentration),
            })
        return result

    def build_training_targets(
            self,
            state_feature: torch.Tensor,
            normalized_action: torch.Tensor,
            coefficient_eps: float = 1e-3) -> Dict[str, torch.Tensor]:
        """Build Eq. 34 targets using a single current-state basis per chunk."""
        # DP3 chunk 近似：B(s_current) 覆盖整段动作，不等同于论文逐时刻 B(s_t)。
        basis = self.basis_generator(state_feature)
        posterior_concentration = self.posterior_gate(
            state_feature, normalized_action)
        posterior_gate = dirichlet_mean(posterior_concentration)
        coefficient_sg, coefficient_rec = build_coefficient_targets(
            basis, normalized_action, posterior_gate, coefficient_eps)
        # 论文 Eq. (35)：重建分支通过 B 和 gate 反向传播。
        reconstructed_action = decode_action(
            basis, posterior_gate, coefficient_rec)
        return {
            "basis": basis,
            "posterior_concentration": posterior_concentration,
            "posterior_gate": posterior_gate,
            "coefficient_sg": coefficient_sg,
            "coefficient_rec": coefficient_rec,
            "reconstructed_action": reconstructed_action,
        }

    def denoise_coefficients(
            self,
            noisy_coefficient: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            state_feature: torch.Tensor) -> torch.Tensor:
        # DP3 chunk 近似：以 s_current 条件化整段系数，不等同于 Eq. (26)/(36) 的逐时刻 s_t。
        return self.expert_bank(
            noisy_coefficient=noisy_coefficient,
            timestep=timestep,
            global_cond=state_feature,
        )

    def decode(self, basis: torch.Tensor, gate: torch.Tensor,
               coefficient: torch.Tensor) -> torch.Tensor:
        return decode_action(basis, gate, coefficient)


class SMPDP3Architecture(nn.Module):
    """DP3 point-cloud encoder composed with the SMP coefficient-space head.

    Inputs are normalized observations. Action normalization and diffusion
    scheduling intentionally remain outside this architecture-only module.
    """

    def __init__(
            self,
            shape_meta: dict,
            obs_horizon: int,
            action_horizon: int,
            num_tasks: int,
            num_experts: int,
            task_embed_dim: int = 16,
            state_dim: int = 256,
            encoder_output_dim: int = 256,
            crop_shape=None,
            use_pc_color: bool = False,
            pointnet_type: str = "pointnet",
            pointcloud_encoder_cfg=None,
            **smp_kwargs):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(
                f"unsupported action shape {action_shape}")

        obs_shapes = {
            key: value["shape"] for key, value in shape_meta["obs"].items()
        }
        self.obs_encoder = DP3Encoder(
            observation_space=obs_shapes,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        self.obs_horizon = obs_horizon
        self.use_pc_color = use_pc_color
        self.smp = SMPModel(
            obs_feature_dim=self.obs_encoder.output_shape(),
            obs_horizon=obs_horizon,
            action_dim=action_dim,
            action_horizon=action_horizon,
            num_tasks=num_tasks,
            num_experts=num_experts,
            task_embed_dim=task_embed_dim,
            state_dim=state_dim,
            **smp_kwargs,
        )

    def encode_observation(
            self, normalized_obs: Dict[str, torch.Tensor],
            task_id: torch.Tensor) -> torch.Tensor:
        """Encode [B, To, ...] observations into a task-conditioned [B, Ds]."""
        if not normalized_obs:
            raise ValueError("normalized_obs must not be empty")
        first_value = next(iter(normalized_obs.values()))
        if first_value.ndim < 2:
            raise ValueError("observations must include batch and horizon dimensions")
        batch_size = first_value.shape[0]
        if first_value.shape[1] < self.obs_horizon:
            raise ValueError(
                f"need {self.obs_horizon} observation steps, got "
                f"{first_value.shape[1]}")

        obs = dict(normalized_obs)
        if not self.use_pc_color:
            obs["point_cloud"] = obs["point_cloud"][..., :3]
        flattened_obs = dict_apply(
            obs,
            lambda value: value[:, :self.obs_horizon].reshape(
                -1, *value.shape[2:]),
        )
        obs_feature = self.obs_encoder(flattened_obs).reshape(
            batch_size, self.obs_horizon, -1)
        return self.smp.encode_state(obs_feature, task_id)

    def build_training_targets(
            self,
            normalized_obs: Dict[str, torch.Tensor],
            task_id: torch.Tensor,
            normalized_action: torch.Tensor,
            coefficient_eps: float = 1e-3) -> Dict[str, torch.Tensor]:
        state_feature = self.encode_observation(normalized_obs, task_id)
        targets = self.smp.build_training_targets(
            state_feature, normalized_action, coefficient_eps)
        targets["state_feature"] = state_feature
        return targets

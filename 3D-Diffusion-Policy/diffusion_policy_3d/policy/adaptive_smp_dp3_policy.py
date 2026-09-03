import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.smp import (
    SMPDP3Architecture,
    router_alignment_kl,
    sticky_gate_kl,
)
from diffusion_policy_3d.policy.base_policy import BasePolicy


def select_experts_by_importance_mass(
        gate: torch.Tensor,
        mass_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the smallest expert set whose squared gate mass reaches tau."""
    if gate.ndim != 3:
        raise ValueError("gate must have shape [batch, horizon, experts]")
    if not 0.0 < mass_threshold <= 1.0:
        raise ValueError("mass_threshold must be in (0, 1]")
    if not gate.is_floating_point():
        raise TypeError("gate must be floating point")
    if not torch.isfinite(gate).all() or (gate < 0).any():
        raise ValueError("gate must contain finite non-negative values")

    importance = gate.square()
    sorted_importance, sorted_indices = importance.sort(
        dim=-1, descending=True)
    total_importance = sorted_importance.sum(dim=-1, keepdim=True)
    target_mass = mass_threshold * total_importance
    mass_before = sorted_importance.cumsum(dim=-1) - sorted_importance
    sorted_mask = mass_before < target_mass
    sorted_mask[..., 0] = True
    mask = torch.zeros_like(sorted_mask).scatter(
        dim=-1, index=sorted_indices, src=sorted_mask)
    retained_mass = (
        (importance * mask).sum(dim=-1)
        / total_importance.squeeze(-1).clamp_min(torch.finfo(gate.dtype).tiny)
    )
    return mask, retained_mass


class AdaptiveSMPDP3Policy(BasePolicy):
    """Standalone SMP-DP3 policy with configurable adaptive inference."""

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
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
            alpha: float = 2.0,
            alpha0: float = 0.5,
            kappa: float = 20.0,
            coefficient_eps: float = 1e-3,
            action_likelihood_std: float = 1.0,
            coeff_weight: float = 1.0,
            recon_weight: float = 1.0,
            gate_weight: float = 1.0,
            align_weight: float = 1.0,
            expert_down_dims=(64, 128, 256),
            expert_diffusion_step_embed_dim: int = 128,
            expert_kernel_size: int = 3,
            expert_n_groups: int = 8,
            num_inference_steps: Optional[int] = None,
            adaptive_inference: bool = True,
            importance_mass_threshold: float = 0.95,
            encoder_output_dim: int = 64,
            crop_shape=None,
            use_pc_color: bool = False,
            pointnet_type: str = "pointnet",
            pointcloud_encoder_cfg=None):
        super().__init__()
        self._dummy_variable.requires_grad_(False)
        if noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError(
                "SMP Eq. (36) requires epsilon prediction, got "
                f"{noise_scheduler.config.prediction_type}")
        if action_likelihood_std <= 0:
            raise ValueError("action_likelihood_std must be positive")
        if not 0.0 < importance_mass_threshold <= 1.0:
            raise ValueError("importance_mass_threshold must be in (0, 1]")

        self.model = SMPDP3Architecture(
            shape_meta=shape_meta,
            obs_horizon=n_obs_steps,
            action_horizon=horizon,
            num_tasks=num_tasks,
            num_experts=num_experts,
            task_embed_dim=task_embed_dim,
            state_dim=state_dim,
            encoder_output_dim=encoder_output_dim,
            crop_shape=crop_shape,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            basis_hidden_dim=basis_hidden_dim,
            basis_qr_fp32=basis_qr_fp32,
            basis_sign_stabilization=basis_sign_stabilization,
            gate_hidden_dim=gate_hidden_dim,
            min_concentration=min_concentration,
            global_initial_concentration=global_initial_concentration,
            expert_down_dims=expert_down_dims,
            expert_diffusion_step_embed_dim=expert_diffusion_step_embed_dim,
            expert_kernel_size=expert_kernel_size,
            expert_n_groups=expert_n_groups,
        )
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.action_dim = self.model.smp.action_dim
        self.alpha = alpha
        self.alpha0 = alpha0
        self.kappa = kappa
        self.coefficient_eps = coefficient_eps
        self.action_likelihood_std = action_likelihood_std
        self.loss_weights = {
            "coeff": coeff_weight,
            "recon": recon_weight,
            "gate": gate_weight,
            "align": align_weight,
        }
        self.use_pc_color = use_pc_color
        self.inference_task_id = 0
        self.num_inference_steps = (
            noise_scheduler.config.num_train_timesteps
            if num_inference_steps is None else num_inference_steps)
        self.adaptive_inference = bool(adaptive_inference)
        self.importance_mass_threshold = float(importance_mass_threshold)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_task_id(self, task_id: int) -> None:
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"task_id must be in [0, {self.num_tasks})")
        self.inference_task_id = int(task_id)

    def _prepare_observation(
            self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = dict(obs)
        expected_state_dim = self.model.obs_encoder.state_shape[0]
        state = prepared["agent_pos"]
        if state.shape[-1] > expected_state_dim:
            raise ValueError(
                f"agent_pos has {state.shape[-1]} values, expected at most "
                f"{expected_state_dim}")
        if state.shape[-1] < expected_state_dim:
            prepared["agent_pos"] = F.pad(
                state, (0, expected_state_dim - state.shape[-1]))
        return prepared

    def _task_tensor(
            self,
            batch_size: int,
            device: torch.device,
            task_id: Optional[torch.Tensor]) -> torch.Tensor:
        if task_id is None:
            return torch.full(
                (batch_size,), self.inference_task_id,
                dtype=torch.long, device=device)
        if not torch.is_tensor(task_id):
            task_id = torch.as_tensor(task_id, device=device)
        task_id = task_id.to(device=device, dtype=torch.long)
        if task_id.ndim == 0:
            task_id = task_id.expand(batch_size)
        if task_id.shape != (batch_size,):
            raise ValueError(
                f"task_id must have shape [{batch_size}], got "
                f"{tuple(task_id.shape)}")
        return task_id

    def compute_loss(self, batch):
        normalized_obs = self.normalizer.normalize(
            self._prepare_observation(batch["obs"]))
        normalized_action = self.normalizer["action"].normalize(batch["action"])
        batch_size = normalized_action.shape[0]
        task_id = self._task_tensor(
            batch_size, normalized_action.device, batch["task_id"])

        targets = self.model.build_training_targets(
            normalized_obs=normalized_obs,
            task_id=task_id,
            normalized_action=normalized_action,
            coefficient_eps=self.coefficient_eps,
        )
        clean_coefficient = targets["coefficient_sg"]
        noise = torch.randn_like(clean_coefficient)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=clean_coefficient.device,
        ).long()
        noisy_coefficient = self.noise_scheduler.add_noise(
            clean_coefficient, noise, timesteps)
        prediction = self.model.smp.denoise_coefficients(
            noisy_coefficient, timesteps, targets["state_feature"])

        coeff_loss = F.mse_loss(
            prediction, noise, reduction="none").sum(dim=(1, 2)).mean()
        recon_squared_error = F.mse_loss(
            targets["reconstructed_action"], normalized_action,
            reduction="none")
        recon_loss = (
            recon_squared_error.sum(dim=(1, 2))
            / (2.0 * self.action_likelihood_std ** 2)
        ).mean()
        prior_concentration = self.model.smp.prior_router(
            targets["state_feature"])
        global_concentration = self.model.smp.global_usage_posterior()
        gate_loss, _ = sticky_gate_kl(
            global_concentration,
            targets["posterior_concentration"],
            alpha=self.alpha,
            alpha0=self.alpha0,
            kappa=self.kappa,
        )
        align_loss = router_alignment_kl(
            targets["posterior_concentration"], prior_concentration)
        total_loss = (
            self.loss_weights["coeff"] * coeff_loss
            + self.loss_weights["recon"] * recon_loss
            + self.loss_weights["gate"] * gate_loss
            + self.loss_weights["align"] * align_loss
        )
        metrics = self._training_metrics(
            total_loss, coeff_loss, recon_loss, gate_loss, align_loss,
            targets, task_id)
        return total_loss, metrics

    def _training_metrics(
            self, total_loss, coeff_loss, recon_loss, gate_loss, align_loss,
            targets, task_id) -> Dict[str, float]:
        gate = targets["posterior_gate"].detach()
        coefficient = targets["coefficient_sg"].detach()
        basis = targets["basis"].detach()
        entropy = -(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=-1)
        switches = (gate[:, 1:].argmax(dim=-1)
                    != gate[:, :-1].argmax(dim=-1)).float()
        usage = F.one_hot(
            gate.argmax(dim=-1), num_classes=self.num_experts).float()
        metrics = {
            "loss_total": total_loss.detach().item(),
            "loss_coeff": coeff_loss.detach().item(),
            "loss_recon": recon_loss.detach().item(),
            "loss_gate": gate_loss.detach().item(),
            "loss_align": align_loss.detach().item(),
            "basis_orthogonality_error": self.model.smp.basis_generator
                .orthogonality_error(basis).mean().item(),
            "basis_max_abs": basis.abs().max().item(),
            "basis_min_abs": basis.abs().min().item(),
            "gate_entropy": entropy.mean().item(),
            "gate_min": gate.min().item(),
            "gate_max": gate.max().item(),
            "expert_switch_rate": switches.mean().item(),
            "z_mean": coefficient.mean().item(),
            "z_std": coefficient.std().item(),
            "z_abs_mean": coefficient.abs().mean().item(),
            "z_abs_max": coefficient.abs().max().item(),
            "fraction_gate_lt_0.01": (gate < 0.01).float().mean().item(),
        }
        for expert_id in range(self.num_experts):
            metrics[f"gate_mean_expert_{expert_id}"] = (
                gate[..., expert_id].mean().item())
            metrics[f"usage_expert_{expert_id}"] = (
                usage[..., expert_id].mean().item())
        for current_task in task_id.unique():
            task_mask = task_id == current_task
            task_usage = usage[task_mask].mean(dim=(0, 1))
            for expert_id in range(self.num_experts):
                metrics[
                    f"task_{current_task.item()}_usage_expert_{expert_id}"
                ] = task_usage[expert_id].item()
        return metrics

    def _adaptive_gate(
            self, gate: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.adaptive_inference:
            mask, retained_mass = select_experts_by_importance_mass(
                gate, self.importance_mass_threshold)
        else:
            mask = torch.ones_like(gate, dtype=torch.bool)
            retained_mass = torch.ones_like(gate[..., 0])
        return gate * mask.to(gate.dtype), mask, retained_mass

    def _denoise_active_experts(
            self,
            noisy_coefficient: torch.Tensor,
            timestep,
            state_feature: torch.Tensor,
            active_experts: torch.Tensor) -> torch.Tensor:
        predictions = torch.zeros_like(noisy_coefficient)
        experts = self.model.smp.expert_bank.experts
        for expert_id in active_experts.nonzero(as_tuple=False).flatten().tolist():
            predictions[..., expert_id:expert_id + 1] = experts[expert_id](
                sample=noisy_coefficient[..., expert_id:expert_id + 1],
                timestep=timestep,
                global_cond=state_feature,
            )
        return predictions

    @torch.no_grad()
    def predict_action(
            self, obs_dict: Dict[str, torch.Tensor],
            task_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_start = time.perf_counter()
        normalized_obs = self.normalizer.normalize(
            self._prepare_observation(obs_dict))
        batch_size = next(iter(normalized_obs.values())).shape[0]
        task_id = self._task_tensor(batch_size, self.device, task_id)
        state_feature = self.model.encode_observation(normalized_obs, task_id)
        basis = self.model.smp.basis_generator(state_feature)
        route = self.model.smp.route(state_feature)
        gate, expert_mask, retained_mass = self._adaptive_gate(
            route["prior_gate"])
        active_experts = expert_mask.any(dim=0).any(dim=0)

        coefficient = torch.randn(
            batch_size, self.horizon, self.num_experts,
            device=self.device, dtype=self.dtype)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self._denoise_active_experts(
                coefficient, timestep, state_feature, active_experts)
            coefficient = self.noise_scheduler.step(
                prediction, timestep, coefficient).prev_sample

        normalized_action = self.model.smp.decode(basis, gate, coefficient)
        action_prediction = self.normalizer["action"].unnormalize(
            normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        active_count = expert_mask.sum(dim=-1).float()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_latency = time.perf_counter() - inference_start
        return {
            "action": action_prediction[:, start:end],
            "action_pred": action_prediction,
            "gate": gate,
            "dense_gate": route["prior_gate"],
            "expert_mask": expert_mask,
            "number_active_experts": active_count.mean(),
            "percentage_active_experts": (
                active_count / self.num_experts).mean(),
            "number_executed_experts": active_experts.sum(),
            "retained_importance_mass": retained_mass.mean(),
            "inference_latency_seconds": inference_latency,
        }

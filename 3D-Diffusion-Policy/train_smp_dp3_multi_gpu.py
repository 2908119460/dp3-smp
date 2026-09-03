if __name__ == "__main__":
    import os
    import pathlib
    import sys

    CODE_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(CODE_DIR))
    os.chdir(CODE_DIR)

import contextlib
import copy
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.distributed_training_runtime import (
    DistributedTrainingRuntime,
)
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from train import TrainDP3Workspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class ComputeLossModule(nn.Module):
    """Give policies with compute_loss a conventional DDP forward method."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


class TrainSMPDP3MultiGPUWorkspace(TrainDP3Workspace):
    def __init__(self, cfg, output_dir, runtime):
        super().__init__(cfg=cfg, output_dir=output_dir)
        self._distributed_runtime = runtime

    @staticmethod
    def _loader_kwargs(loader_cfg):
        kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
        shuffle = bool(kwargs.pop("shuffle", False))
        return kwargs, shuffle

    def _load_normalizer(self, dataset):
        runtime = self._distributed_runtime
        normalizer = dataset.get_normalizer() if runtime.is_main_process else None
        return runtime.broadcast_object(normalizer)

    def _run_rollout_subprocess(self, epoch):
        output_dir = pathlib.Path(self.output_dir)
        result_path = output_dir.joinpath(
            "evaluation", f"success_rates_epoch_{epoch:04d}.json")
        if result_path.is_file():
            print(f"Using existing rollout results: {result_path}")
        else:
            checkpoint_path = self.get_checkpoint_path()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"rollout checkpoint does not exist: {checkpoint_path}")
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            eval_env = os.environ.copy()
            eval_env["CUDA_VISIBLE_DEVICES"] = visible_devices.split(",")[0]
            eval_log_path = output_dir.joinpath("evaluation.log")
            with tempfile.TemporaryDirectory(
                    prefix=f".eval_epoch_{epoch:04d}_",
                    dir=str(output_dir)) as hydra_dir:
                command = [
                    sys.executable,
                    "-u",
                    str(pathlib.Path(__file__).with_name("eval.py")),
                    "--config-name=smp_dp3_adaptive_multi_gpu.yaml",
                    f"evaluation.checkpoint={checkpoint_path}",
                    f"evaluation.epoch={epoch}",
                    "training.device=cuda:0",
                    "logging.mode=disabled",
                    "hydra/job_logging=disabled",
                    f"hydra.run.dir={hydra_dir}",
                ]
                print(f"Starting isolated rollout evaluation for epoch {epoch}")
                with eval_log_path.open("a") as eval_log:
                    subprocess.run(
                        command,
                        cwd=str(pathlib.Path(__file__).parent),
                        env=eval_env,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            if not result_path.is_file():
                raise RuntimeError(
                    f"rollout did not create expected result: {result_path}")

        with result_path.open() as stream:
            success_rates = json.load(stream)
        metrics = {
            f"{task_name}/test_mean_score": float(score)
            for task_name, score in success_rates.items()
            if task_name != "mean"
        }
        metrics["test_mean_score"] = float(success_rates["mean"])
        return metrics

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        runtime = self._distributed_runtime
        device = runtime.device
        is_main = runtime.is_main_process
        output_dir = pathlib.Path(self.output_dir)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 2
            cfg.training.max_val_steps = 2
            cfg.training.checkpoint_every = 1
            cfg.training.sample_every = 1
        run_rollout = bool(cfg.training.get("run_rollout", False))
        run_validation = bool(cfg.training.get("run_validation", False))
        if cfg.checkpoint.save_last_snapshot:
            raise ValueError("distributed training does not support snapshots")

        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        runtime.barrier()

        resumed = False
        if cfg.training.resume:
            checkpoint_path = self.get_checkpoint_path()
            if checkpoint_path.is_file():
                if is_main:
                    print(f"Resuming from checkpoint {checkpoint_path}")
                self.load_checkpoint(path=checkpoint_path)
                resumed = True
        resume_to_total = bool(
            cfg.training.get("resume_to_total_epochs", False))
        if resumed and resume_to_total:
            self.epoch += 1

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"dataset must be BaseDataset, got {type(dataset)}")
        normalizer = self._load_normalizer(dataset)
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        train_kwargs, train_shuffle = self._loader_kwargs(cfg.dataloader)
        train_sampler = runtime.sampler(
            dataset,
            shuffle=train_shuffle,
            seed=cfg.training.seed,
            drop_last=bool(train_kwargs.get("drop_last", False)),
        )
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, **train_kwargs)

        val_dataloader = None
        if is_main and run_validation:
            val_dataset = dataset.get_validation_dataset()
            val_kwargs, _ = self._loader_kwargs(cfg.val_dataloader)
            val_dataloader = DataLoader(val_dataset, **val_kwargs)

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        ddp_model = DistributedDataParallel(
            ComputeLossModule(self.model),
            device_ids=[runtime.local_rank] if device.type == "cuda" else None,
            output_device=runtime.local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(
                cfg.distributed.find_unused_parameters),
        )
        torch.manual_seed(cfg.training.seed + runtime.rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(cfg.training.seed + runtime.rank)

        accumulation = int(cfg.training.gradient_accumulate_every)
        if accumulation < 1:
            raise ValueError("gradient_accumulate_every must be positive")
        batches_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            batches_per_epoch = min(
                batches_per_epoch, int(cfg.training.max_train_steps))
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                optimizer_steps_per_epoch * cfg.training.num_epochs),
            last_epoch=self.global_step - 1,
        )
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        wandb_run = None
        topk_manager = None
        if is_main:
            cfg.logging.name = str(cfg.logging.name)
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({
                "output_dir": self.output_dir,
                "world_size": runtime.world_size,
                "per_device_batch_size": cfg.dataloader.batch_size,
                "global_batch_size": (
                    cfg.dataloader.batch_size
                    * runtime.world_size
                    * accumulation),
            })
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk,
            )
            print(
                f"Distributed training: {runtime.world_size} GPUs, "
                f"per-device batch {cfg.dataloader.batch_size}, "
                f"effective global batch "
                f"{cfg.dataloader.batch_size * runtime.world_size * accumulation}")

        if resumed and run_rollout:
            resumed_epoch = self.epoch - 1 if resume_to_total else self.epoch
            if resumed_epoch % cfg.training.rollout_every == 0:
                if is_main:
                    resume_metrics = self._run_rollout_subprocess(resumed_epoch)
                    resume_metrics.update({
                        "epoch": resumed_epoch,
                        "global_step": self.global_step,
                    })
                    wandb_run.log(resume_metrics, step=self.global_step)
                runtime.barrier()

        num_epochs = cfg.training.num_epochs
        if resume_to_total:
            num_epochs = max(cfg.training.num_epochs - self.epoch, 0)
        train_sampling_batch = None
        self.optimizer.zero_grad(set_to_none=True)

        for _ in range(num_epochs):
            train_sampler.set_epoch(self.epoch)
            self.model.train()
            loss_sum = 0.0
            loss_count = 0
            step_log = {}
            iterator = tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
                disable=not is_main,
            )
            for batch_idx, batch in enumerate(iterator):
                if batch_idx >= batches_per_epoch:
                    break
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                if is_main and train_sampling_batch is None:
                    train_sampling_batch = batch

                window_start = (batch_idx // accumulation) * accumulation
                window_size = min(
                    accumulation, batches_per_epoch - window_start)
                should_step = (
                    (batch_idx + 1) % accumulation == 0
                    or batch_idx + 1 == batches_per_epoch)
                sync_context = (
                    contextlib.nullcontext()
                    if should_step else ddp_model.no_sync())
                with sync_context:
                    raw_loss, local_metrics = ddp_model(batch)
                    (raw_loss / window_size).backward()

                reduced_loss = runtime.reduce_mean(raw_loss).item()
                reduced_metrics = runtime.reduce_metrics(local_metrics)
                loss_sum += reduced_loss
                loss_count += 1
                if is_main:
                    iterator.set_postfix(loss=reduced_loss, refresh=False)
                    step_log = {
                        "train_loss": reduced_loss,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(reduced_metrics)

                if should_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    if ema is not None:
                        ema.step(self.model)
                    if is_main:
                        wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1

            epoch_loss = loss_sum / max(loss_count, 1)
            if is_main:
                step_log["train_loss"] = epoch_loss
            runtime.barrier()

            if is_main:
                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()
                rollout_due = (
                    run_rollout
                    and self.epoch % cfg.training.rollout_every == 0)
                checkpoint_saved_for_rollout = False
                if rollout_due:
                    self.save_checkpoint()
                    checkpoint_saved_for_rollout = True
                    step_log.update(self._run_rollout_subprocess(self.epoch))

                if (run_validation and val_dataloader is not None
                        and self.epoch % cfg.training.val_every == 0):
                    val_losses = []
                    with torch.no_grad():
                        for val_idx, val_batch in enumerate(val_dataloader):
                            val_batch = dict_apply(
                                val_batch,
                                lambda value: value.to(
                                    device, non_blocking=True),
                            )
                            val_loss, _ = self.model.compute_loss(val_batch)
                            val_losses.append(val_loss.item())
                            if (cfg.training.max_val_steps is not None
                                    and val_idx + 1 >= cfg.training.max_val_steps):
                                break
                    if val_losses:
                        step_log["val_loss"] = float(np.mean(val_losses))

                if (train_sampling_batch is not None
                        and self.epoch % cfg.training.sample_every == 0):
                    with torch.no_grad():
                        sample_result = policy.predict_action(
                            train_sampling_batch["obs"],
                            task_id=train_sampling_batch.get("task_id"),
                        )
                        sample_mse = torch.nn.functional.mse_loss(
                            sample_result["action_pred"],
                            train_sampling_batch["action"],
                        )
                        step_log["train_action_mse_error"] = sample_mse.item()
                        for key in (
                                "number_active_experts",
                                "percentage_active_experts",
                                "number_executed_experts",
                                "retained_importance_mass",
                                "inference_latency_seconds"):
                            if key in sample_result:
                                value = sample_result[key]
                                if torch.is_tensor(value):
                                    value = value.item()
                                step_log[f"inference_{key}"] = float(value)

                if not run_rollout:
                    step_log["test_mean_score"] = -epoch_loss

                if (cfg.checkpoint.save_ckpt
                        and self.epoch % cfg.training.checkpoint_every == 0):
                    if (cfg.checkpoint.save_last_ckpt
                            and not checkpoint_saved_for_rollout):
                        self.save_checkpoint()
                    metric_dict = {
                        key.replace("/", "_"): value
                        for key, value in step_log.items()
                    }
                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)
                wandb_run.log(step_log, step=self.global_step)
                policy.train()

            self.epoch += 1
            runtime.barrier()

        if is_main and wandb_run is not None:
            wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        "diffusion_policy_3d", "config")),
    config_name="smp_dp3_adaptive_multi_gpu",
)
def main(cfg):
    runtime = DistributedTrainingRuntime.initialize(
        backend=cfg.distributed.backend,
        timeout_minutes=cfg.distributed.timeout_minutes,
    )
    try:
        output_dir = pathlib.Path(cfg.distributed.output_dir).resolve()
        workspace = TrainSMPDP3MultiGPUWorkspace(
            cfg=cfg,
            output_dir=str(output_dir),
            runtime=runtime,
        )
        workspace.run()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

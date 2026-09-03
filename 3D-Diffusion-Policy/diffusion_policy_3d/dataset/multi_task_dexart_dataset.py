from typing import Dict, Sequence
import copy

import numpy as np
import torch

from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.dexart_dataset import DexArtDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class MultiTaskDexArtDataset(BaseDataset):
    """Task-balanced view over compatible DexArt demonstration datasets.

    The dataset length is ``num_tasks * max_task_length`` and indices alternate
    tasks, so a shuffled DataLoader samples every task with probability 1/N.
    Shorter tasks are cycled without merging replay buffers, preserving each
    source dataset's episode boundaries and validation split.
    """

    def __init__(
            self,
            tasks: Sequence[dict],
            state_dim: int,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None):
        super().__init__()
        if not tasks:
            raise ValueError("at least one DexArt task is required")

        self.state_dim = state_dim
        self.task_names = []
        self.datasets = []
        for task_id, task in enumerate(tasks):
            task_name = str(task["name"])
            dataset = DexArtDataset(
                zarr_path=str(task["zarr_path"]),
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                seed=seed + task_id,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                task_name=task_name,
            )
            self._validate_dataset(task_name, dataset)
            self.task_names.append(task_name)
            self.datasets.append(dataset)

        self.num_tasks = len(self.datasets)
        self.samples_per_task = max(len(dataset) for dataset in self.datasets)

    def _validate_dataset(self, task_name: str, dataset: DexArtDataset) -> None:
        replay = dataset.replay_buffer
        state_width = replay["state"].shape[-1]
        if state_width > self.state_dim:
            raise ValueError(
                f"task {task_name} state dimension {state_width} exceeds "
                f"configured state_dim {self.state_dim}")
        if replay["action"].shape[-1] != 22:
            raise ValueError(
                f"task {task_name} has incompatible action shape "
                f"{replay['action'].shape[1:]}; DexArt SMP expects [22]")
        if replay["point_cloud"].shape[1:3] != (1024, 6):
            raise ValueError(
                f"task {task_name} has incompatible point cloud shape "
                f"{replay['point_cloud'].shape[1:]}")
        if replay["imagin_robot"].shape[1:] != (96, 7):
            raise ValueError(
                f"task {task_name} has incompatible imagined robot shape "
                f"{replay['imagin_robot'].shape[1:]}")

    def __len__(self) -> int:
        return self.num_tasks * self.samples_per_task

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        task_id = index % self.num_tasks
        sample_index = (index // self.num_tasks) % len(self.datasets[task_id])
        sample = self.datasets[task_id][sample_index]
        sample["obs"]["agent_pos"] = self._pad_state(
            sample["obs"]["agent_pos"])
        sample["task_id"] = torch.tensor(task_id, dtype=torch.long)
        return sample

    def _pad_state(self, state: torch.Tensor) -> torch.Tensor:
        pad_width = self.state_dim - state.shape[-1]
        if pad_width == 0:
            return state
        return torch.nn.functional.pad(state, (0, pad_width))

    def get_validation_dataset(self) -> "MultiTaskDexArtDataset":
        validation = copy.copy(self)
        validation.datasets = [
            dataset.get_validation_dataset() for dataset in self.datasets
        ]
        validation.samples_per_task = max(
            (len(dataset) for dataset in validation.datasets), default=0)
        return validation

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        actions = []
        states = []
        for dataset in self.datasets:
            replay = dataset.replay_buffer
            actions.append(np.asarray(replay["action"][:], dtype=np.float32))
            state = np.asarray(replay["state"][:], dtype=np.float32)
            if state.shape[-1] < self.state_dim:
                state = np.pad(
                    state,
                    ((0, 0), (0, self.state_dim - state.shape[-1])),
                    mode="constant",
                )
            states.append(state)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "action": np.concatenate(actions, axis=0),
                "agent_pos": np.concatenate(states, axis=0),
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["imagin_robot"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

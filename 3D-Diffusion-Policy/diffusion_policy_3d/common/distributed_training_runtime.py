import datetime
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


@dataclass
class DistributedTrainingRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @classmethod
    def initialize(
            cls,
            backend: str = "nccl",
            timeout_minutes: int = 180) -> "DistributedTrainingRuntime":
        missing = [
            name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
            if name not in os.environ
        ]
        if missing:
            raise RuntimeError(
                "distributed launch variables are missing: "
                f"{', '.join(missing)}; launch with torchrun")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL training requires CUDA")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only "
                    f"{torch.cuda.device_count()} CUDA devices are visible")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        elif backend == "gloo":
            device = torch.device("cpu")
        else:
            raise ValueError(f"unsupported distributed backend: {backend}")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            backend=backend,
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def sampler(
            self,
            dataset,
            shuffle: bool,
            seed: int,
            drop_last: bool = False) -> DistributedSampler:
        return DistributedSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def broadcast_object(self, value, source: int = 0):
        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source)
        return payload[0]

    def reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.detach().clone().to(self.device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced / self.world_size

    def reduce_metrics(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        key_lists = [None for _ in range(self.world_size)]
        dist.all_gather_object(key_lists, sorted(metrics))
        keys = sorted({key for rank_keys in key_lists for key in rank_keys})
        values = torch.tensor(
            [float(metrics.get(key, 0.0)) for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        counts = torch.tensor(
            [1.0 if key in metrics else 0.0 for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if not self.is_main_process:
            return None
        means = values / counts.clamp_min(1.0)
        return {key: means[index].item() for index, key in enumerate(keys)}

    def close(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

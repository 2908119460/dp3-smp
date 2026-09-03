import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from diffusion_policy_3d.env_runner.base_runner import BaseRunner


class _DexArtPolicyAdapter:
    """Hide non-tensor inference metrics from the legacy DexArt runner."""

    def __init__(self, policy):
        self.policy = policy

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def reset(self):
        return self.policy.reset()

    def predict_action(self, obs):
        result = self.policy.predict_action(obs)
        return {
            key: value for key, value in result.items()
            if hasattr(value, "detach")
        }


class MultiTaskDexArtRunner(BaseRunner):
    """Evaluate one task-conditioned policy on each configured DexArt task."""

    def __init__(
            self,
            output_dir,
            task_names: Sequence[str],
            n_train: int = 20,
            max_steps: int = 50,
            max_steps_by_task=None,
            n_obs_steps: int = 2,
            n_action_steps: int = 8,
            fps: int = 10,
            eval_seeds=None,
            episodes_per_seed=None):
        super().__init__(output_dir)
        self.task_names = list(task_names)
        self.runner_kwargs = {
            "n_train": n_train,
            "max_steps": max_steps,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "fps": fps,
            "eval_seeds": eval_seeds,
            "episodes_per_seed": episodes_per_seed,
        }
        self.max_steps_by_task = {
            str(task_name): int(task_max_steps)
            for task_name, task_max_steps in (max_steps_by_task or {}).items()
        }
        self.evaluation_epoch = None

    def run(self, policy):
        # Keep offline training independent of optional simulator libraries.
        from diffusion_policy_3d.env_runner.dexart_runner import DexArtRunner

        evaluation_dir = Path(self.output_dir) / "evaluation"
        video_dir = evaluation_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        combined = {}
        scores = []
        success_rates = {}
        rollout_policy = _DexArtPolicyAdapter(policy)
        for task_id, task_name in enumerate(self.task_names):
            policy.set_task_id(task_id)
            runner_kwargs = dict(self.runner_kwargs)
            runner_kwargs["max_steps"] = self.max_steps_by_task.get(
                task_name, runner_kwargs["max_steps"])
            runner = DexArtRunner(
                output_dir=self.output_dir,
                task_name=task_name,
                **runner_kwargs,
            )
            task_log = runner.run(rollout_policy)
            score = float(task_log["test_mean_score"])
            scores.append(score)
            success_rates[task_name] = score
            video = task_log.get("sim_video_train")
            if video is not None and getattr(video, "_path", None):
                shutil.copy2(video._path, video_dir / f"{task_name}.mp4")
                if self.evaluation_epoch is not None:
                    shutil.copy2(
                        video._path,
                        video_dir / (
                            f"{task_name}_epoch_"
                            f"{self.evaluation_epoch:04d}.mp4"),
                    )
            for key, value in task_log.items():
                combined[f"{task_name}/{key}"] = value
        combined["test_mean_score"] = float(np.mean(scores))
        success_rates["mean"] = combined["test_mean_score"]
        with (evaluation_dir / "success_rates.json").open("w") as stream:
            json.dump(success_rates, stream, indent=2, sort_keys=True)
        if self.evaluation_epoch is not None:
            result_path = evaluation_dir / (
                f"success_rates_epoch_{self.evaluation_epoch:04d}.json")
            with result_path.open("w") as stream:
                json.dump(success_rates, stream, indent=2, sort_keys=True)
        return combined

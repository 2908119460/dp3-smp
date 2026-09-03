import wandb
import numpy as np
import random
import torch
import collections
import math
import tqdm
from termcolor import cprint
from diffusion_policy_3d.env import DexArtEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 eval_seeds=None,
                 episodes_per_seed=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train
        self.eval_seeds = None if eval_seeds is None else [
            int(seed) for seed in eval_seeds
        ]
        self.episodes_per_seed = episodes_per_seed

        if self.eval_seeds:
            if episodes_per_seed is None or episodes_per_seed <= 0:
                raise ValueError(
                    'episodes_per_seed must be positive when eval_seeds is set')

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []
        success_rates_by_seed = {}
        returns_by_seed = {}

        if self.eval_seeds:
            episode_groups = [
                (seed, self.episodes_per_seed) for seed in self.eval_seeds
            ]
        else:
            episode_groups = [(None, self.episode_train)]

        for seed, num_episodes in episode_groups:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                env_train.seed(seed)

            group_returns = []
            group_successes = []
            seed_suffix = '' if seed is None else f' seed={seed}'
            description = f"DexArt {self.task_name} Eval{seed_suffix}"

            for episode_id in tqdm.tqdm(
                    range(num_episodes), desc=description, leave=False,
                    mininterval=self.tqdm_interval_sec):
                obs = env_train.reset()

                policy.reset()

                done = False
                reward_sum = 0.
                for step_id in range(self.max_steps):
                    np_obs_dict = dict(obs)
                    obs_dict = dict_apply(
                        np_obs_dict,
                        lambda x: torch.from_numpy(x).to(device=device))

                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict[
                            'point_cloud'].unsqueeze(0)
                        obs_dict_input['imagin_robot'] = obs_dict[
                            'imagin_robot'].unsqueeze(0)
                        obs_dict_input['agent_pos'] = obs_dict[
                            'agent_pos'].unsqueeze(0)
                        action_dict = policy.predict_action(obs_dict_input)

                    np_action_dict = dict_apply(
                        action_dict,
                        lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)

                    obs, reward, done, info = env_train.step(action)
                    reward_sum += reward
                    done = np.all(done)

                    if done:
                        break

                success = bool(env_train.is_success())
                group_returns.append(float(reward_sum))
                group_successes.append(success)
                all_returns_train.append(float(reward_sum))
                all_success_rates_train.append(success)

            if seed is not None:
                success_rates_by_seed[seed] = float(np.mean(group_successes))
                returns_by_seed[seed] = float(np.mean(group_returns))

        SR_mean_train = float(np.mean(all_success_rates_train))
        returns_mean_train = float(np.mean(all_returns_train))

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train

        log_data['test_mean_score'] = SR_mean_train
        log_data['eval_num_episodes'] = len(all_success_rates_train)
        log_data['eval_num_successes'] = int(np.sum(all_success_rates_train))

        if success_rates_by_seed:
            seed_rates = list(success_rates_by_seed.values())
            seed_std = float(np.std(seed_rates, ddof=1)) \
                if len(seed_rates) > 1 else 0.0
            log_data['eval_success_rate_seed_std'] = seed_std
            for seed, success_rate in success_rates_by_seed.items():
                log_data[f'eval_success_rate_seed_{seed}'] = success_rate
                log_data[f'eval_mean_return_seed_{seed}'] = returns_by_seed[seed]

        # Wilson interval is more reliable than p +/- z*SE near 0 and 1.
        num_episodes = len(all_success_rates_train)
        z = 1.959963984540054
        z_squared = z ** 2
        denominator = 1 + z_squared / num_episodes
        center = (SR_mean_train + z_squared / (2 * num_episodes)) / denominator
        margin = z * math.sqrt(
            (SR_mean_train * (1 - SR_mean_train)
             + z_squared / (4 * num_episodes)) / num_episodes
        ) / denominator
        log_data['eval_success_rate_ci95_low'] = float(center - margin)
        log_data['eval_success_rate_ci95_high'] = float(center + margin)

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        if success_rates_by_seed:
            formatted_rates = ', '.join(
                f'{seed}:{rate:.3f}'
                for seed, rate in success_rates_by_seed.items())
            cprint(f'SR by seed: {formatted_rates}', 'green')
        cprint(
            f'Mean SR: {SR_mean_train:.3f} '
            f'({log_data["eval_num_successes"]}/{num_episodes}), '
            f'95% CI [{log_data["eval_success_rate_ci95_low"]:.3f}, '
            f'{log_data["eval_success_rate_ci95_high"]:.3f}]',
            'green')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data

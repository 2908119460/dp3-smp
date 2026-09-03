if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf, open_dict
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    env_runner_cfg = cfg.task.env_runner
    runner_target = str(env_runner_cfg.get('_target_', ''))
    dexart_runner_targets = (
        '.DexArtRunner',
        '.MultiTaskDexArtRunner',
    )
    if runner_target.endswith(dexart_runner_targets) and 'evaluation' in cfg:
        with open_dict(env_runner_cfg):
            env_runner_cfg.eval_seeds = list(cfg.evaluation.seeds)
            env_runner_cfg.episodes_per_seed = int(
                cfg.evaluation.episodes_per_seed)

    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()

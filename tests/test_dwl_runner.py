# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from rsl_rl.dwl_model import DwlActorModel, DwlCriticModel
from rsl_rl.dwl_ppo import DwlPPO
from rsl_rl.dwl_runner import DwlRunner, copy_and_prepare_dwl_runner_cfg, validate_dwl_reconstruction_observations


def _obs(num_envs=4):
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 12),
            "privileged": torch.randn(num_envs, 7),
        },
        batch_size=[num_envs],
    )


def _cfg():
    return {
        "num_steps_per_env": 2,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "test_dwl",
        "logger": "tensorboard",
        "check_for_nan": True,
        "obs_groups": {},
        "multi_gpu": None,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": None,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


class FakeEnv:
    num_envs = 4
    num_actions = 3
    max_episode_length = 100
    device = "cpu"
    cfg = SimpleNamespace()

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def get_observations(self):
        return _obs(self.num_envs)


def test_prepare_dwl_runner_cfg_sets_dwl_defaults_and_obs_groups():
    cfg = copy_and_prepare_dwl_runner_cfg(_cfg(), _obs())

    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["privileged"]}
    assert cfg["actor"]["class_name"] == "DwlActorModel"
    assert cfg["critic"]["class_name"] == "DwlCriticModel"
    assert cfg["algorithm"]["class_name"] == "DwlPPO"
    assert cfg["actor"]["decoder_obs_set"] == "critic"
    assert cfg["algorithm"]["reconstruction_loss_coef"] == 1.0
    assert cfg["algorithm"]["latent_l1_loss_coef"] == 0.0


def test_validate_dwl_reconstruction_observations_requires_target_group():
    cfg = copy_and_prepare_dwl_runner_cfg(_cfg(), _obs())
    obs = TensorDict({"policy": torch.randn(4, 12)}, batch_size=[4])

    try:
        validate_dwl_reconstruction_observations(cfg, obs)
    except ValueError as exc:
        assert "missing from rollout observations" in str(exc)
    else:
        raise AssertionError("Expected missing reconstruction target validation to fail.")


def test_dwl_runner_constructs_dwl_algorithm_and_preserves_decoder_targets():
    cfg = _cfg()
    runner = DwlRunner(FakeEnv(), cfg, log_dir=None, device="cpu")

    assert isinstance(runner.alg, DwlPPO)
    assert isinstance(runner.alg.actor, DwlActorModel)
    assert isinstance(runner.alg.critic, DwlCriticModel)
    assert runner.decoder_target_groups == ["privileged"]
    assert runner.alg.actor.reconstruction_target(runner.env.get_observations()).shape == (4, 7)

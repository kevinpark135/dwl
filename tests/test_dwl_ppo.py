# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from types import SimpleNamespace
from tensordict import TensorDict

from rsl_rl.dwl_model import DwlActorModel, DwlCriticModel
from rsl_rl.dwl_ppo import DwlPPO
from rsl_rl.storage import RolloutStorage


def _obs(num_envs=4):
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 12),
            "privileged": torch.randn(num_envs, 7),
        },
        batch_size=[num_envs],
    )


def _make_algorithm(num_envs=4, num_steps=2):
    obs = _obs(num_envs)
    obs_groups = {"actor": ["policy"], "critic": ["privileged"]}
    actor = DwlActorModel(
        obs,
        obs_groups,
        "actor",
        output_dim=3,
        history_length=3,
        encoder_hidden_dim=16,
        latent_dim=8,
        hidden_dims=[16],
        decoder_hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    critic = DwlCriticModel(obs, obs_groups, "critic", output_dim=1, hidden_dims=[16])
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [3], device="cpu")
    return DwlPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        desired_kl=None,
        reconstruction_loss_coef=0.25,
        latent_l1_loss_coef=0.05,
        device="cpu",
    )


def test_dwl_ppo_update_reports_reconstruction_and_latent_losses():
    alg = _make_algorithm()
    for _ in range(alg.storage.num_transitions_per_env):
        obs = _obs()
        alg.act(obs)
        rewards = torch.ones(alg.storage.num_envs)
        dones = torch.zeros(alg.storage.num_envs, dtype=torch.bool)
        alg.process_env_step(obs, rewards, dones, extras={})

    alg.compute_returns(_obs())
    losses = alg.update()

    assert "reconstruction" in losses
    assert "latent_l1" in losses
    assert losses["reconstruction"] > 0.0
    assert losses["latent_l1"] > 0.0
    assert alg.storage.step == 0


def test_dwl_auxiliary_losses_are_zero_for_plain_actor():
    alg = _make_algorithm()
    alg.actor = object()

    reconstruction_loss, latent_l1_loss = alg._dwl_losses(_obs())

    assert torch.allclose(reconstruction_loss, torch.tensor(0.0))
    assert torch.allclose(latent_l1_loss, torch.tensor(0.0))


def test_dwl_ppo_construct_algorithm_resolves_dwl_models_from_config():
    obs = _obs()
    env = SimpleNamespace(num_actions=3, num_envs=4)
    cfg = {
        "num_steps_per_env": 2,
        "obs_groups": {"actor": ["policy"], "critic": ["privileged"]},
        "multi_gpu": None,
        "actor": {
            "class_name": "DwlActorModel",
            "hidden_dims": [16],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
            "history_length": 3,
            "encoder_hidden_dim": 16,
            "latent_dim": 8,
            "decoder_hidden_dims": [16],
        },
        "critic": {
            "class_name": "DwlCriticModel",
            "hidden_dims": [16],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "DwlPPO",
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
            "reconstruction_loss_coef": 0.25,
            "latent_l1_loss_coef": 0.05,
        },
    }

    alg = DwlPPO.construct_algorithm(obs, env, cfg, device="cpu")

    assert isinstance(alg, DwlPPO)
    assert alg.reconstruction_loss_coef == 0.25
    assert alg.latent_l1_loss_coef == 0.05

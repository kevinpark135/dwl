# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from tensordict import TensorDict

from rsl_rl.dwl_model import DwlActorModel, DwlCriticModel


def _obs(num_envs=4):
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 12),
            "privileged": torch.randn(num_envs, 7),
        },
        batch_size=[num_envs],
    )


def test_dwl_actor_samples_actions_and_exposes_distribution_state():
    obs = _obs()
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

    actions = actor(obs, stochastic_output=True)

    assert actions.shape == (4, 3)
    assert actor.output_mean.shape == (4, 3)
    assert actor.output_std.shape == (4, 3)
    assert actor.get_output_log_prob(actions).shape == (4,)
    assert actor.output_entropy.shape == (4,)


def test_dwl_actor_decodes_privileged_reconstruction_target():
    obs = _obs()
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
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )

    reconstruction = actor.reconstruct(obs)
    target = actor.reconstruction_target(obs)

    assert reconstruction.shape == target.shape == (4, 7)


def test_dwl_actor_normalizes_reconstruction_target_for_decoder_loss():
    obs = _obs()
    obs["privileged"] = obs["privileged"] * 100.0 + 50.0
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
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )

    target = actor.normalized_reconstruction_target(obs, update=True)

    assert target.shape == (4, 7)
    assert torch.all(torch.abs(target.mean(dim=0)) < 1.0e-4)


def test_dwl_critic_returns_scalar_values_from_privileged_state():
    obs = _obs()
    obs_groups = {"critic": ["privileged"]}
    critic = DwlCriticModel(obs, obs_groups, "critic", output_dim=1, hidden_dims=[16])

    values = critic(obs)

    assert values.shape == (4, 1)
    assert not critic.is_recurrent
    assert critic.get_hidden_state() is None


def test_dwl_actor_export_wrapper_runs_deterministic_policy():
    obs = _obs()
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
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )

    jit_actor = actor.as_jit()
    actions = jit_actor(obs["policy"])

    assert actions.shape == (4, 3)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch
from isaaclab.utils.dict import class_to_dict
from tensordict import TensorDict

from agents.rsl_rl_ppo_cfg import (
    G1DwlHeightScanActorPPORunnerCfg,
    G1DwlPPORunnerCfg,
    G1DwlPpoNoDenoisingPPORunnerCfg,
    G1DwlPrivilegedActorPPORunnerCfg,
    G1ProprioceptiveBaselinePPORunnerCfg,
)
from rsl_rl.dwl_model import DwlActorModel, DwlCriticModel
from rsl_rl.dwl_ppo import DwlPPO


def test_dwl_agent_cfg_points_to_dwl_runner_model_and_algorithm():
    cfg = G1DwlPPORunnerCfg()

    assert cfg.class_name == "OnPolicyRunner"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["privileged"]}
    assert cfg.actor.class_name.endswith(":DwlActorModel")
    assert cfg.actor.history_length == 5
    assert cfg.actor.encoder_hidden_dim == 256
    assert cfg.actor.latent_dim == 24
    assert cfg.actor.decoder_obs_set == "critic"
    assert cfg.actor.hidden_dims == [48]
    assert cfg.actor.decoder_hidden_dims == [64]
    assert cfg.actor.obs_normalization
    assert cfg.actor.distribution_cfg.init_std == 1.0
    assert cfg.critic.class_name.endswith(":DwlCriticModel")
    assert cfg.critic.hidden_dims == [512, 512, 256]
    assert cfg.critic.obs_normalization
    assert cfg.algorithm.class_name.endswith(":DwlPPO")
    assert cfg.algorithm.reconstruction_loss_coef == 1.0
    assert cfg.algorithm.latent_l1_loss_coef == 0.002
    assert cfg.algorithm.policy_loss_coef == 5.0
    assert cfg.algorithm.value_loss_coef == 5.0
    assert cfg.algorithm.entropy_coef == 0.005
    assert cfg.algorithm.num_learning_epochs == 2
    assert cfg.algorithm.learning_rate == 1.0e-5
    assert cfg.algorithm.gamma == 0.995


def test_dwl_agent_cfg_constructs_dwl_algorithm_from_serialized_config():
    obs = TensorDict(
        {
            "policy": torch.randn(4, 47 * 5),
            "privileged": torch.randn(4, 184),
        },
        batch_size=[4],
    )
    env = SimpleNamespace(num_actions=12, num_envs=4)
    cfg = class_to_dict(G1DwlPPORunnerCfg())
    cfg["multi_gpu"] = None

    alg = DwlPPO.construct_algorithm(obs, env, cfg, device="cpu")

    assert alg.__class__.__name__ == "DwlPPO"
    assert alg.actor.__class__.__name__ == "DwlActorModel"
    assert alg.critic.__class__.__name__ == "DwlCriticModel"
    assert alg.actor.history_length == 5
    assert alg.actor.reconstruction_target(obs).shape == (4, 184)


def test_proprioceptive_baseline_agent_cfg_uses_stock_ppo_components():
    cfg = G1ProprioceptiveBaselinePPORunnerCfg()

    assert cfg.class_name == "OnPolicyRunner"
    assert cfg.experiment_name == "g1_proprio_baseline"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert cfg.actor.class_name == "MLPModel"
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.algorithm.class_name == "PPO"


def test_dwl_ppo_no_denoising_agent_cfg_uses_plain_asymmetric_ppo():
    cfg = G1DwlPpoNoDenoisingPPORunnerCfg()

    assert cfg.experiment_name == "g1_dwl_ppo_no_denoising"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["privileged"]}
    assert cfg.actor.class_name == "MLPModel"
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.algorithm.class_name == "PPO"
    assert cfg.algorithm.value_loss_coef == 5.0
    assert cfg.algorithm.learning_rate == 1.0e-5
    assert cfg.algorithm.gamma == 0.995


def test_height_scan_actor_agent_cfg_uses_dwl_plain_ppo_settings():
    cfg = G1DwlHeightScanActorPPORunnerCfg()

    assert cfg.experiment_name == "g1_dwl_height_scan_actor"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["privileged"]}
    assert cfg.algorithm.class_name == "PPO"


def test_privileged_actor_agent_cfg_uses_privileged_actor_and_critic_inputs():
    cfg = G1DwlPrivilegedActorPPORunnerCfg()

    assert cfg.experiment_name == "g1_dwl_privileged_actor"
    assert cfg.obs_groups == {"actor": ["privileged"], "critic": ["privileged"]}
    assert cfg.algorithm.class_name == "PPO"

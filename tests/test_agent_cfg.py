# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch
from isaaclab.utils.dict import class_to_dict
from tensordict import TensorDict

from agents.rsl_rl_ppo_cfg import G1DwlPPORunnerCfg, G1ProprioceptiveBaselinePPORunnerCfg
from rsl_rl.dwl_model import DwlActorModel, DwlCriticModel
from rsl_rl.dwl_ppo import DwlPPO


def test_dwl_agent_cfg_points_to_dwl_runner_model_and_algorithm():
    cfg = G1DwlPPORunnerCfg()

    assert cfg.class_name == "DwlRunner"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["privileged"]}
    assert cfg.actor.class_name == "DwlActorModel"
    assert cfg.actor.history_length == 5
    assert cfg.actor.encoder_hidden_dim == 128
    assert cfg.actor.latent_dim == 64
    assert cfg.actor.decoder_obs_set == "critic"
    assert cfg.critic.class_name == "DwlCriticModel"
    assert cfg.algorithm.class_name == "DwlPPO"
    assert cfg.algorithm.reconstruction_loss_coef == 1.0
    assert cfg.algorithm.latent_l1_loss_coef == 1.0e-3


def test_dwl_agent_cfg_constructs_dwl_algorithm_from_serialized_config():
    obs = TensorDict(
        {
            "policy": torch.randn(4, 47 * 5),
            "privileged": torch.randn(4, 64),
        },
        batch_size=[4],
    )
    env = SimpleNamespace(num_actions=12, num_envs=4)
    cfg = class_to_dict(G1DwlPPORunnerCfg())
    cfg["multi_gpu"] = None

    alg = DwlPPO.construct_algorithm(obs, env, cfg, device="cpu")

    assert isinstance(alg, DwlPPO)
    assert isinstance(alg.actor, DwlActorModel)
    assert isinstance(alg.critic, DwlCriticModel)
    assert alg.actor.history_length == 5
    assert alg.actor.reconstruction_target(obs).shape == (4, 64)


def test_proprioceptive_baseline_agent_cfg_uses_stock_ppo_components():
    cfg = G1ProprioceptiveBaselinePPORunnerCfg()

    assert cfg.class_name == "OnPolicyRunner"
    assert cfg.experiment_name == "g1_proprio_baseline"
    assert cfg.actor.class_name == "MLPModel"
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.algorithm.class_name == "PPO"

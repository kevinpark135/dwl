# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner configuration for the DWL G1 task."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset


@configclass
class DwlActorModelCfg(RslRlMLPModelCfg):
    """DWL actor with finite-history GRU encoder and decoder head."""

    class_name = "isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.rsl_rl.dwl_model:DwlActorModel"
    history_length = 5
    encoder_hidden_dim = 256
    encoder_num_layers = 1
    latent_dim = 24
    decoder_obs_set = "critic"
    decoder_hidden_dims = [64]


@configclass
class DwlCriticModelCfg(RslRlMLPModelCfg):
    """DWL privileged-state critic."""

    class_name = "isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.rsl_rl.dwl_model:DwlCriticModel"


@configclass
class DwlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with DWL reconstruction and latent regularization losses."""

    class_name = "isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.rsl_rl.dwl_ppo:DwlPPO"
    reconstruction_loss_coef = 1.0
    latent_l1_loss_coef = 0.002
    policy_loss_coef = 5.0


@configclass
class G1DwlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    # Newton needs ~1.7x the PPO iterations to match PhysX on G1. PhysX saturates near iter 3000
    # (reward ≈ +18, ep_len ≈ 980) and does not meaningfully improve on either metric past that —
    # reward oscillates +16 to +19 through iter 7500, ep_len stays flat. Newton reaches the same
    # (reward, ep_len) quality at iter 5000 (+16 / 984). Comparing reward alone is misleading:
    # ep_len confirms the robot is stable in both cases. The gap is sample-efficiency, not a
    # ceiling — no physics or reward tuning closes it.
    max_iterations = preset(default=3000, newton=5000)
    save_interval = 50
    experiment_name = "g1_dwl"
    obs_groups = {"actor": ["policy"], "critic": ["privileged"]}
    actor = DwlActorModelCfg(
        hidden_dims=[48],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = DwlCriticModelCfg(
        hidden_dims=[512, 512, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = DwlPpoAlgorithmCfg(
        value_loss_coef=5.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1ProprioceptiveBaselinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stock MLP PPO baseline with proprioception-only policy observations."""

    num_steps_per_env = 24
    max_iterations = preset(default=3000, newton=5000)
    save_interval = 50
    experiment_name = "g1_proprio_baseline"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

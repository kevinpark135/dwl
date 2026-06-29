# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from actions import DwlJointPositionActionCfg
from baseline_env_cfg import G1ProprioceptiveBaselineEnvCfg, G1ProprioceptiveBaselineEnvCfg_PLAY
from dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY
from observations import CONTROLLED_LEG_JOINT_NAMES


def test_dwl_env_cfg_wires_policy_and_privileged_observations():
    cfg = G1DwlEnvCfg()

    assert not cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.concatenate_terms
    assert cfg.observations.policy.history_length == 5
    assert cfg.observations.policy.flatten_history_dim
    assert cfg.observations.privileged.concatenate_terms
    assert cfg.observations.policy.joint_pos.noise.n_min == -0.3
    assert cfg.observations.policy.joint_vel.noise.n_max == 1.0
    assert cfg.observations.policy.base_ang_vel.noise.n_min == -0.1
    assert cfg.observations.policy.base_orientation.noise.n_max == 0.1


def test_dwl_env_cfg_uses_controlled_leg_actions_and_dwl_rewards():
    cfg = G1DwlEnvCfg()

    assert isinstance(cfg.actions.joint_pos, DwlJointPositionActionCfg)
    assert cfg.actions.joint_pos.joint_names == CONTROLLED_LEG_JOINT_NAMES
    assert cfg.actions.joint_pos.scale == 0.25
    assert cfg.actions.joint_pos.max_delay_steps == 4
    assert cfg.rewards.alive.weight == 1.0
    assert cfg.rewards.double_support.weight == 2.0
    assert cfg.rewards.lin_velocity_tracking.weight == 0.2
    assert cfg.rewards.periodic_force.weight == 0.0
    assert cfg.rewards.track_lin_vel_xy_exp is None
    assert cfg.rewards.dof_torques_l2 is None


def test_dwl_env_cfg_wires_dwl_events():
    cfg = G1DwlEnvCfg()

    assert cfg.events.init_dwl_buffers is not None
    assert cfg.events.store_friction.params["friction_range"] == (0.8, 1.2)
    assert cfg.events.physics_material.params["static_friction_range"] == (0.8, 1.2)
    assert cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert cfg.events.reset_base.params["velocity_range"]["roll"] == (0.0, 0.0)
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert cfg.rewards.feet_movement.params["acceleration_scale"] == 10.0
    assert cfg.events.system_delay.params["delay_range_s"] == (0.0, 0.0)
    assert cfg.events.motor_offset.params["offset_range"] == (0.0, 0.0)
    assert cfg.events.motor_strength.params["strength_distribution_params"] == (0.9, 1.1)
    assert cfg.events.pd_factors.params["pd_factor_distribution_params"] == (0.8, 1.2)
    assert cfg.events.joint_position_observation_noise.params["noise_range"] == (-0.3, 0.3)
    assert cfg.events.push_force_torques is not None
    assert cfg.events.push_force_torques.params["force_range"] == (0.0, 0.0)


def test_play_cfg_disables_policy_noise_and_push_wrenches():
    cfg = G1DwlEnvCfg_PLAY()

    assert not cfg.observations.policy.enable_corruption
    assert cfg.events.store_friction is None
    assert cfg.events.system_delay is None
    assert cfg.events.motor_offset is None
    assert cfg.events.motor_strength is None
    assert cfg.events.pd_factors is None
    assert cfg.events.add_base_mass is None
    assert cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert cfg.events.reset_base.params["pose_range"]["yaw"] == (0.0, 0.0)
    assert cfg.events.reset_base.params["velocity_range"]["roll"] == (0.0, 0.0)
    assert cfg.events.base_external_force_torque is None
    assert cfg.events.push_force_torques is None


def test_proprioceptive_baseline_removes_external_height_scan_sensor():
    cfg = G1ProprioceptiveBaselineEnvCfg()

    assert cfg.observations.policy.height_scan is None
    assert cfg.scene.height_scanner is None
    assert cfg.rewards.track_lin_vel_xy_exp is not None


def test_proprioceptive_baseline_play_removes_external_height_scan_sensor():
    cfg = G1ProprioceptiveBaselineEnvCfg_PLAY()

    assert cfg.observations.policy.height_scan is None
    assert cfg.scene.height_scanner is None
    assert not cfg.observations.policy.enable_corruption

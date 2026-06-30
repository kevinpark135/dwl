# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from actions import DwlJointPositionActionCfg
from baseline_env_cfg import G1ProprioceptiveBaselineEnvCfg, G1ProprioceptiveBaselineEnvCfg_PLAY
from dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY, YAW_CURRICULUM_STEPS
from observations import CONTROLLED_LEG_JOINT_NAMES


def test_dwl_env_cfg_wires_policy_and_privileged_observations():
    cfg = G1DwlEnvCfg()

    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.concatenate_terms
    assert cfg.observations.policy.history_length == 5
    assert cfg.observations.policy.flatten_history_dim
    assert cfg.observations.privileged.concatenate_terms
    assert cfg.observations.privileged.clock is not None
    assert cfg.observations.privileged.velocity_commands is not None
    assert cfg.observations.privileged.joint_pos is not None
    assert cfg.observations.privileged.joint_vel is not None
    assert cfg.observations.privileged.base_ang_vel is not None
    assert cfg.observations.privileged.base_orientation is not None
    assert cfg.observations.privileged.last_action is not None
    assert cfg.observations.policy.joint_pos.noise.n_min == -0.3
    assert cfg.observations.policy.joint_vel.noise.n_max == 1.0
    assert cfg.observations.policy.base_ang_vel.noise.n_min == -0.1
    assert cfg.observations.policy.base_orientation.noise.n_max == 0.1


def test_dwl_env_cfg_uses_controlled_leg_actions_and_dwl_rewards():
    cfg = G1DwlEnvCfg()

    assert isinstance(cfg.actions.joint_pos, DwlJointPositionActionCfg)
    assert cfg.actions.joint_pos.joint_names == CONTROLLED_LEG_JOINT_NAMES
    assert cfg.actions.joint_pos.scale == 1.0
    assert cfg.actions.joint_pos.max_delay_steps == 4
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 1.0
    assert cfg.scene.robot.actuators["legs"].damping[".*_knee_joint"] == 12.0
    assert cfg.scene.robot.actuators["feet"].damping == 8.0
    assert cfg.rewards.alive.weight == 0.05
    assert cfg.rewards.base_motion_penalty.weight == -0.01
    assert cfg.rewards.lin_velocity_tracking.weight == 4.0
    assert cfg.rewards.lin_velocity_tracking.params["tolerance"] == 2.5
    assert cfg.observations.policy.velocity_commands.params["yaw_curriculum_steps"] == YAW_CURRICULUM_STEPS
    assert cfg.observations.privileged.velocity_commands.params["yaw_curriculum_steps"] == YAW_CURRICULUM_STEPS
    assert cfg.rewards.ang_velocity_tracking.weight == 1.0
    assert cfg.rewards.ang_velocity_tracking.params["yaw_curriculum_steps"] == YAW_CURRICULUM_STEPS
    assert cfg.rewards.yaw_drift_penalty.weight == -0.15
    assert cfg.rewards.yaw_drift_penalty.params["yaw_curriculum_steps"] == YAW_CURRICULUM_STEPS
    assert cfg.rewards.forward_progress.weight == 1.0
    assert cfg.rewards.low_forward_speed_penalty.weight == -0.5
    assert cfg.rewards.low_forward_speed_penalty.params["min_forward_speed"] == 0.2
    assert cfg.rewards.low_forward_speed_penalty.params["grace_period_s"] == 0.5
    assert cfg.rewards.periodic_force.weight == 0.6
    assert cfg.rewards.periodic_velocity.weight == 1.0
    assert cfg.rewards.commanded_swing_air_time.weight == 0.4
    assert cfg.rewards.commanded_swing_air_time.params["min_forward_speed"] == 0.15
    assert cfg.rewards.commanded_swing_air_time.params["max_tilt"] == 0.55
    assert cfg.rewards.commanded_swing_air_time.params["target_air_time"] == 0.4
    assert cfg.rewards.commanded_swing_air_time.params["max_air_time"] == 0.8
    assert cfg.rewards.foot_height_tracking.weight == 0.3
    assert cfg.rewards.foot_velocity_tracking.weight == 0.2
    assert cfg.rewards.foot_lateral_tracking.weight == 0.8
    assert cfg.rewards.foot_lateral_tracking.params["target_width"] == 0.16
    assert cfg.rewards.foot_lateral_velocity.weight == -0.04
    assert cfg.rewards.foot_sagittal_tracking.weight == 0.8
    assert cfg.rewards.foot_sagittal_tracking.params["sensor_cfg"] is not None
    assert cfg.rewards.foot_sagittal_symmetry.weight == 0.4
    assert cfg.rewards.hip_deviation.weight == -0.03
    assert cfg.rewards.default_joint_tracking.weight == 0.02
    assert cfg.rewards.action_smoothness.weight == -0.0002
    assert cfg.rewards.energy_cost.weight == -0.00003
    assert cfg.rewards.feet_movement.weight == -0.0005
    assert cfg.rewards.body_contact.weight == -5.0
    assert cfg.rewards.body_contact.params["sensor_cfg"].body_names == "torso_link"
    assert cfg.rewards.track_lin_vel_xy_exp is None
    assert cfg.rewards.dof_torques_l2 is None


def test_dwl_env_cfg_wires_dwl_events():
    cfg = G1DwlEnvCfg()

    assert cfg.events.init_dwl_buffers is not None
    assert cfg.events.store_friction.params["friction_range"] == (0.2, 2.0)
    assert cfg.events.physics_material.params["static_friction_range"] == (0.2, 2.0)
    assert cfg.events.reset_robot_joints.params["position_range"] == (-0.05, 0.05)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (-0.1, 0.1)
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.5, 1.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.scene.robot.init_state.joint_pos[".*_elbow_pitch_joint"] == 0.35
    assert cfg.events.reset_base.params["pose_range"]["yaw"] == (-0.1, 0.1)
    assert cfg.events.reset_base.params["velocity_range"]["roll"] == (-0.05, 0.05)
    assert cfg.scene.terrain.max_init_terrain_level == 1
    assert cfg.scene.height_scanner.pattern_cfg.resolution == 0.1
    assert cfg.scene.height_scanner.pattern_cfg.size == [1.1, 0.7]
    assert cfg.rewards.feet_movement.params["acceleration_scale"] == 10.0
    assert cfg.events.system_delay.params["delay_range_s"] == (0.0, 0.0)
    assert cfg.events.motor_offset.params["offset_range"] == (-0.05, 0.05)
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

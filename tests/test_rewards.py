# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg

from gait import DwlGaitCfg
from rewards import (
    action_smoothness,
    alive,
    base_height_tracking,
    base_motion_penalty,
    body_contact,
    commanded_swing_air_time,
    default_joint_tracking,
    energy_cost,
    foot_height_tracking,
    foot_lateral_tracking,
    foot_lateral_velocity,
    foot_sagittal_symmetry,
    foot_sagittal_tracking,
    foot_velocity_tracking,
    forward_progress,
    hip_deviation,
    large_contact,
    lin_velocity_tracking,
    low_forward_speed_penalty,
    periodic_force,
    periodic_velocity,
    feet_movement,
    tracking_exp,
    yaw_drift_penalty,
)


class MockScene(dict):
    pass


class MockCommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, command_name):
        return self.command


def _mock_env(num_envs=1):
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.25,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        scene=MockScene(),
        command_manager=MockCommandManager(torch.zeros(num_envs, 3)),
        action_manager=SimpleNamespace(
            action=torch.tensor([[1.0, 2.0, 3.0]]).repeat(num_envs, 1),
            prev_action=torch.tensor([[0.5, 1.5, 2.5]]).repeat(num_envs, 1),
        ),
    )
    env.scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            root_ang_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            projected_gravity_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.7]]).repeat(num_envs, 1)),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)),
            body_link_pose_w=SimpleNamespace(torch=torch.zeros(num_envs, 2, 7)),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 2, 6)),
            body_lin_acc_w=SimpleNamespace(torch=torch.zeros(num_envs, 2, 6)),
            joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 2)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 2)),
            joint_vel=SimpleNamespace(torch=torch.ones(num_envs, 2)),
            applied_torque=SimpleNamespace(torch=2.0 * torch.ones(num_envs, 2)),
        )
    )
    env.scene.sensors = {
        "contact_forces": SimpleNamespace(
            data=SimpleNamespace(
                net_forces_w=SimpleNamespace(
                    torch=torch.tensor([[[0.0, 0.0, 400.0], [0.0, 0.0, 0.0]]]).repeat(num_envs, 1, 1)
                )
            )
        )
    }
    return env


def test_tracking_exp_matches_paper_kernel():
    error = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(tracking_exp(error, tolerance=2.0), torch.exp(torch.tensor([-10.0])))


def test_velocity_and_height_tracking_are_one_when_at_target():
    env = _mock_env()

    assert torch.allclose(lin_velocity_tracking(env), torch.ones(1))
    assert torch.allclose(base_height_tracking(env), torch.ones(1))


def test_forward_progress_rewards_commanded_positive_body_velocity():
    env = _mock_env()
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.scene["robot"].data.root_lin_vel_b.torch[:, 0] = 0.25

    assert torch.allclose(forward_progress(env), torch.tensor([0.5]))

    env.command_manager.command = torch.tensor([[0.1, 0.0, 0.0]])
    assert torch.allclose(forward_progress(env), torch.zeros(1))


def test_low_forward_speed_penalty_is_commanded_and_after_grace_period():
    env = _mock_env()
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.episode_length_buf[:] = 3
    env.scene["robot"].data.root_lin_vel_b.torch[:, 0] = 0.1

    assert torch.allclose(low_forward_speed_penalty(env, min_forward_speed=0.2), torch.tensor([0.25]))

    env.episode_length_buf[:] = 1
    assert torch.allclose(low_forward_speed_penalty(env, min_forward_speed=0.2), torch.zeros(1))

    env.episode_length_buf[:] = 3
    env.command_manager.command = torch.tensor([[0.1, 0.0, 0.0]])
    assert torch.allclose(low_forward_speed_penalty(env, min_forward_speed=0.2), torch.zeros(1))


def test_stability_rewards_ignore_commanded_horizontal_motion():
    env = _mock_env()

    assert torch.allclose(alive(env), torch.ones(1))

    env.scene["robot"].data.root_lin_vel_b.torch[:] = torch.tensor([[1.0, 2.0, 3.0]])
    env.scene["robot"].data.root_ang_vel_b.torch[:] = torch.tensor([[4.0, 5.0, 6.0]])
    assert torch.allclose(base_motion_penalty(env), torch.tensor([50.0]))


def test_periodic_force_rewards_stance_foot_contact():
    env = _mock_env()
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    assert torch.allclose(periodic_force(env, sensor_cfg=sensor_cfg), torch.ones(1))


def test_periodic_velocity_rewards_swing_foot_motion():
    env = _mock_env()
    env.scene["robot"].data.body_link_vel_w.torch[:, 1, 0] = 1.0
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

    assert torch.allclose(periodic_velocity(env, asset_cfg=asset_cfg), torch.ones(1))


def test_commanded_swing_air_time_is_gated_by_forward_command_and_rewards_touchdown():
    env = _mock_env()
    env.episode_length_buf[:] = 1
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.scene["robot"].data.root_lin_vel_b.torch[:, 0] = 0.3
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, 2] = 0.06
    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 1, 2] = 0.0
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    swing_reward = commanded_swing_air_time(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    assert torch.allclose(swing_reward, torch.tensor([0.25]))

    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 1, 2] = 350.0
    touchdown_reward = commanded_swing_air_time(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    assert touchdown_reward.item() > 1.0

    env.command_manager.command = torch.zeros(1, 3)
    assert torch.allclose(commanded_swing_air_time(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg), torch.zeros(1))

    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.scene["robot"].data.root_lin_vel_b.torch[:, 0] = 0.0
    assert torch.allclose(
        commanded_swing_air_time(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg, min_forward_speed=0.15),
        torch.zeros(1),
    )


def test_foot_tracking_uses_gait_references_for_swing_foot():
    env = _mock_env()
    env.episode_length_buf[:] = 1
    env.step_dt = 0.25
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, 2] = 0.1
    env.scene["robot"].data.body_link_vel_w.torch[:, 1, 2] = 0.0125
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    gait_cfg = DwlGaitCfg()

    assert torch.allclose(foot_height_tracking(env, gait_cfg=gait_cfg, asset_cfg=asset_cfg), torch.ones(1), atol=1e-6)
    assert torch.allclose(foot_velocity_tracking(env, gait_cfg=gait_cfg, asset_cfg=asset_cfg), torch.ones(1), atol=1e-6)


def test_foot_lateral_tracking_rewards_narrow_body_frame_corridor():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    env.scene["robot"].data.body_link_pose_w.torch[:, 0, :3] = torch.tensor([[0.0, 0.08, 0.0]])
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, :3] = torch.tensor([[0.0, -0.08, 0.0]])

    assert torch.allclose(foot_lateral_tracking(env, asset_cfg=asset_cfg), torch.ones(1), atol=1e-6)

    env.scene["robot"].data.body_link_pose_w.torch[:, 0, 1] = 0.35
    assert foot_lateral_tracking(env, asset_cfg=asset_cfg).item() < 0.2


def test_foot_lateral_velocity_penalizes_side_shuffle_only():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    env.scene["robot"].data.body_link_vel_w.torch[:, :, :3] = torch.tensor([[[2.0, 0.5, 1.0], [3.0, -0.25, 2.0]]])

    assert torch.allclose(foot_lateral_velocity(env, asset_cfg=asset_cfg), torch.tensor([0.3125]))


def test_foot_sagittal_tracking_rewards_alternating_forward_swing():
    env = _mock_env()
    env.episode_length_buf[:] = 1
    env.step_dt = 0.25
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.scene["robot"].data.root_lin_vel_b.torch[:, 0] = 0.5
    env.scene["robot"].data.body_link_vel_w.torch[:, :, :3] = torch.tensor([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]])
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, 2] = 0.07
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    assert torch.allclose(foot_sagittal_tracking(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg), torch.ones(1), atol=1e-6)

    env.scene["robot"].data.body_link_vel_w.torch[:, 1, 0] = 0.0
    assert foot_sagittal_tracking(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg).item() < 0.65

    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 1, 2] = 350.0
    assert foot_sagittal_tracking(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg).item() < 0.6


def test_foot_sagittal_symmetry_rewards_feet_centered_around_base():
    env = _mock_env()
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.0]])
    env.scene["robot"].data.body_link_pose_w.torch[:, 0, :3] = torch.tensor([[-0.12, 0.08, 0.0]])
    env.scene["robot"].data.body_link_pose_w.torch[:, 1, :3] = torch.tensor([[0.12, -0.08, 0.0]])
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

    assert torch.allclose(foot_sagittal_symmetry(env, asset_cfg=asset_cfg), torch.ones(1), atol=1e-6)

    env.scene["robot"].data.body_link_pose_w.torch[:, 1, 0] = 0.35
    assert foot_sagittal_symmetry(env, asset_cfg=asset_cfg).item() < 0.5


def test_yaw_drift_penalty_is_gated_by_effective_yaw_command():
    env = _mock_env()
    env.common_step_counter = 0
    env.command_manager.command = torch.tensor([[0.5, 0.0, 0.4]])
    env.scene["robot"].data.root_ang_vel_b.torch[:, 2] = 0.2
    curriculum_steps = (12000, 18000, 24000, 36000)

    assert torch.allclose(yaw_drift_penalty(env, yaw_curriculum_steps=curriculum_steps), torch.tensor([0.04]))

    env.common_step_counter = 36000
    assert torch.allclose(yaw_drift_penalty(env, yaw_curriculum_steps=curriculum_steps), torch.zeros(1))


def test_regularization_terms_return_expected_costs():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    assert torch.allclose(default_joint_tracking(env, asset_cfg=asset_cfg), torch.ones(1))
    assert torch.allclose(energy_cost(env, asset_cfg=asset_cfg), torch.tensor([4.0]))
    assert torch.allclose(action_smoothness(env), torch.tensor([5.0]))

    env.scene["robot"].data.joint_pos.torch[:] = torch.tensor([[0.5, -0.25]])
    env.scene["robot"].data.joint_vel.torch[:] = torch.tensor([[2.0, 1.0]])
    assert torch.allclose(hip_deviation(env, asset_cfg=asset_cfg, velocity_weight=0.1), torch.tensor([0.8125]))


def test_feet_movement_penalizes_only_vertical_foot_motion():
    env = _mock_env()
    env.scene["robot"].data.body_link_vel_w.torch[:, 0, 0] = 10.0
    env.scene["robot"].data.body_lin_acc_w.torch[:, 0, 1] = 20.0
    env.scene["robot"].data.body_link_vel_w.torch[:, :, 2] = torch.tensor([[1.0, 2.0]])
    env.scene["robot"].data.body_lin_acc_w.torch[:, :, 2] = torch.tensor([[3.0, 4.0]])
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

    assert torch.allclose(feet_movement(env, asset_cfg=asset_cfg), torch.tensor([5.25]))


def test_large_contact_penalizes_force_above_threshold():
    env = _mock_env()
    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 0, 2] = 450.0
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    assert torch.allclose(large_contact(env, sensor_cfg=sensor_cfg), torch.tensor([50.0]))


def test_body_contact_penalizes_non_foot_contact_events():
    env = _mock_env()
    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 0, 2] = 0.5
    env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, 1, 2] = 2.0
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    assert torch.allclose(body_contact(env, sensor_cfg=sensor_cfg), torch.tensor([1.0]))

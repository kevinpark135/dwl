# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg

from observations import (
    base_orientation_rpy,
    make_policy_observation_terms,
    make_privileged_observation_terms,
    policy_clock,
    quat_to_rpy,
    state_body_mass,
    state_feet_contact,
    state_feet_movement,
    state_friction,
    state_push_force_torques,
    state_stance_mask,
)


class MockScene(dict):
    pass


def _mock_env(num_envs=2):
    episode_length_buf = torch.arange(num_envs)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.5,
        episode_length_buf=episode_length_buf,
        scene=MockScene(),
    )
    env.scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)),
            body_link_pose_w=SimpleNamespace(torch=torch.zeros(num_envs, 2, 7)),
            body_link_vel_w=SimpleNamespace(torch=torch.ones(num_envs, 2, 6)),
            body_mass=SimpleNamespace(torch=torch.full((num_envs, 3), 2.0)),
        )
    )
    env.scene["robot"].data.body_link_pose_w.torch[:, :, :3] = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    ).repeat(num_envs, 1, 1)
    env.scene.sensors = {
        "contact_forces": SimpleNamespace(
            data=SimpleNamespace(
                net_forces_w=SimpleNamespace(
                    torch=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]]).repeat(num_envs, 1, 1)
                )
            )
        )
    }
    return env


def test_quat_to_rpy_identity():
    quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    assert torch.allclose(quat_to_rpy(quat_xyzw), torch.zeros(1, 3))


def test_quat_to_rpy_yaw_rotation():
    yaw = math.pi / 2.0
    quat_xyzw = torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]])

    expected = torch.tensor([[0.0, 0.0, yaw]])
    assert torch.allclose(quat_to_rpy(quat_xyzw), expected, atol=1.0e-6)


def test_base_orientation_rpy_reads_robot_root_quat():
    quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=quat_xyzw)),
            )
        }
    )

    assert torch.allclose(base_orientation_rpy(env), torch.zeros(1, 3))


def test_policy_clock_uses_episode_time():
    env = _mock_env(num_envs=2)

    assert torch.allclose(policy_clock(env), torch.tensor([[0.0, 1.0], [0.0, -1.0]]), atol=1.0e-6)


def test_state_stance_mask_uses_episode_time():
    env = _mock_env(num_envs=2)

    assert torch.equal(state_stance_mask(env), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_optional_privileged_terms_have_safe_defaults():
    env = _mock_env(num_envs=2)

    assert torch.allclose(state_friction(env), torch.ones(2, 1))
    assert torch.allclose(state_push_force_torques(env), torch.zeros(2, 6))


def test_state_feet_movement_flattens_foot_positions_and_velocities():
    env = _mock_env(num_envs=1)
    foot_body_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(state_feet_movement(env, foot_body_cfg), expected)


def test_state_feet_contact_thresholds_contact_forces():
    env = _mock_env(num_envs=1)
    contact_sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    assert torch.equal(state_feet_contact(env, contact_sensor_cfg, contact_threshold=1.0), torch.tensor([[0.0, 1.0]]))


def test_state_body_mass_sums_selected_bodies():
    env = _mock_env(num_envs=2)
    robot_cfg = SceneEntityCfg("robot", body_ids=slice(None))

    assert torch.allclose(state_body_mass(env, robot_cfg), torch.full((2, 1), 6.0))


def test_observation_term_factories_include_expected_terms():
    assert set(make_policy_observation_terms()) == {
        "clock",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "base_ang_vel",
        "base_orientation",
        "last_action",
    }
    assert set(make_privileged_observation_terms()) == {
        "base_lin_vel",
        "friction",
        "push_force_torques",
        "cycle_time",
        "stance_mask",
        "feet_movement",
        "feet_contact",
        "body_mass",
        "current_reward",
        "joint_torques",
        "height_scan",
    }

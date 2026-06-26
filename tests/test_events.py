# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg

from events import (
    DWL_FRICTION_ATTR,
    DWL_MOTOR_OFFSET_ATTR,
    DWL_MOTOR_STRENGTH_ATTR,
    DWL_OBSERVATION_NOISE_RANGES_ATTR,
    DWL_PD_FACTORS_ATTR,
    DWL_PUSH_FORCE_TORQUES_ATTR,
    DWL_SYSTEM_DELAY_ATTR,
    clear_push_force_torques,
    init_dwl_event_buffers,
    randomize_body_mass,
    randomize_motor_offset,
    randomize_motor_strength,
    randomize_pd_factors,
    randomize_system_delay,
    randomize_joint_reset_noise,
    randomize_joint_position_observation_noise,
    sample_push_force_torques,
    store_friction,
)


class MockScene(dict):
    num_envs = 2


class MockWrenchComposer:
    def __init__(self):
        self.forces = None
        self.torques = None
        self.body_ids = None
        self.env_ids = None

    def set_forces_and_torques_index(self, forces, torques, body_ids, env_ids):
        self.forces = forces
        self.torques = torques
        self.body_ids = body_ids
        self.env_ids = env_ids


class MockAsset:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_bodies = 2
        self.num_joints = 2
        self.permanent_wrench_composer = MockWrenchComposer()
        self.actuators = {
            "legs": SimpleNamespace(
                joint_indices=torch.tensor([0, 1]),
                effort_limit=torch.full((2, 2), 100.0),
                stiffness=torch.full((2, 2), 20.0),
                damping=torch.full((2, 2), 5.0),
            )
        }
        self.data = SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((2, 2), 10.0)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(2, 2)),
            default_joint_vel=SimpleNamespace(torch=torch.zeros(2, 2)),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.stack((-torch.ones(2, 2), torch.ones(2, 2)), dim=-1)),
            soft_joint_vel_limits=SimpleNamespace(torch=torch.ones(2, 2)),
        )
        self.written_masses = None
        self.written_joint_pos = None
        self.written_joint_vel = None

    def set_masses_index(self, masses, body_ids, env_ids):
        self.written_masses = masses
        self.data.body_mass.torch[env_ids.long()[:, None], body_ids.long()] = masses

    def write_joint_position_to_sim_index(self, position, joint_ids, env_ids):
        self.written_joint_pos = position

    def write_joint_velocity_to_sim_index(self, velocity, joint_ids, env_ids):
        self.written_joint_vel = velocity


def _mock_env():
    scene = MockScene()
    scene["robot"] = MockAsset()
    return SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=scene)


def test_init_and_clear_dwl_event_buffers():
    env = _mock_env()

    init_dwl_event_buffers(env)
    assert torch.allclose(getattr(env, DWL_FRICTION_ATTR), torch.ones(2, 1))
    assert torch.allclose(getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR), torch.zeros(2, 6))
    assert torch.allclose(getattr(env, DWL_SYSTEM_DELAY_ATTR), torch.zeros(2, 1))
    assert getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["joint_position"] == (-0.3, 0.3)

    getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR)[:] = 1.0
    clear_push_force_torques(env, torch.tensor([0]))
    assert torch.allclose(getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR)[0], torch.zeros(6))
    assert torch.allclose(getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR)[1], torch.ones(6))


def test_store_friction_samples_and_stores_range():
    env = _mock_env()

    friction = store_friction(env, None, friction_range=(0.5, 0.5))
    assert torch.allclose(friction, torch.full((2, 1), 0.5))
    assert torch.allclose(getattr(env, DWL_FRICTION_ATTR), torch.full((2, 1), 0.5))


def test_sample_push_force_torques_applies_and_stores_wrench():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", body_ids=[0])

    sample_push_force_torques(env, None, force_range=(1.0, 1.0), torque_range=(2.0, 2.0), asset_cfg=asset_cfg)

    asset = env.scene["robot"]
    assert torch.allclose(asset.permanent_wrench_composer.forces, torch.ones(2, 1, 3))
    assert torch.allclose(asset.permanent_wrench_composer.torques, 2.0 * torch.ones(2, 1, 3))
    assert torch.allclose(getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR), torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]).repeat(2, 1))


def test_randomize_body_mass_adds_payload_and_clamps():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", body_ids=[0])

    randomize_body_mass(env, None, mass_distribution_params=(1.0, 1.0), asset_cfg=asset_cfg)

    assert torch.allclose(env.scene["robot"].written_masses, torch.full((2, 1), 11.0))


def test_randomize_joint_reset_noise_writes_clamped_joint_state():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    randomize_joint_reset_noise(
        env,
        torch.tensor([0, 1]),
        position_range=(0.5, 0.5),
        velocity_range=(0.25, 0.25),
        asset_cfg=asset_cfg,
    )

    assert torch.allclose(env.scene["robot"].written_joint_pos, torch.full((2, 2), 0.5))
    assert torch.allclose(env.scene["robot"].written_joint_vel, torch.full((2, 2), 0.25))


def test_observation_noise_events_record_ranges():
    env = _mock_env()

    randomize_joint_position_observation_noise(env, noise_range=(-0.2, 0.2))

    assert getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["joint_position"] == (-0.2, 0.2)


def test_randomize_system_delay_samples_and_stores_delay():
    env = _mock_env()

    delay = randomize_system_delay(env, None, delay_range_s=(0.01, 0.01))

    assert torch.allclose(delay, torch.full((2, 1), 0.01))
    assert torch.allclose(getattr(env, DWL_SYSTEM_DELAY_ATTR), torch.full((2, 1), 0.01))


def test_randomize_motor_offset_samples_and_stores_offsets():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    offsets = randomize_motor_offset(env, None, offset_range=(0.05, 0.05), asset_cfg=asset_cfg)

    assert torch.allclose(offsets, torch.full((2, 2), 0.05))
    assert torch.allclose(getattr(env, DWL_MOTOR_OFFSET_ATTR), torch.full((2, 2), 0.05))


def test_randomize_motor_strength_scales_actuator_effort_limits():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    strength = randomize_motor_strength(
        env,
        None,
        strength_distribution_params=(1.1, 1.1),
        asset_cfg=asset_cfg,
    )

    assert torch.allclose(strength, torch.full((2, 2), 1.1))
    assert torch.allclose(env.scene["robot"].actuators["legs"].effort_limit, torch.full((2, 2), 110.0))
    assert torch.allclose(getattr(env, DWL_MOTOR_STRENGTH_ATTR), torch.full((2, 2), 1.1))


def test_randomize_pd_factors_scales_actuator_stiffness_and_damping():
    env = _mock_env()
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    factors = randomize_pd_factors(
        env,
        None,
        pd_factor_distribution_params=(1.2, 1.2),
        asset_cfg=asset_cfg,
    )

    assert torch.allclose(factors, torch.full((2, 2, 2), 1.2))
    assert torch.allclose(env.scene["robot"].actuators["legs"].stiffness, torch.full((2, 2), 24.0))
    assert torch.allclose(env.scene["robot"].actuators["legs"].damping, torch.full((2, 2), 6.0))
    assert torch.allclose(getattr(env, DWL_PD_FACTORS_ATTR), torch.full((2, 2, 2), 1.2))

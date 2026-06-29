# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from action_terms import DwlJointPositionAction
from actions import DwlJointPositionActionCfg, delay_steps_from_env


class MockAsset:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_joints = 2
        self.data = SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.tensor([[0.1, -0.2], [0.3, -0.4]]))
        )
        self.position_target = None
        self.position_target_joint_ids = None

    def find_joints(self, joint_names, preserve_order=False):
        return [0, 1], ["left_joint", "right_joint"]

    def set_joint_position_target_index(self, target, joint_ids):
        self.position_target = target.clone()
        self.position_target_joint_ids = joint_ids


def _mock_env():
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.5,
        scene={"robot": MockAsset()},
        dwl_system_delay_s=torch.tensor([[0.5], [0.0]]),
        dwl_motor_offset=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )


def test_delay_steps_from_env_ceilings_and_clamps_delay_seconds():
    env = _mock_env()
    env.dwl_system_delay_s = torch.tensor([[0.01], [1.2]])

    assert torch.equal(delay_steps_from_env(env, max_delay_steps=2), torch.tensor([1, 2]))


def test_dwl_joint_position_action_delays_actions_and_adds_motor_offset():
    env = _mock_env()
    cfg = DwlJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        max_delay_steps=2,
    )
    action = DwlJointPositionAction(cfg, env)

    action.process_actions(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert torch.allclose(action.processed_actions, torch.tensor([[0.2, 0.0], [3.6, 4.0]]))

    action.process_actions(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    action.apply_actions()

    assert torch.allclose(env.scene["robot"].position_target, torch.tensor([[1.2, 2.0], [7.6, 8.0]]))
    assert env.scene["robot"].position_target_joint_ids == slice(None)


def test_dwl_joint_position_action_reset_restores_default_joint_targets():
    env = _mock_env()
    cfg = DwlJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        max_delay_steps=2,
    )
    action = DwlJointPositionAction(cfg, env)
    action.process_actions(torch.ones(2, 2))

    action.reset([0])

    assert torch.allclose(action.processed_actions[0], env.scene["robot"].data.default_joint_pos.torch[0])

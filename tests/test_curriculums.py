# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from curriculums import dwl_terrain_levels


class MockTerrain:
    def __init__(self):
        self.cfg = SimpleNamespace(terrain_generator=SimpleNamespace(size=(4.0, 4.0)))
        self.terrain_levels = torch.tensor([1, 3], dtype=torch.long)
        self.terrain_types = torch.tensor([0, 0], dtype=torch.long)
        self.terrain_origins = torch.zeros(5, 1, 3)
        self.terrain_origins[:, 0, 0] = torch.arange(5, dtype=torch.float32)

    def update_env_origins(self, env_ids, move_up, move_down):
        self.terrain_levels[env_ids] += move_up.to(dtype=torch.long)
        self.terrain_levels[env_ids] -= move_down.to(dtype=torch.long)
        self.terrain_levels.clamp_(0, 4)


class MockCommandManager:
    def get_command(self, command_name):
        return torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


class MockScene(dict):
    pass


def _mock_env():
    terrain = MockTerrain()
    scene = MockScene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.tensor([[3.5, 0.0, 0.0], [3.8, 0.0, 0.0]])))
    )
    scene.terrain = terrain
    scene.env_origins = torch.zeros(2, 3)
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene=scene,
        command_manager=MockCommandManager(),
        episode_length_buf=torch.tensor([500, 500]),
        max_episode_length=1000,
        max_episode_length_s=20.0,
        common_step_counter=0,
    )


def test_dwl_terrain_curriculum_caps_early_levels_and_requires_progress():
    env = _mock_env()

    mean_level = dwl_terrain_levels(env, torch.tensor([0, 1]))

    assert mean_level.item() == 1.0
    assert torch.equal(env.scene.terrain.terrain_levels, torch.tensor([1, 1]))
    assert torch.equal(env.scene.env_origins[:, 0], torch.tensor([1.0, 1.0]))

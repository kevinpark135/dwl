# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from gait import (
    DwlGaitCfg,
    clock_input,
    foot_acceleration_reference,
    foot_height_reference,
    foot_velocity_reference,
    gait_phase,
    quintic_acceleration,
    quintic_height,
    quintic_velocity,
    stance_mask,
)


def test_gait_phase_wraps_with_offset():
    time_s = torch.tensor([0.0, 0.5, 1.0])

    assert torch.allclose(gait_phase(time_s, 1.0), torch.tensor([0.0, 0.5, 0.0]))
    assert torch.allclose(gait_phase(time_s, 1.0, phase_offset=0.25), torch.tensor([0.25, 0.75, 0.25]))


def test_clock_input_shape_and_start_value():
    clock = clock_input(torch.tensor([0.0, 0.25]), cycle_time_s=1.0)

    assert clock.shape == (2, 2)
    assert torch.allclose(clock[0], torch.tensor([0.0, 1.0]))


def test_stance_mask_alternates_feet():
    time_s = torch.tensor([0.0, 0.49, 0.5, 0.99, 1.0])

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    assert torch.equal(stance_mask(time_s, cycle_time_s=1.0), expected)


def test_quintic_reference_matches_table_iv_boundaries():
    time_s = torch.tensor([0.0, 0.5])

    assert torch.allclose(quintic_height(time_s), torch.tensor([0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(quintic_velocity(time_s), torch.tensor([0.1, 0.0]), atol=1.0e-6)
    assert torch.allclose(quintic_acceleration(torch.tensor([0.0])), torch.tensor([10.0]), atol=1.0e-6)


def test_foot_references_are_zero_for_stance_foot():
    cfg = DwlGaitCfg()
    time_s = torch.tensor([0.25, 0.75])

    heights = foot_height_reference(time_s, cfg)
    velocities = foot_velocity_reference(time_s, cfg)
    accelerations = foot_acceleration_reference(time_s, cfg)

    assert torch.allclose(heights[:, 0], torch.tensor([0.0, 0.1]), atol=1.0e-6)
    assert torch.allclose(heights[:, 1], torch.tensor([0.1, 0.0]), atol=1.0e-6)
    assert velocities.shape == (2, 2)
    assert accelerations.shape == (2, 2)

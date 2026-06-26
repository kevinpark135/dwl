# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait utilities for the DWL task.

The DWL paper uses a periodic gait prior in three places:

- policy/state observations receive a clock input, usually ``sin(phase)`` and
  ``cos(phase)``
- privileged state contains cycle time and a periodic stance mask
- rewards use the stance mask and a quintic swing-foot trajectory reference

This module owns those shared definitions so observations and rewards do not
silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


# Table IV in the DWL paper lists f(t) = sum_k a_k t^k with these coefficients.
DEFAULT_QUINTIC_COEFFS = (0.0, 0.1, 5.0, -18.8, 12.0, 9.6)


@dataclass(frozen=True)
class DwlGaitCfg:
    """Configuration for the periodic gait reference.

    The paper reports a 0.5 s swing trajectory. A full gait cycle is one swing
    phase plus one stance phase, so the default cycle time is 1.0 s.
    """

    cycle_time_s: float = 1.0
    swing_time_s: float = 0.5
    phase_offset: float = 0.0
    foot_height_coeffs: tuple[float, ...] = DEFAULT_QUINTIC_COEFFS


def gait_phase(time_s: torch.Tensor, cycle_time_s: float, phase_offset: float = 0.0) -> torch.Tensor:
    """Return normalized gait phase in ``[0, 1)`` for each environment."""

    return torch.remainder(time_s / cycle_time_s + phase_offset, 1.0)


def clock_input(time_s: torch.Tensor, cycle_time_s: float, phase_offset: float = 0.0) -> torch.Tensor:
    """Return the DWL clock observation as ``[sin(phase), cos(phase)]``."""

    phase = gait_phase(time_s, cycle_time_s, phase_offset)
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def stance_mask(time_s: torch.Tensor, cycle_time_s: float, phase_offset: float = 0.0) -> torch.Tensor:
    """Return expected stance mask ``[left, right]``.

    A value of 1 means the foot is expected to be in stance/contact. A value of
    0 means the foot is expected to be in swing. The paper only requires the
    feet to alternate; this default starts with left stance and right swing in
    the first half-cycle.
    """

    phase = gait_phase(time_s, cycle_time_s, phase_offset)
    left_stance = phase < 0.5
    right_stance = ~left_stance
    return torch.stack((left_stance, right_stance), dim=-1).to(dtype=time_s.dtype)


def swing_elapsed_time(time_s: torch.Tensor, cfg: DwlGaitCfg, leg: str) -> torch.Tensor:
    """Return elapsed time inside the current swing phase for one leg.

    Values are clamped to ``[0, cfg.swing_time_s]`` so callers can safely use the
    result for trajectory evaluation even during stance.
    """

    if leg not in ("left", "right"):
        raise ValueError(f"Unsupported leg: {leg!r}. Expected 'left' or 'right'.")

    phase = gait_phase(time_s, cfg.cycle_time_s, cfg.phase_offset)
    if leg == "left":
        swing_phase = (phase - 0.5) * cfg.cycle_time_s
    else:
        swing_phase = phase * cfg.cycle_time_s
    return swing_phase.clamp(min=0.0, max=cfg.swing_time_s)


def quintic_height(time_s: torch.Tensor, coeffs: tuple[float, ...] = DEFAULT_QUINTIC_COEFFS) -> torch.Tensor:
    """Evaluate the DWL quintic swing-foot height reference."""

    height = torch.zeros_like(time_s)
    power = torch.ones_like(time_s)
    for coeff in coeffs:
        height = height + coeff * power
        power = power * time_s
    return height


def quintic_velocity(time_s: torch.Tensor, coeffs: tuple[float, ...] = DEFAULT_QUINTIC_COEFFS) -> torch.Tensor:
    """Evaluate the first derivative of the quintic height reference."""

    velocity = torch.zeros_like(time_s)
    power = torch.ones_like(time_s)
    for order, coeff in enumerate(coeffs[1:], start=1):
        velocity = velocity + order * coeff * power
        power = power * time_s
    return velocity


def quintic_acceleration(time_s: torch.Tensor, coeffs: tuple[float, ...] = DEFAULT_QUINTIC_COEFFS) -> torch.Tensor:
    """Evaluate the second derivative of the quintic height reference."""

    acceleration = torch.zeros_like(time_s)
    power = torch.ones_like(time_s)
    for order, coeff in enumerate(coeffs[2:], start=2):
        acceleration = acceleration + order * (order - 1) * coeff * power
        power = power * time_s
    return acceleration


def foot_height_reference(time_s: torch.Tensor, cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return swing-foot height references as ``[left, right]``.

    Stance feet receive zero height reference. Swing feet follow the quintic
    trajectory from Appendix Table IV.
    """

    mask = stance_mask(time_s, cfg.cycle_time_s, cfg.phase_offset)
    left_t = swing_elapsed_time(time_s, cfg, "left")
    right_t = swing_elapsed_time(time_s, cfg, "right")
    heights = torch.stack(
        (
            quintic_height(left_t, cfg.foot_height_coeffs),
            quintic_height(right_t, cfg.foot_height_coeffs),
        ),
        dim=-1,
    )
    return heights * (1.0 - mask)


def foot_velocity_reference(time_s: torch.Tensor, cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return vertical swing-foot velocity references as ``[left, right]``."""

    mask = stance_mask(time_s, cfg.cycle_time_s, cfg.phase_offset)
    left_t = swing_elapsed_time(time_s, cfg, "left")
    right_t = swing_elapsed_time(time_s, cfg, "right")
    velocities = torch.stack(
        (
            quintic_velocity(left_t, cfg.foot_height_coeffs),
            quintic_velocity(right_t, cfg.foot_height_coeffs),
        ),
        dim=-1,
    )
    return velocities * (1.0 - mask)


def foot_acceleration_reference(time_s: torch.Tensor, cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return vertical swing-foot acceleration references as ``[left, right]``."""

    mask = stance_mask(time_s, cfg.cycle_time_s, cfg.phase_offset)
    left_t = swing_elapsed_time(time_s, cfg, "left")
    right_t = swing_elapsed_time(time_s, cfg, "right")
    accelerations = torch.stack(
        (
            quintic_acceleration(left_t, cfg.foot_height_coeffs),
            quintic_acceleration(right_t, cfg.foot_height_coeffs),
        ),
        dim=-1,
    )
    return accelerations * (1.0 - mask)

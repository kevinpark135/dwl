# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DWL action terms.

The paper randomizes motor target offsets and system delay.  Isaac Lab's stock
joint-position action term already handles the normal scale/default-pose offset,
so this module adds only the DWL-specific pieces around that existing behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DWL_MOTOR_OFFSET_ATTR = "dwl_motor_offset"
DWL_SYSTEM_DELAY_ATTR = "dwl_system_delay_s"


def delay_steps_from_env(
    env: "ManagerBasedEnv",
    max_delay_steps: int,
    delay_attr_name: str = DWL_SYSTEM_DELAY_ATTR,
) -> torch.Tensor:
    """Convert per-env delay seconds into bounded integer control-step delays."""

    device = torch.device(env.device)
    delay_s = getattr(env, delay_attr_name, None)
    if delay_s is None:
        return torch.zeros(env.num_envs, device=device, dtype=torch.long)

    delay_s = torch.as_tensor(delay_s, device=device, dtype=torch.float32).reshape(env.num_envs, -1)[:, 0]
    step_dt = float(getattr(env, "step_dt", 0.0))
    if step_dt <= 0.0:
        return torch.zeros(env.num_envs, device=device, dtype=torch.long)
    return torch.ceil(delay_s / step_dt).to(dtype=torch.long).clamp_(0, max_delay_steps)


@configclass
class DwlJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with DWL motor offset and system-delay consumption."""

    class_type: str = "isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.action_terms:DwlJointPositionAction"

    motor_offset_attr_name: str = DWL_MOTOR_OFFSET_ATTR
    """Name of the env buffer containing per-env/per-joint motor target offsets."""

    delay_attr_name: str = DWL_SYSTEM_DELAY_ATTR
    """Name of the env buffer containing per-env system delay in seconds."""

    max_delay_steps: int = 4
    """Maximum number of control steps represented by the action delay buffer."""

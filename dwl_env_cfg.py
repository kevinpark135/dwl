# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the DWL G1 locomotion task.

This file currently adapts Isaac Lab's rough locomotion base configuration for
G1. As DWL is implemented, it will be wired to the local observation, reward,
event, and gait modules.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

try:
    from . import events as dwl_events
    from . import observations as dwl_obs
    from . import rewards as dwl_rewards
    from .gait import DwlGaitCfg
except ImportError:
    import events as dwl_events
    import observations as dwl_obs
    import rewards as dwl_rewards
    from gait import DwlGaitCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


@configclass
class G1Observations:
    """DWL observation groups for policy and privileged/state inputs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Onboard observations used by actor/encoder."""

        clock = ObsTerm(func=dwl_obs.policy_clock, params={"gait_cfg": DwlGaitCfg()})
        velocity_commands = ObsTerm(func=dwl_obs.policy_velocity_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=dwl_obs.policy_joint_pos,
            params={"asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        joint_vel = ObsTerm(
            func=dwl_obs.policy_joint_vel,
            params={"asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        base_ang_vel = ObsTerm(
            func=dwl_obs.policy_base_ang_vel,
            params={"asset_cfg": dwl_obs.DEFAULT_ROBOT_CFG},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_orientation = ObsTerm(
            func=dwl_obs.policy_base_orientation,
            params={"asset_cfg": dwl_obs.DEFAULT_ROBOT_CFG},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        last_action = ObsTerm(func=dwl_obs.policy_last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Simulation-only state used by critic and decoder targets."""

        base_lin_vel = ObsTerm(func=dwl_obs.state_base_lin_vel, params={"asset_cfg": dwl_obs.DEFAULT_ROBOT_CFG})
        friction = ObsTerm(func=dwl_obs.state_friction)
        push_force_torques = ObsTerm(func=dwl_obs.state_push_force_torques)
        cycle_time = ObsTerm(func=dwl_obs.state_cycle_time, params={"gait_cfg": DwlGaitCfg()})
        stance_mask = ObsTerm(func=dwl_obs.state_stance_mask, params={"gait_cfg": DwlGaitCfg()})
        feet_movement = ObsTerm(func=dwl_obs.state_feet_movement, params={"asset_cfg": dwl_obs.DEFAULT_FOOT_BODY_CFG})
        feet_contact = ObsTerm(
            func=dwl_obs.state_feet_contact,
            params={"sensor_cfg": dwl_obs.DEFAULT_CONTACT_SENSOR_CFG},
        )
        body_mass = ObsTerm(func=dwl_obs.state_body_mass, params={"asset_cfg": dwl_obs.DEFAULT_ROBOT_CFG})
        current_reward = ObsTerm(func=dwl_obs.state_current_reward)
        joint_torques = ObsTerm(
            func=dwl_obs.state_joint_torques,
            params={"asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG},
        )
        height_scan = ObsTerm(
            func=dwl_obs.state_height_scan,
            params={"sensor_cfg": dwl_obs.DEFAULT_HEIGHT_SCAN_CFG},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_velocity_tracking = RewTerm(
        func=dwl_rewards.lin_velocity_tracking,
        weight=1.0,
        params={"command_name": "base_velocity", "tolerance": 5.0},
    )
    ang_velocity_tracking = RewTerm(
        func=dwl_rewards.ang_velocity_tracking,
        weight=1.0,
        params={"command_name": "base_velocity", "tolerance": 7.0},
    )
    orientation_tracking = RewTerm(func=dwl_rewards.orientation_tracking, weight=1.0, params={"tolerance": 5.0})
    base_height_tracking = RewTerm(
        func=dwl_rewards.base_height_tracking,
        weight=0.5,
        params={"target_height": 0.7, "tolerance": 10.0},
    )
    periodic_force = RewTerm(
        func=dwl_rewards.periodic_force,
        weight=1.0,
        params={"gait_cfg": DwlGaitCfg(), "sensor_cfg": dwl_obs.DEFAULT_CONTACT_SENSOR_CFG},
    )
    periodic_velocity = RewTerm(
        func=dwl_rewards.periodic_velocity,
        weight=1.0,
        params={"gait_cfg": DwlGaitCfg(), "asset_cfg": dwl_obs.DEFAULT_FOOT_BODY_CFG},
    )
    foot_height_tracking = RewTerm(
        func=dwl_rewards.foot_height_tracking,
        weight=1.0,
        params={"gait_cfg": DwlGaitCfg(), "asset_cfg": dwl_obs.DEFAULT_FOOT_BODY_CFG, "tolerance": 5.0},
    )
    foot_velocity_tracking = RewTerm(
        func=dwl_rewards.foot_velocity_tracking,
        weight=0.5,
        params={"gait_cfg": DwlGaitCfg(), "asset_cfg": dwl_obs.DEFAULT_FOOT_BODY_CFG, "tolerance": 3.0},
    )
    default_joint_tracking = RewTerm(
        func=dwl_rewards.default_joint_tracking,
        weight=0.2,
        params={"asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG, "tolerance": 2.0},
    )
    energy_cost = RewTerm(
        func=dwl_rewards.energy_cost,
        weight=-0.0001,
        params={"asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG},
    )
    action_smoothness = RewTerm(func=dwl_rewards.action_smoothness, weight=-0.01)
    feet_movement = RewTerm(
        func=dwl_rewards.feet_movement,
        weight=-0.01,
        params={"asset_cfg": dwl_obs.DEFAULT_FOOT_BODY_CFG},
    )
    large_contact = RewTerm(
        func=dwl_rewards.large_contact,
        weight=-0.01,
        params={"sensor_cfg": dwl_obs.DEFAULT_CONTACT_SENSOR_CFG},
    )


@configclass
class G1DwlEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: G1Observations = G1Observations()
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # biped yaw control is harder than quadruped — relax the per-episode-mean yaw
        # threshold to 0.8 rad/s (defaults work for quadrupeds).
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.actions.joint_pos.joint_names = dwl_obs.CONTROLLED_LEG_JOINT_NAMES

        # DWL events and privileged buffers
        self.events.init_dwl_buffers = EventTerm(func=dwl_events.init_dwl_event_buffers, mode="startup")
        self.events.store_friction = EventTerm(
            func=dwl_events.store_friction,
            mode="startup",
            params={"friction_range": (0.2, 2.0)},
        )
        self.events.physics_material.params["static_friction_range"] = (0.2, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 2.0)
        self.events.add_base_mass = EventTerm(
            func=dwl_events.randomize_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-5.0, 20.0),
            },
        )
        self.events.base_com = None
        self.events.clear_push_force_torques = EventTerm(func=dwl_events.clear_push_force_torques, mode="reset")
        self.events.base_external_force_torque = EventTerm(
            func=dwl_events.sample_push_force_torques,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (0.0, 0.0),
                "torque_range": (0.0, 0.0),
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=dwl_events.randomize_joint_reset_noise,
            mode="reset",
            params={
                "asset_cfg": dwl_obs.DEFAULT_CONTROLLED_JOINT_CFG,
                "position_range": (-0.3, 0.3),
                "velocity_range": (-1.0, 1.0),
            },
        )
        self.events.push_force_torques = EventTerm(
            func=dwl_events.sample_push_force_torques,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "force_range": (-50.0, 50.0),
                "torque_range": (-10.0, 10.0),
            },
        )
        self.events.push_robot = None

        # Use DWL rewards instead of the stock rough-locomotion penalties.
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_pos_limits = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1DwlEnvCfg_PLAY(G1DwlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_force_torques = None

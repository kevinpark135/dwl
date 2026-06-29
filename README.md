# DWL Isaac Lab Task

DWL task and environment configuration built for Isaac Lab. Copy this repository into this directory:

~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config

## Registered Tasks

- Isaac-Velocity-DWL-G1-v0
- Isaac-Velocity-DWL-G1-Play-v0
- Isaac-Velocity-DwlBaseline-G1-v0
- Isaac-Velocity-DwlBaseline-G1-Play-v0

## Local Train/Play CLI

All commands below are one-line commands for this machine's Isaac Lab checkout at `/home/kevinpark135/IsaacLab`.

Smoke-test training, small env count and tiny iteration count:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 64 --max_iterations 5 --headless
```

Short debug training:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 128 --max_iterations 50 --headless
```

Medium training run:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 1024 --max_iterations 500 --headless
```

DWL full training run:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 4096 --max_iterations 3000 --headless
```

Proprioception-only stock G1 PPO baseline without DWL-specific model, rewards, observations, domain randomization, or external height-scan policy input:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DwlBaseline-G1-v0 --num_envs 4096 --max_iterations 3000 --headless
```

Full PhysX-style training:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 4096 --max_iterations 1000 --headless
```

Long Newton-style training budget:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 4096 --max_iterations 5000 --headless
```

Play latest checkpoint from `logs/rsl_rl/g1_dwl`:

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-Play-v0 --num_envs 32
```

Play latest checkpoint with fewer envs:

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-Play-v0 --num_envs 8
```

Play a specific checkpoint:

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-Play-v0 --num_envs 50 --checkpoint /home/kevinpark135/IsaacLab/logs/rsl_rl/g1_dwl/<RUN_DIR>/model_<ITER>.pt
```

## File Layout

```text
dwl/
├── __init__.py
├── dwl_env_cfg.py
├── baseline_env_cfg.py
├── actions.py
├── action_terms.py
├── observations.py
├── rewards.py
├── events.py
├── gait.py
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py
├── rsl_rl/
│   ├── __init__.py
│   ├── dwl_model.py
│   ├── dwl_ppo.py
│   └── dwl_runner.py
├── scripts/
├── tests/
├── README.md
└── .gitignore
```

## File Roles

Core task files:

- `__init__.py`: Registers the DWL and baseline train/play Gym environments with Isaac Lab.
- `dwl_env_cfg.py`: Defines the G1 DWL environment configuration by adapting the Isaac Lab rough locomotion base cfg.
- `baseline_env_cfg.py`: Defines the proprioception-only stock PPO baseline task.
- `actions.py`: Defines the lightweight DWL joint-position action config and delay helper used during cfg parsing.
- `action_terms.py`: Defines the runtime DWL joint-position action term that consumes motor offsets and system delay.
- `observations.py`: Defines policy observations and privileged/state observations used by DWL.
- `rewards.py`: Defines paper-specific reward terms such as velocity tracking, periodic gait rewards, foot tracking, and regularization.
- `events.py`: Defines DWL domain randomization buffers and event helpers for friction, pushes, mass, reset noise, motor offsets, motor strength, PD factors, observation noise bookkeeping, and system delay sampling.
- `gait.py`: Defines gait phase, stance masks, clock inputs, and quintic foot trajectory references.

Training configuration:

- `agents/__init__.py`: Marks the agent configuration package.
- `agents/rsl_rl_ppo_cfg.py`: Holds the RSL-RL training configuration for DWL and the proprioception-only baseline. DWL uses Isaac Lab's stock `OnPolicyRunner` entrypoint with fully-qualified custom actor, critic, and PPO class names.
- `scripts/`: Holds baseline training pipeline

RSL-RL extension files:

- `rsl_rl/__init__.py`: Marks the local RSL-RL extension package for DWL.
- `rsl_rl/dwl_model.py`: Defines the DWL actor/critic modules with GRU history encoding, latent decoding, actor head, and privileged critic.
- `rsl_rl/dwl_ppo.py`: Extends PPO with DWL decoder reconstruction and latent L1 losses.
- `rsl_rl/dwl_runner.py`: Optional helper module for preparing DWL observation groups and validating privileged decoder targets. The default Isaac Lab train CLI now runs through `OnPolicyRunner`.

Other:

- `tests/`: Code tests.
- `.gitignore`: Keeps local caches, logs, checkpoints, `DWL.pdf`, and local error-log captures out of git.

## `gait.py`

`gait.py` contains the shared periodic gait utilities used by DWL observations and rewards. It translates the gait-related parts of the paper into pure torch helpers:

- `DwlGaitCfg`: Stores gait timing, phase offset, and the Appendix Table IV quintic coefficients.
- `gait_phase`: Converts simulation time into normalized cycle phase in `[0, 1)`.
- `clock_input`: Produces the policy/state clock input `[sin(phase), cos(phase)]`.
- `stance_mask`: Produces the periodic stance mask `[left, right]`, where `1` means stance/contact and `0` means swing.
- `swing_elapsed_time`: Computes each foot's elapsed time inside its current swing phase.
- `quintic_height`, `quintic_velocity`, `quintic_acceleration`: Evaluate the Appendix Table IV swing-foot trajectory and derivatives.
- `foot_height_reference`, `foot_velocity_reference`, `foot_acceleration_reference`: Return left/right vertical swing-foot references while zeroing the stance foot.

The file is intentionally independent from Isaac Lab manager APIs. The expected connections are:

- `observations.py` will call `clock_input`, `gait_phase`, and `stance_mask` to expose clock, cycle time, and periodic stance mask terms.
- `rewards.py` will call `stance_mask` for periodic force/velocity rewards and the foot reference helpers for foot height/velocity tracking.
- `dwl_env_cfg.py` will configure the observation and reward terms that use these helpers.
- `rsl_rl/dwl_model.py` will indirectly receive these signals through the policy and privileged/state observation groups.
- `tests/test_gait.py` keeps the gait conventions stable while observations and rewards are implemented.

## `actions.py` and `action_terms.py`

The DWL action path is split into a lightweight config module and a runtime
action-term module. This avoids importing Isaac Sim/USD runtime classes while
Isaac Lab is only parsing task configuration.

- `DwlJointPositionActionCfg`: Configures the DWL action term while preserving
  the stock scale/default-joint-offset behavior. Its `class_type` points to
  `isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.action_terms:DwlJointPositionAction`
  as a lazy fully-qualified reference.
- `delay_steps_from_env`: Converts sampled delay seconds to integer control
  steps using `ceil(delay_s / env.step_dt)`.
- `DwlJointPositionAction`: Runtime action term in `action_terms.py`. It stores
  raw action history, consumes
  `env.dwl_system_delay_s` as a per-env bounded control-step delay, and adds
  `env.dwl_motor_offset` to processed joint position targets.

The action path applies delay before action scaling/default-pose offset, then
adds the sampled motor target offset before sending joint position targets to
the articulation.

Keeping `DwlJointPositionAction` out of `actions.py` is intentional: importing
the runtime class too early can load `pxr`/USD modules before `SimulationApp`
starts, which may crash Isaac Sim during startup.

## `observations.py`

`observations.py` defines the DWL policy and privileged/state observation terms. The policy group is restricted to signals that should be available on the real robot, while the privileged/state group contains simulation-only information for the critic and decoder target.

Policy terms:

- `policy_clock`: Gait clock `[sin(phase), cos(phase)]`.
- `policy_velocity_commands`: Commanded base velocity.
- `policy_joint_pos`: Controlled leg joint positions relative to default.
- `policy_joint_vel`: Controlled leg joint velocities relative to default.
- `policy_base_ang_vel`: Base angular velocity.
- `policy_base_orientation`: Paper-aligned base orientation as RPY.
- `policy_last_action`: Previous action.

The policy terms wired by `dwl_env_cfg.py` use delayed variants of these
functions. Each delayed policy term keeps a per-term history buffer, consumes
`env.dwl_system_delay_s`, and returns the per-env delayed signal. Privileged
state terms remain immediate simulator state for the critic/decoder target.

Privileged/state terms:

- `state_base_lin_vel`: Base linear velocity.
- `state_friction`: Per-env friction scalar, currently read from `env.dwl_friction` with a neutral fallback.
- `state_push_force_torques`: External push force/torque vector, currently read from `env.dwl_push_force_torques` with a zero fallback.
- `state_cycle_time`: Gait cycle time.
- `state_stance_mask`: Periodic stance mask.
- `state_feet_movement`: Foot positions and linear velocities.
- `state_feet_contact`: Binary foot contact mask from contact forces.
- `state_body_mass`: Total selected body mass.
- `state_current_reward`: Current reward buffer.
- `state_joint_torques`: Applied torques for controlled joints.
- `state_height_scan`: Privileged terrain height scan.

Helpers and conventions:

- `quat_to_rpy`: Converts Isaac Lab XYZW root quaternions into `[roll, pitch, yaw]`.
- `base_orientation_rpy`: Paper-aligned orientation term matching Table I's Euler orientation component.
- `base_orientation_projected_gravity`: Isaac Lab locomotion-style orientation proxy for experiments that prefer projected gravity.
- `CONTROLLED_LEG_JOINT_NAMES`: The 12 leg joint regex patterns used for DWL control.
- `FOOT_BODY_NAMES`: The foot body regex patterns used by foot movement/contact terms.
- `make_policy_observation_terms`: Builds the policy observation term map for `dwl_env_cfg.py`.
- `make_privileged_observation_terms`: Builds the privileged/state observation term map for `dwl_env_cfg.py`.

The DWL default direction is to keep `base_lin_vel` and `height_scan` out of the policy observation group and place them in the privileged/state group. Orientation is explicit because the paper uses Euler angles, while many Isaac Lab locomotion baselines use projected gravity.

## `rewards.py`

`rewards.py` implements the DWL paper reward table as Isaac Lab reward terms:

- `lin_velocity_tracking`: Tracks commanded base linear velocity with zero vertical command.
- `lin_velocity_tracking_yaw_frame`: Compatibility wrapper for Isaac Lab's yaw-frame XY velocity tracking.
- `ang_velocity_tracking`: Tracks commanded yaw rate while keeping roll/pitch rates near zero.
- `orientation_tracking`: Tracks upright orientation.
- `base_height_tracking`: Tracks the target base height.
- `periodic_force`: Rewards contact force on the foot currently expected to be in stance.
- `periodic_velocity`: Rewards movement of the foot currently expected to be in swing.
- `foot_height_tracking`: Tracks the quintic swing-foot height reference.
- `foot_velocity_tracking`: Tracks the quintic swing-foot vertical velocity reference.
- `default_joint_tracking`: Rewards staying near the default controlled-joint posture.
- `energy_cost`: Computes `sum(|tau| * |qdot|)`.
- `action_smoothness`: Computes the second-order action difference.
- `feet_movement`: Penalizes vertical foot velocity and acceleration (`z` axis only) using Isaac Lab 3.0's `body_lin_acc_w` acceleration field.
- `large_contact`: Penalizes excessive foot contact force.

The gait helpers matter here because four rewards are phase-aware: `periodic_force`, `periodic_velocity`, `foot_height_tracking`, and `foot_velocity_tracking`. They must use the same clock, stance mask, and foot trajectory reference as the observations; otherwise the policy could observe one gait phase while rewards score another.

## `events.py`

`events.py` maps the DWL paper domain-randomization table into named Isaac Lab event hooks and privileged buffers.

Event helpers:

- `init_dwl_event_buffers`: Creates buffers consumed by privileged observations.
- `clear_push_force_torques`: Clears the stored external wrench buffer.
- `sample_push_force_torques`: Samples external force/torque, applies it to the robot, and stores a 6D privileged value.
- `store_friction`: Samples/stores per-env friction values for `state_friction`.
- `randomize_body_mass`: Adds bounded payload mass offsets to selected bodies.
- `randomize_joint_reset_noise`: Applies additive joint position/velocity reset noise.
- `randomize_joint_position_observation_noise`: Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`.
- `randomize_joint_velocity_observation_noise`: Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`.
- `randomize_angular_velocity_observation_noise`: Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`.
- `randomize_orientation_observation_noise`: Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`.
- `randomize_system_delay`: Samples/stores `env.dwl_system_delay_s` for action and policy-observation delay buffers.
- `randomize_motor_offset`: Samples/stores `env.dwl_motor_offset` for action target processing.
- `randomize_motor_strength`: Scales actuator effort limits and records `env.dwl_motor_strength`.
- `randomize_pd_factors`: Scales actuator stiffness/damping and records `env.dwl_pd_factors`.

The event helpers use Isaac Lab 3.0-compatible signatures with `env_ids` as the
second argument. PhysX/Warp index tensors passed back into simulation APIs are
converted to `int32`.

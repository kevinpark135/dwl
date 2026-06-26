# DWL Isaac Lab Task

DWL task and environment configuration built for Isaac Lab. Copy this repository into this directory:

~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config

## Registered Tasks

- Isaac-Velocity-DWL-G1-v0
- Isaac-Velocity-DWL-G1-Play-v0

## File Layout

```text
dwl/
├── __init__.py
├── dwl_env_cfg.py
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
├── tests/
│   └── test_gait.py
├── README.md
└── .gitignore
```

## File Roles

- `__init__.py`: Registers the DWL train/play Gym environments with Isaac Lab.
- `dwl_env_cfg.py`: Defines the G1 DWL environment configuration by adapting the Isaac Lab rough locomotion base cfg.
- `observations.py`: Placeholder for policy observations and privileged/state observations used by DWL.
- `rewards.py`: Placeholder for paper-specific reward terms such as velocity tracking, periodic gait rewards, foot tracking, and regularization.
- `events.py`: Placeholder for DWL domain randomization and perturbation events.
- `gait.py`: Placeholder for gait phase, stance masks, clock inputs, and quintic foot trajectory references.
- `agents/__init__.py`: Marks the agent configuration package.
- `agents/rsl_rl_ppo_cfg.py`: Holds the RSL-RL training configuration and will later point to the DWL custom model/algorithm/runner.
- `rsl_rl/__init__.py`: Marks the local RSL-RL extension package for DWL.
- `rsl_rl/dwl_model.py`: Placeholder for the GRU encoder, latent decoder, actor, and critic modules.
- `rsl_rl/dwl_ppo.py`: Placeholder for PPO with DWL denoising and latent regularization losses.
- `rsl_rl/dwl_runner.py`: Placeholder for the runner that wires Isaac Lab observation groups into the DWL training loop.
- `tests/test_gait.py`: Regression tests for gait phase wrapping, clock inputs, stance masks, and quintic foot references.
- `tests/test_observations.py`: Regression tests for DWL observation helper conventions such as quaternion-to-RPY orientation.
- `.gitignore`: Keeps local caches, logs, checkpoints, and `DWL.pdf` out of git.

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

## Implementation Order

1. Implement `gait.py` clock signals, stance masks, and quintic foot trajectory helpers. Basic utilities and tests are now in place.
2. Implement `observations.py` with separate policy and privileged/state observation terms.
3. Implement `rewards.py` using the DWL paper reward table and the gait helpers.
4. Implement `events.py` domain randomization for noise, friction, mass/payload, motor, PD, push, and delay effects.
5. Wire the new observation, reward, and event terms into `dwl_env_cfg.py`.
6. Implement `rsl_rl/dwl_model.py` with the encoder, decoder, actor, and critic.
7. Implement `rsl_rl/dwl_ppo.py` by adding reconstruction and latent L1 losses to PPO.
8. Implement `rsl_rl/dwl_runner.py` to pass privileged reconstruction targets through rollout and training.
9. Update `agents/rsl_rl_ppo_cfg.py` with DWL architecture paths and paper-aligned hyperparameters.
10. Train and compare the DWL policy against the current PPO baseline.

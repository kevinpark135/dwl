# DWL Isaac Lab Task

DWL task and environment configuration built for Isaac Lab. Copy this repository into this directory:

~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config

## Registered Tasks

| Task | Use |
| --- | --- |
| `Isaac-Velocity-DWL-G1-v0` | Main DWL training task. |
| `Isaac-Velocity-DWL-G1-Play-v0` | Main DWL play/evaluation task. |
| `Isaac-Velocity-DWL-StockProprio-G1-v0` | True stock PPO proprioception-only baseline with no base linear velocity and no height scan. |
| `Isaac-Velocity-DWL-StockProprio-G1-Play-v0` | Play/evaluation task for the true proprioception-only baseline. |
| `Isaac-Velocity-DWL-PrivilegedActor-G1-v0` | Oracle baseline where the actor receives privileged state. |
| `Isaac-Velocity-DWL-PrivilegedActor-G1-Play-v0` | Play/evaluation task for the privileged actor baseline. |

## System Environment

| Item | Value |
| --- | --- |
| OS/kernel | Ubuntu 24.04, Linux `6.17.0-35-generic` |
| CPU | Intel Xeon Silver 4114 @ 2.20 GHz, 20 CPUs, 20 cores, 1 socket |
| RAM | 62 GiB |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11264 MiB VRAM |
| NVIDIA driver / CUDA | Driver `595.71.05`, CUDA `13.2` |
| Isaac Lab checkout | `/home/kevinpark135/IsaacLab`, branch `release/3.0.0-beta2`, commit `d8ec040d8c` |
| Isaac Lab version | `3.0.0` from `IsaacLab/VERSION`; Python package `isaaclab==6.1.6` |
| Isaac Sim version | Python package `isaacsim==6.0.0.1` |
| Task package path | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/dwl` |
| RL library | `rsl_rl` through Isaac Lab's train/play CLI |

## Local Train/Play CLI

DWL training run:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-v0 --num_envs 4096 --max_iterations 5000 --headless
```

Play a specific checkpoint:

```bash
./isaaclab.sh play --rl_library rsl_rl --task Isaac-Velocity-DWL-G1-Play-v0 --num_envs 16 --checkpoint /home/kevinpark135/IsaacLab/logs/rsl_rl/g1_dwl/<RUN_DIR>/model_<ITER>.pt --viz kit
```

## Baseline Cases

True stock G1 PPO proprioception-only:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-StockProprio-G1-v0 --num_envs 4096 --max_iterations 1000 --headless
```

Privileged actor oracle:

```bash
./isaaclab.sh train --rl_library rsl_rl --task Isaac-Velocity-DWL-PrivilegedActor-G1-v0 --num_envs 4096 --max_iterations 1000 --headless
```

Run both baselines sequentially:

```bash
NUM_ENVS=4096 MAX_ITERATIONS=1000 ./scripts/train_baselines.sh
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

| File | Role |
| --- | --- |
| `__init__.py` | Registers the DWL and baseline train/play Gym environments with Isaac Lab. |
| `dwl_env_cfg.py` | Defines the G1 DWL environment configuration by adapting the Isaac Lab rough locomotion base cfg. |
| `baseline_env_cfg.py` | Defines executable baseline and ablation environment configs. |
| `actions.py` | Defines the lightweight DWL joint-position action config and delay helper used during cfg parsing. |
| `action_terms.py` | Defines the runtime DWL joint-position action term that consumes motor offsets and system delay. |
| `observations.py` | Defines policy observations and privileged/state observations used by DWL. |
| `rewards.py` | Defines paper-specific reward terms such as velocity tracking, periodic gait rewards, foot tracking, and regularization. |
| `events.py` | Defines DWL domain randomization buffers and event helpers for friction, pushes, mass, reset noise, motor offsets, motor strength, PD factors, observation noise bookkeeping, and system delay sampling. |
| `gait.py` | Defines gait phase, stance masks, clock inputs, and quintic foot trajectory references. |

Training configuration:

| Path | Role |
| --- | --- |
| `agents/__init__.py` | Marks the agent configuration package. |
| `agents/rsl_rl_ppo_cfg.py` | Holds the RSL-RL training configuration for DWL and the baseline cases. DWL uses Isaac Lab's `OnPolicyRunner` with fully-qualified custom `DwlActorModel`, `DwlCriticModel`, and `DwlPPO` class names. |
| `scripts/` | Holds baseline training helpers. |

RSL-RL extension files:

| File | Role |
| --- | --- |
| `rsl_rl/__init__.py` | Marks the local RSL-RL extension package for DWL. |
| `rsl_rl/dwl_model.py` | Defines the DWL actor/critic modules with GRU history encoding, latent decoding, actor head, and privileged critic. |
| `rsl_rl/dwl_ppo.py` | Extends PPO with DWL decoder reconstruction and latent L1 losses. |
| `rsl_rl/dwl_runner.py` | Helper runner for preparing DWL observation groups and validating privileged decoder targets. The default Isaac Lab train CLI currently runs through `OnPolicyRunner`. |

Other:

| Path | Role |
| --- | --- |
| `tests/` | Code tests. |
| `.gitignore` | Keeps local caches, logs, checkpoints, `DWL.pdf`, and local error-log captures out of git. |

## `gait.py`

`gait.py` contains the shared periodic gait utilities used by DWL observations and rewards. It translates the gait-related parts of the paper into pure torch helpers:

| Helper | Role |
| --- | --- |
| `DwlGaitCfg` | Stores gait timing, phase offset, and the Appendix Table IV quintic coefficients. |
| `gait_phase` | Converts simulation time into normalized cycle phase in `[0, 1)`. |
| `clock_input` | Produces the policy/state clock input `[sin(phase), cos(phase)]`. |
| `stance_mask` | Produces the periodic stance mask `[left, right]`, where `1` means stance/contact and `0` means swing. |
| `swing_elapsed_time` | Computes each foot's elapsed time inside its current swing phase. |
| `quintic_height`, `quintic_velocity`, `quintic_acceleration` | Evaluate the Appendix Table IV swing-foot trajectory and derivatives. |
| `foot_height_reference`, `foot_velocity_reference`, `foot_acceleration_reference` | Return left/right vertical swing-foot references while zeroing the stance foot. |

The file is intentionally independent from Isaac Lab manager APIs. The expected connections are:

| Consumer | Connection |
| --- | --- |
| `observations.py` | Calls `clock_input`, `gait_phase`, and `stance_mask` to expose clock, cycle time, and periodic stance mask terms. |
| `rewards.py` | Calls `stance_mask` for periodic force/velocity rewards and the foot reference helpers for foot height/velocity tracking. |
| `dwl_env_cfg.py` | Configures the observation and reward terms that use these helpers. |
| `rsl_rl/dwl_model.py` | Indirectly receives these signals through the policy and privileged/state observation groups. |
| `tests/test_gait.py` | Keeps the gait conventions stable while observations and rewards are implemented. |

## `actions.py` and `action_terms.py`

The DWL action path is split into a lightweight config module and a runtime
action-term module. This avoids importing Isaac Sim/USD runtime classes while
Isaac Lab is only parsing task configuration.

| Component | Role |
| --- | --- |
| `DwlJointPositionActionCfg` | Configures the DWL action term while preserving the stock scale/default-joint-offset behavior. Its `class_type` points to `isaaclab_tasks.manager_based.locomotion.velocity.config.dwl.action_terms:DwlJointPositionAction` as a lazy fully-qualified reference. |
| `delay_steps_from_env` | Converts sampled delay seconds to integer control steps using `ceil(delay_s / env.step_dt)`. |
| `DwlJointPositionAction` | Runtime action term in `action_terms.py`. It stores raw action history, consumes `env.dwl_system_delay_s` as a per-env bounded control-step delay, and adds `env.dwl_motor_offset` to processed joint position targets. |

The action path applies delay before action scaling/default-pose offset, then
adds the sampled motor target offset before sending joint position targets to
the articulation.

Keeping `DwlJointPositionAction` out of `actions.py` is intentional: importing
the runtime class too early can load `pxr`/USD modules before `SimulationApp`
starts, which may crash Isaac Sim during startup.

## `observations.py`

`observations.py` defines the DWL policy and privileged/state observation terms. The policy group is restricted to signals that should be available on the real robot, while the privileged/state group contains simulation-only information for the critic and decoder target.

Policy terms:

| Term | Signal |
| --- | --- |
| `policy_clock` | Gait clock `[sin(phase), cos(phase)]`. |
| `policy_velocity_commands` | Commanded base velocity. |
| `policy_joint_pos` | Controlled leg joint positions relative to default. |
| `policy_joint_vel` | Controlled leg joint velocities relative to default. |
| `policy_base_ang_vel` | Base angular velocity. |
| `policy_base_orientation` | Paper-aligned base orientation as RPY. |
| `policy_last_action` | Previous action. |

The policy terms wired by `dwl_env_cfg.py` use delayed variants of these
functions. Each delayed policy term keeps a per-term history buffer, consumes
`env.dwl_system_delay_s`, and returns the per-env delayed signal. Privileged
state terms remain immediate simulator state for the critic/decoder target.

Privileged/state terms:

| Term | Signal |
| --- | --- |
| `state_base_lin_vel` | Base linear velocity. |
| `state_friction` | Per-env friction scalar, currently read from `env.dwl_friction` with a neutral fallback. |
| `state_push_force_torques` | External push force/torque vector, currently read from `env.dwl_push_force_torques` with a zero fallback. |
| `state_cycle_time` | Gait cycle time. |
| `state_stance_mask` | Periodic stance mask. |
| `state_feet_movement` | Foot positions and linear velocities. |
| `state_feet_contact` | Binary foot contact mask from contact forces. |
| `state_body_mass` | Total selected body mass. |
| `state_current_reward` | Current reward buffer. |
| `state_joint_torques` | Applied torques for controlled joints. |
| `state_height_scan` | Privileged terrain height scan. |

Helpers and conventions:

| Helper | Role |
| --- | --- |
| `quat_to_rpy` | Converts Isaac Lab XYZW root quaternions into `[roll, pitch, yaw]`. |
| `base_orientation_rpy` | Paper-aligned orientation term matching Table I's Euler orientation component. |
| `base_orientation_projected_gravity` | Isaac Lab locomotion-style orientation proxy for experiments that prefer projected gravity. |
| `CONTROLLED_LEG_JOINT_NAMES` | The 12 leg joint regex patterns used for DWL control. |
| `FOOT_BODY_NAMES` | The foot body regex patterns used by foot movement/contact terms. |
| `make_policy_observation_terms` | Builds the policy observation term map for `dwl_env_cfg.py`. |
| `make_privileged_observation_terms` | Builds the privileged/state observation term map for `dwl_env_cfg.py`. |

The DWL default direction is to keep `base_lin_vel` and `height_scan` out of the policy observation group and place them in the privileged/state group. Orientation is explicit because the paper uses Euler angles, while many Isaac Lab locomotion baselines use projected gravity.

## `rewards.py`

`rewards.py` implements the DWL paper reward table as Isaac Lab reward terms:

Tracking and task rewards:

| Reward | Type | Purpose |
| --- | --- | --- |
| `lin_velocity_tracking` | Command tracking | Tracks commanded base linear velocity with zero vertical command. |
| `ang_velocity_tracking` | Command tracking | Tracks commanded yaw rate while keeping roll/pitch rates near zero. |
| `low_forward_speed_penalty` | Anti-stall penalty | Penalizes commanded forward episodes that remain near-stationary after a short grace period. |
| `orientation_tracking` | Pose tracking | Tracks upright orientation. |
| `base_height_tracking` | Pose tracking | Tracks the target base height. |

Phase-aware gait rewards:

| Reward | Type | Purpose |
| --- | --- | --- |
| `periodic_force` | Stance reward | Rewards contact force on the foot currently expected to be in stance. |
| `periodic_velocity` | Swing reward | Rewards movement of the foot currently expected to be in swing. |
| `foot_height_tracking` | Swing trajectory | Tracks the quintic swing-foot height reference. |
| `foot_velocity_tracking` | Swing trajectory | Tracks the quintic swing-foot vertical velocity reference. |
| `foot_sagittal_tracking` | Swing/stance coordination | Rewards the expected swing foot moving forward relative to the base while the stance foot anchors backward, reducing one-foot pivot gaits. |

Foot-shape and gait-quality rewards:

| Reward | Type | Purpose |
| --- | --- | --- |
| `foot_lateral_tracking` | Foot placement | Rewards a narrow body-frame left/right foot corridor to discourage wide A-frame walking. |
| `foot_lateral_velocity` | Side-shuffle penalty | Penalizes side-shuffling foot velocity while leaving forward swing motion free. |
| `foot_sagittal_symmetry` | Symmetry reward | Rewards left/right feet staying balanced around the base in the sagittal axis. |
| `yaw_drift_penalty` | Direction regularizer | Penalizes unintended yaw drift most strongly while the staged yaw target is small. |
| `hip_deviation` | Posture regularizer | Penalizes excessive hip yaw/roll displacement and velocity, targeting bow-legged gait artifacts. |

Regularization and safety rewards:

| Reward | Type | Purpose |
| --- | --- | --- |
| `default_joint_tracking` | Joint regularizer | Rewards staying near the default controlled-joint posture. |
| `energy_cost` | Energy penalty | Computes `sum(|tau| * |qdot|)`. |
| `action_smoothness` | Action regularizer | Computes the second-order action difference. |
| `feet_movement` | Foot motion penalty | Penalizes vertical foot velocity and acceleration (`z` axis only) using Isaac Lab 3.0's `body_lin_acc_w` acceleration field. The DWL cfg keeps the paper weight at `-0.01` but scales vertical acceleration by `10.0` before squaring so SI-unit accelerations do not dominate early learning. |
| `large_contact` | Contact penalty | Penalizes excessive foot contact force. |
| `body_contact` | Failure-prevention penalty | Penalizes torso contact before it becomes a stable failure mode. |

The gait helpers matter here because five rewards are phase-aware: `periodic_force`, `periodic_velocity`, `foot_height_tracking`, `foot_velocity_tracking`, and `foot_sagittal_tracking`. They must use the same clock, stance mask, and foot trajectory reference as the observations; otherwise the policy could observe one gait phase while rewards score another.

## `events.py`

`events.py` maps the DWL paper domain-randomization table into named Isaac Lab event hooks and privileged buffers.

Event helpers:

| Helper | Role |
| --- | --- |
| `init_dwl_event_buffers` | Creates buffers consumed by privileged observations. |
| `clear_push_force_torques` | Clears the stored external wrench buffer. |
| `sample_push_force_torques` | Samples external force/torque, applies it to the robot, and stores a 6D privileged value. |
| `store_friction` | Samples/stores per-env friction values for `state_friction`. |
| `randomize_body_mass` | Adds bounded payload mass offsets to selected bodies. |
| `randomize_joint_reset_noise` | Applies additive joint position/velocity reset noise. |
| `randomize_joint_position_observation_noise` | Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`. |
| `randomize_joint_velocity_observation_noise` | Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`. |
| `randomize_angular_velocity_observation_noise` | Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`. |
| `randomize_orientation_observation_noise` | Records the paper noise range; actual noise is applied by `ObservationTermCfg.noise`. |
| `randomize_system_delay` | Samples/stores `env.dwl_system_delay_s` for action and policy-observation delay buffers. |
| `randomize_motor_offset` | Samples/stores `env.dwl_motor_offset` for action target processing. |
| `randomize_motor_strength` | Scales actuator effort limits and records `env.dwl_motor_strength`. |
| `randomize_pd_factors` | Scales actuator stiffness/damping and records `env.dwl_pd_factors`. |
| `store_foot_height_baseline` | Stores reset-time foot heights so foot trajectory rewards use terrain-relative swing clearance instead of raw world height. |

The event helpers use Isaac Lab 3.0-compatible signatures with `env_ids` as the
second argument. PhysX/Warp index tensors passed back into simulation APIs are
converted to `int32`.

The DWL train cfg currently starts at `max_init_terrain_level = 1` so distance
curriculum can promote the policy after it learns stable forward translation.

## Patch Notes

Major changes only. This table keeps the trial-and-error path visible without
listing every small constant change.

| Focus | Issue | Change |
| --- | --- | --- |
| Baseline cases | Needed a smaller and cleaner comparison set after DWL began walking reliably. | Kept two baseline families: true `StockProprio` PPO with no base linear velocity or height scan, and `PrivilegedActor` oracle with full privileged actor input. |
| Long-run stabilization prep | Rough-terrain learning became jumpy after reward-scale alignment produced visible walking. | Foot-height rewards now refresh a reset-time terrain-relative baseline; DWL uses a conservative terrain curriculum that promotes on sustained progress, demotes only clear short base-contact failures, and caps early terrain levels by global step. |
| Standing local optimum | Checkpoint play still showed standing or creeping in place on commanded forward tasks. | Sharpened forward tracking, increased capped forward-progress reward, and made the low-speed penalty scale against a command-relative speed floor. |
| Post-spike locomotion stabilization | A sharp `base_contact` spike near the first yaw-curriculum step led to near-stationary gait. | Temporarily disabled yaw commands, kept curriculum terrain init, gated swing air-time by real forward speed/uprightness, added command-gated low-speed penalty, added torso-contact penalty, and removed dead `double_support`/yaw-frame wrapper plumbing. |
| Human-like gait refinement (`8ae54cb`) | Visual checks showed wide stance, high cadence, one-foot pivoting, circular drift, and stiff raised arms. | Slowed the gait clock to `1.2s`, narrowed target foot width to `0.16m`, added contact/clearance-aware sagittal tracking and left/right symmetry, staged yaw at 500/750/1000/1500 iterations, softened linear velocity tolerance, lengthened swing air-time, and relaxed the G1 arm pose. |
| First gait-shape pass (`418896c`) | The first walking policy reached positive terrain curriculum but looked wide and A-framed. | Added lateral foot-corridor, side-shuffle, and hip yaw/roll regularization while keeping forward velocity dominant. |
| Straight-walk curriculum (`24f3878`, `6f8308a`) | Early turning made the biped exploit circular motion before stable forward walking. | Staged yaw learning and kept actor observations, privileged observations, and `ang_velocity_tracking` on the same yaw target. |
| Paper dimension alignment (`6dbc9eb`) | DWL model and PPO path needed to match the paper tables. | Aligned encoder input, privileged decoder/critic state, height scan, network widths, and PPO coefficients. |
| Exploration and local-optimum removal (`fce87ae`, `0809d43`, `6f8308a`) | Zero/low forward commands plus strong stability rewards produced passive crouch or upright no-translation policies. | Shifted reward balance toward forward velocity, capped forward progress, commanded swing air-time, and weaker default/smoothness/energy penalties. |
| Early stand-up debugging | Initial rollouts collapsed or contacted the torso too quickly. | Softened reset noise, friction/push randomization, actuator damping, joint limits, and action initialization; tried and backed out a pure stand-first setup because it encouraged standing instead of walking. |

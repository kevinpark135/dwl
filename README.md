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
- `.gitignore`: Keeps local caches, logs, checkpoints, and `DWL.pdf` out of git.

## Implementation Order

1. Implement `gait.py` clock signals, stance masks, and quintic foot trajectory helpers.
2. Implement `observations.py` with separate policy and privileged/state observation terms.
3. Implement `rewards.py` using the DWL paper reward table and the gait helpers.
4. Implement `events.py` domain randomization for noise, friction, mass/payload, motor, PD, push, and delay effects.
5. Wire the new observation, reward, and event terms into `dwl_env_cfg.py`.
6. Implement `rsl_rl/dwl_model.py` with the encoder, decoder, actor, and critic.
7. Implement `rsl_rl/dwl_ppo.py` by adding reconstruction and latent L1 losses to PPO.
8. Implement `rsl_rl/dwl_runner.py` to pass privileged reconstruction targets through rollout and training.
9. Update `agents/rsl_rl_ppo_cfg.py` with DWL architecture paths and paper-aligned hyperparameters.
10. Train and compare the DWL policy against the current PPO baseline.

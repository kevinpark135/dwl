# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DWL runner integration for RSL-RL.

RSL-RL stores the full observation TensorDict in rollout storage.  The DWL
runner makes that contract explicit for this task: actor observations come from
the onboard ``policy`` group, while the critic and decoder reconstruction target
come from the privileged ``privileged`` group.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner


DEFAULT_ACTOR_GROUP = "policy"
DEFAULT_RECONSTRUCTION_GROUP = "privileged"


def prepare_dwl_runner_cfg(cfg: dict[str, Any], obs: TensorDict | None = None) -> dict[str, Any]:
    """Fill DWL-specific runner defaults in a mutable RSL-RL config dict."""

    obs_groups = cfg.setdefault("obs_groups", {})
    if obs_groups is None:
        obs_groups = {}
        cfg["obs_groups"] = obs_groups

    if "actor" not in obs_groups:
        obs_groups["actor"] = [_first_available_group(obs, (DEFAULT_ACTOR_GROUP, "actor", "policy"))]
    if "critic" not in obs_groups:
        obs_groups["critic"] = [_first_available_group(obs, (DEFAULT_RECONSTRUCTION_GROUP, "critic", "policy"))]

    actor_cfg = cfg.setdefault("actor", {})
    critic_cfg = cfg.setdefault("critic", {})
    algorithm_cfg = cfg.setdefault("algorithm", {})

    if actor_cfg.get("class_name", "MLPModel") in ("MLPModel", "RNNModel"):
        actor_cfg["class_name"] = "DwlActorModel"
    if critic_cfg.get("class_name", "MLPModel") in ("MLPModel", "RNNModel"):
        critic_cfg["class_name"] = "DwlCriticModel"
    if algorithm_cfg.get("class_name", "PPO") == "PPO":
        algorithm_cfg["class_name"] = "DwlPPO"

    actor_cfg.setdefault("decoder_obs_set", "critic")
    algorithm_cfg.setdefault("reconstruction_loss_coef", 1.0)
    algorithm_cfg.setdefault("latent_l1_loss_coef", 0.0)
    return cfg


def validate_dwl_reconstruction_observations(cfg: dict[str, Any], obs: TensorDict) -> list[str]:
    """Return decoder target groups after validating that rollout obs contains them."""

    obs_groups = cfg.get("obs_groups", {})
    decoder_obs_set = cfg.get("actor", {}).get("decoder_obs_set", "critic")
    if decoder_obs_set in obs_groups:
        target_groups = list(obs_groups[decoder_obs_set])
    elif decoder_obs_set in obs:
        target_groups = [decoder_obs_set]
    else:
        raise ValueError(
            f"DWL decoder_obs_set '{decoder_obs_set}' is neither an obs set nor an observation group. "
            f"Available obs sets: {list(obs_groups.keys())}; available groups: {list(obs.keys())}."
        )

    missing = [group for group in target_groups if group not in obs]
    if missing:
        raise ValueError(
            f"DWL reconstruction target groups {missing} are missing from rollout observations. "
            f"Available groups: {list(obs.keys())}."
        )
    return target_groups


class DwlRunner(OnPolicyRunner):
    """On-policy runner that preserves privileged decoder targets for DWL."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        obs = env.get_observations()
        prepare_dwl_runner_cfg(train_cfg, obs)
        validate_dwl_reconstruction_observations(train_cfg, obs)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.decoder_target_groups = validate_dwl_reconstruction_observations(self.cfg, obs)
        self._validate_actor_decoder(obs.to(self.device))

    def _validate_actor_decoder(self, obs: TensorDict) -> None:
        actor = self.alg.get_policy()
        if not hasattr(actor, "reconstruction_target"):
            raise TypeError("DWL runner requires an actor with a reconstruction_target(obs) method.")
        actor.reconstruction_target(obs)


def _first_available_group(obs: TensorDict | None, candidates: tuple[str, ...]) -> str:
    if obs is None:
        return candidates[0]
    for candidate in candidates:
        if candidate in obs:
            return candidate
    return candidates[0]


def copy_and_prepare_dwl_runner_cfg(cfg: dict[str, Any], obs: TensorDict | None = None) -> dict[str, Any]:
    """Return a prepared copy of a runner config for tests or dry-run inspection."""

    return prepare_dwl_runner_cfg(deepcopy(cfg), obs)


DwlOnPolicyRunner = DwlRunner

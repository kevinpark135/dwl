# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DWL neural network modules for RSL-RL.

The current RSL-RL release used by Isaac Lab constructs actor and critic models
separately.  DWL still needs an encoder/decoder pair on the actor side, so this
module provides RSL-RL-compatible model classes:

- ``DwlActorModel``: policy observations -> GRU encoder -> latent -> actor head
  and latent -> decoder reconstruction.
- ``DwlCriticModel``: privileged/state observations -> value head.

The actor intentionally keeps ``is_recurrent = False``.  Its GRU encodes a
finite observation history window supplied in the observation tensor, which
keeps rollout storage feed-forward until ``dwl_runner.py`` owns recurrent
rollouts explicitly.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict


HiddenState = Any


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _as_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Iterable[int],
    activation: str = "elu",
    last_activation: str | None = None,
) -> nn.Sequential:
    dims = [input_dim, *_as_tuple(hidden_dims), output_dim]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(_activation(activation))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if last_activation is not None:
        layers.append(_activation(last_activation))
    return nn.Sequential(*layers)


def _init_mlp(module: nn.Module, output_gain: float = 1.0) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, gain=1.0)
        nn.init.zeros_(layer.bias)
    if linear_layers:
        nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
        nn.init.zeros_(linear_layers[-1].bias)


def _selected_obs_groups(obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    groups = obs_groups[obs_set]
    obs_dim = 0
    for group in groups:
        if group not in obs:
            raise KeyError(f"Observation group '{group}' from set '{obs_set}' is missing.")
        if len(obs[group].shape) != 2:
            raise ValueError(f"DWL models expect 1D observations, got {obs[group].shape} for '{group}'.")
        obs_dim += int(obs[group].shape[-1])
    return groups, obs_dim


def _concat_obs(obs: TensorDict, groups: list[str]) -> torch.Tensor:
    return torch.cat([obs[group] for group in groups], dim=-1)


class RunningNormalizer(nn.Module):
    """Small empirical normalizer matching the subset of RSL-RL behavior we need."""

    def __init__(self, size: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("count", torch.tensor(eps))
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        batch_count = torch.as_tensor(flat.shape[0], device=flat.device, dtype=self.count.dtype)
        if batch_count <= 0:
            return

        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
        self.count.copy_(total_count)


class GaussianActionDistribution(nn.Module):
    """Diagonal Gaussian action distribution with optional state-independent std."""

    def __init__(self, action_dim: int, init_std: float = 1.0, std_type: str = "scalar"):
        super().__init__()
        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown std_type '{std_type}'. Expected 'scalar' or 'log'.")
        self.action_dim = int(action_dim)
        self.std_type = std_type
        if std_type == "scalar":
            self.std_param = nn.Parameter(torch.full((self.action_dim,), float(init_std)))
        else:
            self.log_std_param = nn.Parameter(torch.log(torch.full((self.action_dim,), float(init_std))))
        self._distribution: torch.distributions.Normal | None = None
        torch.distributions.Normal.set_default_validate_args(False)

    @property
    def input_dim(self) -> int:
        return self.action_dim

    def update(self, mean: torch.Tensor) -> None:
        if self.std_type == "scalar":
            std = self.std_param.clamp_min(1.0e-6).expand_as(mean)
        else:
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = torch.distributions.Normal(mean, std)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()  # type: ignore[union-attr]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions).sum(dim=-1)  # type: ignore[union-attr]

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = torch.distributions.Normal(old_mean, old_std)
        new_dist = torch.distributions.Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std


class HeteroscedasticGaussianActionDistribution(GaussianActionDistribution):
    """Diagonal Gaussian action distribution with state-dependent std head."""

    def __init__(self, action_dim: int, init_std: float = 1.0, std_type: str = "scalar"):
        nn.Module.__init__(self)
        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown std_type '{std_type}'. Expected 'scalar' or 'log'.")
        self.action_dim = int(action_dim)
        self.init_std = float(init_std)
        self.std_type = std_type
        self._distribution: torch.distributions.Normal | None = None
        torch.distributions.Normal.set_default_validate_args(False)

    @property
    def input_dim(self) -> int:
        return 2 * self.action_dim

    def update(self, raw_output: torch.Tensor) -> None:
        mean, raw_std = raw_output.split(self.action_dim, dim=-1)
        if self.std_type == "scalar":
            std = torch.nn.functional.softplus(raw_std) + 1.0e-6
        else:
            std = torch.exp(raw_std)
        self._distribution = torch.distributions.Normal(mean, std)


def _make_distribution(output_dim: int, distribution_cfg: dict | None) -> GaussianActionDistribution | None:
    if distribution_cfg is None:
        return None

    cfg = dict(distribution_cfg)
    class_name = cfg.pop("class_name")
    if class_name.endswith("GaussianDistribution") and not class_name.endswith("HeteroscedasticGaussianDistribution"):
        return GaussianActionDistribution(output_dim, **cfg)
    if class_name.endswith("HeteroscedasticGaussianDistribution"):
        return HeteroscedasticGaussianActionDistribution(output_dim, **cfg)
    raise ValueError(f"Unsupported DWL action distribution '{class_name}'.")


class DwlActorModel(nn.Module):
    """DWL actor with GRU history encoder, latent actor head, and decoder."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        history_length: int = 1,
        encoder_hidden_dim: int = 128,
        encoder_num_layers: int = 1,
        latent_dim: int = 64,
        decoder_obs_set: str = "critic",
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        decoder_activation: str | None = None,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = _selected_obs_groups(obs, obs_groups, obs_set)
        self.history_length = int(history_length)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1.")
        if self.obs_dim % self.history_length != 0:
            raise ValueError(
                f"Actor obs dim {self.obs_dim} is not divisible by history_length={self.history_length}."
            )

        self.step_obs_dim = self.obs_dim // self.history_length
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer = RunningNormalizer(self.obs_dim) if obs_normalization else nn.Identity()

        self.encoder = nn.GRU(
            input_size=self.step_obs_dim,
            hidden_size=int(encoder_hidden_dim),
            num_layers=int(encoder_num_layers),
            batch_first=True,
        )
        self.latent_head = _make_mlp(int(encoder_hidden_dim), int(latent_dim), [], activation)

        self.distribution = _make_distribution(output_dim, distribution_cfg)
        actor_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.actor = _make_mlp(int(latent_dim), actor_output_dim, hidden_dims, activation)

        self.decoder_obs_set = decoder_obs_set
        if decoder_obs_set in obs_groups:
            self.decoder_obs_groups, self.decoder_output_dim = _selected_obs_groups(obs, obs_groups, decoder_obs_set)
        elif decoder_obs_set in obs:
            self.decoder_obs_groups, self.decoder_output_dim = [decoder_obs_set], int(obs[decoder_obs_set].shape[-1])
        else:
            self.decoder_obs_groups, self.decoder_output_dim = [], 0
        decoder_activation = decoder_activation if decoder_activation is not None else activation
        self.decoder = (
            _make_mlp(int(latent_dim), self.decoder_output_dim, decoder_hidden_dims, decoder_activation)
            if self.decoder_output_dim > 0
            else nn.Identity()
        )

        self._last_latent: torch.Tensor | None = None
        _init_mlp(self.latent_head)
        _init_mlp(self.actor, output_gain=0.01)
        if isinstance(self.decoder, nn.Sequential):
            _init_mlp(self.decoder)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        raw_output = self.actor(latent)
        if self.distribution is None:
            return raw_output
        self.distribution.update(raw_output)
        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.mean

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del masks, hidden_state
        flat_obs = self.obs_normalizer(_concat_obs(obs, self.obs_groups))
        sequence = flat_obs.reshape(*flat_obs.shape[:-1], self.history_length, self.step_obs_dim)
        if sequence.ndim != 3:
            sequence = sequence.flatten(0, -3)
        _, hidden = self.encoder(sequence)
        latent = self.latent_head(hidden[-1])
        self._last_latent = latent
        return latent

    def decode(self, latent: torch.Tensor | None = None) -> torch.Tensor:
        """Decode a latent state into the configured privileged reconstruction target."""

        if self.decoder_output_dim <= 0:
            raise RuntimeError("DWL decoder has no output target. Configure decoder_obs_set in obs_groups.")
        if latent is None:
            if self._last_latent is None:
                raise RuntimeError("No latent is available yet. Call the actor or get_latent first.")
            latent = self._last_latent
        return self.decoder(latent)

    def reconstruct(self, obs: TensorDict, detach_latent: bool = False) -> torch.Tensor:
        """Encode observations and decode the privileged/state reconstruction."""

        latent = self.get_latent(obs)
        if detach_latent:
            latent = latent.detach()
        return self.decode(latent)

    def reconstruction_target(self, obs: TensorDict) -> torch.Tensor:
        """Return the decoder target from the configured observation groups."""

        if not self.decoder_obs_groups:
            raise RuntimeError("No decoder observation groups are configured.")
        return _concat_obs(obs, self.decoder_obs_groups)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(_concat_obs(obs, self.obs_groups))  # type: ignore[attr-defined]

    @property
    def latent(self) -> torch.Tensor | None:
        return self._last_latent

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        return _TorchDwlActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        return _OnnxDwlActor(self)


class DwlCriticModel(nn.Module):
    """DWL critic that consumes privileged/state observations."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        del distribution_cfg
        self.obs_groups, self.obs_dim = _selected_obs_groups(obs, obs_groups, obs_set)
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer = RunningNormalizer(self.obs_dim) if obs_normalization else nn.Identity()
        self.critic = _make_mlp(self.obs_dim, output_dim, hidden_dims, activation)
        _init_mlp(self.critic)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output
        return self.critic(self.get_latent(obs))

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del masks, hidden_state
        return self.obs_normalizer(_concat_obs(obs, self.obs_groups))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(_concat_obs(obs, self.obs_groups))  # type: ignore[attr-defined]


class _TorchDwlActor(nn.Module):
    """TorchScript-friendly deterministic actor export."""

    def __init__(self, model: DwlActorModel):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_length = model.history_length
        self.step_obs_dim = model.step_obs_dim
        self.encoder = copy.deepcopy(model.encoder)
        self.latent_head = copy.deepcopy(model.latent_head)
        self.actor = copy.deepcopy(model.actor)
        self.action_dim = model.distribution.action_dim if model.distribution is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        sequence = x.reshape(x.shape[0], self.history_length, self.step_obs_dim)
        _, hidden = self.encoder(sequence)
        latent = self.latent_head(hidden[-1])
        output = self.actor(latent)
        if self.action_dim is not None and output.shape[-1] == 2 * self.action_dim:
            output = output[..., : self.action_dim]
        return output

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxDwlActor(_TorchDwlActor):
    """ONNX export wrapper for deterministic DWL actor inference."""

    is_recurrent: bool = False

    def __init__(self, model: DwlActorModel):
        super().__init__(model)
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


# Backwards-compatible alias for configs that name the model generically.
DwlModel = DwlActorModel

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the DWL task.

This module will define the observation groups used by Denoising World Model
Learning:

- policy observations: onboard/proprioceptive signals available on the real robot
- privileged/state observations: simulation-only targets used by the critic and
  the decoder reconstruction loss

Implementation will be added once the DWL observation contract is finalized.
"""


def placeholder():
    """Keep this module importable until DWL observation terms are implemented."""
    raise NotImplementedError("DWL observation terms are not implemented yet.")

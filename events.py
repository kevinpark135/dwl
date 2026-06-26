# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Event and domain-randomization terms for the DWL task.

This module will collect DWL-specific randomization hooks such as sensor noise,
friction variation, payload/mass changes, motor offset/strength changes, PD
factor randomization, pushes, and latency-style perturbations.

Implementation will be added when the environment configuration is wired to the
DWL training setup.
"""


def placeholder():
    """Keep this module importable until DWL event terms are implemented."""
    raise NotImplementedError("DWL event terms are not implemented yet.")

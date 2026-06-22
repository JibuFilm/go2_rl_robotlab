# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0

"""Dynamic tracking-sigma RESTORATION (go2_rl_gym parity).

The upstream IsaacLab port dropped go2_rl_gym's dynamic tracking sigma (fixed ``std=0.5``,
README-admitted). This module restores it, transcribed from the reference implementation:

  * ``legged_gym/envs/base/legged_robot.py:1288-1334`` (``_get_dynamic_sigma`` + the two
    tracking rewards)
  * ``legged_gym/envs/go2/go2_config.py:166-175`` (the dynamic_sigma config block)

This file is PURE TORCH — no isaaclab imports — so the math is unit-testable offline
(``tests/test_dynamic_sigma.py``) before any GPU run. The env-facing reward terms that consume
it live in ``rewards.py``.

Sigma-space note: the reference reward is ``exp(-err^2 / sigma)`` with sigma values given
directly (default 0.25, per-terrain maxima below). The port's tracking terms take ``std`` with
``exp(-err^2 / std^2)``; the reward wrappers convert via ``default_sigma = std**2``.
"""
from __future__ import annotations

import torch

# go2_config.py:168-171 — command-magnitude bands for the sigma interpolation.
MIN_LIN_VEL, MAX_LIN_VEL = 0.5, 1.5
MIN_ANG_VEL, MAX_ANG_VEL = 1.0, 2.0

# go2_config.py:174 — per-terrain sigma maxima, gym id order
# [wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]
# = [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4].
# Mapped BY NAME here because the port splits gym's single `slope` bucket (id 1, half
# positive / half negative inside make_terrain) into `slope_up` + `slope_down` sub-terrains.
MAX_SIGMA_BY_NAME: dict[str, float] = {
    "wave": 5.0 / 12.0,
    "slope_up": 1.0 / 4.0,
    "slope_down": 1.0 / 4.0,
    "rough_slope": 1.0 / 4.0,
    "stairs_up": 1.0 / 2.0,
    "stairs_down": 1.0 / 2.0,
    "obstacles": 3.0 / 4.0,
    "stepping_stones": 1.0,
    "gap": 1.0,
    "flat": 1.0 / 4.0,
}


def cols_to_max_sigma(
    names: list[str],
    proportions: list[float],
    num_cols: int,
    max_sigma_by_name: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-terrain-column sigma_max lookup, via the EXACT column→sub-terrain binning BOTH
    generators use:

      * gym ``utils/terrain.py:65``: ``choice = j / num_cols + 0.001`` vs cumulative
        normalized proportions (``make_terrain`` bucket thresholds);
      * IsaacLab ``TerrainGenerator._generate_curriculum_terrains``:
        ``np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions)))`` —
        the identical formula, iterating ``sub_terrains`` in dict order.

    Returns a float32 tensor of shape ``(num_cols,)``: column index → sigma_max.
    """
    if max_sigma_by_name is None:
        max_sigma_by_name = MAX_SIGMA_BY_NAME
    props = torch.tensor(proportions, dtype=torch.float64)
    cum = torch.cumsum(props / props.sum(), dim=0)
    out = []
    for j in range(num_cols):
        choice = j / num_cols + 0.001
        idx = int(torch.nonzero(choice < cum, as_tuple=False)[0].item())
        out.append(float(max_sigma_by_name[names[idx]]))
    return torch.tensor(out, dtype=torch.float32)


def dynamic_sigma(
    cmd_abs: torch.Tensor,
    max_sigma_per_env: torch.Tensor,
    terrain_levels: torch.Tensor,
    default_sigma: float,
    v_min: float | torch.Tensor,
    v_max: float | torch.Tensor,
) -> torch.Tensor:
    """Per-env tracking sigma — transcribed from ``legged_robot.py:1288-1308``
    (``_get_dynamic_sigma``):

      * ``cmd_abs < v_min``                → default sigma
      * ``v_min <= cmd_abs < v_max``       → linear interpolation default → sigma_max
      * ``cmd_abs >= v_max``               → sigma_max
      * then the terrain-level ramp: ``level_scale = clamp(exp((level+1)/10) - 1, max=1)``,
        ``sigma = default + level_scale * (sigma - default)`` — low terrain levels keep sigma
        near default; the relaxation phases in as the curriculum promotes.
    """
    target = max_sigma_per_env
    sigma = torch.full_like(cmd_abs, default_sigma)
    v_min_t = torch.as_tensor(v_min, device=cmd_abs.device, dtype=cmd_abs.dtype)
    v_max_t = torch.as_tensor(v_max, device=cmd_abs.device, dtype=cmd_abs.dtype)
    v_min_t = torch.broadcast_to(v_min_t, cmd_abs.shape)
    v_max_t = torch.broadcast_to(v_max_t, cmd_abs.shape)
    denom = torch.clamp(v_max_t - v_min_t, min=1e-6)
    mask = (cmd_abs >= v_min_t) & (cmd_abs < v_max_t)
    if mask.any():
        ratio = (cmd_abs[mask] - v_min_t[mask]) / denom[mask]
        sigma[mask] = default_sigma + ratio * (target[mask] - default_sigma)
    mask = cmd_abs >= v_max_t
    if mask.any():
        sigma[mask] = target[mask]
    level_scale = torch.clamp(torch.exp((terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
    return default_sigma + level_scale * (sigma - default_sigma)

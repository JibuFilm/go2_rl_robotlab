# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A3 isaaclab glue: the dome-pattern RayCaster pattern + the fused-range observation term.

This is the THIN isaaclab-touching layer over `a3_range_core` (pure torch, offline-tested):
  * `A3DomePatternCfg` / `a3_dome_pattern` — a custom RayCaster pattern whose directions are
    LOADED from the pattern contract JSON (never re-derived; PatternBaseCfg.func protocol:
    (cfg, device) -> (ray_starts, ray_directions)). Both sensors share the sensor-frame
    directions; their site quats (RayCasterCfg.offset.rot) orient them.
  * `dome_ranges` — the V5 student observation term: per-sensor terrain ranges from the two
    RayCasters (static terrain mesh only — isaaclab raycasting cannot see the robot) fused
    with the analytic SELF-OCCLUSION kernel (true-first-hit) and normalized to [0,1].
    Output: (num_envs, 512), sensor-major [front, rear].
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from robot_lab.tasks.a2.a3_range_core import (
    A3SelfOcclusionKernel,
    load_pattern,
    sensor_frame_directions,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------------- pattern
def a3_dome_pattern(cfg: "A3DomePatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """PatternBaseCfg.func: ray starts (at the sensor origin) + the CONTRACT's directions."""
    contract = load_pattern(cfg.json_path)
    dirs = sensor_frame_directions(contract, device=device, dtype=torch.float32)
    starts = torch.zeros_like(dirs)
    return starts, dirs


@configclass
class A3DomePatternCfg(patterns.PatternBaseCfg):
    """The A3 dome pattern, loaded from the contract JSON (train==deploy parity object)."""

    func = a3_dome_pattern
    json_path: str = MISSING


# ----------------------------------------------------------------------------- obs term
_KERNEL_CACHE: dict[int, tuple] = {}


def _get_kernel(env: "ManagerBasedRLEnv", asset_name: str):
    """Build (once per env instance) the self-occlusion kernel + the body-name -> index map."""
    key = id(env)
    if key not in _KERNEL_CACHE:
        kernel = A3SelfOcclusionKernel(device=env.device)
        asset = env.scene[asset_name]
        body_names = list(asset.body_names)
        missing = [b for b in kernel.required_bodies if b not in body_names]
        if missing:
            raise RuntimeError(
                f"A3 kernel: articulation lacks geom bodies {missing} "
                f"(have {body_names}) — MJCF/URDF body-name mismatch")
        idx = {b: body_names.index(b) for b in kernel.required_bodies}
        _KERNEL_CACHE[key] = (kernel, idx)
    return _KERNEL_CACHE[key]


def dome_ranges(
    env: "ManagerBasedRLEnv",
    sensor_names: tuple[str, ...] = ("dome_front", "dome_rear"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The V5 student range block: fused min(terrain, self-body) ranges, normalized [0,1].

    TRUE-FIRST-HIT: the self-occlusion kernel supplies the robot-body hits the static-mesh
    RayCaster cannot see; no masking, no fill rules — the fused measurement IS the observation.
    """
    kernel, idx = _get_kernel(env, asset_cfg.name)
    # terrain ranges from the two RayCasters (miss -> inf -> normalizes to 1.0)
    terrain = []
    for name in sensor_names:
        s = env.scene.sensors[name]
        r = (s.data.ray_hits_w - s.data.pos_w.unsqueeze(1)).norm(dim=-1)   # (B, 256)
        terrain.append(r)
    terrain_r = torch.cat(terrain, dim=1)                                   # (B, 512)
    # self-body first hits from the articulation's body poses
    asset = env.scene[asset_cfg.name]
    body_poses = {
        b: (asset.data.body_pos_w[:, i], asset.data.body_quat_w[:, i]) for b, i in idx.items()
    }
    self_r = kernel.self_ranges(body_poses)                                 # (B, 512)
    return kernel.fuse_normalize(terrain_r, self_r)

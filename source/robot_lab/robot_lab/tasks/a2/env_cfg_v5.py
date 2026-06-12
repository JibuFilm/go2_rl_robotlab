# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V5 env cfg — the A3 perceptive student (PATH_A_SESSION_BRIEF.md §7 A3, LOCKED 2026-06-12).

DELTA over A2EnvCfg (everything else inherited byte-for-byte — recipe untouched):
  * scene: + two dome RayCasters (`dome_front` / `dome_rear`) on base_link at the FIXED lidar
    sites (offsets read from the pattern contract JSON at import — never hand-typed), custom
    A3DomePatternCfg (256 rays each, loaded from the same JSON), `ray_alignment="base"` (full
    body-frame rotation — these are dome sensors, not height scanners), terrain mesh only.
  * observations: STUDENT groups only — policy + single_obs gain the trailing `dome_ranges`
    term (45 -> 557 per frame; history x5 = 2785 term-major). CriticCfg (teacher/critic
    privileged obs incl. the ideal 187-grid scan) inherited UNTOUCHED — the teacher keeps its
    oracle; the student's ranges never feed it, and vice versa.
  * runner cfg (rsl_rl_cfg.A2V5MoECTSRunnerCfg): `proprio_dim=45` — the shared actor stays
    `latent + obs(45)`; the range block enters ONLY the student MoE encoder (flat MLP, wider
    first layer — architecture family untouched).

The proprio `RobotLab-A2-v0` task is untouched (the post-V5 blind ablation baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from robot_lab.tasks.a2 import mdp_a3
from robot_lab.tasks.a2.env_cfg import (
    A2EnvCfg,
    A2SceneCfg,
    BASE_LINK_NAME,
    ObservationsCfg,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_PATTERN_JSON = _DATA_DIR / "a3_pattern_contract_v1.json"

# Mount offsets come from the CONTRACT (which read them from the locked MJCF) — one source.
with open(_PATTERN_JSON, "r", encoding="utf-8") as _f:
    _CONTRACT = json.load(_f)
_FRONT, _REAR = _CONTRACT["sensors"]
assert _FRONT["site"] == "a2_front_lidar_frame" and _REAR["site"] == "a2_rear_lidar_frame"
# normalization clamps at r_max; cast a little past it so "beyond r_max" reads as no-hit=1.0
_MAX_CAST_DISTANCE = float(_CONTRACT["r_max_m"]) + 5.0


def _dome_caster(sensor: dict) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}",
        offset=RayCasterCfg.OffsetCfg(
            pos=tuple(sensor["site_pos_body"]),
            rot=tuple(sensor["site_quat_wxyz_body"]),       # (w, x, y, z) — site quat verbatim
        ),
        ray_alignment="base",                                # full base-frame rotation (dome)
        pattern_cfg=mdp_a3.A3DomePatternCfg(json_path=str(_PATTERN_JSON)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=_MAX_CAST_DISTANCE,
    )


@configclass
class A2V5SceneCfg(A2SceneCfg):
    """A2 scene + the two A3 dome RayCasters (terrain hits; self-hits come from the kernel)."""

    dome_front = _dome_caster(_FRONT)
    dome_rear = _dome_caster(_REAR)


@configclass
class V5PolicyCfg(ObservationsCfg.PolicyCfg):
    """Student per-frame obs: the 45-dim proprio block + the 512-dim fused range block.

    dome_ranges is APPENDED (dataclass subclass field ordering) => term-major layout
    [ang_vel 3, gravity 3, cmd 3, q 12, qd 12, action 12, ranges 512] = 557."""

    dome_ranges = ObsTerm(
        func=mdp_a3.dome_ranges,
        clip=(0.0, 1.0),
        scale=1.0,         # already normalized range/r_max in [0,1]; no added noise in v1
    )
    # __post_init__ inherited: history_length=5, flatten_history_dim=True -> 2785 term-major


@configclass
class V5SingleObsCfg(V5PolicyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.history_length = 1                              # current frame: 557


@configclass
class V5ObservationsCfg(ObservationsCfg):
    policy: V5PolicyCfg = V5PolicyCfg()
    single_obs: V5SingleObsCfg = V5SingleObsCfg()
    # critic: inherited UNTOUCHED (teacher oracle + privileged dynamics)


@configclass
class A2V5EnvCfg(A2EnvCfg):
    """A2 rough-terrain cfg + the A3 perceptive-student sensorium (V5)."""

    scene: A2V5SceneCfg = A2V5SceneCfg(num_envs=8192, env_spacing=0.5)
    observations: V5ObservationsCfg = V5ObservationsCfg()

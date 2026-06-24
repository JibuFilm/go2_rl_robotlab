# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V16 - GOAL-ORIENTED TRAVERSAL on top of V15.

WHY (2026-06-23): V15's earned-only command gate works (records climbing, range holding at ~1.5 until
speed competence earns it), but a comprehensive bench of the iter-3600 checkpoint exposed the deeper
gap the user flagged: on stairs / obstacle / slope the policy reaches ZERO goals (robogauge `success`
0.0 across all terrain tasks; stairs it tips over). The velocity numbers MASKED this — obstacle showed
lin_vel_err 0.44 and orientation 0.998 yet success 0.0. The root cause is structural, not a tuning miss:

  * the task is velocity-command-driven, and the dominant reward (track_lin_vel_xy_exp, weight 1.0)
    optimizes velocity-tracking PRECISION everywhere, including where the right move is to slow down and
    COMMIT to getting over the feature;
  * the terrain-progress dial that DOES credit climbing was DEMOTED in V14 (0.35 -> 0.1) to prioritize
    speed; and
  * the terrain curriculum promotes on raw distance walked only — nothing explicitly credits CLEARING a
    stair / summiting a slope / getting over an obstacle.

So on hard terrain the policy keeps optimizing speed precision right up until it tips. The user's frame:
stairs/obstacle are GOAL-ORIENTED (reach-goal first, velocity a SECOND metric); flat/slope stay
velocity-tracking.

THE V16 CHANGE (deliberately ONE reward addition - stack + gate, don't pile on):
  A new additive reward `terrain_traversal_progress` pays the SAME bounded along-terrain progress as the
  V12 dial, SCALED by local height-scan relief: ~0 on flat (relief gate 0 -> V15 flat behavior is
  byte-unchanged) and full on stairs/obstacle/slope. At weight 0.6 it competes with the 1.0 velocity
  tracker exactly where traversal should lead, WITHOUT touching the primary velocity reward (so flat
  competence carried over from the V15 warm-start is preserved). It reuses track_lin_vel_terrain_dial +
  _height_scan_relief (call-don't-copy).

NEXT STAGED LEVER (NOT in V16 - apply only if the smoke/eval shows velocity still dominating on hard
terrain): relief-scale the velocity tracker DOWN on hard terrain so velocity is explicitly secondary
there. Held back to keep V16 a single, attributable change.

WARM-START: arch/obs/action contract is identical to V15 -> warm-start the latest V15 best
(model_best.pt). Competence mode rebuilds the command curriculum at stage 0 (range restarts low). The
command gate, terrain, MoE, bravery, posture, energy are all V15, unchanged.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg_v15 import A2V15EnvCfg


@configclass
class A2V16EnvCfg(A2V15EnvCfg):
    """V15 + a terrain-relief-gated traversal reward. See module docstring."""

    def __post_init__(self):
        super().__post_init__()  # all of V15 (gate) + V14 (caps/speed/terrain/energy/MoE/bravery/posture)

        # Goal-oriented traversal credit: bounded along-terrain progress, paid only where the terrain is
        # hard (relief-gated). Flat -> ~0 (V15 unchanged); stairs/obstacle/slope -> full. weight 0.6 puts
        # "get over it" on par with the 1.0 velocity tracker on hard terrain; velocity stays primary on flat.
        self.rewards.terrain_traversal = RewTerm(
            func=mdp.terrain_traversal_progress,
            weight=0.6,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("height_scanner_small"),
                "backtrack_cap": 0.5,
                "relief_threshold": 0.03,
                "relief_full": 0.12,
            },
        )

# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V15 — V14 with the command-curriculum gate FIXED so the range ramps only on EARNED, sustained,
low-drift competence at the CURRENT speed (the whole config is otherwise V14, byte-for-byte).

WHY (2026-06-22): the live V14 run exposed that its competence gate is too loose. Warm-started from a
competent walker, V14 ramped the command range +0.5 every ~16-17 iters, clearing ±0.5 -> ±5.0 in ~130
iters (log: nine `Command range updated` lines 2016-2150, zero HOLDs after the first). It did NOT earn
those speeds — on flat at iter 2500 it tracks cmd 1.0 at 86% but cmd 5.0 at only 41%, with heavy lateral
veer (+7 to +14 m) at cmd >= 2. Three structural reasons in commands.py:
  (a) speed_ratio is a FLAT AVERAGE over moving envs (`speed_ratio[moving].mean()`). Commands are sampled
      uniformly and per-terrain-clamped, so most envs get modest commands the policy already tracks; the
      high-command-on-flat tail (the new capability) is a minority averaged away. The gate asks "can you
      track the AVERAGE command?", never "can you track the TOP of the range?".
  (b) DRIFT is invisible: the metric is the along-command projection only, so a robot veering sideways
      still scores well. "without much drift" was literally not measured.
  (c) the bar (0.60) and consolidation window (min_samples 400 ~= 17 iters) are both lenient -> it flashes
      competence rather than sustaining it.

V15 flips on the three opt-in gate features added to Go2RLGymCommand (default-off, so go2/v5-v14 are
unchanged): magnitude-WEIGHTED speed ratio (fast commands dominate -> gates the top of the range), a
lateral-DRIFT gate, a higher bar (0.80) and a longer window (800 samples). The drift bound 0.15 is
calibrated from the V14 sweep (cmd 1.0 -> ~0.01 drift-ratio = fine; cmd 2.0 -> ~0.31 = veering).

The policy ARCH / OBS / ACTION contract is identical to V14 -> warm-start the latest V14 weights; in
competence mode a fresh env rebuilds the command curriculum at stage 0, so the range RESTARTS at ±0.5
and re-ramps under the strict gate. NOTHING else about V14 changes (terrain, rewards, MoE, stance).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from robot_lab.tasks.a2.env_cfg_v14 import A2V14EnvCfg


@configclass
class A2V15EnvCfg(A2V14EnvCfg):
    """V14 + an earned-only command-curriculum gate. See module docstring."""

    def __post_init__(self):
        super().__post_init__()  # all of V14 (caps, speed rework, terrain, energy, MoE, bravery, posture)

        cmd = self.commands.base_velocity
        # gate the ramp on tracking the TOP of the range (not the easy average) ...
        cmd.command_curriculum_speed_weighting = "magnitude"
        # ... at a higher bar ...
        cmd.command_curriculum_min_speed_ratio = 0.80
        # ... with low lateral drift (veering blocks advancement; 0.15 from the V14 sweep) ...
        cmd.command_curriculum_max_drift_ratio = 0.15
        # ... sustained over a longer consolidation window (was 400 ~= 17 iters).
        cmd.command_curriculum_min_samples = 800

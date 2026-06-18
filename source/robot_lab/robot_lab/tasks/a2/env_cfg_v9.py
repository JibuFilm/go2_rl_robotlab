# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V9 env cfg — the STAIR-INITIATION fix on top of the V8 self-finding dial.

WHY THIS FILE EXISTS — the V8 stairs post-mortem (2026-06-18):
  V8 (the self-finding speed dial) was validated on FLAT (speed 1.33->1.75) and on the AGGREGATE
  terrain_levels curriculum, but a local RoboGauge probe (`/tmp/rg_stairs_probe.py`) showed it scores
  0.00 on stairs_fd/bd at iter15k/18k — it ROLLS OVER. The user then found the tell in the drive-sim:
  the A2 can physically walk the stairs but DOES NOT TAKE THE FIRST STEP. So this is an INITIATION
  gap, not a capability gap, and a reward-stack audit (HANDOFF V9 block) found why:

    - The dial (mdp.track_lin_vel_dial) rewards PLANAR forward speed only. The first step is a vertical
      lift with ~0 horizontal speed -> the dial pays nothing for initiating the climb.
    - Nothing penalizes a commanded standstill (the dial has no undershoot penalty by design) ->
      hesitating at the base is free.
    - base_height_l2 is curriculum-ramped to -10.0 (the STRONGEST term in the stack) and penalizes
      (belly-clearance - 0.40)^2. `_get_base_height` estimates ground as the MEAN of the height-scan
      rays, so at the flat->riser boundary the patch straddles two levels, the estimated clearance
      spikes, and the L2 kernel turns that transient into a WALL the dial can't pay to cross.

  => The policy is REWARD-OPTIMAL refusing the step. The emergence-faithful fix (user-chosen
     2026-06-18, "option C") is NOT a per-terrain gate (anti-emergent: hand-sets per terrain) nor an
     added climb reward (hand-shapes the answer) but RESHAPING the blocking penalty: make the
     base-height term transient-tolerant so the dial's OWN forward-progress drive climbs the stairs
     endogenously. (The dial already rewards up-stairs motion: once the body pitches into the climb,
     going up registers as +root_lin_vel_b[x] = forward progress along the command.)

DELTA over A2V8EnvCfg (inherits EVERYTHING — dial, cap ramp, robustness, sensorium, MoE/CTS runner):
  1. Swap base_height_l2's kernel: mdp.base_height_l2 (quadratic) -> mdp.base_height_huber (L2 within
     `delta`, LINEAR beyond; value+slope continuous at `delta`). SAME term name (so the inherited
     -1->-10 weight curriculum still applies), SAME target_height/sensor, SAME -10 mature weight — only
     the kernel SHAPE changes, and only for the large transient excursions that wall off a step.
     Flat-ground discipline (small/sustained clearance error) is unchanged.
  2. delta = 0.06 m (the quad->linear knob; smoke-tuned).
  Nothing else changes. flat_orientation_l2 (-1.0) is the philosophically-similar but 10x-gentler
  sibling (also a flat-posture prior); left UNTOUCHED — it is the WATCHED SECONDARY if the smoke shows
  the step initiates but the climb doesn't sustain (HANDOFF V9 block). One change at a time.

WARM-START: V8 model_25000 (the parked robust dial-walker) — keep the clean gait + flat speed, just
  unlock stair initiation. The cap-ramp / curricula clocks resume at iter 25000 (train.py continuity
  fix): cap ±3.0, base_height weight already -10, lin_vel_z already 0 -> the huber fix is ACTIVE from
  step 1. SMOKE (~500-800 iters) -> RoboGauge stairs probe: confirm stairs_fd > 0 AND flat speed held.
"""
from __future__ import annotations

from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg_v8 import A2V8EnvCfg


@configclass
class A2V9EnvCfg(A2V8EnvCfg):
    """V8 dial + robustness, with base_height_l2 reshaped to a transient-tolerant kernel so the dial's
    forward-progress drive can initiate stair-climbing. See module docstring. No new reward FIELDS —
    the kernel swap is a mutation in __post_init__ (keeps the term name + the -1->-10 weight curriculum)."""

    def __post_init__(self):
        super().__post_init__()  # inherits ALL V8 deltas (dial, cap ramp, robustness, terrains, sensorium)

        # STAIR-INITIATION FIX (option C): reshape base_height_l2 from a stiff L2 spring into a
        # transient-tolerant kernel (L2 within delta, linear beyond). Keeps the term NAME (so the
        # CurriculumCfg.base_height_l2 weight ramp -1->-10 still targets it) + target/sensor; only the
        # kernel shape changes, de-escalating the flat->riser scan-straddle spike that walls off the
        # first step. Terrain-agnostic — the dial then climbs endogenously.
        self.rewards.base_height_l2.func = mdp.base_height_huber
        self.rewards.base_height_l2.params["delta"] = 0.06

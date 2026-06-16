# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V8 env cfg — the SELF-FINDING SPEED DIAL (the user's literal vision, realized in the reward).

WHY THIS FILE EXISTS — the V7 post-mortem (2026-06-16):
  V7 commanded a UNIFORM ±5.0 m/s on every terrain and relied on the dynamic-σ exp-kernel tracker
  (`track_lin_vel_xy_exp_dynamic_sigma`) to "let the policy find each terrain's feasible max" (V7
  docstring §1). The RoboGauge trend disproved it: robustness REGRESSED as the command rose
  (stairs_fd 0.00→0.134@±2.0→0.00@±3.0; slopes down; flat low). The mechanism is now understood:

    A setpoint tracker rewards achieved == commanded. On a staircase where ~1 m/s is the physical
    ceiling, a 5 m/s command is a permanent error the policy can only shrink by going FASTER →
    rushing → falling. A plain exp kernel at least VANISHES when the command is hopeless (gradient
    ≈ 0, so the policy ignores it). The dynamic-σ deliberately WIDENS σ at high commands to keep
    that reward alive — which re-introduces a monotonic "go faster" gradient even when 5 m/s is
    impossible. So uniform-±5 + dynamic-σ is structurally a "chase the number" engine; it CANNOT
    represent "1 m/s is the correct answer here." It punishes feasible-max behaviour identically
    to loafing. (Full analysis: info/locomotion/HANDOFF.md, the V8 reward-redesign block.)

  The upstream/Wu recipe achieves "fastest-feasible per terrain" the OTHER way — per-terrain
  command caps (`terrain_max_command_ranges`: stairs ±1.0, slopes ±1.5, flat ±2.0), so the policy
  is only ever ASKED feasible speeds and the tracker stays sane. That works, but it puts the
  terrain-feasibility knowledge in the COMMAND layer (a table we hand-set), not in the policy.

  The user chose (2026-06-16) the PURER realization: put it in the POLICY. The command becomes a
  CAP + DIRECTION ("go as fast as you safely can, up to this, that way"), and the policy discovers
  each terrain's ceiling itself. That requires a reward whose optimum is at feasible-max, not at
  the command — which a setpoint tracker is not. Hence the new reward below.

THE MECHANISM (mdp.track_lin_vel_dial, added to go2/mdp/rewards.py):
  reward = clamp(  dot(v_xy, cmd_dir),  0,  |cmd| )      # forward progress along command, capped
  - NO penalty for achieving less than |cmd| → over-commanding is harmless (no reckless pull).
  - Bounded + modest WEIGHT → the SAFETY stack (orientation / contact / vertical-bounce / fall-
    termination), which grows with speed AND terrain difficulty, overtakes the speed bonus at a
    terrain-appropriate speed. That crossover IS the emergent ceiling: ≈ flat-feasible on flat,
    ≈1 m/s on stairs — from ONE uniform high cap. The self-finding dial.
  Heading is kept by mdp.lin_vel_lateral_l2 (penalize velocity ⟂ to command) + the unchanged
  yaw-rate tracker (turning is a genuine setpoint, so it stays a setpoint).

DELTA over A2V7EnvCfg (inherits ALL V7 robustness deltas — terrains, push/mass DR, flat_orientation,
feet_air_time, ang_vel_xy, smoothness, AND the uniform ±5.0 per-terrain caps, which are now CORRECT:
the cap that was wrong for a setpoint tracker is exactly right as a dial maximum):
  1. DISABLE the setpoint tracker: track_lin_vel_xy_exp.weight 1.0 → 0.0.
  2. ADD track_lin_vel_dial (weight 0.35 — THE tuning knob; see note) as the speed driver.
  3. ADD lin_vel_lateral_l2 (weight -0.5) for heading-keeping.
  4. Command = a CAP RAMP ±2.0 → ±5.0 (the dial's MAX grows with competence). NOTE: the first V8 smoke
     tried ±5.0 from iter 0 and the warm-started walker COLLAPSED (99% falls by iter ~50) — the dial's
     no-undershoot-penalty does not stop the policy *trying* to reach an out-of-distribution command, so
     a velocity cap ramp is still required for warm-start. At each cap level the dial self-limits per
     terrain (orthogonal). Uniform ±5.0 per-terrain caps kept from V7 (they don't bind below the ramp).

TUNING (the one empirical unknown — set by a smoke run, NOT guessable a priori):
  The dial weight (0.35) sets where the emergent ceiling lands. Too high → reckless everywhere (the
  V7 failure); too low → lazy everywhere (crawls even on flat). The launch plan is: SHORT smoke
  (~500-800 iters, warm-started from the cleanest non-reckless walker = V5 model_7000) → RoboGauge
  eval → confirm flat speed > V5 AND stairs/slopes survive → adjust weight if needed → full run.

WARM-START: V5 model_7000 (2026-06-13_06-17-32_a2-patha-v5) — the cleanest gait NOT corrupted by
  V7's over-command recklessness. terrain_levels + the curriculum clock reset on launch regardless,
  so V5's "never climbed terrain" is not inherited as a handicap; only its (clean) gait carries over.

RUN LENGTH (set at launch): the 10-row gym curriculum needs real iters — target ≥40-50k once the
  dial weight is validated (same as V7's recipe note).
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg_v7 import A2V7EnvCfg, A2V7RewardsCfg


@configclass
class A2V8RewardsCfg(A2V7RewardsCfg):
    """V7 rewards + the dial speed reward and the lateral-drift keeper (declared as fields →
    guaranteed registration). The inherited setpoint tracker is disabled in __post_init__."""

    # The self-finding speed dial: forward progress along the command, capped at |cmd|, no
    # undershoot penalty. Replaces the setpoint tracker as the speed driver. WEIGHT is the knob.
    track_lin_vel_dial = RewTerm(
        func=mdp.track_lin_vel_dial,
        weight=0.35,
        params={"command_name": "base_velocity"},
    )

    # Heading-keeping for the dial (which only rewards on-axis speed): penalize velocity ⟂ command.
    lin_vel_lateral_l2 = RewTerm(
        func=mdp.lin_vel_lateral_l2,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )


@configclass
class A2V8EnvCfg(A2V7EnvCfg):
    """V7 robustness + sensorium, with the speed objective swapped from setpoint-tracking to the
    self-finding dial. See module docstring."""

    rewards: A2V8RewardsCfg = A2V8RewardsCfg()

    def __post_init__(self):
        super().__post_init__()  # inherits ALL V7 deltas (uniform ±5 caps, terrains, DR, robustness)

        cmd = self.commands.base_velocity

        # CAP RAMP — start near the warm-start walker's competence, grow the dial's MAX as competence
        # builds. The terrain_max_command_ranges are uniform ±5.0 (from V7) but DON'T bind below the
        # global range (env cap = min(global, terrain)), so the global ramp IS the dial cap everywhere.
        #
        # !! SMOKE-1 LESSON (2026-06-16): the original V8 set ±5.0 from iter 0 ("the dial has no undershoot
        #    cliff so a high cap is safe"). FALSE for a WARM-START — the dial's no-undershoot-penalty stops
        #    the SETPOINT pathology, but it does NOT stop the policy from *trying* to accelerate toward a
        #    wildly out-of-distribution command. A V5 walker (trained ±0.5, probes to ~1.33) handed ±5
        #    instantly lunged, fell, and collapsed: illegal_contact 0.77→0.99 by iter ~50, terrain_levels
        #    1.53→0.43 (curriculum floored). So the dial needs a competence-tracking cap ramp like any
        #    velocity curriculum. (At each cap level the dial STILL self-limits per terrain — orthogonal.)
        #
        # Milestones are ABSOLUTE cumulative iters (train.py resume-continuity fix). Warm-start from V5
        # model_7000 → clock starts at 7000, so a SMOKE (~700 iters) stays at the ±2.0 start (clean test of
        # "does the dial keep the walker alive at a safe cap"); the FULL run (≥40-50k) climbs to ±5.0.
        # LAUNCH-COUPLED: these iters assume the V5-warm-start full run; re-tune to the chosen length.
        cmd.ranges.lin_vel_x = [-2.0, 2.0]
        cmd.ranges.lin_vel_y = [-1.0, 1.0]
        cmd.ranges.ang_vel_yaw = [-2.5, 2.5]
        cmd.command_range_curriculum = [
            {'iter': 15000, 'lin_vel_x': [-3.0, 3.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 28000, 'lin_vel_x': [-4.0, 4.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 42000, 'lin_vel_x': [-5.0, 5.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
        ]

        # Disable the setpoint tracker — track_lin_vel_dial replaces it. (Kept at weight 0 rather than
        # removed so its per-component reward still logs as 0; the dynamic-σ compute at weight 0 is cheap.)
        self.rewards.track_lin_vel_xy_exp.weight = 0.0

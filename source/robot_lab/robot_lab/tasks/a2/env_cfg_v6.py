# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V6 env cfg — SPRINT. V5's A3 perceptive student + the speed domain TRANSLATED to the A2's
real physical envelope (info/locomotion/A2_CAPABILITY_SSOT.md §3), targeting the official 5 m/s bar.

WHY THIS FILE EXISTS — the go2->A2 transcription drift (2026-06-14):
  The recipe is forked from go2_rl_robotlab. The body/actuators were correctly TRANSLATED to the A2
  (120/180 N.m, 41.55 kg), but the SPEED domain was TRANSCRIBED from the Go2 verbatim. Worse, it was
  traced and found compounded: the initial command range is go2's `ranges.lin_vel_x = +/-0.5`, and the
  `command_range_curriculum` that widens it only fires at iter 20k/50k -- never reached in a 10k run.
  So V5 trained the WHOLE run at +/-0.5 m/s (its 1.33 m/s probe is ~2.7x extrapolation). The A2 is a
  41.55 kg / 120-180 N.m / 22 rad/s / 0.55 m-leg machine with a ~3.6-5.4 m/s envelope (SSOT) -- we were
  training a sprinter at half a meter per second.

DELTA over A2V5EnvCfg (sensorium, reward balance, Gate-T energy weights all inherited byte-for-byte):
  1. initial `ranges` raised off the +/-0.5 floor (the value short runs actually train at).
  2. command_range_curriculum re-scheduled + ramped to +/-5.0 m/s on a schedule matched to the run.
  3. terrain_max_command_ranges['flat'] -> +/-5.0 (sprint is a flat behavior; non-flat caps unchanged).
  4. dynamic-sigma v_max 1.5 -> 5.0 so tracking tolerance scales across the full A2 speed range.
  5. base-mass DR +/-1.0 kg (go2-absolute = +/-2.4% on the A2) -> +/-2.6 kg (~+/-6.3%, matching go2's fraction).

PROVENANCE: every value below traces to A2_CAPABILITY_SSOT.md, NOT to the Go2 fork.

!! LAUNCH-COUPLED (read before running) -- the command_range_curriculum `iter` milestones are gated by
   `env.common_step_counter // num_steps_per_iter`, and that clock RESETS TO 0 on every launch/resume
   (VERIFIED 2026-06-14: common_step_counter is an env attribute incremented in go2_env.step; rsl_rl's
   runner.load() restores only the policy + logging iteration, NOT the env step counter -- which is also
   why V5 never escaped +/-0.5: its clock reset on each relaunch and never neared the 20k trigger). So
   these iters are RELATIVE TO V6's OWN run from 0, regardless of warm-start.
   DEFINITIVE CHECK before trusting: the command term prints "Command range updated at iter X" when a
   stage fires -- confirm that line in a smoke (the only residual is whether the forked rsl_rl restores
   the counter; the print settles it).
   Schedule below assumes WARM-START from V5's ~10k walker (already walks -> ramp early, from 0). For a
   FRESH run: lower initial `ranges.lin_vel_x` to +/-0.5..1.0 and stretch the milestones. Retune to the
   chosen run length -- do NOT run as-is under a different plan.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from robot_lab.tasks.a2.env_cfg_v5 import A2V5EnvCfg


@configclass
class A2V6EnvCfg(A2V5EnvCfg):
    """V5 sensorium + the A2-translated speed domain (5 m/s sprint target). See module docstring."""

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity

        # 1. INITIAL range -- binds from iter 0 until the first curriculum stage. The curriculum clock
        #    resets to 0 on launch (docstring), so a WARM-STARTED walker is commanded this from the start.
        #    PROVENANCE: TRANSLATED from go2 ranges.lin_vel_x +/-0.5 -> A2 working band (SSOT §3).
        #    +/-1.5 ~ where the V5 walker already extrapolates to; for a FRESH run drop to +/-0.5..1.0.
        cmd.ranges.lin_vel_x = [-1.5, 1.5]
        cmd.ranges.lin_vel_y = [-1.0, 1.0]
        cmd.ranges.ang_vel_yaw = [-2.0, 2.0]

        # 2. FAST ramp to the 5 m/s peak (OFFICIAL spec: 0-3.7 sustained, ~5 peak -- SSOT §3), iters
        #    RELATIVE TO V6's OWN run from 0 (clock resets -- docstring). WARM-START pace: the V5 seed
        #    already walks, so ramp early; full +/-5.0 reached by clock-iter 10000 (~33h @ 8192).
        #    PROVENANCE: re-derived for the A2 envelope, NOT go2's +/-2.0. Retune iters to the run length.
        cmd.command_range_curriculum = [
            {'iter': 500,   'lin_vel_x': [-2.5, 2.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            {'iter': 2500,  'lin_vel_x': [-3.5, 3.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 6000,  'lin_vel_x': [-4.5, 4.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 10000, 'lin_vel_x': [-5.0, 5.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
        ]

        # 3. Flat = the sprint axis -> full envelope. Non-flat caps stay conservative (you can't sprint
        #    up 0.24 m stairs); they remain go2-inherited PENDING a per-terrain A2 review (SSOT §4).
        cmd.terrain_max_command_ranges = {
            **cmd.terrain_max_command_ranges,
            'flat': {'lin_vel_x': [-5.0, 5.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
        }

        # 4. dynamic-sigma: widen the tracking-tolerance band to the full A2 speed range so high-speed
        #    commands are not punished by a go2-narrow sigma. PROVENANCE: TRANSLATED v_max 1.5 -> 5.0.
        self.rewards.track_lin_vel_xy_exp.params["v_max"] = 5.0

        # 5. base-mass DR: fraction-match go2 (+/-1.0/16.09 = +/-6.2%) on the A2 (x41.55 ~ +/-2.6 kg).
        #    PROVENANCE: TRANSLATED from go2 +/-1.0 kg absolute (was +/-2.4% on the A2 -- under-randomized).
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-2.6, 2.6)

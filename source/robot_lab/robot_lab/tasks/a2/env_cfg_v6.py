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

!! LAUNCH-COUPLED (read before running) -- the command_range_curriculum `iter` milestones depend on
   the launch plan. rsl_rl's step counter CONTINUES on resume, so a WARM-START from V5's ~10k walker
   means thresholds must be > 10000. The schedule below assumes the RECOMMENDED plan:
       WARM-START from V5's ~10k walker, ramping to +/-5.0 by ~iter 42k (+32k of training; "not in a rush").
   For a FRESH-from-scratch run instead: lower `ranges.lin_vel_x` to +/-0.5..1.0 (let it learn to walk
   slow first) and shift the curriculum to start later. Retune the iters to the chosen run length --
   do NOT run as-is under a different plan.
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

        # 1. INITIAL range -- binds on any run that hasn't reached the curriculum yet.
        #    PROVENANCE: TRANSLATED from go2 ranges.lin_vel_x +/-0.5 -> A2 working band (SSOT §3).
        #    +/-2.0 assumes a WARM-STARTED walker; for a FRESH run drop to +/-0.5..1.0.
        cmd.ranges.lin_vel_x = [-2.0, 2.0]
        cmd.ranges.lin_vel_y = [-1.0, 1.0]
        cmd.ranges.ang_vel_yaw = [-2.0, 2.0]

        # 2. Graded ramp to the 5 m/s bar. PROVENANCE: re-derived for the A2 envelope (SSOT §3).
        #    !! iters assume WARM-START from ~10k (counter continues) -- see module docstring to retune.
        cmd.command_range_curriculum = [
            {'iter': 18000, 'lin_vel_x': [-3.0, 3.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            {'iter': 28000, 'lin_vel_x': [-4.0, 4.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 42000, 'lin_vel_x': [-5.0, 5.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.5, 2.5]},
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

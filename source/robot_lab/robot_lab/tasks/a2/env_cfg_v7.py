# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V7 env cfg — ROBUSTNESS first (info/locomotion/V7_ROBUSTNESS_RECIPE.md).

Redirected from V6's forward-speed sprint to balance / landing / turning / agility, after driving the
iter-7000 V5 student exposed those gaps (and the deterministic forward probe was blind to them — see
[[feedback-call-off-failing-runs]]). Inherits V5's A3 perceptive-student sensorium byte-for-byte;
the deltas are the speed needed to CLIMB the terrain curriculum + the rewards/events/terrains that
force balance.

THE PIVOT THESIS (V7_ROBUSTNESS_RECIPE §Thesis, verified against the fork here):
  The terrain is NOT the gap — we already train on Wu's rich TERRAIN_CFG (gym-parity, 10 difficulty
  rows, stairs→0.257 m / obstacles→0.275 m / slopes→0.568 / wave). The gap is the policy never
  CLIMBED it. Two clamps stalled the curriculum on the easy rows:
    1. `terrain_levels_vel_gym` promotes a row only if `max_move_dist > terrain_length/2`
       (= 8.0/2 = 4.0 m; curriculums.py:215) per 25 s episode.
    2. `terrain_max_command_ranges` clamps the command PER TERRAIN — on stairs/obstacles/stepping/gap
       the go2 cap is lin_vel_x ±1.0 (commands.py:388-397). So the global curriculum could ramp to
       ±5.0 and the robot on stairs would still be commanded ≤1.0 m/s → slow traversal → never crosses
       4 m on the harder rows → never promoted → never trains hard terrain → no balance.
  V5 compounded this: it ran the whole run at the ±0.5 floor (its curriculum clock resets to 0 on every
  launch — see V6 docstring / commit 070ded5). Baseline scorecard (the "before", /tmp/rg_baseline.json):
  stairs_fd/bd 0.00, wave CRASH, flat 0.11 — exactly the "never climbed + no balance" signature.

  => SPEED IS THE KEY THAT UNLOCKS THE ROBUSTNESS CURRICULUM. V7 raises BOTH the global ranges AND the
     per-terrain caps (moderate — this is traversal, not the 5 m/s sprint) so the robot can cross 4 m
     on the hard rows and climb into the terrain that teaches balance.

DELTA over A2V5EnvCfg (sensorium, CTS/MoE wiring, Gate-T energy weights all inherited byte-for-byte):
  1. Command ranges + per-terrain caps raised to traversal speeds (unlock the terrain gym).
  2. resampling_time 5.0 → 3.0 — more frequent command CHANGES (trains direction transitions, not just
     steady holds; the "not robust to direction" symptom).
  3. Turning: track_ang_vel_z_exp.weight 0.5 → 1.0 (parity with linear tracking).
  4. Balance: ang_vel_xy_l2 -0.05 → -0.1; ADD flat_orientation_l2 (gentle -1.0 — see note); push x/y ±0.4→±0.6.
  5. Landing / clean gait: ADD feet_air_time (IsaacLab quadruped default 0.125 / 0.5 s threshold).
  6. Smoothness: action_rate_l2 / action_smoothness_l2 -0.01 → -0.02.
  7. Terrain: enable the balance terrains stepping_stones (0→0.08) + gap (0→0.05), renormalized.
  8. base-mass DR ±1.0 → ±2.6 kg (the V6-blessed fraction-match; aids mass/push robustness).
  9. dynamic-σ v_max 1.5 → 3.5 (track the V7 top speed, NOT V6's 5.0 sprint).

PROVENANCE (the load-bearing rule — A2_CAPABILITY_SSOT.md §Provenance): every value is re-derived for
the A2 or blessed; no silent go2 transcription. Speeds are set for TRAVERSAL (cross 4 m on hard rows),
keyed to the A2 envelope ~3.6–5.4 m/s (SSOT §3), not copied from go2's ±0.5/±2.0.

!! LAUNCH-COUPLED (inherits the V6 lesson, commit 070ded5): the command_range_curriculum `iter`
   milestones are gated by `env.common_step_counter // num_steps_per_iter` (=24), and that clock
   RESETS TO 0 on every launch/resume (rsl_rl runner.load restores policy+logging iter, NOT the env
   step counter). So the milestones below are RELATIVE TO V7's OWN run from 0, regardless of warm-start.
   Confirm in a smoke: the term prints "Command range updated at iter X" when a stage fires.
   Schedule assumes WARM-START from V5's iter-7000 walker (already walks ±0.5 → ramp gently off that
   floor, NOT a 4× shock). For a FRESH run, lower the initial ranges and stretch the milestones.

RUN LENGTH (set at launch, not here): the 10-row gym curriculum needs real iters to reach the high rows
   — target ≥40–50k, or until terrain levels plateau (V7_ROBUSTNESS_RECIPE §1). V5/V6's 8–10k never
   got there.

DEVIATIONS from the recipe draft, with reasons:
  * resampling jitter: the recipe asked to "widen resampling_time_range". Go2RLGymCommand uses the FIXED
    scalar `resampling_time` in _resample (commands.py:110,139); resampling_time_range is vestigial for
    this term. Setting it would be dead config + touching it needs a fork code change (scope). We get
    more frequent direction changes via the lower scalar (5.0→3.0) instead — same intent, no code risk.
  * flat_orientation_l2 weight: recipe range -1.0..-2.0. Chose the GENTLE end (-1.0) — this term
    penalizes xy projected-gravity, i.e. base tilt, which the robot MUST do on slopes/stairs; a strong
    weight would fight the very terrain-climbing V7 is trying to enable. -1.0 nudges flatness on flat
    ground without forbidding necessary tilt. The baseline scorecard refines it.
"""
from __future__ import annotations

import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg import FOOT_LINK_NAME, RewardsCfg
from robot_lab.tasks.a2.env_cfg_v5 import A2V5EnvCfg


@configclass
class A2V7RewardsCfg(RewardsCfg):
    """V5/base rewards + the two NEW robustness terms (declared as fields → guaranteed registration).

    Weight TWEAKS to existing terms (track_ang_vel_z_exp, ang_vel_xy_l2, action_rate/smoothness) are
    applied in A2V7EnvCfg.__post_init__ (mutating the inherited fields), to keep the deltas in one place.
    """

    # ADD: penalize non-flat base orientation (xy projected gravity). GENTLE weight — see module docstring
    # (a strong flat-orientation penalty fights slope/stair climbing). PROVENANCE: recipe §3, gentle end.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # ADD: reward proper swing time → cleaner steps + landings. IsaacLab-canonical quadruped default
    # (weight 0.125, threshold 0.5 s). Uses the contact_forces sensor (track_air_time=True). recipe §4.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "threshold": 0.5,
        },
    )


@configclass
class A2V7EnvCfg(A2V5EnvCfg):
    """A2 V5 perceptive-student sensorium + the robustness-first deltas (V7). See module docstring."""

    rewards: A2V7RewardsCfg = A2V7RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # 1. UNLOCK THE TERRAIN GYM — global ranges + per-terrain caps to traversal speeds.
        # ------------------------------------------------------------------ #
        cmd = self.commands.base_velocity

        # Initial range — binds from iter 0 (curriculum clock resets, docstring). Gentle step off the V5
        # ±0.5 walker (avoid a 4× warm-start shock); yaw generous from the start (turning is a V7 goal).
        cmd.ranges.lin_vel_x = [-1.0, 1.0]
        cmd.ranges.lin_vel_y = [-0.6, 0.6]
        cmd.ranges.ang_vel_yaw = [-2.0, 2.0]

        # Moderate ramp to ±3.5 (traversal, NOT V6's 5 m/s sprint). Iters RELATIVE to V7's own run from 0.
        # PROVENANCE: A2 envelope ~3.6–5.4 (SSOT §3) — set so the robot crosses 4 m on hard rows, not for peak.
        cmd.command_range_curriculum = [
            {'iter': 500,   'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-0.6, 0.6], 'ang_vel_yaw': [-2.0, 2.0]},
            {'iter': 3000,  'lin_vel_x': [-2.5, 2.5], 'lin_vel_y': [-0.6, 0.6], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 8000,  'lin_vel_x': [-3.0, 3.0], 'lin_vel_y': [-0.6, 0.6], 'ang_vel_yaw': [-2.5, 2.5]},
            {'iter': 15000, 'lin_vel_x': [-3.5, 3.5], 'lin_vel_y': [-0.6, 0.6], 'ang_vel_yaw': [-2.5, 2.5]},
        ]

        # THE ACTUAL UNLOCK: raise the PER-TERRAIN caps. The global ramp above is clamped per terrain by
        # this dict (commands.py:_update_env_command_ranges). go2 capped hard terrain at ±1.0 → the robot
        # could never be commanded fast enough to cross 4 m on stairs → curriculum stalled. Moderate raises
        # below let it traverse + climb. Non-listed (robotlab-default) terrains keep their inherited caps.
        cmd.terrain_max_command_ranges = {
            **cmd.terrain_max_command_ranges,
            'flat':            {'lin_vel_x': [-2.5, 2.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'slope_up':        {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'slope_down':      {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'rough_slope':     {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'wave':            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'stairs_up':       {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'stairs_down':     {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'obstacles':       {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
            'stepping_stones': {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-0.8, 0.8], 'ang_vel_yaw': [-1.5, 1.5]},
            'gap':             {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-0.8, 0.8], 'ang_vel_yaw': [-1.5, 1.5]},
        }

        # 2. More frequent command CHANGES (direction transitions, not steady holds). Couples (benignly)
        #    into the curriculum target_dist (curriculums.py:216) — lower → gentler move_down threshold.
        cmd.resampling_time = 3.0

        # ------------------------------------------------------------------ #
        # 3/4/5/6. REWARD TWEAKS to inherited terms (the two NEW terms are fields on A2V7RewardsCfg).
        # ------------------------------------------------------------------ #
        # Turning to parity with linear tracking.
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # dynamic-σ tolerance tracks the V7 top speed (NOT V6's 5.0 sprint). PROVENANCE: V7 ramp top ±3.5.
        self.rewards.track_lin_vel_xy_exp.params["v_max"] = 3.5
        # Balance: stronger roll/pitch-rate penalty.
        self.rewards.ang_vel_xy_l2.weight = -0.1
        # Smoothness: firmer action-rate / action-smoothness.
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness_l2.weight = -0.02

        # ------------------------------------------------------------------ #
        # 4 (cont.) / 8. EVENTS — push-recovery + corrected mass DR.
        # ------------------------------------------------------------------ #
        # Stronger push-recovery on the linear axes (roll/pitch/yaw already ±0.6 in base EventCfg).
        push_vr = self.events.randomize_push_robot.params["velocity_range"]
        push_vr["x"] = (-0.6, 0.6)
        push_vr["y"] = (-0.6, 0.6)
        # base-mass DR: V6-blessed fraction-match of go2 (±1.0/16.09 = ±6.2%) on the A2 (×41.55 ≈ ±2.6 kg).
        # PROVENANCE: A2_CAPABILITY_SSOT §4 (go2 ±1.0 kg was only ±2.4% on the A2 — under-randomized).
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-2.6, 2.6)

        # ------------------------------------------------------------------ #
        # 7. TERRAIN — enable the balance terrains, renormalized to sum 1.0.
        #    Deep-copy first so we never mutate the shared module-level TERRAIN_CFG singleton (would leak
        #    into V5/V6/go2 within a process). configclass already deep-copies field defaults, but this is
        #    the explicit, audit-proof guarantee.
        # ------------------------------------------------------------------ #
        tg = self.scene.terrain.terrain_generator
        tg = copy.deepcopy(tg)
        self.scene.terrain.terrain_generator = tg
        sub = tg.sub_terrains
        # Enable the foot-placement / balance terrains (were proportion 0.0 = OFF).
        sub["stepping_stones"].proportion = 0.08
        sub["gap"].proportion = 0.05
        # Renormalize: trim flat (0.15→0.10) + stairs_up (0.25→0.17) by the +0.13 we added. Sum stays 1.00:
        #   wave .05 + slope_up .10 + slope_down .10 + rough_slope .05 + stairs_up .17 + stairs_down .10
        #   + obstacles .20 + stepping_stones .08 + gap .05 + flat .10 = 1.00
        sub["flat"].proportion = 0.10
        sub["stairs_up"].proportion = 0.17

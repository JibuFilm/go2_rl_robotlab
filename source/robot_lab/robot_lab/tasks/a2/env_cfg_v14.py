# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V14 — the CLEAN-SLATE capability-matched config (fresh weights, no warm-start).

WHY (2026-06-21): an audit of the whole A2 port found the lineage was a go2-defaults shell corrected
only by an 8-deep override stack, plus two real go2-copy bugs that capped the robot below its real
envelope. This config re-bases ALL of our genuine reward-shaping fixes onto a CORRECTED physical
foundation, and is meant to be trained from FRESH weights (the old weights were trained on the flawed
foundation). The recipe is kept; only the weights restart.

CORRECTED AT THE SOURCE (baked into the base, not here — so a short run can't miss an override):
  * STANCE — unitree.py A2_CFG_UNITREE: thigh 0.9/calf -1.8 (over-folded, ~0.44 m physical, 62% reach)
    -> thigh 0.8/calf -1.05 (URDF FK: base_link ~0.474 m -> physical ~0.566 m == the real A2's ~0.57 m
    operating height; ~81% reach). init z 0.45 -> 0.55.
  * BASE_HEIGHT_TARGET (env_cfg.py): 0.40 -> 0.47 (matches the corrected stance; was inconsistent with
    the -1.8 calf, which only reaches ~0.35 m). The two MUST move together (base_height_l2 -> -10).
  * base-mass DR (env_cfg.py): ±1.0 kg (go2-absolute, ±2.4% on the A2) -> ±2.6 kg (fraction-matched).

APPLIED HERE (the clean-slate command/terrain/reward corrections):
  1. Wu's PER-TERRAIN COMMAND CAPS, A2-scaled. Our V7 clobbered Wu's terrain-graded caps to a uniform
     ±5.0 on every terrain — so the robot was commanded 5 m/s INTO staircases, which is what made the
     dynamic-σ setpoint tracker look broken (V8 then replaced it with the dial). Wu's caps are his own
     mechanism (commands.py, wertyuilife) and bind per-env as intersection(global_curriculum, terrain
     cap). Restored, scaled to the A2: flat ±5.0, slopes/rough ±2.5, stairs/obstacles/gap ±1.5.
  1b. SPEED REWARD REWORKED. dynamic-σ is NOT Wu's thesis (his is MoE-CTS; his IsaacLab port used FIXED
     σ; dynamic-σ is a legged_gym detail we restored). The reckless-overspeed failure that made us
     replace the velocity tracker with the dial (V8) was caused by OUR uniform ±5 caps (item 1), NOT the
     tracker — with Wu's caps restored, the tracker is commanded a sane ±1.5 on stairs and never lunges.
     So the velocity tracker is RE-PROMOTED to primary (the proven approach that yields clean speed-
     matched gaits + handles lateral), and the dial is DEMOTED to a small additive terrain-tangent
     CLIMB-CREDIT bonus (its one genuine merit: it credits the vertical climb motion the planar tracker
     structurally can't see). track_lin_vel_xy_exp 0.0 -> 1.0; track_lin_vel_dial 0.35 -> 0.1 (kept
     small — note in code: the dial scales with commanded speed, so it's a general progress nudge, not
     a climb-ONLY term; a clean climb-isolated bonus is a future refinement).
  2. FRESH-WALKER command curriculum. V8's schedule (init ±2.0; milestones 15k/28k/42k) is warm-start-
     coupled. A fresh walker starts at ±0.5 and ramps to the A2's ±5.0 peak over ~35k.
  3. ENERGY weights restored to FULL Gate-T candidates. V12 HALVED joint_torques_l2/joint_power as a
     bravery band-aid on top of unvalidated candidates; the clean slate uses the full translated values
     (re-anchor via the Gate-T step-2 a2 smoke is still owed — flagged, not silently inherited halved).
  4. undesired_contacts: go2's 5 N trip-wire (absolute, not mass-scaled) penalized the exact leg-on-
     step contact a climb NEEDS. Raised to 50 N and scoped to the thigh only (the calf rests on steps
     freely; a hard thigh slam is still discouraged).
  5. TERRAIN to the A2's real envelope. 0.30 m is the COMFORTABLE stair, not the ceiling (obstacle climb
     spec is 0.5-1 m). Stairs 0.05-0.28 -> 0.05-0.35 (top curriculum row ~0.32 m, past comfortable);
     discrete obstacles 0.05-0.30 -> 0.05-0.60 (toward the single-clamber climb envelope; the terrain
     curriculum self-caps at what's achievable, so a high ceiling just exposes the headroom — a literal
     1 m single wall is not clamber-able by a 0.57 m robot, so 0.60 is the realistic stretch); slopes
     0.62 -> 0.84 tan (~40°, toward the 45° spec). Done on a DEEP-COPY so Wu's shared TERRAIN_CFG
     singleton (used by go2) is never mutated.

INHERITED, UNCHANGED (our genuine fixes — all carry forward; posture auto-retargets to the corrected
default stance): goal-driven terrain-tangent dial (V10), goal-relaxed style priors (V10), the bravery
arm (V12: backtrack, termination grace, recover_and_progress, recovery-state relaxation, bad-init
spawns), the stand-still posture penalties (V13 — now pulling to the CORRECT taller default), and the
gate-selection tuning (z_loss 3e-4 / load_balance 0.002).
"""
from __future__ import annotations

import copy

from isaaclab.utils import configclass

from robot_lab.tasks.a2.env_cfg_v13 import A2V13EnvCfg


def _cap(vx: float, vy: float = 1.0, wz: float = 2.5) -> dict:
    return {"lin_vel_x": [-vx, vx], "lin_vel_y": [-vy, vy], "ang_vel_yaw": [-wz, wz]}


# Per-terrain cap classes (Wu's terrain-graded discipline, A2-scaled). Names cover BOTH the go2-terrain
# and robotlab-default keysets so whatever terrain_type2idx contains is covered (the cap consumer
# raises if a present terrain type is missing).
_SLOW = {"stairs_up", "stairs_down", "pyramid_stairs", "pyramid_stairs_inv",
         "obstacles", "stepping_stones", "gap", "boxes"}          # step/clamber: ±1.5
_MID = {"slope_up", "slope_down", "rough_slope", "wave", "random_rough",
        "hf_pyramid_slope", "hf_pyramid_slope_inv"}               # slopes/rough: ±2.5
# everything else (flat, ...) -> ±5.0


@configclass
class A2V14EnvCfg(A2V13EnvCfg):
    """Clean-slate capability-matched A2 (fresh-train). See module docstring."""

    def __post_init__(self):
        super().__post_init__()  # all inherited fixes (dial, goal-relax, bravery, posture, gate-tune)

        cmd = self.commands.base_velocity

        # --- (1) Wu's per-terrain caps, A2-scaled. REPLACE V7's uniform ±5.0, preserving the keyset
        # (the cap consumer ValueErrors on any present-but-unmapped terrain).
        cmd.terrain_max_command_ranges = {
            k: (_cap(1.5) if k in _SLOW else _cap(2.5) if k in _MID else _cap(5.0))
            for k in cmd.terrain_max_command_ranges
        }

        # --- (1b) SPEED-REWARD REWORK. Re-promote the dynamic-σ velocity tracker to PRIMARY (the proven
        # go2_rl_gym mechanism, clean speed-matched gaits, penalizes lateral too); DEMOTE the dial to a
        # small additive terrain-tangent climb-credit bonus (its one merit the planar tracker lacks). The
        # V8 swap was a misdiagnosis — the reckless-overspeed was OUR uniform ±5 caps, now fixed in (1).
        self.rewards.track_lin_vel_xy_exp.weight = 1.0   # was 0.0 (V8 disabled); v_max 5.0 baked in base
        # CAVEAT (honest): the dial's value is a clamped velocity (~[0,|cmd|]), so it scales with
        # commanded SPEED while the tracker is bounded [0,1]. A fixed weight thus makes it a general
        # PROGRESS nudge (largest on flat-fast), not a clean climb-ONLY bonus — its terrain-tangent
        # crediting still gives the stairs-climb credit the planar tracker lacks, just not isolated.
        # Kept small (0.1) so it never competes with the tracker for gait shape; smoke-tunable. A truly
        # climb-isolated term (credit only the v_z·slope component) is a future refinement if needed.
        self.rewards.track_lin_vel_dial.weight = 0.1     # was 0.35 (primary) -> small progress/climb bonus

        # --- (2) fresh-walker command curriculum: start ±0.5, ramp to the A2's ±5.0 peak over ~35k.
        cmd.ranges.lin_vel_x = [-0.5, 0.5]
        cmd.ranges.lin_vel_y = [-0.5, 0.5]
        cmd.ranges.ang_vel_yaw = [-1.0, 1.0]
        cmd.command_range_curriculum = [
            {"iter":  4000, "lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-0.6, 0.6], "ang_vel_yaw": [-1.5, 1.5]},
            {"iter": 10000, "lin_vel_x": [-2.5, 2.5], "lin_vel_y": [-0.8, 0.8], "ang_vel_yaw": [-2.0, 2.0]},
            {"iter": 18000, "lin_vel_x": [-3.5, 3.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.5, 2.5]},
            {"iter": 26000, "lin_vel_x": [-4.5, 4.5], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.5, 2.5]},
            {"iter": 34000, "lin_vel_x": [-5.0, 5.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_yaw": [-2.5, 2.5]},
        ]

        # --- (3) restore the FULL Gate-T energy candidates (undo V12's bravery-arm halving).
        self.rewards.joint_torques_l2.weight = -5.571765636567876e-06
        self.rewards.joint_power.weight = -7.389186728023717e-06

        # --- (4) un-trip-wire the climbing contact: 5 N -> 50 N, thigh-only (calf rests on steps freely).
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.undesired_contacts.params["threshold"] = 50.0

        # --- (5) terrain to the A2's real envelope, on a DEEP-COPY (never mutate Wu's shared singleton).
        tg = copy.deepcopy(self.scene.terrain.terrain_generator)
        sub = tg.sub_terrains
        if "stairs_up" in sub:
            sub["stairs_up"].step_height_range = (0.05, 0.35)     # top row ~0.32 m (past 0.30 comfortable)
        if "stairs_down" in sub:
            sub["stairs_down"].step_height_range = (0.05, 0.35)
        if "obstacles" in sub:
            sub["obstacles"].obstacle_height_range = (0.05, 0.60)  # toward the single-clamber climb envelope
        for k in ("slope_up", "slope_down"):
            if k in sub:
                sub[k].slope_range = (0.1, 0.84)                   # ~40 deg top, toward the 45 deg spec
        self.scene.terrain.terrain_generator = tg

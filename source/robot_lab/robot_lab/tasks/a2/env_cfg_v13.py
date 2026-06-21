# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V13 env cfg - V12 bravery arm + a STAND-STILL POSTURE fix.

Why this exists (stacks on V12, drops nothing):
  V12's bravery arm climbs better (cleared ~0.28 in sim), but the policy's REST pose is a learned
  collapse: body sagged below the 0.40 nominal stand, hips splayed wide, feet tucked under. That
  shape is the energy-minimal, fall-safest configuration — and at rest NOTHING in the reward fought
  it:
    * the energy terms (joint_power, joint_torques_l2) PAY for using less torque, and a splayed
      squat holds itself with less effort than a tall stand;
    * the risk-averse MoE ("honor-guard", gate still averaging) prefers the widest, lowest base;
    * the two posture penalties that SHOULD hold the default stand were toothless when stationary —
      hip_pos_penalty_l1 (w -0.05) and joint_pos_penalty_l1 (w -0.01) both had stand_still_scale=1.0,
      i.e. NO extra enforcement at standstill.
  base_height_l2 (-10) holds HEIGHT but never sees the splay / leg-tuck (that's joint posture).

The fix (V13 adds, command-gated so it CANNOT fight the climb):
  Both posture penalties multiply by stand_still_scale ONLY when cmd<thr AND body_vel<thr
  (rewards.py: torch.where(cmd_large, running_reward, stand_still_scale * running_reward)). So raising
  the scale snaps the joints back to default_joint_pos (hip +/-0.1, thigh 0.9, calf -1.8 = the clean
  0.40 stand) AT REST, while the moving/​climbing weight is left ~unchanged — the gate is open during
  locomotion, so the big thigh/calf excursions stairs need are untouched.
    P1 - hip splay (primary; hip is ~constant in a trot, so a heavier hip weight is safe even while
         moving). hip_pos_penalty_l1: weight -0.05 -> -0.15, stand_still_scale 1.0 -> 6.0.
         GATING NUANCE (verified in rewards.py:hip_pos_penalty_l1): this term's stand-still branch
         keys on the command's [vy, wz] components, NOT vx. So a straight-line forward trot/climb
         (vx>0, vy=wz=0) leaves the hip gate CLOSED -> the 6.0x scale DOES apply while running
         straight. That is intended here: hips SHOULD stay home (no splay) running straight, and
         forward stair-climbing needs thigh/calf excursion, not hip abduction, so this can't fight
         the climb. The gate OPENS for turns/strafes (vy/wz commanded), where hip motion is wanted
         and only the -0.15 base weight acts.
    P2 - leg tuck/sag at rest. joint_pos_penalty_l1 (thigh/calf): weight -0.01 -> -0.03 (kept small so
         the MOVING penalty barely changes), stand_still_scale 1.0 -> 6.0 (the strong pull is rest-only).
  TUNING NOTE: these are a first cut. If the V13 read still sags, raise the scales (rest-only, free of
  climb cost) before the base weights. The one term to WATCH for a climb regression is P2's base
  weight (-0.03): it acts during motion too; if the 0.28 climb regresses, drop it back toward -0.01
  and lean entirely on the stand_still_scale.

Gate-side lever (stacks the V12 conditional, does NOT replace it):
  V12 lowered load_balance 0.02 -> 0.005 to let the differentiated experts be SELECTED. Through ~92%
  of V12 the gate entropy is still pinned at max (ln 8 = 2.079) — i.e. selection has NOT emerged at
  0.005. V13 takes the documented next step: load_balance 0.005 -> 0.002 (A2V13MoECTSRunnerCfg).
  CONDITIONAL: if V12's FINAL checkpoint shows entropy dropping below max (selection did emerge in the
  last 8%), revert that one line and keep 0.005. z_loss stays 3e-4. B1/B2 (logit temperature, entropy
  floor) remain consciously OMITTED — they re-force uniformity.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from robot_lab.tasks.a2.env_cfg_v12 import A2V12EnvCfg


@configclass
class A2V13EnvCfg(A2V12EnvCfg):
    """V12 bravery arm + the stand-still posture fix (P1 hip splay, P2 leg tuck at rest)."""

    def __post_init__(self):
        super().__post_init__()

        # --- P1: kill the hip splay. Heavier base weight (safe — hip is ~constant in a trot) plus a
        # strong stand-still multiplier so the hips snap home to the +/-0.1 default when stationary.
        self.rewards.hip_pos_penalty_l1.weight = -0.15
        self.rewards.hip_pos_penalty_l1.params["stand_still_scale"] = 6.0

        # --- P2: lift the leg tuck/sag at rest. Keep the MOVING weight small (-0.03) so stair
        # articulation is barely touched; the real correction is the stand-still multiplier, which
        # only fires when cmd<thr AND body_vel<thr (pure rest), pulling thigh/calf to the 0.9/-1.8
        # default = the clean 0.40 stand.
        self.rewards.joint_pos_penalty_l1.weight = -0.03
        self.rewards.joint_pos_penalty_l1.params["stand_still_scale"] = 6.0

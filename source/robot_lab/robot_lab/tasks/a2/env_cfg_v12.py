# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V12 env cfg - the BRAVERY ARM on top of V10.

Why this exists:
  V10 made the speed goal a terrain-aware dial and let flat-ground style priors yield on commanded
  relief. But the policy still has a "retreat from hard spots" bias: a stumble is all downside (style
  penalties slam back, base contact ends the episode), so the safe move is to back off the goal. V12
  is a coordinated set of small changes that re-price a SAVE as something worth attempting — making
  bravery (push through a recoverable stumble toward the goal) the locally-rewarded choice, without
  loosening any true safety bound.

Each fix (the "bravery arm"):
  A1 - backtrack penalty. ``track_lin_vel_terrain_dial`` now charges a small cost for commanded-
       direction backtracking (``backtrack_cap=0.5``), so retreating from a hard spot is no longer
       free. Default behavior (cap 0.0) is unchanged elsewhere.
  A2 - termination grace + drag penalty. The base no longer terminates on first contact (base env
       TerminationsCfg swaps illegal_contact -> bad_orientation at limit_angle 1.4 rad); a brief base
       scrape is survivable, only a real flip ends the episode. To keep base-DRAGGING discouraged
       (no crawl-on-belly gait), this cfg adds a modest per-step base-contact PENALTY reward term.
  A3 - recovery reward. A new positive term (``recover_and_progress``) pays the SAME bounded forward
       progress as the dial, but only while the recovery gate is high — crediting the robot for
       driving toward the goal THROUGH a stumble. Bounded by the dial cap, paid only for actual
       progress, so not farmable.
  A4 - recovery-aware relaxation. The four goal-relaxed style priors (base height, flat orientation,
       vertical velocity, foot regulation) now also relax on the robot's OWN recovery state (tilt /
       low base / base contact), not just on sensed terrain relief — so style penalties yield during
       a mid-stumble save even on flat ground. Threaded via ``recovery=True`` + ``recovery_cfg``.
  A5 - bad-initiation spawns. ``events.reset_base`` is widened so the robot sometimes starts in an
       awkward/imbalanced pose (larger roll/pitch, full-circle heading error, a small z drop, modest
       velocity) — teaching "this is recoverable, not a reason to retreat," not "always start clean."
  D3 - cool the unvalidated energy weights. ``joint_torques_l2`` and ``joint_power`` are halved; they
       are unvalidated Gate-T CANDIDATEs and, at full strength, over-penalize the corrective motion a
       save needs. ``joint_acc_l2`` is left as-is.

CONSCIOUSLY OMITTED (NOT forgotten):
  B1 - gate logit temperature, and B2 - gate entropy floor. Both push the MoE gate back toward
       UNIFORM routing. That directly fights V12's goal of letting the now-differentiated experts be
       SELECTED (a specialist for the save), so they are intentionally left out here. The gate-side
       lever V12 does pull is the algorithm cfg (A2V12MoECTSRunnerCfg): LOWER z_loss / load-balance
       coefficients so selection is allowed, not averaged.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg import BASE_LINK_NAME, BASE_HEIGHT_TARGET
from robot_lab.tasks.a2.env_cfg_v10 import A2V10EnvCfg


@configclass
class A2V12EnvCfg(A2V10EnvCfg):
    """V10 terrain dial + the bravery arm (A1-A5, D3); B1/B2 consciously omitted (see module docstring)."""

    def __post_init__(self):
        super().__post_init__()

        # Shared recovery descriptor for the A4 relaxation drivers and the A3 recovery reward. Keys
        # the robot's own recovery state: tilt (uprightness), low base height vs terrain, base contact.
        recovery_cfg = {
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "base_sensor_cfg": SceneEntityCfg("height_scanner_small"),
            "base_target": BASE_HEIGHT_TARGET,  # tracks the corrected stance (0.47), not a hardcoded 0.40
        }

        # --- A1: backtrack penalty on the terrain dial (V10 already swapped this term to the terrain
        # dial func). 0.5 lets commanded-direction retreat go mildly negative instead of being free.
        self.rewards.track_lin_vel_dial.params["backtrack_cap"] = 0.5

        # --- A4: make the four goal-relaxed style priors ALSO relax on the robot's own recovery
        # state, so style penalties yield during a save even when the terrain sensor reads no relief.
        for term_name in ("base_height_l2", "flat_orientation_l2", "lin_vel_z_l2", "feet_regulation"):
            getattr(self.rewards, term_name).params.update(
                {"recovery": True, "recovery_cfg": recovery_cfg}
            )

        # --- A3: recovery reward — bounded forward progress paid only inside the save window.
        self.rewards.recover_progress = RewTerm(
            func=mdp.recover_and_progress,
            weight=0.15,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("height_scanner_small"),
                "recovery_cfg": recovery_cfg,
            },
        )

        # --- A2: base contact is now survivable (base env termination swap), so keep base-DRAGGING
        # discouraged with a modest per-step base-contact penalty (mirrors the thigh/calf
        # undesired_contacts pattern, scoped to the base link).
        self.rewards.base_contact_pen = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
                # THRESHOLD TAILORED: 5 N was a small-robot trip-wire. On the A2's ~408 N body a 5 N
                # threshold fires on feather-touch and over-suppresses the SURVIVABLE base-scrape the
                # termination-grace is meant to ALLOW. 30 N charges real belly-DRAGGING (sustained load),
                # not incidental contact — mass-scaled, mirroring the thigh undesired_contacts fix.
                "threshold": 30.0,
            },
        )

        # --- A5: bad-initiation spawns — sometimes start awkward/imbalanced so a stumble at reset is
        # learned as recoverable. Widen pose (roll/pitch tilt, full-circle heading, small z drop) and
        # keep velocity modest. Override here only; the base env reset stays clean.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.3),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # --- D3: cool the unvalidated Gate-T CANDIDATE energy weights (they over-penalize the
        # corrective motion a save needs). Halve torques + power; leave joint_acc_l2 untouched.
        self.rewards.joint_torques_l2.weight *= 0.5
        self.rewards.joint_power.weight *= 0.5

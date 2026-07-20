# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
# A2Z1-L1 — the ARMED WALKER env (W2/W3-L1, locomotion strand).
#
# = A2V16EnvCfg (the full env/reward moat, UNCHANGED — same 12 leg actions, same obs
# contract 2785/263/557, same rewards/curriculum/terrain) with the robot swapped for the
# armed body (A2Z1_CFG_UNITREE: sensorized A2 + Z1 arm servo-held at stow) and the arm's
# domain randomization added. The policy learns the arm as PHYSICS (mass, momentum, load
# moments), never as action — W3-L2 widens the action space later; this cfg does not.
#
# Recon-verified invariants this file RELIES on (2026-07-18, see PerceptionGame
# info/locomotion/W2 docs): every obs/action/reward term is name-pinned to the 12 leg
# joints (arm joints leak nowhere); height scanners + dome RayCasters cast only against
# /World/ground (the arm is structurally invisible to them); contact/reward body regexes
# (.*_foot, .*_thigh, base_link) cannot match z1_* names (audited at URDF generation).
#
# Dome self-occlusion: this cfg points the kernel at the ARMED occluder set
# (data/a3_body_geoms_a2z1_v1.json — parity-debt item 1 CLOSED 2026-07-18; behaviorally
# validated: pose sweep +22 arm rays, mj_ray corroborates 7/7). Design fact from that
# validation: the mid-roof mount sits largely in the two domes' MUTUAL BLIND WEDGE (each
# 96° cone looks away from the roof centre), so per-pose arm visibility is ~3-7 rays —
# physically correct for a 3-5 cm limb on a 16×16 pattern; train==deploy share the wedge
# by construction (the pattern contract IS the deploy pattern).

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.assets.unitree import A2Z1_CFG_UNITREE
from robot_lab.tasks.a2.env_cfg import JOINT_NAMES
from robot_lab.tasks.a2.env_cfg_v16 import A2V16EnvCfg

# Arm joint selectors (URDF names audited against every leg regex at generation time).
ARM_JOINT_EXPR = ["z1_joint[1-6]"]          # the 6 arm DOF; gripper excluded from pose DR
ARM_ALL_JOINT_EXPR = ["z1_joint[1-6]", "z1_jointGripper"]   # all 7 (target-hold set)
ARM_LINK_EXPR = "z1_.*"                      # all 8 arm links


def hold_arm_position_targets(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Write the arm joints' position TARGETS to their current (post-jitter) positions.

    THE fix for the review-found blocker (2026-07-18): IsaacLab's ``joint_pos_target``
    buffer is written ONLY by action terms — and ours is pinned to the 12 legs — so
    without this the arm's kp-1000 servos drive from stow to the ALL-ZERO pose (which
    sits ON joint2/joint3's limits) within the first fraction of every episode.
    ``reset_joints_by_offset`` writes joint STATE, never targets. Runs as a reset event
    AFTER the arm-pose jitter (dict order); targets persist for the whole episode
    because nothing else writes the arm's target rows.
    """
    asset = env.scene[asset_cfg.name]
    targets = asset.data.joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


@configclass
class A2Z1L1EnvCfg(A2V16EnvCfg):
    """V16 moat + armed body + arm DR. Action/obs contracts unchanged (12 legs)."""

    def __post_init__(self):
        super().__post_init__()

        # ---- the armed body ----
        self.scene.robot = A2Z1_CFG_UNITREE.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- armed OCCLUDER set for the dome self-occlusion kernel (W2 parity-debt item 1,
        # closed 2026-07-18): the a2z1-keyed geoms file (base 40 + all 23 Z1 collision geoms;
        # gripper mesh hulls as AABB boxes) so the domes SEE the rear-mounted arm exactly as
        # the deploy sensor will. Emitted by PerceptionGame a3_pattern.py --emit --fork
        # --robot a2z1; the armless tasks keep a3_body_geoms_v1.json untouched.
        _a2z1_geoms = str(Path(__file__).resolve().parent / "data" / "a3_body_geoms_a2z1_v1.json")
        self.observations.policy.dome_ranges.params = {"body_geoms_json": _a2z1_geoms}
        self.observations.single_obs.dome_ranges.params = {"body_geoms_json": _a2z1_geoms}

        # ---- THE one mandatory inherited-event fix (recon 2026-07-18): the base cfg's
        # reset_robot_joints passes NO asset_cfg -> IsaacLab defaults to ALL joints ->
        # the arm's stow pose would be scale-randomized 0.5–1.5x every reset. Pin it to
        # the legs; arm-pose DR is its own deliberate term below.
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=JOINT_NAMES, preserve_order=True
        )

        # ---- arm-pose DR: jitter the servo-held stow at reset (W2-09). Offsets, not
        # scales — half the stow angles are 0 and scale-DR would never move them.
        # ±0.15 rad keeps the pose family stow-like; wider pose work is W3-L2's.
        # (Known asymmetry: reset_joints_by_offset clamps to SOFT limits, so joint3 —
        # stow −0.261, soft upper −0.144 — samples +0.117 max instead of +0.15.)
        self.events.reset_arm_pose = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_EXPR),
                "position_range": (-0.15, 0.15),
                "velocity_range": (0.0, 0.0),
            },
        )

        # ---- arm target hold (MUST follow reset_arm_pose — dict order is call order):
        # writes the servo TARGETS to the jittered pose; see hold_arm_position_targets.
        self.events.hold_arm_targets = EventTerm(
            func=hold_arm_position_targets,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_ALL_JOINT_EXPR),
            },
        )

        # ---- EE payload DR (W2-09/19/21): mass added AT THE GRIPPER (~0.55 m lever from
        # the mount at stow) so the policy learns to carry a sustained, un-sensed load
        # moment — the bench's measured mortality driver (ARM_PLAYABLE_PASS_BRIEF §4.3/4.5:
        # transitions under load; compensation must be LEARNED, never commanded). Range
        # tops at the bench's measured working class (0.25–0.5 kg props).
        self.events.randomize_arm_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="z1_gripperMover"),
                "mass_distribution_params": (0.0, 0.5),
                "operation": "add",
                "recompute_inertia": True,
            },
        )

        # ---- protect-the-arm contact penalty (deliberate addition, recipe-optional):
        # V14's undesired_contacts covers only .*_thigh; arm-ground/arm-trunk contact is
        # otherwise free. Mild penalty — falls already terminate via bad_orientation.
        self.rewards.arm_undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_LINK_EXPR),
                "threshold": 30.0,
            },
        )

        # ---- L1 PHASE-IN deltas (r1 post-mortem, 2026-07-19 — run a2z1-l1-r1 collapsed into
        # the die-fast basin: 100% base_fall at ~19 steps, ep length SHRINKING, per-episode
        # reward improving by dying sooner. The armless V17 smoke entered the same basin at
        # iters 49-98 and escaped by 147; the armed body could not. Paired MuJoCo A/B
        # (64 spawns, servo-hold): the V12-A5 spawn severity dooms ~60% of episodes AT BIRTH
        # for BOTH bodies — those spawns were designed to teach recovery to a COMPETENT
        # V12 walker, not for from-scratch armed learning.)
        # (1) moderate bad-init spawns: survivable variety, not recovery training. The full
        #     A5 severity returns with the W3 get-up teacher, where it belongs.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.15, 0.15),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        }
        # (2) termination penalty: one-time −5 on death (is_terminated excludes time-outs).
        #     ⚠ isaaclab reward terms are dt-SCALED RATES (reward_manager.py:150:
        #     value = func × weight × dt) — r2 shipped weight −5.0 which became −0.1/death
        #     (Episode_Reward/termination_penalty read −0.00; the policy learned suicide:
        #     iter-20 episodes were 325 steps at 24% falls, by iter 60 it dove to 18-step
        #     100%-fall episodes). −250 × dt 0.02 = the intended −5.
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-250.0,
        )
        # (3) alive bonus (+0.15/step = 7.5 × dt): sized against the MEASURED early per-step
        #     penalty floor (~−0.13/step for a flailing non-tracking policy on this recipe) so
        #     LIVING is net-positive from step one — the armless smoke escaped the suicide
        #     basin on tracking-reward traction alone; the armed body demonstrably cannot.
        #     L1 PHASE-IN crutch: action-independent (no behavior distortion among survivors),
        #     revisit/anneal at W3-L2. Corridor eval judges behavior, not reward magnitude.
        self.rewards.alive_bonus = RewTerm(
            func=mdp.is_alive,
            weight=7.5,
        )

        # ---- r4 COMMAND-GATE RECALIBRATION (2026-07-19, resume phase from the r3 checkpoint;
        # full evidence in W2_A2Z1_TASK.md r4 section). r3 proved the V15 gate structurally
        # locked for a FRESH walker: nobody ever SUSTAINED the 0.80 magnitude-weighted bar
        # (V15 elite held 0.69-0.72 at ±1.5; V16 elite 0.75-0.77 at ±3.0) — every historical
        # advance was a post-reset EMA transient a warm-started walker gets and a from-scratch
        # one never does (V16 log: ±0.5→±3.0 cascades in ~130 iters after each resume). And
        # drift = v_perp/|cmd| — the 0.15 bound was calibrated at cmd ≥ 1.0; at stage-0 norms
        # (~0.2-0.7) the same ABSOLUTE wobble reads 2-5× larger (r3: 0.236 at ±0.5 ≈ V15's
        # passing 0.12 at ±1.5). Recalibration, all measured-evidence-grounded:
        #   * speed bar 0.70 per stage = at/below the armless-elite STEADY STATE (0.72-0.77),
        #     r3's walker sits at 0.703 at ±0.5 — earnable, but each widening must still be
        #     HELD at the new range (magnitude weighting makes the new top dominate);
        #   * drift bar scaled so the implied ABSOLUTE lateral bound stays ~constant
        #     (0.15 × 1.0 m/s at the V15 calibration point): ±1.0→0.30, ±1.5→0.22,
        #     ±2.0→0.18, ±2.5+→0.15; fall bar 0.12 unchanged;
        #   * floors compressed to consolidation SPACING (~500 iters/stage): the resume
        #     restarts env.common_step_counter, so the inherited 4k/7k/10k... floors would
        #     re-bind from zero and cap an 8k resume at ±1.5 regardless of competence.
        #     Competence, not the clock, is the binding constraint — as V15 intended.
        #   * warmup 200 (was 1000): warm-walker resume; the EMA consolidates in ~33 iters.
        # Table ends at ±3.0 (V16-class, the no-regression bar; drive-sim --vx-max pins to
        # the TRAINED range). Everything else — rewards, terrain, phase-in deltas — untouched.
        cmd = self.commands.base_velocity
        cmd.command_curriculum_warmup_iters = 200
        cmd.command_range_curriculum = [
            {"iter":  300, "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_yaw": [-1.25, 1.25],
             "min_speed_ratio": 0.70, "max_drift_ratio": 0.30},
            {"iter":  800, "lin_vel_x": [-1.5, 1.5], "lin_vel_y": [-0.6, 0.6], "ang_vel_yaw": [-1.5, 1.5],
             "min_speed_ratio": 0.70, "max_drift_ratio": 0.22},
            {"iter": 1300, "lin_vel_x": [-2.0, 2.0], "lin_vel_y": [-0.7, 0.7], "ang_vel_yaw": [-1.75, 1.75],
             "min_speed_ratio": 0.70, "max_drift_ratio": 0.18},
            {"iter": 1800, "lin_vel_x": [-2.5, 2.5], "lin_vel_y": [-0.8, 0.8], "ang_vel_yaw": [-2.0, 2.0],
             "min_speed_ratio": 0.70, "max_drift_ratio": 0.15},
            {"iter": 2300, "lin_vel_x": [-3.0, 3.0], "lin_vel_y": [-0.9, 0.9], "ang_vel_yaw": [-2.25, 2.25],
             "min_speed_ratio": 0.70, "max_drift_ratio": 0.15},
        ]

        # ---- r5 ε-TAIL CO-TRAINING (2026-07-19, user call: "slow-and-pushing-terrain and
        # fast-and-pushing-terrain are TWO DIFFERENT gaits — as much as it sounds like staging,
        # it isn't"). Evidence agrees: v16full's own 19.4k-iter log ends still fighting ±2.0 at
        # 0.79/0.80 — under a serial earned gate the fast basin is never practiced and the slow
        # gait consolidates alone. So: 15% of command resamples draw from the FULL ±3.0 target
        # band (per-terrain caps still bind — sprint commands land on flat-class cells, stairs
        # stay ±1.0); tail envs are masked out of the gate EMA so the official range stays
        # earned. Fast + slow gaits co-develop for the remaining ~5k iters. Constant fraction
        # for r5; anneal/schedule decisions belong to W3-L2.
        cmd.command_curriculum_explore_frac = 0.15
        cmd.command_curriculum_explore_ranges = {
            "lin_vel_x": [-3.0, 3.0],
            "lin_vel_y": [-0.9, 0.9],
            "ang_vel_yaw": [-2.25, 2.25],
        }

        # Inherited events that now ALSO cover the arm — kept deliberately (recon-audited):
        #  * randomize_actuator_gains (joint_names '.*'): ±10% arm servo-gain DR — desirable.
        #  * randomize_rigid_body_mass_others ('^(?!.*base).*'): ±10% per-link arm mass DR.
        #  * randomize_rigid_body_material ('.*'): arm link friction/restitution DR.
        #  * randomize_push_robot / reset_base: root-level, arm-agnostic.

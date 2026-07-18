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
# ⚠ KNOWN PARITY DEBT (tracked in the W2 docs, NOT closed here): the dome_ranges
# self-occlusion kernel reads tasks/a2/data/a3_body_geoms_v1.json — A2-only occluders.
# Until that file is regenerated with the arm's geoms, the policy's lidar does NOT see
# the arm that the deploy sensor will see.

from __future__ import annotations

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

        # Inherited events that now ALSO cover the arm — kept deliberately (recon-audited):
        #  * randomize_actuator_gains (joint_names '.*'): ±10% arm servo-gain DR — desirable.
        #  * randomize_rigid_body_mass_others ('^(?!.*base).*'): ±10% per-link arm mass DR.
        #  * randomize_rigid_body_material ('.*'): arm link friction/restitution DR.
        #  * randomize_push_robot / reset_base: root-level, arm-agnostic.

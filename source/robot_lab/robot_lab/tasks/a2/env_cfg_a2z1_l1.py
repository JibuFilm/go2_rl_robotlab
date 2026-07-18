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
ARM_LINK_EXPR = "z1_.*"                      # all 8 arm links


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
        self.events.reset_arm_pose = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_EXPR),
                "position_range": (-0.15, 0.15),
                "velocity_range": (0.0, 0.0),
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

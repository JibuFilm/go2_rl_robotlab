# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A2 V10 env cfg - goal-aware terrain-progress dial on top of V9.

Why this exists:
  V8/V9 made the velocity command a goal/dial instead of a rigid setpoint, but several inherited
  choices still treated flat-ground form as the universal answer. The main bug was the dial itself:
  it credited horizontal velocity only, so the first stair step's vertical lift was invisible. Several
  body-style priors also acted like flat-ground laws: base height, flat orientation, vertical velocity,
  foot scuffing, and a fairly strong fixed swing-time hint. That fights the point of MoE: the policy
  should be able to route rough/stair contexts into different gait regimes.

V10 keeps hard safety intact:
  illegal base contact, joint limits, undesired thigh/calf contacts, and energy costs are unchanged.

V10 changes the positive goal term:
  track_lin_vel_dial -> track_lin_vel_terrain_dial, which still reads command as cap+direction but
  measures progress along the local sensed terrain tangent. Flat ground behaves like V8/V9; stair
  initiation can receive positive credit for commanded upward progress.

V10 also relaxes style/posture penalties only when BOTH are true:
  1. the command asks for movement, and
  2. the height scan reports relief under the body.

The relaxation is sensor/goal-aware, not a terrain table. Flat/idle behavior still sees the full
flat-ground prior; rough commanded motion gets room to climb, and the MoE is allowed to discover the
gait instead of being forced into one flat-ground template.
"""
from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.a2.env_cfg_v9 import A2V9EnvCfg


@configclass
class A2V10EnvCfg(A2V9EnvCfg):
    """V9 huber plus a terrain-progress dial and goal-aware relaxation of flat style priors."""

    def __post_init__(self):
        super().__post_init__()

        relief_sensor = SceneEntityCfg("height_scanner_small")
        relax = {
            "command_name": "base_velocity",
            "sensor_cfg": relief_sensor,
            "command_threshold": 0.1,
            "command_full": 0.8,
            "relief_threshold": 0.03,
            "relief_full": 0.12,
        }

        # Replace the planar-only dial with the same cap+direction objective measured along the
        # sensed terrain tangent. This is the actual goal-reading fix: the first stair step is no
        # longer invisible just because useful progress is partly vertical.
        self.rewards.track_lin_vel_dial.func = mdp.track_lin_vel_terrain_dial
        self.rewards.track_lin_vel_dial.params = {
            "command_name": "base_velocity",
            "sensor_cfg": relief_sensor,
            "max_slope": 1.0,
            "min_span": 0.05,
        }

        # Keep the same term name so the inherited -1 -> -10 curriculum still targets it, but
        # make the mature height prior yield when the movement goal meets sensed terrain relief.
        self.rewards.base_height_l2.func = mdp.base_height_huber_goal_relaxed
        self.rewards.base_height_l2.params.update({**relax, "delta": 0.06, "max_relax": 0.75})

        # V7's flat-orientation prior is a style prior, not a safety term. Full strength on flat,
        # softened on commanded relief so the body can pitch/roll into a climb.
        self.rewards.flat_orientation_l2.func = mdp.flat_orientation_l2_goal_relaxed
        self.rewards.flat_orientation_l2.params = {**relax, "max_relax": 0.80}

        # Vertical motion is expected during stair/slope initiation. This term is still ramped
        # toward zero by the inherited curriculum; the relaxed function keeps early/resumed runs
        # from over-penalizing commanded climb transients.
        self.rewards.lin_vel_z_l2.func = mdp.lin_vel_z_l2_goal_relaxed
        self.rewards.lin_vel_z_l2.params = {**relax, "max_relax": 0.85}

        # Foot scuffing remains a gait-quality prior, but not a hidden veto on fast foot placement
        # at a step edge.
        self.rewards.feet_regulation.func = mdp.feet_regulation_goal_relaxed
        self.rewards.feet_regulation.params.update({**relax, "max_relax": 0.50})

        # The fixed swing-time hint should not dominate the MoE's terrain-specific gait choice.
        self.rewards.feet_air_time.weight = 0.05

        # Back off V7's doubled smoothness costs so rough-terrain corrective actions are less
        # expensive while retaining a mild regularizer.
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01

# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""A2 task registration (Path-A transcription of the go2 task).

The env class is robot-agnostic (12-DoF FL/FR/RL/RR x hip/thigh/calf — the A2
matches the go2 exactly), so we REUSE `Go2Env` (action-manager override included).
Only the env cfg (A2EnvCfg) and runner cfg (A2MoECTSRunnerCfg) are A2-specific.
"""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##
gym.register(
    id="RobotLab-A2-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:A2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2MoECTSRunnerCfg",
    },
)

gym.register(
    id="RobotLab-A2-V5-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v5:A2V5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# V6 = V5 sensorium + the speed domain translated to the A2's real envelope (5 m/s sprint).
# Reuses the V5 runner cfg (network architecture unchanged — the delta is env-side command ranges).
gym.register(
    id="RobotLab-A2-V6-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v6:A2V6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# V7 = V5 sensorium + robustness-first deltas (unlock terrain via speed + per-terrain caps; balance/
# turning/landing rewards; stepping_stones+gap enabled). Reuses the V5 runner cfg (network unchanged —
# the delta is env-side: command ranges, rewards, events, terrain proportions). See env_cfg_v7.py.
gym.register(
    id="RobotLab-A2-V7-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v7:A2V7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# V8 = V7 robustness + sensorium, with the speed objective swapped from setpoint-tracking to the
# self-finding speed dial (mdp.track_lin_vel_dial): command = cap+direction, policy discovers each
# terrain's feasible max itself. Reuses the V5 runner cfg (network unchanged). See env_cfg_v8.py.
gym.register(
    id="RobotLab-A2-V8-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v8:A2V8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# V9 = V8 dial + the STAIR-INITIATION fix: base_height_l2's kernel reshaped to a transient-tolerant
# (huber) form so the curriculum-ramped -10 clearance spring stops walling off the first step at the
# flat->riser scan-straddle. Terrain-agnostic reward shaping; the dial's forward-progress drive then
# climbs endogenously. Reuses the V5 runner cfg (network unchanged). See env_cfg_v9.py.
gym.register(
    id="RobotLab-A2-V9-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v9:A2V9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

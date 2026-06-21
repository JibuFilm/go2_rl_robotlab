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

# V10 = V9 huber plus a terrain-progress dial and goal-aware relaxation of flat-ground style priors.
# Hard safety stays unchanged; the positive speed goal reads sensed terrain tangent, while style
# penalties loosen only when command+relief say flat form is the wrong prior. See env_cfg_v10.py.
gym.register(
    id="RobotLab-A2-V10-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v10:A2V10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V5MoECTSRunnerCfg",
    },
)

# V12 = V10 terrain dial + the BRAVERY ARM (A1 backtrack penalty, A2 termination grace + base-drag
# penalty, A3 recovery reward, A4 recovery-aware style relaxation, A5 bad-initiation spawns, D3 cooled
# energy weights) + gate-selection tuning (lower z_loss/load-balance so differentiated experts get
# SELECTED). B1/B2 consciously omitted (re-force uniform routing). See env_cfg_v12.py.
gym.register(
    id="RobotLab-A2-V12-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v12:A2V12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V12MoECTSRunnerCfg",
    },
)

# V13 = V12 bravery arm + a STAND-STILL POSTURE fix (P1 kill hip splay, P2 lift the leg-tuck/sag at
# rest — both command-gated so they can't fight the climb) + the deeper gate-selection lever
# (load_balance 0.005 -> 0.002, since selection never emerged at 0.005 through V12). See env_cfg_v13.py.
gym.register(
    id="RobotLab-A2-V13-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v13:A2V13EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:A2V13MoECTSRunnerCfg",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

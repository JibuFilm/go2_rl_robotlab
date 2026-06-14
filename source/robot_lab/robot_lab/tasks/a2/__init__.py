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

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

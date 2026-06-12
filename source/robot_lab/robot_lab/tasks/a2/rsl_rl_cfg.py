# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
# A2 runner cfg: the go2 MoE-CTS runner with ONLY experiment_name changed to
# `a2_moe_cts` (brief §3). All CTS/MoE/PPO hyperparameters are inherited
# byte-for-byte from the go2 runner (the recipe is not re-tuned).

from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import (
    MoECTSRunnerCfg,
    RslRlMoeCtsActorCriticCfg,
    RslRlMoeCtsAlgorithmCfg,
)


@configclass
class A2MoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "a2_moe_cts"
    # class_name / num_steps_per_env / max_iterations / save_interval / policy /
    # algorithm all inherited from MoECTSRunnerCfg (go2) — unchanged.

@configclass
class RslRlMoeCtsV5ActorCriticCfg(RslRlMoeCtsActorCriticCfg):
    """V5 (A3 perceptive student): the shared actor consumes ONLY the leading 45-dim proprio
    slice of single_obs (latent + obs(45) — interface unchanged; the 512-dim range block
    reaches actions through the latent only). Encoder/teacher/critic/latent/losses inherited
    byte-for-byte; the student MoE encoder's first layer widens automatically from the obs."""

    proprio_dim = 45


@configclass
class A2V5MoECTSRunnerCfg(A2MoECTSRunnerCfg):
    experiment_name = "a2_v5_moe_cts"
    policy = RslRlMoeCtsV5ActorCriticCfg()

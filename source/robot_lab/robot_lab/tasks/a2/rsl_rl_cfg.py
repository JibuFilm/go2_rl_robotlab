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


@configclass
class RslRlMoeCtsV12AlgorithmCfg(RslRlMoeCtsAlgorithmCfg):
    """V12 gate-selection tuning (bravery arm). The go2 defaults push the gate toward UNIFORM
    routing (z_loss 1e-3 anti-saturation + load_balance 0.02 post-softmax balancer); now that the
    experts have differentiated, that averaging is what we want to RELAX so a specialist (e.g. a
    save/recovery expert) can actually be SELECTED. Lower both coefficients; everything else inherited
    byte-for-byte. This is the gate-side lever V12 DOES pull — B1/B2 (logit temperature, entropy
    floor) are consciously omitted because they re-force uniformity (see env_cfg_v12.py)."""

    z_loss_coef = 3e-4  # lowered from 1e-3 — let gate logits differentiate
    load_balance_coef = 0.005  # lowered from 0.02 — let experts be selected, not averaged


@configclass
class A2V12MoECTSRunnerCfg(A2V5MoECTSRunnerCfg):
    """V12 runner = V5 perceptive-student policy + the gate-selection algorithm cfg."""

    experiment_name = "a2_v12_moe_cts"
    algorithm = RslRlMoeCtsV12AlgorithmCfg()

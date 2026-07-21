# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
# A2 runner cfg: the go2 MoE-CTS runner with ONLY experiment_name changed to
# `a2_moe_cts` (brief §3). All CTS/MoE/PPO hyperparameters are inherited
# byte-for-byte from the go2 runner (the recipe is not re-tuned).

from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import (
    MoECTSRunnerCfg,
    PPORunnerCfg,
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


@configclass
class RslRlMoeCtsV13AlgorithmCfg(RslRlMoeCtsV12AlgorithmCfg):
    """V13 = V12's gate-selection tuning, one step further. V12 dropped load_balance 0.02 -> 0.005,
    but through ~92% of V12 the gate entropy stayed pinned at max (ln 8 = 2.079) — selection never
    emerged at 0.005. V13 takes the documented next step: load_balance 0.005 -> 0.002. z_loss stays
    3e-4 (the conditional next lever was always load_balance, NOT z_loss). CONDITIONAL: if V12's FINAL
    checkpoint shows entropy dropping below max, revert to 0.005 before launch."""

    load_balance_coef = 0.002  # 0.005 (V12) -> 0.002 — selection didn't emerge at 0.005


@configclass
class A2V13MoECTSRunnerCfg(A2V5MoECTSRunnerCfg):
    """V13 runner = V5 perceptive-student policy + the V13 (deeper) gate-selection algorithm cfg.
    Env-side stand-still posture fix rides on A2V13EnvCfg."""

    experiment_name = "a2_v13_moe_cts"
    algorithm = RslRlMoeCtsV13AlgorithmCfg()


@configclass
class A2V14MoECTSRunnerCfg(A2V5MoECTSRunnerCfg):
    """V14 (clean-slate) runner = V5 perceptive-student policy + the V13 gate-selection algorithm cfg
    (z_loss 3e-4 / load_balance 0.002). The clean-slate corrections are all env-side (A2V14EnvCfg);
    the algorithm is unchanged from V13. Intended for a FRESH train (no --resume)."""

    experiment_name = "a2_v14_moe_cts"
    algorithm = RslRlMoeCtsV13AlgorithmCfg()


@configclass
class A2V15MoECTSRunnerCfg(A2V5MoECTSRunnerCfg):
    """V15 runner = identical to V14 (V5 student + V13 gate-selection algo). V15's only change is the
    env-side command-curriculum gate (A2V15EnvCfg); the algorithm is unchanged. Intended to WARM-START
    the latest V14 weights (--resume --load_run <v14 dir> --checkpoint model_<N>.pt) so the policy carries
    over while the command curriculum restarts at ±0.5 under the strict gate."""

    experiment_name = "a2_v15_moe_cts"
    algorithm = RslRlMoeCtsV13AlgorithmCfg()
    # checkpoint cadence aligned to the every-400 eval-curate loop (RUNPOD_SETUP.md §E): the in-pod
    # daemon benches each new model_<N>.pt, keeps the rolling best (deletes the superseded one), and
    # prunes non-record checkpoints — baseline stays resumable. V14 was 500; V15-only override.
    save_interval = 400


@configclass
class A2V16MoECTSRunnerCfg(A2V5MoECTSRunnerCfg):
    """V16 runner = identical algo/policy to V14/V15 (V5 student + V13 gate-selection algo). V16's only
    change is the env-side goal-oriented terrain-traversal reward (A2V16EnvCfg); the algorithm is
    unchanged. WARM-START the latest V15 best (--resume --load_run <v15 dir> --checkpoint model_<N>.pt).
    save_interval=400 to feed the same every-400 eval-curate loop."""

    experiment_name = "a2_v16_moe_cts"
    algorithm = RslRlMoeCtsV13AlgorithmCfg()
    # DENSE cadence for the corridor-competition outer loop (the select-and-continue loop evals a range
    # of ~20 checkpoints at Δ100 on the comprehensive course, 1 distinct seed each, fits the score-vs-
    # iter trend, promotes the smoothed peak). Δ100 because the eval noise is seed-driven, not iteration-
    # driven, so density + trend-fit smooths it cheaply. ~22 GB of checkpoints over a full run (fine on
    # the 200 GB volume). See tools/sim/run_corridor_competition.py + COMPETITION_LOOP design.
    save_interval = 100


@configclass
class A2V17CleanPPORunnerCfg(PPORunnerCfg):
    """V17 (W1 recipe surgery): the de-MoE'd head — stock OnPolicyRunner + plain gradiented
    actor-critic on the UNCHANGED V16 env (the env/reward moat is untouched by construction; the
    registration pairs this with env_cfg_v16:A2V16EnvCfg as-is). Inherits go2 PPORunnerCfg
    byte-for-byte: its PPO hyperparams are identical to the MoE-CTS cfg (lr 1e-3 adaptive, clip 0.2,
    entropy 0.01, gamma 0.99, lam 0.95, 5 epochs, 4 minibatches, grad-norm 1.0) — the MoE cfg differs
    only by the encoder losses + student lr + teacher_env_ratio + class names, all absent here.

    Deliberate differences vs V16 (write them in the run log, they are NOT bugs):
      * 100% of envs collect on-policy for the actor (no teacher_env_ratio=0.75 split).
      * The actor consumes the FULL `policy` obs group directly (2785 = [45 proprio + 512 dome
        ranges] x 5-frame history) — no latent bottleneck; the raw range block reaching the actor
        is the accepted Phase-2-encoder-deferred trade. The critic consumes the privileged `critic`
        group (263, incl. height_scan 187). The `single_obs` group (557) goes unused (CTS-only).
      * No gate/latent/z-loss channels in logs — expected, not a logging bug.

    NO V16 warm-start (its weights live in the MoE head) — from-scratch on the recipe."""

    experiment_name = "a2_v17_clean_ppo"
    # Pinned explicitly (train.py dispatches on it); the isaaclab default is the same but the go2
    # template leaves it implicit. Stock RslRlPpoActorCriticCfg (512/256/128, elu, no obs-norm —
    # matching V16's choices) + stock RslRlPpoAlgorithmCfg inherited from PPORunnerCfg (go2).
    class_name = "OnPolicyRunner"
    # DENSE cadence for the corridor-competition outer loop, same rationale as V16.
    save_interval = 100


@configclass
class A2Z1L1PPORunnerCfg(A2V17CleanPPORunnerCfg):
    """W3-L1 armed walker on the W1 clean recipe (V17 smoke PASSED 5/5, 2026-07-18).
    The A2Z1-L1 env keeps the exact V16/V17 obs+action contract (12 leg actions; every
    term name-pinned, the arm leaks nowhere), so the whole runner cfg inherits —
    only the experiment name forks the run dir."""

    experiment_name = "a2z1_l1_ppo"
    # SIDECAR-GLOB FIX (2026-07-20): `--resume` without an explicit `--checkpoint` resolves through
    # get_checkpoint_path, which sorts the glob and takes the LAST match — and the terrain sidecar
    # (now also the command-range sidecar) shares the `model_*` prefix, so it picked
    # `model_<it>_terrain.pt` and runner.load() died on the state dict. Anchor to the weights file.
    load_checkpoint = "model_[0-9]*.pt"

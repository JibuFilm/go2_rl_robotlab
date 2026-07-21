# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Preload native extensions before isaacsim/Kit modules are imported to avoid
# Windows DLL loader conflicts when Isaac Lab/RSL-RL import them later.
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [r"python", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

# local imports
from utils import Logger

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["robogauge"] = {
        "enabled": args_cli.robogauge,
        "port": args_cli.robogauge_port,
        # eval-side: consumed by OnPolicyRunnerCTS.robogauge_task_name (WP2c param) — without
        # this the A2 runs would silently score against the go2_lab task.
        "task_name": args_cli.robogauge_task,
    }
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerCTS":
        runner = OnPolicyRunnerCTS(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # TERRAIN CURRICULUM PERSISTENCE (PerceptionGame fix 2026-06-24) — mirrors the common_step_counter
    # resume fix below. terrain_levels lives on the ENV's TerrainImporter and is re-randomized every
    # process launch (torch.randint(0, max_init_terrain_level+1); mean ~2.5 with max_init=5), so each
    # resume threw away the EARNED terrain ladder and re-climbed from scratch on already-competent
    # weights (~300 wasted iters/resume — and the competition loop resumes every round). Fix: snapshot
    # terrain_levels next to each checkpoint (model_<it>.pt -> model_<it>_terrain.pt) and restore it on
    # resume, repositioning env origins to the restored rows (keeping each env's current column/type).
    def _terrain_sidecar(ckpt_path):
        return ckpt_path[:-3] + "_terrain.pt" if ckpt_path.endswith(".pt") else ckpt_path + ".terrain.pt"

    def _get_terrain():
        t = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
        if t is None or getattr(t, "terrain_origins", None) is None or getattr(t, "terrain_levels", None) is None:
            return None  # not a curriculum/generator terrain (e.g. plane) — nothing to persist
        return t

    # COMMAND-RANGE PERSISTENCE (PerceptionGame fix 2026-07-20) — same class of bug as the terrain
    # one above, found by the r6 pre-restart audit. common_step_counter IS restored on resume (see
    # below), but the command term rebuilds `command_ranges` from cfg.ranges (±0.5) and refills
    # `cfg.command_range_curriculum` with ALL stages on every process launch. In competence mode the
    # iter floor is retired, so a resumed run re-earns the entire ramp from ±0.5 — measured on r5:
    # ~400 iters per restart spent re-climbing a range the previous segment had already earned.
    def _cmdrange_sidecar(ckpt_path):
        return ckpt_path[:-3] + "_cmdrange.pt" if ckpt_path.endswith(".pt") else ckpt_path + ".cmdrange.pt"

    def _get_cmd_term():
        try:
            return env.unwrapped.command_manager.get_term("base_velocity")
        except Exception:
            return None

    _orig_runner_save = runner.save
    def _save_with_terrain(path, *a, **k):
        out = _orig_runner_save(path, *a, **k)
        try:
            t = _get_terrain()
            if t is not None:
                torch.save(t.terrain_levels.detach().cpu(), _terrain_sidecar(path))
        except Exception as _e:
            print(f"[WARN]: terrain_levels snapshot failed for '{path}': {_e!r}")
        try:
            c = _get_cmd_term()
            if c is not None and getattr(c, "command_ranges", None) is not None:
                torch.save(
                    {"command_ranges": c.command_ranges,
                     "remaining_stages": list(getattr(c.cfg, "command_range_curriculum", []))},
                    _cmdrange_sidecar(path),
                )
        except Exception as _e:
            print(f"[WARN]: command_ranges snapshot failed for '{path}': {_e!r}")
        return out
    runner.save = _save_with_terrain

    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

        # RESUME CURRICULUM CONTINUITY (PerceptionGame fix 2026-06-15):
        # The command-range + reward-weight curricula key off env.common_step_counter
        # (go2_env.step), which lives on the ENV and is rebuilt to 0 every process launch.
        # runner.load() restores the policy + logging iteration but NOT this env counter, so
        # without this the curricula RESET to iter 0 on every resume (the bug that pinned V5 at
        # ±0.5 and would cap a segmented run's speed ramp). Restore the counter from the runner's
        # loaded checkpoint iter so the curriculum clock continues across segments.
        # (num_steps_per_env == command cfg num_steps_per_iter == 24 → counter//24 == iteration.)
        if agent_cfg.resume:
            try:
                _loaded_iter = int(getattr(runner, "current_learning_iteration"))
                if _loaded_iter <= 0:
                    raise ValueError(f"loaded checkpoint iter is {_loaded_iter}")
                env.unwrapped.common_step_counter = _loaded_iter * agent_cfg.num_steps_per_env
                print(
                    f"[INFO]: resume curriculum continuity — common_step_counter set to "
                    f"{env.unwrapped.common_step_counter} (iter {_loaded_iter} x {agent_cfg.num_steps_per_env})"
                )
            except Exception as _e:
                print(f"[WARN]: could not restore common_step_counter from loaded runner state "
                      f"for '{resume_path}': {_e!r} "
                      f"— curricula will reset to iter 0 (legacy behavior)")

            # restore the EARNED terrain-curriculum ladder snapshotted next to the checkpoint (see save
            # hook above) so a resume continues from the achieved difficulty instead of re-climbing 2.5.
            try:
                _t = _get_terrain()
                _tl_path = _terrain_sidecar(resume_path)
                if _t is not None and os.path.exists(_tl_path):
                    _tl = torch.load(_tl_path, map_location=_t.terrain_levels.device)
                    if tuple(_tl.shape) == tuple(_t.terrain_levels.shape):
                        _t.terrain_levels[:] = _tl.to(_t.terrain_levels.device).clamp_(0, _t.max_terrain_level - 1)
                        _t.env_origins[:] = _t.terrain_origins[_t.terrain_levels, _t.terrain_types]
                        print(f"[INFO]: resume terrain continuity — restored terrain_levels "
                              f"(mean {_t.terrain_levels.float().mean():.2f}) from {_tl_path}")
                    else:
                        print(f"[WARN]: terrain_levels sidecar shape {tuple(_tl.shape)} != "
                              f"env {tuple(_t.terrain_levels.shape)} — terrain curriculum resets (legacy)")
                else:
                    print(f"[INFO]: no terrain_levels sidecar at {_tl_path} — terrain curriculum starts fresh")
            except Exception as _e:
                print(f"[WARN]: could not restore terrain_levels for '{resume_path}': {_e!r} — terrain resets")

            # restore the EARNED command range + remaining stage list (see the save hook above), so a
            # resumed segment continues from the range it earned instead of re-climbing from ±0.5.
            try:
                _c = _get_cmd_term()
                _cr_path = _cmdrange_sidecar(resume_path)
                if _c is not None and os.path.exists(_cr_path):
                    _cr = torch.load(_cr_path, map_location="cpu", weights_only=False)
                    _c.command_ranges = _cr["command_ranges"]
                    _c.cfg.command_range_curriculum = _cr["remaining_stages"]
                    _c.max_lin_vel = max(
                        abs(_c.command_ranges["lin_vel_x"][0]), abs(_c.command_ranges["lin_vel_x"][1]),
                        abs(_c.command_ranges["lin_vel_y"][0]), abs(_c.command_ranges["lin_vel_y"][1]),
                    )
                    _c._update_env_command_ranges()
                    if hasattr(_c, "_update_explore_eligibility"):
                        _c._update_explore_eligibility()
                    print(f"[INFO]: resume command continuity — restored range "
                          f"{_c.command_ranges['lin_vel_x']} with {len(_c.cfg.command_range_curriculum)} "
                          f"stage(s) remaining from {_cr_path}")
                else:
                    print(f"[INFO]: no command-range sidecar at {_cr_path} — command curriculum starts fresh")
            except Exception as _e:
                print(f"[WARN]: could not restore command_ranges for '{resume_path}': {_e!r} — range resets")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    sys.stdout = Logger(os.path.join(log_dir, "train.log"))
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# -*- coding: utf-8 -*-
'''
@File    : commands_go2_rl_gym.py
@Time    : 2026/04/01 17:16:44
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : CommandTerm for go2_rl_gym style command generation, reference to https://github.com/wty-yy/go2_rl_gym

IsaacLab CommandTerm working flow:
Env: after compute reward call command.compute(dt)
1. self._update_metrics(): update self.metrics dict for logging
2. self.time_left -= dt
3. self._resample(self.time_left <= 0)
4. self._update_command(): update command if needed

Get command from self.command property, return command, shape=(num_envs, command_dim)

Note:
1. We don't use original self._resample(env_ids) and self._resample_command(env_ids), because it will randomize time_left
2. Remove heading command
3. We don't use curriculum item to update curriculum, inplace update
'''

from __future__ import annotations  # For forward reference of type hints

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:  # Avoid circular import for type checking
    from robot_lab.tasks.go2.env.go2_env import ActionDelayGo2Env

from itertools import product

import torch

from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

from robot_lab.tasks.go2.mdp.utils import is_robot_on_terrain, sample_disjoint_intervals, sample_single_interval


class Go2RLGymCommand(CommandTerm):
    cfg: Go2RLGymCommandCfg
    _env: ActionDelayGo2Env
    
    def __init__(self, cfg: Go2RLGymCommandCfg, env: ActionDelayGo2Env):
        """Reference: https://github.com/wty-yy/go2_rl_gym/blob/master/legged_gym/envs/base/legged_robot.py
        LeggedRobot._resample_command() and LeggedRobot._post_physics_step_callback()
        """
        super().__init__(cfg, env)
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        self.command_ranges = self.cfg.ranges.to_dict()
        self.env_command_ranges = {
            'lin_vel_x': torch.tensor(self.command_ranges['lin_vel_x'], device=self.device).repeat(self.num_envs, 1),
            'lin_vel_y': torch.tensor(self.command_ranges['lin_vel_y'], device=self.device).repeat(self.num_envs, 1),
            'ang_vel_yaw': torch.tensor(self.command_ranges['ang_vel_yaw'], device=self.device).repeat(self.num_envs, 1),
        }
        self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                               abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.limit_vel["lin_vel_x"],
            self.cfg.limit_vel["lin_vel_y"],
            self.cfg.limit_vel["ang_vel_yaw"]
        )), device=self.device)
        self._init_terrain_infos()
        self._update_env_command_ranges()
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.zero_command_prob = 0
        self.max_command_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._curriculum_ema_speed_ratio = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_fall_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_terrain_level = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_drift_ratio = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_samples = 0
        self._last_curriculum_hold_iter = -1

        # ε-tail exploration (opt-in): per-env FULL-band ranges (terrain caps applied exactly like
        # _update_env_command_ranges) + the tail-membership mask the gate EMA excludes.
        self._is_explore_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_explore_ranges = None
        if self.cfg.command_curriculum_explore_frac > 0.0:
            if not self.cfg.command_curriculum_explore_ranges:
                raise ValueError("command_curriculum_explore_frac > 0 requires command_curriculum_explore_ranges")
            er = self.cfg.command_curriculum_explore_ranges
            self.env_explore_ranges = {
                k: torch.tensor(er[k], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
                for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
            }
            for terrain_type, tcr in self.cfg.terrain_max_command_ranges.items():
                if terrain_type not in self.terrain_type2idx:
                    continue
                t_ids = (self.terrain_idxs == self.terrain_type2idx[terrain_type]).nonzero().flatten()
                for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
                    self.env_explore_ranges[k][t_ids, 0] = max(tcr[k][0], er[k][0])
                    self.env_explore_ranges[k][t_ids, 1] = min(tcr[k][1], er[k][1])
            print(f"Command explore tail ACTIVE: frac={self.cfg.command_curriculum_explore_frac} ranges={er}")

        # r6: eligibility + tail telemetry. `_explore_eligible` marks envs whose terrain-capped
        # explore ceiling actually EXCEEDS the current official ceiling — i.e. envs where the tail
        # can deliver real headroom. Recomputed whenever the official range moves.
        self._explore_eligible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._update_explore_eligibility()

        # Tail telemetry (pure logging; see _update_metrics). Registered HERE so the keys exist in
        # ep_extras from step 0 — rsl_rl's Logger iterates the keys of the FIRST episode dict only,
        # so a conditionally-created metric key never appears in tensorboard. Every buffer is a
        # (num_envs,) tensor written by full BROADCAST each step: CommandManager reduces with
        # mean(buf[env_ids]) over an arbitrary reset subset and then zeroes those rows, so only a
        # row-uniform value survives that contract.
        self._tel_explore_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_explore_drift = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_explore_cmd = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_headroom_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_headroom_n = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_max_cmd_issued = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._tel_explore_n = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.cfg.command_range_curriculum = sorted(self.cfg.command_range_curriculum, key=lambda x: x['iter'], reverse=True)

    def _update_explore_eligibility(self):
        """Mark envs where the tail has genuine headroom over the official range.

        r6 fix (audit 2026-07-20): the explore band is intersected with `terrain_max_command_ranges`
        per env, and on this A2 terrain mix 60% of envs sit on ±1.5-capped stair/obstacle cells. Once
        the official range passed ±1.5 those tail envs were drawing commands BELOW the official band
        while still being masked out of the gate EMA — wasted budget AND a widened lateral/yaw command
        on the hardest terrain. Only envs with real headroom are tail-eligible now.
        """
        if self.env_explore_ranges is None:
            return
        self._explore_eligible = (
            self.env_explore_ranges["lin_vel_x"][:, 1] > self.env_command_ranges["lin_vel_x"][:, 1] + 1e-3
        )

    def __str__(self) -> str:
        """Return a string representation of the command term."""
        msg = (f"""Go2RLGymCommand:\n"""
               f"""Command shape: {self.commands.shape}""")
        return msg

    def _init_terrain_infos(self):
        """Initialize terrain types and indices for each environment."""
        self.terrain_types = list(self._env.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        for terrain_type in self.terrain_types:
            if terrain_type not in self.cfg.terrain_max_command_ranges:
                raise ValueError(f"Terrain type '{terrain_type}' is not defined in cfg.terrain_max_command_ranges.")
        self.terrain_type2idx = {terrain_type: idx for idx, terrain_type in enumerate(self.terrain_types)}
        self.terrain_idxs = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for terrain_type in self.terrain_types:
            idxs = is_robot_on_terrain(self._env, terrain_type).nonzero().flatten()
            if len(idxs) > 0:
                self.terrain_idxs[idxs] = self.terrain_type2idx[terrain_type]
        terrain_cfg = self._env.scene.terrain.cfg.terrain_generator
        sub_terrain_border_width = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
        self.terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * sub_terrain_border_width)

    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _update_metrics(self):
        self.max_command_x[:] = self.command_ranges["lin_vel_x"][1]
        self.metrics["max_command_x"] = self.max_command_x
        # r6 tail telemetry — the signal the r5 run had no way to show: how well the FAST commands
        # are actually tracked, separated from the on-band population the gate already reports.
        # Buffers are broadcast-filled every step in _update_command_curriculum_stats (see the
        # contract note there); publishing them here keeps the keys present from step 0.
        if self.env_explore_ranges is not None:
            self.metrics["explore_speed_ratio"] = self._tel_explore_speed
            self.metrics["explore_drift_ratio"] = self._tel_explore_drift
            self.metrics["explore_cmd_norm"] = self._tel_explore_cmd
            self.metrics["headroom_speed_ratio"] = self._tel_headroom_speed
            self.metrics["headroom_n"] = self._tel_headroom_n
            self.metrics["max_cmd_x_issued"] = self._tel_max_cmd_issued
            self.metrics["explore_n"] = self._tel_explore_n

    def reset(self, env_ids: Sequence[int] | None = None):
        self.time_left[env_ids] = self.cfg.resampling_time
        self.commands_xy_accumulation[env_ids] = 0.0
        self.max_move_distance[env_ids] = 0.0
        return super().reset(env_ids)
        
    def _resample(self, env_ids: Sequence[int]):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        env = self._env
        if len(env_ids) == 0:
            return
        # update command curriculum with train steps
        if len(self.cfg.command_range_curriculum):
            current_iter = env.common_step_counter // self.cfg.num_steps_per_iter
            for i in range(len(self.cfg.command_range_curriculum)-1, -1, -1):  # iterate backwards to be able to pop entries
                cfg = self.cfg.command_range_curriculum[i]
                # 'time' mode: cfg['iter'] is an eligibility FLOOR. 'competence' mode: the iter floor is
                # RETIRED -- cfg['iter'] only sets stage ORDER (stages are iter-sorted), and readiness is
                # the SOLE gate (global warmup + EMA speed-ratio/fall-rate). So a stage opens as soon as
                # the robot has earned the previous one, not when an arbitrary iteration count elapses.
                if self.cfg.command_range_curriculum_mode == "competence" or current_iter >= cfg["iter"]:
                    if not self._command_curriculum_ready(cfg, current_iter):
                        self._maybe_log_curriculum_hold(cfg, current_iter)
                        break
                    self.command_ranges["lin_vel_x"] = cfg["lin_vel_x"]
                    self.command_ranges["lin_vel_y"] = cfg["lin_vel_y"]
                    self.command_ranges["ang_vel_yaw"] = cfg["ang_vel_yaw"]
                    self.max_lin_vel = max(abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                                           abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
                    self.cfg.command_range_curriculum.pop(i)
                    self._update_env_command_ranges()
                    self._update_explore_eligibility()   # r6: headroom shrinks as the range opens
                    print(f"Command range updated at iter {current_iter}: {self.command_ranges}")
                    if self.cfg.command_range_curriculum_mode == "competence":
                        self._reset_command_curriculum_stats()
                        break
        remaining_dist = torch.clip(0.625 * self.terrain_length - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.resampling_time, 0.0)
        self.time_left[env_ids] = self.cfg.resampling_time
        if self.cfg.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance
            if ((env.max_episode_length - env.episode_length_buf[env_ids]) + 1 == 0).any():
                raise ValueError("Some envs have zero remaining episode length during command resampling")
            vel_low_bound = torch.clip(remaining_dist / ((env.max_episode_length - env.episode_length_buf[env_ids] + 1 + 1e-9) * env.step_dt), 0.0)
            self.commands[env_ids, 0] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            r = torch.rand(len(env_ids), device=self.device)
            lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
            upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
            self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 1] = sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device
            )
            self.commands[env_ids, 2] = sample_single_interval(
                env_ids,
                self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                self.device
            )

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.cfg.limit_vel_prob > 0.0:
            max_prob += self.cfg.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1],
                )
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1]
                )
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1]
                )
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.cfg.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.zero_command_curriculum is not None:
            self.zero_command_prob = self.get_current_scale(self.cfg.zero_command_curriculum)
        if self.zero_command_prob > 0.0:
            max_prob += self.zero_command_prob
            next_time_left = torch.clip(
                env.max_episode_length_s - env.episode_length_buf[env_ids] * env.step_dt - (remaining_dist / (0.8 * self.max_lin_vel + 1e-9)),
                min=0.0,
                max=self.cfg.resampling_time,
            )
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_time_left > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.commands[zero_env_ids, :2] = 0.0
                self.time_left[zero_env_ids] = next_time_left[zero_mask]
                if self.cfg.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device)  # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1]
                        )
            min_prob += self.zero_command_prob

        # ---- ε-tail exploration (opt-in; default OFF): a fraction of resamples draw from the FULL
        # target band (terrain-capped) so fast-command gaits co-develop with the earned ramp — the
        # serial gate otherwise consolidates a slow gait first, and a fast gait is a DIFFERENT
        # contact schedule, not an extension (v16full's 19.4k-iter log ends still fighting ±2.0).
        # Applied LAST so limit-vel/zero-command overrides can't snap tail envs back to the stage
        # band; tail envs are masked out of the gate EMA in _update_command_curriculum_stats.
        self._is_explore_env[env_ids] = False
        if self.env_explore_ranges is not None:
            # r6: draw the tail ONLY from eligible envs (real headroom), and renormalise the
            # fraction against the eligible population so the configured dose is delivered to the
            # envs that can use it instead of being diluted ~10x across capped terrain.
            elig = self._explore_eligible[env_ids]
            elig_frac = elig.float().mean().clamp_min(1e-6)
            eff_frac = torch.clamp(
                torch.tensor(self.cfg.command_curriculum_explore_frac, device=self.device) / elig_frac,
                max=1.0,
            )
            tail_sel = (torch.rand(len(env_ids), device=self.device) < eff_frac) & elig
            tail_ids = env_ids[tail_sel]
            if len(tail_ids) > 0:
                # FORWARD command drawn from the HEADROOM interval [official_max, capped_explore_max]
                # with a random sign — so every tail draw is genuinely above the trained band. Drawing
                # uniformly over the whole explore band (r5) put ~89% of tail draws back inside the
                # official band, which is why the tail read as a no-op.
                lo = self.env_command_ranges["lin_vel_x"][tail_ids, 1]
                hi = self.env_explore_ranges["lin_vel_x"][tail_ids, 1]
                mag = (hi - lo) * torch.rand(len(tail_ids), device=self.device) + lo
                sign = torch.where(
                    torch.rand(len(tail_ids), device=self.device) < 0.5,
                    -torch.ones_like(mag),
                    torch.ones_like(mag),
                )
                self.commands[tail_ids, 0] = mag * sign
                # lateral/yaw stay on the OFFICIAL stage band: the tail's job is a fast gait, not a
                # fast spin (r5 commanded ±2.25 rad/s yaw on 0.32 m stairs — a fall driver, not
                # sprint practice). Leaving cols 1,2 untouched keeps whatever the base sampler drew.
                self._is_explore_env[tail_ids] = True

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

    def _update_command(self):
        current_dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2], dim=1)
        self.max_move_distance = torch.max(self.max_move_distance, current_dist)
        self._update_command_curriculum_stats()

    def _update_command_curriculum_stats(self):
        if self.cfg.command_range_curriculum_mode != "competence":
            return

        cmd_xy = self.commands[:, :2]
        cmd_norm = torch.linalg.norm(cmd_xy, dim=1)
        moving_all = cmd_norm > self.cfg.command_curriculum_min_cmd
        # ε-tail envs run off-curriculum commands by design — exclude them from the earned-advancement EMA.
        moving = moving_all & ~self._is_explore_env
        self._update_tail_telemetry(cmd_xy, cmd_norm, moving_all)
        if moving.any():
            cmd_dir = cmd_xy / cmd_norm.clamp_min(1e-6).unsqueeze(1)
            v_xy = self.robot.data.root_lin_vel_b[:, :2]
            v_along = torch.sum(v_xy * cmd_dir, dim=1)
            speed_ratio = torch.clamp(v_along / cmd_norm.clamp_min(1e-6), min=0.0, max=1.0)
            # lateral DRIFT: speed perpendicular to the command, normalized by |cmd|. speed_ratio sees only
            # the along-command component, so a robot veering hard still scores well -> gate drift separately.
            v_perp = torch.linalg.norm(v_xy - v_along.unsqueeze(1) * cmd_dir, dim=1)
            drift_ratio = torch.clamp(v_perp / cmd_norm.clamp_min(1e-6), min=0.0, max=5.0)
            if self.cfg.command_curriculum_speed_weighting == "magnitude":
                # weight by command magnitude so FAST commands dominate -> the gate reflects tracking at the
                # TOP of the range, not the easy low-command average that hides the high-command tail.
                w = cmd_norm[moving]
                wsum = w.sum().clamp_min(1e-6)
                speed_metric = (speed_ratio[moving] * w).sum() / wsum
                drift_metric = (drift_ratio[moving] * w).sum() / wsum
            else:
                speed_metric = speed_ratio[moving].mean()
                drift_metric = drift_ratio[moving].mean()
        else:
            speed_metric = torch.tensor(1.0, dtype=torch.float, device=self.device)
            drift_metric = torch.tensor(0.0, dtype=torch.float, device=self.device)

        reset_terminated = getattr(self._env, "reset_terminated", None)
        if reset_terminated is None:
            fall_metric = torch.tensor(0.0, dtype=torch.float, device=self.device)
        else:
            fall_metric = reset_terminated.float().mean()

        terrain = getattr(self._env.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None) if terrain is not None else None
        if terrain_levels is None:
            terrain_metric = torch.tensor(0.0, dtype=torch.float, device=self.device)
        else:
            terrain_metric = terrain_levels.float().mean()

        alpha = float(self.cfg.command_curriculum_ema_alpha)
        if self._curriculum_ema_samples == 0:
            self._curriculum_ema_speed_ratio = speed_metric.detach()
            self._curriculum_ema_fall_rate = fall_metric.detach()
            self._curriculum_ema_terrain_level = terrain_metric.detach()
            self._curriculum_ema_drift_ratio = drift_metric.detach()
        else:
            self._curriculum_ema_speed_ratio = (
                (1.0 - alpha) * self._curriculum_ema_speed_ratio + alpha * speed_metric.detach()
            )
            self._curriculum_ema_fall_rate = (
                (1.0 - alpha) * self._curriculum_ema_fall_rate + alpha * fall_metric.detach()
            )
            self._curriculum_ema_terrain_level = (
                (1.0 - alpha) * self._curriculum_ema_terrain_level + alpha * terrain_metric.detach()
            )
            self._curriculum_ema_drift_ratio = (
                (1.0 - alpha) * self._curriculum_ema_drift_ratio + alpha * drift_metric.detach()
            )
        self._curriculum_ema_samples += 1

    def _update_tail_telemetry(self, cmd_xy, cmd_norm, moving_all):
        """Fill the tail-vs-on-band telemetry buffers. PURE LOGGING — touches no reward, no obs,
        no RNG draw, no curriculum state, so it cannot alter training dynamics.

        Contract (IsaacLab CommandManager): metric buffers are (num_envs,) tensors reduced with
        ``mean(buf[env_ids])`` over the reset subset and then zeroed in place. Only a value that is
        UNIFORM across rows survives that faithfully, so each buffer is written by full broadcast
        every step (same pattern as ``max_command_x``). No ``.item()``/``float()`` here — keeping the
        path sync-free costs nothing and avoids a per-step host stall.
        """
        if self.env_explore_ranges is None:
            return
        v_xy = self.robot.data.root_lin_vel_b[:, :2]
        cmd_dir = cmd_xy / cmd_norm.clamp_min(1e-6).unsqueeze(1)
        v_along = torch.sum(v_xy * cmd_dir, dim=1)
        speed_ratio = torch.clamp(v_along / cmd_norm.clamp_min(1e-6), min=0.0, max=1.0)
        v_perp = torch.linalg.norm(v_xy - v_along.unsqueeze(1) * cmd_dir, dim=1)
        drift_ratio = torch.clamp(v_perp / cmd_norm.clamp_min(1e-6), min=0.0, max=5.0)

        tail = moving_all & self._is_explore_env
        tw = (cmd_norm * tail.float()).sum().clamp_min(1e-6)
        self._tel_explore_speed[:] = (speed_ratio * cmd_norm * tail.float()).sum() / tw
        self._tel_explore_drift[:] = (drift_ratio * cmd_norm * tail.float()).sum() / tw
        self._tel_explore_cmd[:] = (cmd_norm * tail.float()).sum() / tail.float().sum().clamp_min(1e-6)
        self._tel_explore_n[:] = tail.float().sum()

        # HEADROOM bucket: commands above the current official ceiling, wherever they came from.
        # Anchored to the live range (not a hard-coded 2.0/2.5) so it keeps meaning "above trained
        # band" as the ramp opens. NOTE: with terrain caps this bucket is fed by flat/mid cells only
        # — read it as fast-gait on fast-capable terrain, never as terrain-general.
        head = moving_all & (self.commands[:, 0].abs() > self.command_ranges["lin_vel_x"][1] + 1e-3)
        hw = (cmd_norm * head.float()).sum().clamp_min(1e-6)
        self._tel_headroom_speed[:] = (speed_ratio * cmd_norm * head.float()).sum() / hw
        self._tel_headroom_n[:] = head.float().sum()
        self._tel_max_cmd_issued[:] = self.commands[:, 0].abs().max()

    def _reset_command_curriculum_stats(self):
        self._curriculum_ema_speed_ratio = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_fall_rate = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_terrain_level = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_drift_ratio = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self._curriculum_ema_samples = 0

    def _command_curriculum_ready(self, stage: dict, current_iter: int) -> bool:
        mode = self.cfg.command_range_curriculum_mode
        if mode == "time":
            return True
        if mode != "competence":
            raise ValueError(f"Unknown command_range_curriculum_mode={mode!r}")

        if current_iter < self.cfg.command_curriculum_warmup_iters:
            return False
        if self._curriculum_ema_samples < self.cfg.command_curriculum_min_samples:
            return False

        min_speed_ratio = stage.get("min_speed_ratio", self.cfg.command_curriculum_min_speed_ratio)
        max_fall_rate = stage.get("max_fall_rate", self.cfg.command_curriculum_max_fall_rate)
        min_terrain_level = stage.get("min_terrain_level", self.cfg.command_curriculum_min_terrain_level)
        max_drift_ratio = stage.get("max_drift_ratio", self.cfg.command_curriculum_max_drift_ratio)
        return (
            float(self._curriculum_ema_speed_ratio) >= min_speed_ratio
            and float(self._curriculum_ema_fall_rate) <= max_fall_rate
            and float(self._curriculum_ema_terrain_level) >= min_terrain_level
            and float(self._curriculum_ema_drift_ratio) <= max_drift_ratio
        )

    def _maybe_log_curriculum_hold(self, stage: dict, current_iter: int):
        interval = int(self.cfg.command_curriculum_log_interval)
        if interval <= 0:
            return
        if current_iter == self._last_curriculum_hold_iter or current_iter % interval != 0:
            return
        self._last_curriculum_hold_iter = current_iter
        print(
            "Command range HOLD at iter "
            f"{current_iter}: next={stage} "
            f"speed_ratio={float(self._curriculum_ema_speed_ratio):.3f}/"
            f"{stage.get('min_speed_ratio', self.cfg.command_curriculum_min_speed_ratio):.3f} "
            f"fall={float(self._curriculum_ema_fall_rate):.3f}/"
            f"{stage.get('max_fall_rate', self.cfg.command_curriculum_max_fall_rate):.3f} "
            f"drift={float(self._curriculum_ema_drift_ratio):.3f}/"
            f"{stage.get('max_drift_ratio', self.cfg.command_curriculum_max_drift_ratio):.3f} "
            f"terrain={float(self._curriculum_ema_terrain_level):.2f}/"
            f"{stage.get('min_terrain_level', self.cfg.command_curriculum_min_terrain_level):.2f}"
            + (
                f" explore_n={int(self._is_explore_env.sum())}"
                if self.env_explore_ranges is not None
                else ""
            )
        )

    def _update_env_command_ranges(self):
        """ Update environment-wise command ranges based on current command ranges and terrain type """
        for terrain_type, terrain_command_ranges in self.cfg.terrain_max_command_ranges.items():
            if terrain_type not in self.terrain_type2idx:
                continue
            terrain_idx = self.terrain_type2idx[terrain_type]
            env_ids = (self.terrain_idxs == terrain_idx).nonzero().flatten()
            self.env_command_ranges['lin_vel_x'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_x'][0],
                self.command_ranges['lin_vel_x'][0],
            )
            self.env_command_ranges['lin_vel_x'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_x'][1],
                self.command_ranges['lin_vel_x'][1]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 0] = max(
                terrain_command_ranges['lin_vel_y'][0],
                self.command_ranges['lin_vel_y'][0]
            )
            self.env_command_ranges['lin_vel_y'][env_ids, 1] = min(
                terrain_command_ranges['lin_vel_y'][1],
                self.command_ranges['lin_vel_y'][1]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 0] = max(
                terrain_command_ranges['ang_vel_yaw'][0],
                self.command_ranges['ang_vel_yaw'][0]
            )
            self.env_command_ranges['ang_vel_yaw'][env_ids, 1] = min(
                terrain_command_ranges['ang_vel_yaw'][1],
                self.command_ranges['ang_vel_yaw'][1]
            )
    
    def get_current_scale(self, config: dict):
        """config: {'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0}"""
        current_iter = self._env.common_step_counter // self.cfg.num_steps_per_iter
        cfg_start_iter = config['start_iter']
        cfg_end_iter = config['end_iter']
        cfg_start_val = config['start_value']
        cfg_end_val = config['end_value']

        percentage = (current_iter - cfg_start_iter) / (cfg_end_iter - cfg_start_iter)
        percentage = max(min(percentage, 1.0), 0.0)
        
        current_scale = (1.0 - percentage) * cfg_start_val + percentage * cfg_end_val
        return current_scale

    """Debug Visualization"""

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resample_command(self):
        ...

@configclass
class Go2RLGymCommandCfg(CommandTermCfg):
    class_type: type = Go2RLGymCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    dynamic_resample_commands: bool = True
    """Sample commands with low bounds"""
    limit_vel_invert_when_continuous: bool = True
    """Invert the limit logic when using continuous sample limit velocity commands"""

    zero_command_curriculum: dict = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
    """Start training with zero commands and then gradually increase zero command probability"""
    limit_vel: dict = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}
    """Sample vel commands from min [-1] or zero [0] or max [1] range only"""
    command_range_curriculum: list[dict] = [{
        'iter': 20000, # training iteration at which the command ranges are updated
        'lin_vel_x': [-1.0, 1.0], # min max [m/s]
        'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
    }, {
        'iter': 50000, # training iteration at which the command ranges are updated
        'lin_vel_x': [-2.0, 2.0], # min max [m/s]
        'lin_vel_y': [-1.0, 1.0], # min max [m/s]
        'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
    }]
    """List for command range curriculums at specific training iterations"""
    command_range_curriculum_mode: str = "time"
    """'time' applies stages at their iter (iter = eligibility floor). 'competence' RETIRES the iter floor:
    iter only sets stage ORDER, and readiness (global warmup + EMA speed-ratio/fall-rate) is the SOLE gate."""
    command_curriculum_min_cmd: float = 0.2
    """Commands below this norm are ignored when measuring achieved-speed ratio."""
    command_curriculum_min_speed_ratio: float = 0.60
    """Competence mode: EMA achieved speed along command / command magnitude required to open the next range."""
    command_curriculum_max_fall_rate: float = 0.12
    """Competence mode: maximum EMA termination rate allowed before opening the next range."""
    command_curriculum_min_terrain_level: float = 0.0
    """Competence mode: minimum EMA terrain level required before opening the next range."""
    command_curriculum_ema_alpha: float = 0.01
    """EMA rate for command-curriculum health signals."""
    command_curriculum_min_samples: int = 200
    """Minimum command-term updates before a competence-gated range can open."""
    command_curriculum_warmup_iters: int = 0
    """Minimum training iteration before competence-gated command stages can open."""
    command_curriculum_log_interval: int = 500
    """Print held competence-gated stages every N iterations. <=0 disables."""
    command_curriculum_max_drift_ratio: float = 1e9
    """Competence mode: max EMA lateral-drift ratio (|v_perp|/|cmd|) allowed before opening the next range.
    Default 1e9 = OFF (go2/v5-v13 byte-identical); set tight (e.g. 0.15) to refuse advancing while veering."""
    command_curriculum_speed_weighting: str = "mean"
    """Competence-mode speed metric: 'mean' (flat average over moving envs, the original) or 'magnitude'
    (command-magnitude-weighted, so fast commands dominate -> gates on the TOP of the range, not the easy
    low-command average that hides the high-command tail)."""
    command_curriculum_explore_frac: float = 0.0
    """ε-tail co-training (default 0.0 = OFF, go2/v5-v16 unchanged): fraction of command resamples drawn
    from command_curriculum_explore_ranges (terrain-capped) instead of the current curriculum stage, so
    gaits for the FULL target band are practiced in parallel with the earned ramp rather than after it —
    a fast gait is a different contact schedule, not an extension of the slow one, and a serial gate
    otherwise consolidates the slow gait alone. Tail envs are excluded from the gate EMA, so official
    range advancement stays earned on the stage band."""
    command_curriculum_explore_ranges: dict | None = None
    """Full target band for the ε-tail, e.g. {'lin_vel_x': [-3.0, 3.0], 'lin_vel_y': [-0.9, 0.9],
    'ang_vel_yaw': [-2.25, 2.25]}. Required when command_curriculum_explore_frac > 0. Per-terrain caps
    (terrain_max_command_ranges) still bind, so tail sprints land on flat-class cells, not stairs."""
    terrain_max_command_ranges: dict[str, dict] = {
        #### go2 terrains ####
        'wave':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'slope_up':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'slope_down':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'rough_slope':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'stairs_up':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'stairs_down':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'obstacles':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'stepping_stones':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'gap':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'flat':
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0]},
        #### robotlab default terrains ####
        'random_rough':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'hf_pyramid_slope':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'hf_pyramid_slope_inv':
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'pyramid_stairs':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'pyramid_stairs_inv':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},
        'boxes':
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5]},     
    }
    resampling_time: float = 5.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    """Time before command are changed [s]"""
    limit_ang_vel_at_zero_command_prob: float = 0.2
    """Probability of add limiting angular velocity commands when zero command is sampled"""
    limit_vel_prob: float = 0.2
    """Probability of limiting linear velocity command"""
    num_steps_per_iter: int = 24
    """Number of envs steps for each training iteration"""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = [-0.5, 0.5]
        """Range for the linear-x velocity command [m/s]"""
        lin_vel_y: tuple[float, float] = [-0.5, 0.5]
        """Range for the linear-y velocity command [m/s]"""
        ang_vel_yaw: tuple[float, float] = [-1.0, 1.0]
        """Range for the angular-z velocity command [rad/s]"""

    ranges: Ranges = Ranges()

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

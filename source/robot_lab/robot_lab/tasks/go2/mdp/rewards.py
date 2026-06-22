# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_base_height(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Estimate base height above ground.

    If a height scanner is provided, this returns:
        base_height = base_z - estimated_ground_z

    Otherwise, it falls back to the world-frame root height, which matches the flat-ground
    interpretation used by Gym-style rewards.

    Invalid individual rays are ignored. Only an all-invalid scan falls back to
    ``estimated_ground_z = base_z - base_height_target``, which makes
    ``base_height == base_height_target`` for that environment. This preserves
    the no-sensor fallback without letting one bad ray erase the penalty.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]

    if sensor_cfg is None:
        return base_z

    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits_z) & (torch.abs(ray_hits_z) <= 1e6)
    valid_count = valid.sum(dim=1)
    safe_hits_z = torch.where(valid, ray_hits_z, torch.zeros_like(ray_hits_z))
    estimated_ground_z = safe_hits_z.sum(dim=1) / valid_count.clamp_min(1).to(ray_hits_z.dtype)
    fallback_ground_z = base_z - base_height_target
    estimated_ground_z = torch.where(valid_count > 0, estimated_ground_z, fallback_ground_z)
    return base_z - estimated_ground_z


def _height_scan_relief(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Height variation under the base from a ray scanner.

    This is a terrain cue, not a terrain label: flat ground returns ~0, slopes/stairs/edges return
    positive relief. Invalid rays are ignored; all-invalid scans return zero so they do not relax
    posture constraints by accident.
    """
    if sensor_cfg is None:
        return torch.zeros(env.num_envs, device=env.device)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits_z) & (torch.abs(ray_hits_z) <= 1e6)
    any_valid = valid.any(dim=1)
    finfo = torch.finfo(ray_hits_z.dtype)
    z_min = torch.min(torch.where(valid, ray_hits_z, torch.full_like(ray_hits_z, finfo.max)), dim=1).values
    z_max = torch.max(torch.where(valid, ray_hits_z, torch.full_like(ray_hits_z, finfo.min)), dim=1).values
    relief = torch.where(any_valid, z_max - z_min, torch.zeros_like(z_max))
    return relief


def _recovery_gate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg | None = None,
    base_sensor_cfg: SceneEntityCfg | None = None,
    upright_lo: float = 0.55,
    upright_hi: float = 0.90,
    base_target: float = 0.40,
    base_drop: float = 0.12,
    contact_thresh: float = 1.0,
) -> torch.Tensor:
    """[0,1] gate that rises when the robot is in a RECOVERABLE bad state (V12 / bravery arm).

    Bravery diagnosis A4: style/posture penalties must not slam back during a mid-stumble save,
    even when the height scan does not read terrain relief. This gate keys on the robot's OWN state:
    tilt (low uprightness) OR low base height OR base-in-contact. It is used to relax (not remove)
    style penalties so the policy can flail back under itself. Hard safety terms never use it.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # tilt: uprightness u = -projected_gravity_z (1 = upright, 0 = on its side); gate rises as u drops.
    u = (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    gate = torch.clamp((upright_hi - u) / max(upright_hi - upright_lo, 1e-6), 0.0, 1.0)
    # low base height relative to terrain (reuses the validity-aware base-height helper).
    # NOTE _get_base_height signature is (env, base_height_target: float, asset_cfg, sensor_cfg).
    if base_sensor_cfg is not None:
        h = _get_base_height(env, base_target, asset_cfg, base_sensor_cfg)
        low_gate = torch.clamp((base_target - h) / max(base_drop, 1e-6), 0.0, 1.0)
        gate = torch.maximum(gate, low_gate)
    # base-in-contact is intentionally NOT read from the contact sensor here: a SceneEntityCfg
    # nested inside a dict param is not reliably resolved by the manager (body_ids would default to
    # ALL bodies, firing on every footstep). Base contact is already captured by the low-base gate
    # above (a base resting on the ground has height ~0 << target). contact_sensor_cfg is accepted
    # for forward-compat but unused; tilt + low-base are the robust, resolution-free recovery signals.
    return gate.clamp(0.0, 1.0)


def _goal_relax_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    command_threshold: float = 0.1,
    command_full: float = 0.8,
    relief_threshold: float = 0.03,
    relief_full: float = 0.12,
    max_relax: float = 0.75,
    recovery: bool = False,
    recovery_cfg: dict | None = None,
) -> torch.Tensor:
    """Scale for posture/style penalties that should loosen when the goal conflicts with flat form.

    The MoE policy needs room to route rough-terrain commands into different gait regimes. This
    scale keeps flat/idle penalties unchanged, then reduces them when the command asks for motion
    and the height scan shows terrain relief. Safety terms should not use this helper.

    V12 (bravery A4): when ``recovery=True`` the relaxation ALSO triggers on the robot's own
    recovery-state (tilt / low base / base contact) via ``_recovery_gate``, so the style penalties
    yield during a save even if the terrain sensor does not read relief. The two drivers are OR'd:
    ``relax_drive = max(cmd_gate*relief_gate, recovery_gate)``.
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    cmd_gate = torch.clamp(
        (cmd_norm - command_threshold) / max(command_full - command_threshold, 1e-6),
        min=0.0,
        max=1.0,
    )
    relief = _height_scan_relief(env, sensor_cfg)
    relief_gate = torch.clamp(
        (relief - relief_threshold) / max(relief_full - relief_threshold, 1e-6),
        min=0.0,
        max=1.0,
    )
    relax_drive = cmd_gate * relief_gate
    if recovery:
        relax_drive = torch.maximum(relax_drive, _recovery_gate(env, **(recovery_cfg or {})))
    relax = torch.clamp(torch.as_tensor(max_relax, device=env.device, dtype=cmd_norm.dtype), 0.0, 0.95)
    return 1.0 - relax * relax_drive


def _terrain_slope_along_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eps: float = 0.1,
    min_span: float = 0.05,
    max_slope: float = 1.0,
) -> torch.Tensor:
    """Estimate local terrain slope in the commanded direction from scanner hits.

    The result is not a terrain label. It is a local geometric cue from the same height scanner the
    policy observes: positive when the commanded direction points uphill, negative downhill, and zero
    on flat/invalid scans. All-invalid or one-sided scans deliberately return zero so missing sensor
    data cannot create free vertical reward.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)

    if sensor_cfg is None:
        return torch.zeros_like(cmd_norm)

    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    ray_hits_z = ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits_w).all(dim=-1) & (torch.abs(ray_hits_z) <= 1e6)

    cmd_dir_b = cmd / cmd_norm.clamp_min(eps).unsqueeze(1)
    cmd_dir_b3 = torch.cat((cmd_dir_b, torch.zeros_like(cmd_norm).unsqueeze(1)), dim=1)
    cmd_dir_w = math_utils.quat_apply(yaw_quat(asset.data.root_quat_w), cmd_dir_b3)[:, :2]

    rel_xy = ray_hits_w[..., :2] - asset.data.root_pos_w[:, None, :2]
    proj = torch.sum(rel_xy * cmd_dir_w[:, None, :], dim=-1)
    safe_proj = torch.where(valid, proj, torch.zeros_like(proj))
    safe_hits_z = torch.where(valid, ray_hits_z, torch.zeros_like(ray_hits_z))

    front_w = torch.clamp(safe_proj, min=0.0)
    rear_w = torch.clamp(-safe_proj, min=0.0)
    front_sum = front_w.sum(dim=1)
    rear_sum = rear_w.sum(dim=1)

    front_z = (safe_hits_z * front_w).sum(dim=1) / front_sum.clamp_min(1e-6)
    rear_z = (safe_hits_z * rear_w).sum(dim=1) / rear_sum.clamp_min(1e-6)
    front_dist = (safe_proj * front_w).sum(dim=1) / front_sum.clamp_min(1e-6)
    rear_dist = ((-safe_proj) * rear_w).sum(dim=1) / rear_sum.clamp_min(1e-6)
    span = front_dist + rear_dist

    has_two_sides = (cmd_norm > eps) & (front_sum > 1e-6) & (rear_sum > 1e-6) & (span > min_span)
    slope = (front_z - rear_z) / span.clamp_min(min_span)
    slope = torch.where(has_two_sides, slope, torch.zeros_like(slope))
    return torch.clamp(slope, min=-max_slope, max=max_slope)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1, ord=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    return reward


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)

def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
    return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    base_height = _get_base_height(env, target_height, asset_cfg, sensor_cfg)
    reward = torch.square(base_height - target_height)
    return reward


def base_height_huber(
    env: ManagerBasedRLEnv,
    target_height: float,
    delta: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Transient-tolerant base-height penalty — the V9 stair-INITIATION fix (Path-A fork).

    Identical to ``base_height_l2`` for clearance errors within ``delta`` (flat-ground stance
    discipline is UNCHANGED — same quadratic spring), but beyond ``delta`` the penalty grows only
    LINEARLY instead of quadratically (value AND slope are continuous at ``delta``).

    WHY (info/locomotion/HANDOFF.md, the V9 block): ``_get_base_height`` estimates ground as the MEAN
    of the height-scan ray hits. At a stair's flat->riser boundary the scan patch under the base
    straddles two levels, so the estimated clearance spikes transiently — and with the
    curriculum-ramped -10.0 weight, the L2 (quadratic) kernel turns that brief spike into a WALL the
    dial reward cannot pay to cross, so the policy never takes the first step (confirmed: A2 walks
    stairs in-sim but won't initiate; RoboGauge stairs_fd/bd = 0.00, rolls over). This kernel
    de-escalates the brief LARGE excursion of mounting a step while still spring-loading the SUSTAINED
    clearance error that keeps the flat sprint from crouching/bouncing. Terrain-AGNOSTIC reward
    SHAPING — no per-terrain gate, no added positive — so the dial's own forward-progress drive climbs
    the stairs endogenously (emergent, per the dial design). ``delta`` is the quad->linear knob
    (smoke-tuned; 0.06 m forgives a typical step's scan straddle while staying tight on flat bounce).
    """
    base_height = _get_base_height(env, target_height, asset_cfg, sensor_cfg)
    err = torch.abs(base_height - target_height)
    quad = torch.square(err)
    # linear continuation matching the quadratic's value (delta^2) and slope (2*delta) at err == delta
    lin = delta * delta + 2.0 * delta * (err - delta)
    return torch.where(err <= delta, quad, lin)


def base_height_huber_goal_relaxed(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    delta: float = 0.06,
    max_relax: float = 0.75,
    command_threshold: float = 0.1,
    command_full: float = 0.8,
    relief_threshold: float = 0.03,
    relief_full: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    recovery: bool = False,
    recovery_cfg: dict | None = None,
) -> torch.Tensor:
    """Huber base-height penalty, relaxed when a movement goal meets non-flat terrain.

    This keeps the flat-ground body-height prior intact, but avoids turning it into a global law
    that vetoes stair/slope initiation. Illegal contacts and joint limits remain hard elsewhere.
    """
    penalty = base_height_huber(env, target_height, delta, asset_cfg, sensor_cfg)
    scale = _goal_relax_scale(
        env,
        command_name,
        sensor_cfg,
        command_threshold=command_threshold,
        command_full=command_full,
        relief_threshold=relief_threshold,
        relief_full=relief_full,
        max_relax=max_relax,
        recovery=recovery,
        recovery_cfg=recovery_cfg,
    )
    return penalty * scale


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    return reward


def lin_vel_z_l2_goal_relaxed(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    max_relax: float = 0.8,
    command_threshold: float = 0.1,
    command_full: float = 0.8,
    relief_threshold: float = 0.03,
    relief_full: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recovery: bool = False,
    recovery_cfg: dict | None = None,
) -> torch.Tensor:
    """Vertical-velocity penalty that yields during commanded rough-terrain traversal."""
    scale = _goal_relax_scale(
        env,
        command_name,
        sensor_cfg,
        command_threshold=command_threshold,
        command_full=command_full,
        relief_threshold=relief_threshold,
        relief_full=relief_full,
        max_relax=max_relax,
        recovery=recovery,
        recovery_cfg=recovery_cfg,
    )
    return lin_vel_z_l2(env, asset_cfg) * scale


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward


def flat_orientation_l2_goal_relaxed(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    max_relax: float = 0.8,
    command_threshold: float = 0.1,
    command_full: float = 0.8,
    relief_threshold: float = 0.03,
    relief_full: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recovery: bool = False,
    recovery_cfg: dict | None = None,
) -> torch.Tensor:
    """Flat-orientation prior that stays strong on flat/idle and loosens on commanded relief."""
    scale = _goal_relax_scale(
        env,
        command_name,
        sensor_cfg,
        command_threshold=command_threshold,
        command_full=command_full,
        relief_threshold=relief_threshold,
        relief_full=relief_full,
        max_relax=max_relax,
        recovery=recovery,
        recovery_cfg=recovery_cfg,
    )
    return flat_orientation_l2(env, asset_cfg) * scale


def hip_pos_penalty_l1(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        stand_still_scale: float,
        command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, [1, 2]]
    cmd_large = torch.any(torch.abs(command) > command_threshold, dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1, ord=1
    )
    reward = torch.where(
        cmd_large,
        running_reward,
        stand_still_scale * running_reward
    )
    return reward

def feet_regulation(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize fast horizontal foot motion near the ground.
    
    Feet that are close to the ground receive a much larger penalty for lateral motion, 
    while feet that are lifted during swing are penalized much less. 
    
    Physically, this discourages foot scuffing / dragging, and encourages 
    the robot to lift its feet before moving them quickly in the xy plane.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_ids = asset_cfg.body_ids

    feet_pos_w = asset.data.body_pos_w[:, feet_ids, :]
    base_pos_w = asset.data.root_pos_w.unsqueeze(1)
    feet_xy_vel_w = asset.data.body_lin_vel_w[:, feet_ids, :2]
    base_height = _get_base_height(env, base_height_target, asset_cfg, sensor_cfg)

    gravity_w = torch.tensor(env.sim.cfg.gravity, device=env.device, dtype=feet_pos_w.dtype)
    down_w = gravity_w / torch.norm(gravity_w)
    
    delta_feet_w = feet_pos_w - base_pos_w
    feet2base_height = torch.sum(delta_feet_w * down_w.view(1, 1, 3), dim=-1)
    feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)

    reward = (feet_xy_vel_w.pow(2).sum(dim=-1) * torch.exp(-feet_height / (0.025 * base_height_target))).sum(dim=-1)

    return reward


def feet_regulation_goal_relaxed(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    command_name: str,
    max_relax: float = 0.5,
    command_threshold: float = 0.1,
    command_full: float = 0.8,
    relief_threshold: float = 0.03,
    relief_full: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    recovery: bool = False,
    recovery_cfg: dict | None = None,
) -> torch.Tensor:
    """Foot scuffing regularizer that loosens at commanded terrain discontinuities.

    It remains active as a gait-quality prior, but it stops being a hidden veto on quick foot
    placement when the robot is actually climbing over relief.
    """
    penalty = feet_regulation(env, base_height_target, asset_cfg, sensor_cfg)
    scale = _goal_relax_scale(
        env,
        command_name,
        sensor_cfg,
        command_threshold=command_threshold,
        command_full=command_full,
        relief_threshold=relief_threshold,
        relief_full=relief_full,
        max_relax=max_relax,
        recovery=recovery,
        recovery_cfg=recovery_cfg,
    )
    return penalty * scale


# ---------------------------------------------------------------------------
# Dynamic tracking sigma — go2_rl_gym parity RESTORATION (Path-A fork).
# The port fixed std=0.5 (README-admitted deviation); these terms restore the
# reference behavior (legged_robot.py:1288-1334). Pure math + offline tests in
# dynamic_sigma_core.py / tests/test_dynamic_sigma.py.
# ---------------------------------------------------------------------------
import os  # noqa: E402

from .dynamic_sigma_core import cols_to_max_sigma, dynamic_sigma  # noqa: E402


def _terrain_command_cap_per_env(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis_key: str,
    fallback: float,
) -> torch.Tensor:
    """Per-env command cap from the terrain table, independent of the current global curriculum range."""
    out = torch.full((env.num_envs,), float(fallback), device=env.device)
    try:
        command = env.command_manager.get_term(command_name)
        terrain_idxs = command.terrain_idxs
        terrain_types = command.terrain_types
        ranges = command.cfg.terrain_max_command_ranges
    except (AttributeError, KeyError, LookupError):
        return out

    for terrain_idx, terrain_name in enumerate(terrain_types):
        terrain_range = ranges.get(terrain_name, {}).get(axis_key)
        if terrain_range is None:
            continue
        cap = max(abs(float(terrain_range[0])), abs(float(terrain_range[1])))
        out = torch.where(terrain_idxs == terrain_idx, torch.full_like(out, cap), out)
    return out


def _dynamic_sigma_per_env(
    env: ManagerBasedRLEnv,
    cmd_abs: torch.Tensor,
    default_sigma: float,
    v_min: float,
    v_max: float,
    command_name: str | None = None,
    axis_key: str | None = None,
    terrain_cap_aware: bool = False,
) -> torch.Tensor:
    """Per-env sigma from terrain type + level. Falls back to the default sigma when terrain
    curriculum state is absent (reference behavior when curriculum is off,
    legged_robot.py:1291-1292)."""
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None) if terrain is not None else None
    types = getattr(terrain, "terrain_types", None) if terrain is not None else None
    if levels is None or types is None:
        return torch.full_like(cmd_abs, default_sigma)
    per_col = getattr(env, "_dyn_sigma_per_col", None)
    if per_col is None:
        gen = terrain.cfg.terrain_generator
        names = list(gen.sub_terrains.keys())
        props = [gen.sub_terrains[n].proportion for n in names]
        per_col = cols_to_max_sigma(names, props, gen.num_cols).to(cmd_abs.device)
        env._dyn_sigma_per_col = per_col
    effective_v_max = v_max
    if terrain_cap_aware and command_name is not None and axis_key is not None:
        cap = _terrain_command_cap_per_env(env, command_name, axis_key, v_max)
        effective_v_max = torch.minimum(cap, torch.full_like(cap, float(v_max)))
        effective_v_max = torch.maximum(effective_v_max, torch.full_like(cap, float(v_min) + 1e-6))
    sigma = dynamic_sigma(cmd_abs, per_col[types], levels, default_sigma, v_min, effective_v_max)
    # G0' live-wiring acceptance check (Path-A brief §7 A1): the sigma math is proven offline,
    # but the terrain_levels/terrain_types glue only executes on GPU — with this env var set the
    # smoke logs sigma stats so we can see sigma move with terrain level + command magnitude.
    if os.environ.get("ROBOT_LAB_DYN_SIGMA_DEBUG"):
        n = getattr(env, "_dyn_sigma_dbg_n", 0)
        env._dyn_sigma_dbg_n = n + 1
        if n % 200 == 0:
            print(f"[DYN_SIGMA] call={n} sigma min/mean/max="
                  f"{float(sigma.min()):.4f}/{float(sigma.mean()):.4f}/{float(sigma.max()):.4f} "
                  f"cmd_abs mean/max={float(cmd_abs.mean()):.3f}/{float(cmd_abs.max()):.3f} "
                  f"levels mean/max={float(levels.float().mean()):.2f}/{int(levels.max())} "
                  f"n_default={int((sigma - default_sigma).abs().lt(1e-6).sum())}/{sigma.numel()}")
            # synthetic high-command probe: same LIVE per-col lookup + terrain levels, cmd=2.0
            # (>= v_max -> sigma_max, level-scaled). At stage-1 command ranges the training sigma
            # is legitimately all-default (band starts at v_min); this line proves the live glue
            # moves sigma without waiting 20k iters for stage-2 commands.
            synth = dynamic_sigma(torch.full_like(cmd_abs, 2.0), per_col[types], levels,
                                  default_sigma, v_min, effective_v_max)
            print(f"[DYN_SIGMA_SYNTH] cmd=2.0 sigma min/mean/max="
                  f"{float(synth.min()):.4f}/{float(synth.mean()):.4f}/{float(synth.max()):.4f}")
    return sigma


def track_lin_vel_xy_exp_dynamic_sigma(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    v_min: float = 0.5,
    v_max: float = 1.5,
    terrain_cap_aware: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear-velocity tracking with per-axis dynamic sigma (legged_robot.py:1310-1322):
    sigma_x from |cmd_x|, sigma_y from |cmd_y|; reward = exp(-(err_x^2/sigma_x + err_y^2/sigma_y)).
    With sigma_x == sigma_y == std**2 this reduces exactly to the port's track_lin_vel_xy_exp."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    default_sigma = std**2
    sigma_x = _dynamic_sigma_per_env(
        env, torch.abs(cmd[:, 0]), default_sigma, v_min, v_max,
        command_name=command_name, axis_key="lin_vel_x", terrain_cap_aware=terrain_cap_aware
    )
    sigma_y = _dynamic_sigma_per_env(
        env, torch.abs(cmd[:, 1]), default_sigma, v_min, v_max,
        command_name=command_name, axis_key="lin_vel_y", terrain_cap_aware=terrain_cap_aware
    )
    err_sq = torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2])
    return torch.exp(-(err_sq[:, 0] / sigma_x + err_sq[:, 1] / sigma_y))


def track_ang_vel_z_exp_dynamic_sigma(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    v_min: float = 1.0,
    v_max: float = 2.0,
    terrain_cap_aware: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-rate tracking with dynamic sigma from |cmd_wz| (legged_robot.py:1324-1333)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    default_sigma = std**2
    sigma = _dynamic_sigma_per_env(
        env, torch.abs(cmd[:, 2]), default_sigma, v_min, v_max,
        command_name=command_name, axis_key="ang_vel_yaw", terrain_cap_aware=terrain_cap_aware
    )
    ang_vel_error_sq = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error_sq / sigma)


def track_lin_vel_dial(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eps: float = 0.1,
) -> torch.Tensor:
    """The 'self-finding speed dial' linear-velocity reward — NOT a setpoint tracker.

    Reads the velocity command as a CAP + DIRECTION, never as a target to hit. Rewards the body's
    forward speed PROJECTED onto the commanded direction, clamped to [0, |cmd|]. Crucially there is
    NO penalty for achieving less than the command, so commanding an infeasible 5 m/s on a staircase
    does NOT drag the policy into reckless over-speeding — the pathology of the exp-kernel setpoint
    tracker (`track_lin_vel_xy_exp[_dynamic_sigma]`), whose dynamic-sigma keeps a 'go faster' gradient
    alive even when the command is physically unreachable.

    With a single UNIFORM high cap on every terrain, the terrain-appropriate ceiling emerges
    ENDOGENOUSLY: the policy accelerates until the marginal safety penalties (orientation / contact /
    vertical-bounce / fall-termination), which grow with speed and with terrain difficulty, overtake
    this bounded speed bonus. So one policy sprints on flat and steps carefully on stairs from the
    same command — the dial the user wants. Zero-command envs get 0 (stand-still is shaped elsewhere).

    Tuning: the term WEIGHT sets where the emergent ceiling lands (too high → reckless everywhere;
    too low → lazy everywhere). It is the primary knob and is set empirically from a smoke run.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    moving = (cmd_norm > eps).float()
    direction = cmd / cmd_norm.clamp_min(eps).unsqueeze(1)
    v_along = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    # forward progress along the command, capped at the commanded magnitude; never negative
    reward = torch.clamp(v_along, min=0.0)
    reward = torch.minimum(reward, cmd_norm)
    return reward * moving


def track_lin_vel_terrain_dial(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eps: float = 0.1,
    min_span: float = 0.05,
    max_slope: float = 1.0,
    backtrack_cap: float = 0.0,
) -> torch.Tensor:
    """Self-finding speed dial measured along the sensed terrain, not just the horizontal plane.

    `track_lin_vel_dial` reads the command as cap+direction, but it only credits planar velocity.
    That makes the first stair step a blind spot: the useful motion is partly vertical, so a cautious
    lift can receive no positive progress. This variant estimates the terrain tangent from local
    height-scanner relief in the commanded direction and rewards velocity along that tangent.

    Flat terrain is byte-for-byte the same objective in effect: slope is zero, so only planar progress
    counts. Near a rising edge, upward body velocity contributes to commanded progress; near a descent,
    controlled downward velocity can contribute. The reward stays clamped to [0, |cmd|], so vertical
    bouncing cannot exceed the same dial cap that governs flat running.

    ``backtrack_cap`` (V12 / bravery A1): the lower clamp on along-terrain velocity. The default 0.0
    keeps the original behavior (no credit and no charge for retreating). A positive value lets the
    progress go mildly NEGATIVE down to ``-backtrack_cap``, so commanded-direction backtracking is
    penalized instead of being free — discouraging the "retreat from a hard spot" failure mode.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    moving = (cmd_norm > eps).float()

    cmd_dir_b = cmd / cmd_norm.clamp_min(eps).unsqueeze(1)
    cmd_dir_b3 = torch.cat((cmd_dir_b, torch.zeros_like(cmd_norm).unsqueeze(1)), dim=1)
    cmd_dir_w = math_utils.quat_apply(yaw_quat(asset.data.root_quat_w), cmd_dir_b3)[:, :2]
    slope = _terrain_slope_along_command(
        env,
        command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        eps=eps,
        min_span=min_span,
        max_slope=max_slope,
    )

    v_planar_along = torch.sum(asset.data.root_lin_vel_w[:, :2] * cmd_dir_w, dim=1)
    tangent_norm = torch.sqrt(1.0 + torch.square(slope))
    v_along_terrain = (v_planar_along + asset.data.root_lin_vel_w[:, 2] * slope) / tangent_norm

    reward = torch.clamp(v_along_terrain, min=-backtrack_cap)
    reward = torch.minimum(reward, cmd_norm)
    return reward * moving


def recover_and_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recovery_cfg: dict | None = None,
    eps: float = 0.1,
    min_span: float = 0.05,
    max_slope: float = 1.0,
) -> torch.Tensor:
    """Reward commanded forward progress made WHILE in a recovery state — the 'save' bonus (V12 / A3).

    Bravery diagnosis A3: the policy learns to retreat from hard spots because the only thing paid
    during a stumble is the risk; nothing rewards fighting back to the goal mid-save. This term pays
    the SAME bounded forward progress as the terrain dial, but ONLY while the recovery gate is high
    (tilted / low base / base in contact). So it credits the robot for driving toward the command
    through a stumble — not for being in a bad state.

    It is not farmable: ``track_lin_vel_terrain_dial`` already returns progress clamped to
    [0, |cmd|] and multiplied by the moving mask, so the reward is bounded by the dial cap and is paid
    only for ACTUAL commanded forward progress; the recovery gate just scopes it to the save window.
    """
    progress = track_lin_vel_terrain_dial(
        env,
        command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        eps=eps,
        min_span=min_span,
        max_slope=max_slope,
    )
    gate = _recovery_gate(env, **(recovery_cfg or {}))
    return progress * gate


def lin_vel_lateral_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eps: float = 0.1,
) -> torch.Tensor:
    """Penalize planar body velocity PERPENDICULAR to the commanded direction (heading-keeping for the
    `track_lin_vel_dial` reward, which only rewards on-axis speed — without this, 'as fast as you can'
    could be satisfied by veering). Motion ALONG the command (including a commanded y component) is not
    penalized. For zero-command envs the commanded direction collapses to ~0, so v_perp == v_xy and the
    term penalizes ALL planar drift — i.e. it doubles as a 'don't wander while stopped' penalty."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    v_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_norm = torch.norm(cmd, dim=1, keepdim=True)
    direction = cmd / cmd_norm.clamp_min(eps)
    v_along = torch.sum(v_xy * direction, dim=1, keepdim=True)
    v_perp = v_xy - v_along * direction
    return torch.sum(torch.square(v_perp), dim=1)

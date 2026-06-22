# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
# A2 transcription of the go2 env_cfg (PATH_A_SESSION_BRIEF.md §3). This file is a
# COPY of robot_lab/tasks/go2/env_cfg.py with ONLY the §3 robot-identity rows
# changed (asset/name, base_link naming + termination, init z 0.55, A2 default
# angles incl. OPPOSITE hip signs via A2_CFG_UNITREE, kp/kd via A2_CFG_UNITREE,
# base_height_target 0.50, max_contact_force ~408) + the Gate-T CANDIDATE
# body-coupled cost weights (torques / dof_power / dof_acc — translated for the
# 41.55 kg A2, pending P1 confirmation; see RewardsCfg).
#
# Everything else (dynamic-sigma rewards, curricula, DR, CTS/MoE wiring, terrain
# gym-parity maxima) is REUSED BYTE-FOR-BYTE from robot_lab.tasks.go2 by import —
# the recipe is not re-implemented here (Path-A "transcribe, don't rebalance").

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# REUSE the go2 recipe machinery verbatim (mdp functions, terrain, CTS env).
import robot_lab.tasks.go2.mdp as mdp
from robot_lab.assets.unitree import A2_CFG_UNITREE
from robot_lab.tasks.go2.mdp.terrains import TERRAIN_CFG

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# §3: A2 base link is `base_link` (go2's is `base`).
BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = ".*_foot"
# base_height_l2 measures base_link clearance above terrain. The corrected stance (thigh 0.8 /
# calf -1.05, unitree.py A2_CFG_UNITREE) has a ~0.50 m natural contact-settled base (URDF FK
# foot-center drop ~0.4685 + foot sphere r=0.032; MuJoCo kinematic settle confirms ~0.50). The
# TARGET is set ~3 cm BELOW that natural settle so the policy holds a compliant, slightly-bent
# stand rather than being driven to full kinematic extension (user decision 2026-06-22; also matches
# the deploy stand_base_z 0.47 and the originally-launched V14 run).
BASE_HEIGHT_TARGET = 0.47

##
# Scene definition
##

@configclass
class A2SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the A2 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False
    )

    # §3: the A2 robot (A2_CFG_UNITREE — A2 URDF + OS0 payload, A2 init + gains).
    robot: ArticulationCfg = A2_CFG_UNITREE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # §3: sensor/contact prim paths anchored on `base_link` (go2's were `base`).
    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_small = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_LINK_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # ⚠️ PROVENANCE / KNOWN-DRIFT: Go2RLGymCommandCfg ships GO2's speed numbers (initial
    # ranges.lin_vel_x ±0.5, curriculum opening at iter 20k/50k, terrain caps ±1.5/2.0). The A2 is
    # a 41.55 kg / 120-180 N·m machine good for ~3.6-5.4 m/s (info/locomotion/A2_CAPABILITY_SSOT.md).
    # These go2 values are FROZEN here for V5 reproducibility; the A2-TRANSLATED speed domain lives in
    # env_cfg_v6.py (RobotLab-A2-V6-v0). Do NOT silently reuse these for a new A2 run — see the SSOT.
    base_velocity = mdp.Go2RLGymCommandCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 腿部关节：位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_hip_joint": 0.25, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.068,  # TAILORED: go2 0.05 x(30/22) — span the A2 22 rad/s joint-vel limit
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.068,  # TAILORED: go2 0.05 x(30/22) — span the A2 22 rad/s joint-vel limit
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.0013,  # TAILORED (critic): go2 0.01 x(23.5/180) — A2 180 N·m torque range
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=3.8e-4,  # TAILORED (critic): go2 1e-3 x(150/408) — A2 408 N foot-load range
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=2.5,
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg() # Used to obtain the current-timestep observation for the MoE CTS model

@configclass
class EventCfg:
    """Configuration for events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            # CORRECTED (clean-slate 2026-06-21): TRANSLATED from go2's ±1.0 kg = ±6.2% on 16.09 kg →
            # fraction-matched to the A2 → ±6.2% × 41.55 ≈ ±2.6 kg. (±1.0 kg was only ±2.4% on the A2 —
            # under-randomized; A2_CAPABILITY_SSOT.md §4.) Baked into the BASE here so it isn't reliant
            # on a vN override (the old env_cfg_v6.py patch).
            "mass_distribution_params": (-2.6, 2.6),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",
            "offset_range": (-0.035, 0.035),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6)
            }
        }
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Recipe weights are BYTE-FAITHFUL go2 EXCEPT the three body-coupled cost terms
    (torques / dof_power / dof_acc), which carry the Gate-T CANDIDATE translation
    for the 41.55 kg A2. This is the sanctioned §A2 carve-out to "no reward
    rebalancing" — share-preserving body normalization (translation), NOT a
    hierarchy change. CANDIDATES pending P1 confirmation (Gate-T step 2 a2 smoke
    share-match vs out/gate_t_step1_go2ref.json); provenance below.
    """

    # Path-A restoration: dynamic sigma (go2_rl_gym parity — the port had fixed std).
    # Bands + per-terrain maxima from go2_config.py:166-175; std**2 = the default sigma 0.25.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_dynamic_sigma,
        weight=1.0,
        # σ BAND TAILORED (clean-slate): v_max is the A2 sprint ceiling (~5 m/s peak, SSOT §3), NOT
        # go2's 1.5. The dynamic-σ tolerance must stay alive across the A2's full 0→5 m/s band or
        # high-speed tracking error reads as a permanent miss. Baked into the BASE so go2's 1.5 can't
        # leak via any path. (std 0.5 = dimensionless tolerance floor, body-invariant.) This is the
        # PRIMARY speed reward in the clean slate (re-promoted in env_cfg_v14.py).
        params={"command_name": "base_velocity", "std": 0.5, "v_min": 0.5, "v_max": 5.0}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_dynamic_sigma,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5, "v_min": 1.0, "v_max": 2.0}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # --- GATE-T CANDIDATE body-coupled cost weights (PATH_A_SESSION_BRIEF §7 A2) ---
    # Provenance: out/gate_t_step0.json candidate_a2_weights (converged-walker energy
    # anchor, go2-moe-cts vs a2-stairs-v2[a2-sensorized], matched vx 0.5, Earth-g).
    # Per-term A2/go2 energy ratios -> translated weights. CANDIDATES pending P1.
    #   torques  : go2 -1e-4   -> A2 -5.571765636567876e-06  (ratio x17.95)
    #   dof_power: go2 -2e-5   -> A2 -7.389186728023717e-06  (ratio x2.71)
    #   dof_acc  : go2 -2.5e-7 -> A2 -2.295907501939312e-07  (ratio x1.09, ~body-invariant)
    # NOTE: the port's joint_acc_l2 is on the physics-step signal (go2 used -1.0e-7,
    # a Lab recalibration of gym's -2.5e-7). The Gate-T ratio (1.0888940420675897) was
    # measured on the gym-semantics dof_acc; we apply it to the PORT's joint_acc_l2
    # baseline (-1.0e-7) to stay self-consistent with this repo's signal semantics ->
    # -1.0888940420675897e-07. (Translating the gym -2.5e-7 by the same ratio would
    # mix signal semantics across repos — forbidden by the audit.) The step-2 a2 smoke
    # share-match is what confirms either choice; flagged for P1.
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0888940420675897e-07,  # CANDIDATE (Gate-T ratio x1.0889 on the port's -1.0e-7)
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )

    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-7.389186728023717e-06,  # CANDIDATE (Gate-T: go2 -2e-5 x ratio 2.71)
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.571765636567876e-06,  # CANDIDATE (Gate-T: go2 -1e-4 x ratio 17.95)
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        }
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 5.0},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 1.0,
            "command_threshold": 0.1,
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,
            "command_threshold": 0.1,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # V12 (bravery A2): a brief base scrape is now SURVIVABLE — the episode ends only on a true
    # fall/flip, not on the first base contact. `mdp.bad_orientation` (isaaclab mdp, re-exported by
    # go2.mdp's `from isaaclab.envs.mdp import *`) terminates when the base tilt exceeds limit_angle.
    # limit_angle=1.4 rad (~80 deg) lets the robot stumble/lean far and still try to recover; past
    # that it has effectively gone over and the episode resets. Base-DRAGGING is discouraged instead
    # by a per-step base-contact PENALTY wired in the V12 cfg (not a hard termination here).
    base_fall = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),  # §3: base_link
            "limit_angle": 1.4,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    base_linear_velocity = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "lin_vel_z_l2", "initial_weight": -2.0, "final_weight": -0.0, "start_it": 0, "end_it": 1500
        })
    base_height_l2 = CurrTerm(mdp.gradual_reward_weight_modification, params={
        # TAILORED: end_it 5000 was go2-run-tuned — it slams the -10 posture penalty to full strength while
        # the A2's V14 speed curriculum still commands only ~±1.5 (±2.5 not until iter 10k). Stretch to 9000
        # so the strongest posture-hold term arrives roughly as mid-band speed arrives (SSOT §3).
        "term_name": "base_height_l2", "initial_weight": -1.0, "final_weight": -10.0, "start_it": 0, "end_it": 9000
        })

##
# Environment configuration
##

@configclass
class A2EnvCfg(ManagerBasedRLEnvCfg):
    """Merged configuration for the A2 robot on rough terrain (Path-A transcription)."""

    # Scene settings
    scene: A2SceneCfg = A2SceneCfg(num_envs=8192, env_spacing=0.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # §3: max_contact_force ~408 (41.55 kg x 9.81) — same weight-anchored rule as
    # go2's 147 (15 kg x g). Recorded as an env attribute mirroring go2's plumbing;
    # the audit notes go2's 147 has no live consumer in either repo (no
    # feet_contact_forces reward term), so this is informational/parity-only here.
    max_contact_force: float = 408.0

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 25.0
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Physics material settings from subclass
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = int(1 * 1024 * 1024)  # 1 million
        self.sim.physx.gpu_collision_stack_size = int(512 * 1024 * 1024)  # 128 MB
        self.sim.physx.enable_external_forces_every_iteration = True

        # Update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Handle curriculum for terrain generator
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

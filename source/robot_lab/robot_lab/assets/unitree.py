# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.unitree_actuator import UnitreeActuatorCfg_Go2HV  # noqa: F401

##
# Configuration
##
@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)

# Go2 config from robot_lab [https://github.com/fan-ziqi/robot_lab]
GO2_CFG_ROBOTLAB = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/go2/urdf/go2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)



# Go2 config from unitree_rl_lab [https://github.com/unitreerobotics/unitree_rl_lab]
GO2_CFG_UNITREE = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/go2/urdf/go2.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            min_delay=0,
            max_delay=4,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)


# =============================================================================
# Unitree A2 (Path-A fork) — the A2 transcription (PATH_A_SESSION_BRIEF.md §3).
#
# A copy of GO2_CFG_UNITREE with ONLY the §3 robot-identity rows changed. Sources
# (all transcribed, never invented):
#   * URDF (base_link naming + 17 STLs + 3 OS0 payload fixed links) vendored from
#     unitree_ros/a2_description -> resources/a2/urdf/a2.urdf. Effort/velocity
#     limits come from the URDF: 120/120/180 N·m, 22/22/14.6667 rad/s (VERIFIED
#     against the vendored URDF 2026-06-12 — see PORT_AUDIT / report).
#   * init z 0.55 (corrected stand target + reset drop margin).
#   * default angles incl. the OPPOSITE hip signs (A2: L_hip=-0.1, R_hip=+0.1 —
#     the MIRROR of go2's L=+0.1/R=-0.1), thigh 0.8, calf -1.05
#     (V14 corrected A2 default_stand; a2_constants.py INIT_STATE).
#   * per-joint-group gains kp 100/100/150, kd 4/4/6, effort 120/120/180,
#     armature 0.03 (robot_config.py A2 kp/kd/effort; a2_constants.py actuators).
#     The go2 used one Go2HV torque-speed actuator (kp 25); the A2 actuator is
#     OURS — a plain per-group PD mirrored in the deploy driver (PORT_AUDIT line:
#     "the A2 actuator config is ours, mirrored in our deploy driver"). Three
#     DCMotorCfg groups express the per-group kp/kd a single group cannot.
#
# The OS0 sensor payload (+1.479 kg, total 41.550 kg) is baked into the URDF as
# fixed links and folded into base_link by merge_fixed_joints — teacher and
# student share this body (HANDOFF A2-SENSORIZED lock).
# =============================================================================
A2_CFG_UNITREE = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/a2/urdf/a2.urdf",
        # Explicit: fold the 3 OS0 payload fixed links into base_link (+1.479 kg ->
        # 41.550 kg) at conversion. The 4 *_foot links carry dont_collapse="true"
        # in the URDF, so they SURVIVE the merge (foot collision preserved) — same
        # mechanism go2's Head_upper/Head_lower + feet rely on.
        merge_fixed_joints=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # corrected stand target ~0.50 m + reset drop margin
        joint_pos={
            # A2 hip signs are OPPOSITE go2 (brief §3): left hips -0.1, right hips +0.1.
            ".*L_hip_joint": -0.1,
            ".*R_hip_joint": 0.1,
            # CORRECTED STANCE (clean-slate 2026-06-21). The prior 0.9/-1.8 over-folded the leg to
            # ~0.44 m PHYSICAL height (62% leg reach) — vs the real A2's ~0.57 m operating height.
            # URDF forward-kinematics (thigh swings the leg fwd from vertical; foot sphere r=0.032;
            # trunk top +0.092 m above base_link) gives thigh 0.8 / calf -1.05 -> foot-center drop
            # ~0.4685 m -> contact-settled base_link ~0.50 m -> physical top ~0.59 m. BASE_HEIGHT_TARGET
            # in tasks/a2/env_cfg.py is 0.47 — deliberately ~3 cm BELOW this natural settle (compliant
            # slightly-bent stand, not full extension); keep them coherent if either moves.
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.05,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Per-group PD (kp/kd/effort/velocity from the locked A2 contract). Plain
        # DCMotorCfg — the A2's own actuator, not go2's torque-speed Go2HV model.
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=22.0,   # URDF hip velocity limit
            stiffness=100.0,
            damping=4.0,
            armature=0.03,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=22.0,   # URDF thigh velocity limit
            stiffness=100.0,
            damping=4.0,
            armature=0.03,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=180.0,
            saturation_effort=180.0,
            velocity_limit=14.6667,  # URDF calf velocity limit
            stiffness=150.0,
            damping=6.0,
            armature=0.03,
            friction=0.0,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ],
    # fmt: on
)

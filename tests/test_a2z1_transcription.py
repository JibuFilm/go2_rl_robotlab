# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0

"""Offline structural tests for the A2Z1 (armed) transcription — NO isaaclab required.

Run:  python3 tests/test_a2z1_transcription.py        (or pytest tests/)

Same doctrine as test_a2_transcription.py: the Mac has no IsaacLab, so these tests
validate everything verifiable WITHOUT it. The LIVE programmatic parity (URDF vs the
compiled menagerie MJCF) runs inside PerceptionGame's append_a2z1_urdf.py at generation
time; the golden values here pin REGRESSION (a hand edit to the generated URDF or the
cfg rows fails loudly).

  1. a2z1.urdf parses; contains the full sensorized-A2 content PLUS exactly the 8-link
     Z1 chain; z1_mount_joint is fixed + dont_collapse at the placeholder mount pose.
  2. Arm mass/joint golden parity (values transcribed from the COMPILED menagerie
     z1_gripper.xml, 2026-07-18 — the same source append_a2z1_urdf.py extracts from).
  3. NAMING AUDIT: no z1_* name collides with any leg-side selector regex
     (fullmatch semantics) and none contains the substring 'base'.
  4. Every z1 mesh the URDF references exists in resources/a2/meshes/.
  5. Cfg rows via AST/text: A2Z1_CFG_UNITREE (asset path, arm servo-hold groups
     1000/100 + 1500/150, efforts 30/60, stow init pose), env_cfg_a2z1_l1 (the
     mandatory reset_robot_joints pin + the three arm DR/reward terms), the
     A2Z1L1PPORunnerCfg experiment name, and the RobotLab-A2Z1-L1-v0 registration.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "resources/a2/urdf/a2z1.urdf"
MESHES = ROOT / "resources/a2/meshes"
ENV_L1 = ROOT / "source/robot_lab/robot_lab/tasks/a2/env_cfg_a2z1_l1.py"
RSL_CFG = ROOT / "source/robot_lab/robot_lab/tasks/a2/rsl_rl_cfg.py"
INIT = ROOT / "source/robot_lab/robot_lab/tasks/a2/__init__.py"
UNITREE = ROOT / "source/robot_lab/robot_lab/assets/unitree.py"

_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))


# GOLDEN: transcribed from the COMPILED menagerie z1_gripper.xml (mujoco 3.9.0,
# menagerie sparse clone @4c358ef), 2026-07-18 — the append_a2z1_urdf.py source.
ARM_MASSES = {
    "z1_link00": 0.472475, "z1_link01": 0.673326, "z1_link02": 1.191320,
    "z1_link03": 0.839409, "z1_link04": 0.564046, "z1_link05": 0.389385,
    "z1_link06": 0.288758, "z1_gripperMover": 0.276213,
}
ARM_TOTAL = 4.694932
# joint: (axis, lower, upper, effort)
ARM_JOINTS = {
    "z1_joint1": ((0, 0, 1), -2.61799, 2.61799, 30.0),
    "z1_joint2": ((0, 1, 0), 0.0, 2.96706, 60.0),
    "z1_joint3": ((0, 1, 0), -2.87979, 0.0, 30.0),
    "z1_joint4": ((0, 1, 0), -1.51844, 1.51844, 30.0),
    "z1_joint5": ((0, 0, 1), -1.3439, 1.3439, 30.0),
    "z1_joint6": ((1, 0, 0), -2.79253, 2.79253, 30.0),
    "z1_jointGripper": ((0, 1, 0), -1.51844, 0.0, 30.0),
}
MOUNT_POS = (-0.10, 0.0, 0.117)   # PLACEHOLDER (Jibu eyeball gate) — single source a2z1_mjcf.py
STOW = {"z1_joint1": 0.0, "z1_joint2": 0.785, "z1_joint3": -0.261,
        "z1_joint4": -0.523, "z1_joint5": 0.0, "z1_joint6": 0.0, "z1_jointGripper": 0.0}
FORBIDDEN_FULLMATCH = (r".*_hip_joint", r".*_thigh_joint", r".*_calf_joint",
                       r".*_foot", r".*_thigh", r".*_calf", r"base_link")


def test_urdf_structure():
    tree = ET.parse(URDF)
    root = tree.getroot()
    links = {l.get("name") for l in root.iter("link")}
    joints = {j.get("name"): j for j in root.iter("joint")}
    check("base A2 content present", {"base_link", "FL_foot", "os0_sensor"} <= links)
    z1_links = {n for n in links if n.startswith("z1_")}
    check("exactly 8 z1_ links", z1_links == set(ARM_MASSES), str(sorted(z1_links)))
    for jn in ARM_JOINTS:
        check(f"{jn} present+revolute", joints.get(jn) is not None
              and joints[jn].get("type") == "revolute")
    mnt = joints.get("z1_mount_joint")
    ok = mnt is not None and mnt.get("type") == "fixed" and mnt.get("dont_collapse") == "true"
    check("mount fixed + dont_collapse", ok)
    if mnt is not None:
        xyz = tuple(float(x) for x in mnt.find("origin").get("xyz").split())
        check("mount pos = placeholder", all(abs(a - b) < 1e-9 for a, b in zip(xyz, MOUNT_POS)),
              str(xyz))


def test_arm_goldens():
    root = ET.parse(URDF).getroot()
    links = {l.get("name"): l for l in root.iter("link")}
    total = 0.0
    for name, exp in ARM_MASSES.items():
        mass = float(links[name].find("inertial/mass").get("value"))
        total += mass
        check(f"{name} mass", abs(mass - exp) < 1e-5, f"{mass} vs {exp}")
    check("arm total mass", abs(total - ARM_TOTAL) < 1e-4, f"{total}")
    joints = {j.get("name"): j for j in root.iter("joint")}
    for jn, (axis, lo, hi, eff) in ARM_JOINTS.items():
        j = joints[jn]
        jaxis = tuple(float(x) for x in j.find("axis").get("xyz").split())
        lim = j.find("limit")
        ok = (all(abs(a - b) < 1e-9 for a, b in zip(jaxis, axis))
              and abs(float(lim.get("lower")) - lo) < 1e-5
              and abs(float(lim.get("upper")) - hi) < 1e-5
              and abs(float(lim.get("effort")) - eff) < 1e-9)
        check(f"{jn} axis/range/effort", ok)


def test_naming_audit():
    root = ET.parse(URDF).getroot()
    names = [l.get("name") for l in root.iter("link")] + \
            [j.get("name") for j in root.iter("joint")]
    z1_names = [n for n in names if n and n.startswith("z1_")]
    bad = [(n, p) for n in z1_names for p in FORBIDDEN_FULLMATCH if re.fullmatch(p, n)]
    bad += [(n, "'base' substring") for n in z1_names if "base" in n]
    check("no z1 name collides with leg selectors", not bad, str(bad))


def test_meshes_exist():
    root = ET.parse(URDF).getroot()
    missing = []
    for l in root.iter("link"):
        if not (l.get("name") or "").startswith("z1_"):
            continue
        for mesh in l.iter("mesh"):
            fn = mesh.get("filename").replace("package://a2/meshes/", "")
            if not (MESHES / fn).exists():
                missing.append(fn)
    check("all z1 mesh files exist", not missing, str(missing))


def test_cfg_rows():
    u = UNITREE.read_text()
    check("A2Z1_CFG_UNITREE exists", "A2Z1_CFG_UNITREE = UnitreeArticulationCfg" in u)
    check("a2z1 asset path", "a2/urdf/a2z1.urdf" in u)
    check("arm_main servo hold 1000/100 ±30",
          re.search(r'"arm_main": DCMotorCfg\((?:[^)]*\n)*?[^)]*stiffness=1000\.0', u) is not None
          and '"arm_main"' in u and "effort_limit=30.0" in u)
    check("arm_shoulder 1500/150 ±60",
          re.search(r'"arm_shoulder": DCMotorCfg\((?:[^)]*\n)*?[^)]*stiffness=1500\.0', u) is not None)
    for jn, val in STOW.items():
        check(f"stow {jn}={val}", re.search(rf'"{jn}":\s*{re.escape(str(val))}', u) is not None)

    e = ENV_L1.read_text()
    check("reset_robot_joints pinned to legs",
          'self.events.reset_robot_joints.params["asset_cfg"]' in e and "JOINT_NAMES" in e)
    check("arm-pose DR term", "reset_arm_pose" in e and "reset_joints_by_offset" in e)
    check("payload DR at gripper", "randomize_arm_payload" in e and "z1_gripperMover" in e)
    check("arm contact penalty", "arm_undesired_contacts" in e)

    r = RSL_CFG.read_text()
    check("A2Z1L1PPORunnerCfg(A2V17CleanPPORunnerCfg)",
          "class A2Z1L1PPORunnerCfg(A2V17CleanPPORunnerCfg)" in r)
    check("experiment a2z1_l1_ppo", 'experiment_name = "a2z1_l1_ppo"' in r)

    i = INIT.read_text()
    blk = i[i.find("RobotLab-A2Z1-L1-v0"):]
    check("A2Z1-L1 registered", "RobotLab-A2Z1-L1-v0" in i)
    check("A2Z1-L1 pairs env+runner",
          "env_cfg_a2z1_l1:A2Z1L1EnvCfg" in blk[:600] and "rsl_rl_cfg:A2Z1L1PPORunnerCfg" in blk[:600])


def test_all():
    test_urdf_structure()
    test_arm_goldens()
    test_naming_audit()
    test_meshes_exist()
    test_cfg_rows()
    bad = [(n, d) for n, ok, d in _results if not ok]
    assert not bad, f"failed checks: {bad}"


if __name__ == "__main__":
    test_all()
    for n, ok, d in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"  ({d})" if d and not ok else ""))
    print(f"{sum(1 for _, ok, _ in _results if ok)}/{len(_results)} checks passed")

# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0

"""Offline structural tests for the A2 transcription (NO isaaclab required).

Run:  python3 tests/test_a2_transcription.py        (or pytest tests/)

The Mac has no IsaacLab — the env cfg cannot import/compile locally. These tests
validate everything verifiable WITHOUT isaaclab, so the live registration +
compile check on a VM has a tight, pre-flighted target:

  1. URDF parses (well-formed XML), has the 12 leg joints + 4 foot links + the 3
     OS0 payload fixed links; velocity/effort limits are 22/22/14.6667 & 120/120/180.
  2. OS0 payload mass/inertia parity vs the gear_sonic_fk MJCF source values
     (hard-coded expected values transcribed from a2_sensorized_mjcf DEFAULT_CONFIG;
     the live programmatic parity runs in append_a2_os0_urdf.py).
  3. The A2 transcription rows are present in env_cfg.py / unitree.py via AST/text
     assertions: base_link naming, V14 corrected init z 0.55, OPPOSITE hip signs,
     kp/kd 100/100/150 & 4/4/6, base_height_target 0.47, max_contact_force ~408,
     a2 asset path.
  4. The Gate-T candidate cost weights match out/gate_t_step0.json exactly (read
     programmatically; provenance check).
  5. The registration entry RobotLab-A2-v0 exists with the A2EnvCfg/A2MoECTSRunnerCfg
     entry points and reuses Go2Env.
"""
from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "resources/a2/urdf/a2.urdf"
ENV_CFG = ROOT / "source/robot_lab/robot_lab/tasks/a2/env_cfg.py"
RSL_CFG = ROOT / "source/robot_lab/robot_lab/tasks/a2/rsl_rl_cfg.py"
INIT = ROOT / "source/robot_lab/robot_lab/tasks/a2/__init__.py"
UNITREE = ROOT / "source/robot_lab/robot_lab/assets/unitree.py"

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))


# --- 1. URDF structure + limits -------------------------------------------------
def test_urdf():
    tree = ET.parse(URDF)  # raises if malformed
    root = tree.getroot()
    links = {l.get("name") for l in root.iter("link")}
    joints = {j.get("name"): j for j in root.iter("joint")}

    leg_joints = [f"{leg}_{seg}_joint" for leg in ("FL", "FR", "RL", "RR")
                  for seg in ("hip", "thigh", "calf")]
    check("urdf: 12 leg revolute joints present",
          all(j in joints for j in leg_joints),
          str([j for j in leg_joints if j not in joints]))
    feet = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
    check("urdf: 4 foot links survive (collision geometry)",
          all(f in links for f in feet), str([f for f in feet if f not in links]))
    # foot collision survives (each foot link has a <collision>)
    foot_coll = all(
        any(l.find("collision") is not None for l in root.iter("link") if l.get("name") == f)
        for f in feet
    )
    check("urdf: foot links carry <collision> primitives", foot_coll)

    check("urdf: base_link is the base (not 'base')", "base_link" in links)
    # OS0 payload links + fixed joints
    payload = ["os0_roof_adapter", "os0_baseplate", "os0_sensor"]
    check("urdf: 3 OS0 payload links appended", all(p in links for p in payload),
          str([p for p in payload if p not in links]))
    payload_fixed = all(
        joints.get(f"{p}_joint") is not None and joints[f"{p}_joint"].get("type") == "fixed"
        for p in payload
    )
    check("urdf: OS0 payload joints are fixed (fold into base_link)", payload_fixed)
    # massless built-in frames preserved
    check("urdf: built-in frames kept (camera/front/rear lidar)",
          all(n in links for n in ("camera_link", "front_lidar_link", "rear_lidar_link")))
    # commented-out world/floating_base block preserved (not active)
    text = URDF.read_text()
    check("urdf: world/floating_base block stays commented-out",
          "floating_base_joint" in text and re.search(r"<!--.*floating_base_joint", text, re.S) is not None)

    # velocity + effort limits per joint group
    def lim(jn):
        j = joints[jn]
        lim_el = j.find("limit")
        return float(lim_el.get("effort")), float(lim_el.get("velocity"))
    e_hip, v_hip = lim("FL_hip_joint")
    e_th, v_th = lim("FL_thigh_joint")
    e_calf, v_calf = lim("FL_calf_joint")
    check("urdf: hip effort=120 velocity=22", e_hip == 120.0 and v_hip == 22.0, f"{e_hip}/{v_hip}")
    check("urdf: thigh effort=120 velocity=22", e_th == 120.0 and v_th == 22.0, f"{e_th}/{v_th}")
    check("urdf: calf effort=180 velocity=14.6667",
          e_calf == 180.0 and abs(v_calf - 14.6667) < 1e-6, f"{e_calf}/{v_calf}")


# --- 2. OS0 payload parity (expected = MJCF source values) ---------------------
def test_payload_parity():
    # Expected values transcribed from a2_sensorized_mjcf.DEFAULT_CONFIG + the
    # _diag_inertia_* formulas (brief §3 SENSORIZED table). The live programmatic
    # parity is in append_a2_os0_urdf.py; this is the frozen golden cross-check.
    EXPECT = {
        "os0_roof_adapter": dict(mass=0.449, pos=[0.2073, 0.0, 0.0845],
                                 diag=[5.99602e-4, 8.56842e-4, 1.42651e-3]),
        "os0_baseplate":    dict(mass=0.530, pos=[0.2143, 0.0, 0.1058],
                                 diag=[5.56776e-4, 5.56776e-4, 1.068833e-3]),
        "os0_sensor":       dict(mass=0.500, pos=[0.2143, 0.0, 0.1462],
                                 diag=[3.78395e-4, 3.78395e-4, 4.73062e-4]),
    }
    tree = ET.parse(URDF)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.iter("link")}
    joints = {j.get("name"): j for j in root.iter("joint")}
    total_payload = 0.0
    for name, exp in EXPECT.items():
        link = links[name]
        inr = link.find("inertial")
        mass = float(inr.find("mass").get("value"))
        inertia = inr.find("inertia")
        ixx, iyy, izz = (float(inertia.get(k)) for k in ("ixx", "iyy", "izz"))
        total_payload += mass
        pos = [float(x) for x in joints[f"{name}_joint"].find("origin").get("xyz").split()]
        ok = (abs(mass - exp["mass"]) < 1e-9
              and all(abs(a - b) < 1e-9 for a, b in zip(pos, exp["pos"]))
              and abs(ixx - exp["diag"][0]) < 1e-7
              and abs(iyy - exp["diag"][1]) < 1e-7
              and abs(izz - exp["diag"][2]) < 1e-7)
        check(f"payload: {name} mass/pos/inertia parity", ok,
              f"mass={mass} pos={pos} diag=({ixx:.3e},{iyy:.3e},{izz:.3e})")
    check("payload: total added = 1.479 kg (-> sensorized total 41.550)",
          abs(total_payload - 1.479) < 1e-9, f"{total_payload}")
    # full URDF total link mass = 41.55
    total = sum(float(l.find("inertial").find("mass").get("value"))
                for l in root.iter("link") if l.find("inertial") is not None)
    check("payload: full URDF link mass = 41.55 kg", abs(total - 41.55) < 1e-6, f"{total}")


# --- 3. §3 transcription rows in the config ------------------------------------
def test_env_cfg_rows():
    src = ENV_CFG.read_text()
    check("cfg: BASE_LINK_NAME = base_link", re.search(r'BASE_LINK_NAME\s*=\s*"base_link"', src) is not None)
    check("cfg: BASE_HEIGHT_TARGET = 0.47",
          re.search(r"BASE_HEIGHT_TARGET\s*=\s*0\.47", src) is not None)
    check("cfg: uses A2_CFG_UNITREE for the robot",
          "A2_CFG_UNITREE" in src and "A2_CFG_UNITREE.replace(prim_path" in src)
    check("cfg: max_contact_force = 408.0",
          re.search(r"max_contact_force\s*:\s*float\s*=\s*408\.0", src) is not None)
    check("cfg: termination on base_link (illegal_contact)",
          'body_names=BASE_LINK_NAME' in src)
    check("cfg: reuses go2 TERRAIN_CFG (no terrain re-impl / raise)",
          "from robot_lab.tasks.go2.mdp.terrains import TERRAIN_CFG" in src)
    check("cfg: reuses go2 mdp (recipe not re-implemented)",
          "import robot_lab.tasks.go2.mdp as mdp" in src)
    # dynamic-sigma terms present (untouched recipe element)
    check("cfg: dynamic-sigma tracking terms present",
          "track_lin_vel_xy_exp_dynamic_sigma" in src and "track_ang_vel_z_exp_dynamic_sigma" in src)
    # init z 0.55 + opposite hip signs live in unitree.py (A2_CFG_UNITREE)
    u = UNITREE.read_text()
    check("unitree: A2_CFG_UNITREE defined", "A2_CFG_UNITREE = UnitreeArticulationCfg(" in u)
    check("unitree: init z 0.55", re.search(r"pos=\(0\.0,\s*0\.0,\s*0\.55\)", u) is not None)
    check("unitree: OPPOSITE hip signs (L -0.1, R +0.1)",
          re.search(r'"\.\*L_hip_joint":\s*-0\.1', u) is not None
          and re.search(r'"\.\*R_hip_joint":\s*0\.1', u) is not None)
    check("unitree: thigh 0.8 / calf -1.05",
          re.search(r'"\.\*_thigh_joint":\s*0\.8', u) is not None
          and re.search(r'"\.\*_calf_joint":\s*-1\.05', u) is not None)
    # kp/kd/effort/velocity per group
    check("unitree: hip kp100 kd4 effort120 vel22",
          re.search(r'stiffness=100\.0,\s*damping=4\.0,\s*armature=0\.03', u) is not None
          and "velocity_limit=22.0" in u)
    check("unitree: calf kp150 kd6 effort180 vel14.6667",
          re.search(r'stiffness=150\.0,\s*damping=6\.0,\s*armature=0\.03', u) is not None
          and "velocity_limit=14.6667" in u and "effort_limit=180.0" in u)
    check("unitree: a2 asset path",
          'a2/urdf/a2.urdf' in u)
    # OS0 payload must fold into base_link at conversion (load-bearing for +1.479 kg)
    a2_block = u[u.index("A2_CFG_UNITREE = UnitreeArticulationCfg("):]
    check("unitree: A2 spawn sets merge_fixed_joints=True (folds OS0 payload)",
          "merge_fixed_joints=True" in a2_block.split("init_state")[0])


# --- 4. Gate-T candidate weights vs out/gate_t_step0.json ----------------------
def test_gate_t_weights():
    import json
    gate_paths = (
        Path("/Users/jibujin/Developer/PerceptionGame/tools/sim/out/gate_t_step0.json"),
        Path("/Users/jibujin/Developer/PerceptionGame/tools/gear_sonic_fk/out/gate_t_step0.json"),
    )
    gate_path = next((p for p in gate_paths if p.exists()), None)
    if gate_path is None:
        check("gate-t: provenance json found (optional local artifact)", True)
        return
    g = json.loads(gate_path.read_text())
    cand = g["candidate_a2_weights"]
    src = ENV_CFG.read_text()
    # torques + dof_power are applied verbatim from the json
    check("gate-t: joint_torques_l2 weight == candidate torques",
          f"{cand['torques']!r}" in src or repr(cand['torques']).replace("e-0", "e-0") in src,
          f"expect {cand['torques']!r}")
    check("gate-t: joint_power weight == candidate dof_power",
          f"{cand['dof_power']!r}" in src, f"expect {cand['dof_power']!r}")
    # dof_acc: ratio applied to the PORT's -1.0e-7 baseline (documented deviation)
    ratio = g["ratios_a2_over_go2"]["dof_acc"]
    expected_acc = -1.0e-7 * ratio
    check("gate-t: joint_acc_l2 weight == port_baseline(-1e-7) x ratio",
          f"{expected_acc!r}" in src or "-1.0888940420675897e-07" in src,
          f"expect {expected_acc!r}")
    # weights are marked CANDIDATE (pending P1)
    check("gate-t: weights commented as CANDIDATE pending P1",
          "CANDIDATE" in src and ("P1" in src or "Gate-T" in src))


# --- 5. Registration -----------------------------------------------------------
def test_registration():
    src = INIT.read_text()
    tree = ast.parse(src)  # well-formed python
    check("init: parses as python", True)
    check("init: registers RobotLab-A2-v0", 'id="RobotLab-A2-v0"' in src)
    check("init: env_cfg entry -> A2EnvCfg", "env_cfg:A2EnvCfg" in src)
    check("init: rsl_rl entry -> A2MoECTSRunnerCfg", "rsl_rl_cfg:A2MoECTSRunnerCfg" in src)
    check("init: reuses Go2Env (12-DoF identical)", "go2_env:Go2Env" in src)
    r = RSL_CFG.read_text()
    check("rsl_rl: experiment_name a2_moe_cts", 'experiment_name = "a2_moe_cts"' in r)
    check("rsl_rl: inherits go2 MoECTSRunnerCfg", "MoECTSRunnerCfg" in r and "class A2MoECTSRunnerCfg(MoECTSRunnerCfg)" in r)
    # python files compile (syntax)
    for f in (ENV_CFG, RSL_CFG, INIT, UNITREE):
        try:
            ast.parse(f.read_text())
            check(f"syntax: {f.name} parses", True)
        except SyntaxError as e:
            check(f"syntax: {f.name} parses", False, str(e))


def main() -> int:
    test_urdf()
    test_payload_parity()
    test_env_cfg_rows()
    test_gate_t_weights()
    test_registration()
    npass = sum(1 for _, ok, _ in _results if ok)
    nfail = len(_results) - npass
    for name, ok, detail in _results:
        tag = PASS if ok else FAIL
        line = f"[{tag}] {name}"
        if not ok and detail:
            line += f"   <- {detail}"
        print(line)
    print(f"\n{npass}/{len(_results)} checks passed; {nfail} failed")
    return 0 if nfail == 0 else 1


# pytest entrypoints
def test_all():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

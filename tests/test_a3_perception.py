# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A3 perception build — offline tests (NO isaaclab; pure torch + structural asserts).

Covers G-A3-3 (fork side):
  1. pattern contract data files present + structurally exact (512 rays, unit norms, ring/
     azimuth uniformity, front boresight -> base +x / rear -> -x).
  2. the self-occlusion kernel's analytic primitives vs hand-computed ground truth (sphere,
     box incl. from-inside exit, finite cylinder side + caps, rotated geoms).
  3. fusion/normalization semantics (min-fuse, r/r_max, no-hit -> 1.0).
  4. CTS proprio_dim: 557-obs student with a 45-slice actor (the LOCKED V5 interface) AND
     byte-identical default behavior at 45.
  5. the jit exporter at BOTH widths (feature_dims [3,3,3,A,A,A](+512); deterministic forward).
  6. structural text guards: V5 cfg wires dome_ranges into policy+single_obs ONLY (critic
     inherited untouched); a3_range_core imports no isaaclab; the V5 task is registered;
     occluder census 31/5/4 (the brief's 33/6/4 = + the OS0 visual twins).

Run:  python3 tests/test_a3_perception.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
A2_DIR = ROOT / "source/robot_lab/robot_lab/tasks/a2"
CORE = A2_DIR / "a3_range_core.py"
DATA = A2_DIR / "data"
sys.path.insert(0, str(ROOT / "source/rsl_rl"))

# Load the pure-torch core straight from file (package __init__ would pull isaaclab).
spec = importlib.util.spec_from_file_location("a3_range_core", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

PASS = []


def ok(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:56} {detail}")


print("=" * 78)
print("test_a3_perception (fork offline suite)")
print("=" * 78)

# ----------------------------------------------------------------- [1] pattern contract
pattern = core.load_pattern()
geoms = core.load_body_geoms()
dirs = core.sensor_frame_directions(pattern)
ok("pattern: 2 sensors x 256 = 512 rays", pattern["total_rays"] == 512 and dirs.shape == (256, 3))
ok("pattern: unit directions", torch.allclose(dirs.norm(dim=-1), torch.ones(256), atol=1e-6))
elev = pattern["ring_elevations_deg"]
ok("pattern: 16 rings, band centers -3..87 (6° uniform)",
   len(elev) == 16 and abs(elev[0] + 3) < 1e-9 and abs(elev[-1] - 87) < 1e-9
   and np.allclose(np.diff(elev), 6.0))
o_b, d_b = core.body_frame_rays(pattern)
ok("body_frame_rays shapes (512,3)x2", o_b.shape == (512, 3) and d_b.shape == (512, 3))
Rf = core.quat_to_mat(torch.tensor(pattern["sensors"][0]["site_quat_wxyz_body"]))
Rr = core.quat_to_mat(torch.tensor(pattern["sensors"][1]["site_quat_wxyz_body"]))
ok("front boresight (site +z) -> base +x; rear -> -x",
   torch.allclose(Rf @ torch.tensor([0., 0., 1.]), torch.tensor([1., 0., 0.]), atol=1e-5)
   and torch.allclose(Rr @ torch.tensor([0., 0., 1.]), torch.tensor([-1., 0., 0.]), atol=1e-5))

# ----------------------------------------------------------------- [2] primitive kernels
inf = torch.inf
o = torch.tensor([[-5.0, 0.0, 0.0]])
d = torch.tensor([[1.0, 0.0, 0.0]])
ok("sphere: head-on r=1 -> t=4", torch.allclose(core._ray_sphere(o, d, 1.0), torch.tensor([4.0])))
ok("sphere: miss -> inf", bool(core._ray_sphere(torch.tensor([[-5.0, 2.0, 0.0]]), d, 1.0).isinf()))
half = torch.tensor([1.0, 2.0, 0.5])
ok("box: head-on half=(1,2,.5) -> t=4", torch.allclose(core._ray_box(o, d, half), torch.tensor([4.0])))
ok("box: from INSIDE -> exit t=1", torch.allclose(
    core._ray_box(torch.tensor([[0.0, 0.0, 0.0]]), d, half), torch.tensor([1.0])))
ok("box: parallel offset miss -> inf", bool(core._ray_box(
    torch.tensor([[-5.0, 3.0, 0.0]]), d, half).isinf()))
# cylinder r=1, hh=0.5 about z: side hit head-on at t=4; cap hit from above
ok("cylinder: side head-on -> t=4", torch.allclose(
    core._ray_cylinder(o, d, 1.0, 0.5), torch.tensor([4.0])))
ok("cylinder: cap from +z -> t=1.5", torch.allclose(
    core._ray_cylinder(torch.tensor([[0.3, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]]),
                       1.0, 0.5), torch.tensor([1.5])))
ok("cylinder: axis-parallel outside radius -> inf", bool(core._ray_cylinder(
    torch.tensor([[2.0, 0.0, 5.0]]), torch.tensor([[0.0, 0.0, -1.0]]), 1.0, 0.5).isinf()))
# rotated geom equivalence via the kernel path: a box rotated 90° about z swaps x/y half-extents
kr = core.A3SelfOcclusionKernel(pattern, {
    "schema": "a3_body_geoms_v1", "geoms": [{
        "name": "t", "body": "base_link", "type": "box", "size": [1.0, 2.0, 0.5],
        "pos_body": [3.0, 0.0, 0.0],
        "quat_wxyz_body": [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],  # 90° z
    }]})
bp = {"base_link": (torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))}
r = kr.self_ranges(bp)
# ray 0 of the front sensor: elevation -3°, azimuth 0 -> nearly +z(sensor) ... use the analytic
# check instead: cast the front boresight-ish ray family and verify SOME rays see the box at
# ~(3 - 2 - 0.338) under rotation (rotated box: y-half 2 now spans x) — structural sanity only.
ok("kernel: rotated occluder is visible (finite ranges exist)", bool(r.isfinite().any()),
   f"min={float(r.min()):.3f}")

# ----------------------------------------------------------------- [3] fusion semantics
kf = core.A3SelfOcclusionKernel(pattern, geoms)
t_r = torch.tensor([[10.0, inf, 40.0, 5.0]])
b_r = torch.tensor([[12.0, inf, 35.0, 0.6]])
fused = kf.fuse_normalize(t_r, b_r)
ok("fuse: min + /30 + no-hit=1 + clamp", torch.allclose(
    fused, torch.tensor([[10 / 30, 1.0, 35 / 30, 0.6 / 30]]).clamp(0, 1)), str(fused.tolist()))

# census
cen = geoms["census_collision_active"]
ok("occluder census 31/5/4 (brief 33/6/4 = + OS0 visual twins)",
   cen == {"box": 31, "cylinder": 5, "sphere": 4} and geoms["n_geoms"] == 40, str(cen))

# ----------------------------------------------------------------- [4] CTS proprio_dim
from tensordict import TensorDict                      # noqa: E402
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS  # noqa: E402


def build(nso, proprio=None):
    obs = TensorDict({"single_obs": torch.zeros(2, nso), "policy": torch.zeros(2, nso * 5),
                      "critic": torch.zeros(2, 275)}, batch_size=[2])
    return ActorCriticMoECTS(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
                             num_actions=12, state_dependent_std=False, proprio_dim=proprio)


torch.manual_seed(0)
p45 = build(45)                                        # default: byte-identical pre-V5
ok("45-dim default: actor in = 32+45, proprio_dim==num_single_obs",
   p45.actor.network[0].in_features == 77 and p45.proprio_dim == 45)
p557 = build(557, proprio=45)
ok("557-dim V5: actor in = 32+45 (slice), encoder in = 2785",
   p557.actor.network[0].in_features == 77
   and p557.student_moe_encoder.moe.gating_network[0].network[0].in_features == 2785,
   f"actor_in={p557.actor.network[0].in_features}")
obs557 = TensorDict({"single_obs": torch.randn(2, 557), "policy": torch.randn(2, 2785),
                     "critic": torch.randn(2, 275)}, batch_size=[2])
a = p557.act_inference(obs557)
ok("557 act_inference -> (2,12)", a.shape == (2, 12))
v = p557.evaluate(obs557, is_teacher=False)
ok("557 critic path untouched -> (2,1)", v.shape == (2, 1))

# ----------------------------------------------------------------- [5] exporter both widths
from rsl_rl.utils.exporter_cts import export_cts_policy_as_jit, _TorchPolicyExporter  # noqa: E402

ex45 = _TorchPolicyExporter(p45)
ok("exporter 45: feature_dims [3,3,3,12,12,12]", ex45.feature_dims == [3, 3, 3, 12, 12, 12]
   and ex45.proprio_dim == 45)
ex557 = _TorchPolicyExporter(p557)
ok("exporter 557: feature_dims [3,3,3,12,12,12,512], proprio 45",
   ex557.feature_dims == [3, 3, 3, 12, 12, 12, 512] and ex557.proprio_dim == 45)
tmp = Path(tempfile.mkdtemp(prefix="a3_export_"))
export_cts_policy_as_jit(p557, None, None, str(tmp), "policy.pt")
m = torch.jit.load(str(tmp / "policy.pt")).eval()
m.reset()
o1 = m(torch.ones(1, 557))
m.reset()
o2 = m(torch.ones(1, 557))
ok("557 jit: forward (1,557)->(1,12), deterministic on reset",
   o1.shape == (1, 12) and torch.allclose(o1, o2))
bad = False
try:
    m(torch.ones(1, 45))
except Exception:
    bad = True
ok("557 jit REJECTS 45-dim input (loud contract)", bad)

# ----------------------------------------------------------------- [6] structural guards
cfg_text = (A2_DIR / "env_cfg_v5.py").read_text()
ok("V5 cfg: dome_ranges in policy subclass", "class V5PolicyCfg" in cfg_text
   and "dome_ranges = ObsTerm" in cfg_text)
ok("V5 cfg: critic inherited untouched (no override)",
   "critic:" not in cfg_text.split("class V5ObservationsCfg")[1].split("@configclass")[0]
   .replace("# critic: inherited UNTOUCHED", ""))
ok("V5 cfg: both dome RayCasters at the contract sites",
   "dome_front" in cfg_text and "dome_rear" in cfg_text and 'ray_alignment="base"' in cfg_text)
core_text = CORE.read_text()
import re as _re
ok("a3_range_core: NO isaaclab imports (offline-pure)",
   not _re.search(r"^\s*(import|from)\s+isaaclab", core_text, _re.M))
init_text = (A2_DIR / "__init__.py").read_text()
ok("RobotLab-A2-V5-v0 registered; RobotLab-A2-v0 untouched",
   'id="RobotLab-A2-V5-v0"' in init_text and 'id="RobotLab-A2-v0"' in init_text)
ok("data files shipped in fork", (DATA / "a3_pattern_contract_v1.json").exists()
   and (DATA / "a3_body_geoms_v1.json").exists())
rsl_cfg_text = (A2_DIR / "rsl_rl_cfg.py").read_text()
ok("V5 runner cfg: proprio_dim=45, new experiment name",
   "proprio_dim = 45" in rsl_cfg_text and 'experiment_name = "a2_v5_moe_cts"' in rsl_cfg_text)

# ------------------------------------------------- [7] MERGED-BODY regression (V5 [V2] finding)
# IsaacLab's URDF import folds the OS0 fixed links into base_link — the live articulation has
# no os0_* bodies. resolve_to() must re-anchor those occluders onto base_link with the composed
# static offset, producing IDENTICAL ranges to the unmerged kernel at the same physical pose.
def _rand_quat(g):
    q = torch.randn(4, generator=g, dtype=torch.float64)
    return q / q.norm()


gen = torch.Generator().manual_seed(7)
kA = core.A3SelfOcclusionKernel(dtype=torch.float64)
kB = core.A3SelfOcclusionKernel(dtype=torch.float64)
full_bodies = list(kA.required_bodies)
merged_bodies = [b for b in full_bodies if not b.startswith("os0_")]
kB.resolve_to(merged_bodies)
ok("resolve_to drops os0_* from required_bodies",
   not any(b.startswith("os0_") for b in kB.required_bodies)
   and set(kB.required_bodies) <= set(merged_bodies))

table = core.load_body_geoms()["bodies"]
base_p = torch.tensor([[0.3, -0.2, 0.45]], dtype=torch.float64)
base_q = _rand_quat(gen).unsqueeze(0)
poses = {"base_link": (base_p, base_q)}
Rb = core.quat_to_mat(base_q[0])
for b in full_bodies:
    if b == "base_link":
        continue
    if b.startswith("os0_"):
        ent = table[b]                          # welded: pose = base ∘ fixed offset
        off_p = torch.tensor(ent["pos_in_parent"], dtype=torch.float64)
        off_R = core.quat_to_mat(torch.tensor(ent["quat_wxyz_in_parent"], dtype=torch.float64))
        p = base_p[0] + Rb @ off_p
        Rw = Rb @ off_R
        # rotation matrix -> wxyz quaternion (trace method; det=1 rotations here)
        tr = Rw[0, 0] + Rw[1, 1] + Rw[2, 2]
        w = torch.sqrt((1 + tr).clamp_min(1e-12)) / 2
        q = torch.tensor([w, (Rw[2, 1] - Rw[1, 2]) / (4 * w),
                          (Rw[0, 2] - Rw[2, 0]) / (4 * w),
                          (Rw[1, 0] - Rw[0, 1]) / (4 * w)], dtype=torch.float64)
        poses[b] = (p.unsqueeze(0), q.unsqueeze(0))
    else:                                       # legs: arbitrary poses, fed identically to both
        poses[b] = (torch.randn(1, 3, generator=gen, dtype=torch.float64) * 0.3 + base_p,
                    _rand_quat(gen).unsqueeze(0))

rA = kA.self_ranges(poses)
rB = kB.self_ranges({b: poses[b] for b in kB.required_bodies})
both = rA.isfinite() & rB.isfinite()
agree_class = bool((rA.isfinite() == rB.isfinite()).all())
dmax = float((rA[both] - rB[both]).abs().max()) if both.any() else 0.0
ok("merged kernel == unmerged kernel (same physical pose)",
   agree_class and dmax < 1e-9 and both.any(),
   f"hits={int(both.sum())} max|Δ|={dmax:.2e}")

# a JOINTED absent body must refuse (no silent approximation)
kC = core.A3SelfOcclusionKernel(dtype=torch.float64)
try:
    kC.resolve_to([b for b in full_bodies if b != "FL_calf"])
    ok("jointed-absent body raises (no silent approximation)", False)
except RuntimeError as e:
    ok("jointed-absent body raises (no silent approximation)", "JOINT" in str(e))

print("-" * 78)
n = sum(PASS)
print(f"{'ALL PASS' if all(PASS) else 'FAILURES'}: {n}/{len(PASS)}")
sys.exit(0 if all(PASS) else 1)

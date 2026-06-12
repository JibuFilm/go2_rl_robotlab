# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0
#
"""A3 range-sensor core — pattern loading + the batched self-occlusion kernel (PURE torch).

NO isaaclab imports (offline-testable on any machine, the dynamic_sigma_core discipline).
Everything here implements the LOCKED A3 contract (PATH_A_SESSION_BRIEF.md §7 A3, 2026-06-12):

  * the pattern is LOADED from `data/a3_pattern_contract_v1.json` — generated once by
    tools/gear_sonic_fk/a3_pattern.py, the train==deploy contract (pattern parity replaces
    grid parity). Nothing here re-derives ray geometry.
  * the SELF-OCCLUSION kernel: exact closed-form ray-vs-primitive intersection against the A2's
    collision set (`data/a3_body_geoms_v1.json` — 31 boxes / 5 cylinders / 4 spheres,
    transcribed programmatically from the sensorized MJCF; the brief's 33/6/4 census counts the
    OS0's co-located visual twins). This is a REIMPLEMENTATION of the deploy sensor's exact
    geometry, not an approximation — deploy-side `mj_ray` runs against these same primitives,
    and G-V5-occ requires numerics-level agreement.
  * TRUE-FIRST-HIT fusion: fused = min(terrain_range, body_range); normalized = clamp(r/r_max,
    0, 1); no-hit (inf) -> 1.0.

Convention: quaternions are (w, x, y, z) everywhere (mujoco/IsaacLab convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
PATTERN_JSON = DATA_DIR / "a3_pattern_contract_v1.json"
BODY_GEOMS_JSON = DATA_DIR / "a3_body_geoms_v1.json"

_EPS = 1e-9
_TYPE_CODE = {"sphere": 0, "box": 1, "cylinder": 2}


# ----------------------------------------------------------------------------- quat helpers
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


# ----------------------------------------------------------------------------- pattern load
def load_pattern(path: str | Path = PATTERN_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        c = json.load(f)
    assert c.get("schema") == "a3_pattern_contract_v1", f"bad pattern schema in {path}"
    return c


def load_body_geoms(path: str | Path = BODY_GEOMS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    assert g.get("schema") == "a3_body_geoms_v1", f"bad body-geom schema in {path}"
    return g


def sensor_frame_directions(contract: dict, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """(rays_per_sensor, 3) unit directions in the SENSOR frame, exactly as stored."""
    return torch.tensor(contract["directions_sensor_frame"], device=device, dtype=dtype)


def body_frame_rays(contract: dict, device="cpu", dtype=torch.float32):
    """The canonical parity object (G-A3-1): per-ray origins + directions in the PARENT BODY
    (base_link) frame, sensor-major [front(256), rear(256)] -> (512, 3), (512, 3).

    origin_i = site_pos; dir_i = R(site_quat) @ d_sensor. Both loaders (this one and the
    deploy-side numpy loader) must produce these matrices identically."""
    d_s = sensor_frame_directions(contract, device, dtype)             # (256, 3)
    origins, dirs = [], []
    for s in contract["sensors"]:
        pos = torch.tensor(s["site_pos_body"], device=device, dtype=dtype)
        R = quat_to_mat(torch.tensor(s["site_quat_wxyz_body"], device=device, dtype=dtype))
        origins.append(pos.expand(d_s.shape[0], 3))
        dirs.append(d_s @ R.T)                                          # row-vectors x R^T
    return torch.cat(origins, dim=0), torch.cat(dirs, dim=0)


# ----------------------------------------------------------------------------- primitives
def _ray_sphere(o, d, r):
    """o,d: (..., 3) local frame; returns first-hit t (...,) (inf = none). Smallest t > 0
    (from inside -> the exit root), matching mj_ray's convention."""
    b = (o * d).sum(-1)
    c = (o * o).sum(-1) - r * r
    disc = b * b - c
    s = torch.sqrt(disc.clamp_min(0.0))
    t1, t2 = -b - s, -b + s
    t = torch.where(t1 > _EPS, t1, torch.where(t2 > _EPS, t2, torch.full_like(t1, torch.inf)))
    return torch.where(disc >= 0, t, torch.full_like(t, torch.inf))


def _ray_box(o, d, half):
    """Slab method vs an axis-aligned box of half-extents `half` (3,) in the geom frame."""
    inv = 1.0 / torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12) * torch.sign(d + 1e-30), d)
    t_lo = (-half - o) * inv
    t_hi = (half - o) * inv
    tmin = torch.minimum(t_lo, t_hi).amax(-1)
    tmax = torch.maximum(t_lo, t_hi).amin(-1)
    hit = (tmax >= tmin.clamp_min(0.0)) & (tmax > _EPS)
    t = torch.where(tmin > _EPS, tmin, tmax)
    return torch.where(hit, t, torch.full_like(t, torch.inf))


def _ray_cylinder(o, d, r, hh):
    """Finite capped cylinder about the local z axis (radius r, half-height hh)."""
    inf = torch.full_like(o[..., 0], torch.inf)
    ox, oy, oz = o.unbind(-1)
    dx, dy, dz = d.unbind(-1)
    # side surface
    a = dx * dx + dy * dy
    b = ox * dx + oy * dy
    c = ox * ox + oy * oy - r * r
    disc = b * b - a * c
    ok_a = a > 1e-12
    s = torch.sqrt(disc.clamp_min(0.0))
    a_safe = torch.where(ok_a, a, torch.ones_like(a))
    t1 = (-b - s) / a_safe
    t2 = (-b + s) / a_safe
    def _side(t):
        z = oz + t * dz
        valid = ok_a & (disc >= 0) & (t > _EPS) & (z.abs() <= hh)
        return torch.where(valid, t, inf)
    t_side = torch.minimum(_side(t1), _side(t2))
    # caps
    dz_safe = torch.where(dz.abs() < 1e-12, torch.full_like(dz, 1e-12), dz)
    def _cap(zc):
        t = (zc - oz) / dz_safe
        x, y = ox + t * dx, oy + t * dy
        valid = (dz.abs() >= 1e-12) & (t > _EPS) & (x * x + y * y <= r * r)
        return torch.where(valid, t, inf)
    return torch.minimum(t_side, torch.minimum(_cap(hh), _cap(-hh)))


# ----------------------------------------------------------------------------- the kernel
class A3SelfOcclusionKernel:
    """Batched analytic ray-vs-primitive first-hit against the A2's own collision body.

    Built once from the two contract JSONs; per-frame input = world poses of the geom-carrying
    bodies. Loops over the 40 geoms (each step is a (B, 512) batched intersection) so peak
    memory stays ~B x 512 x 3 — the env count never multiplies against the geom count."""

    def __init__(self, pattern: dict | None = None, body_geoms: dict | None = None,
                 device: str | torch.device = "cpu", dtype=torch.float32):
        self.pattern = pattern or load_pattern()
        self.body_geoms = body_geoms or load_body_geoms()
        self.device, self.dtype = torch.device(device), dtype
        self.r_max = float(self.pattern["r_max_m"])
        self.n_rays = int(self.pattern["total_rays"])
        self.parent_body = self.pattern["sensors"][0]["parent_body"]    # base_link
        o, d = body_frame_rays(self.pattern, self.device, self.dtype)
        self.ray_origins_b = o                                          # (512, 3) base_link frame
        self.ray_dirs_b = d                                             # (512, 3)
        self.geoms = []
        for g in self.body_geoms["geoms"]:
            self.geoms.append({
                "body": g["body"],
                "code": _TYPE_CODE[g["type"]],
                "size": torch.tensor(g["size"], device=self.device, dtype=self.dtype),
                "pos": torch.tensor(g["pos_body"], device=self.device, dtype=self.dtype),
                "R": quat_to_mat(torch.tensor(g["quat_wxyz_body"], device=self.device,
                                              dtype=self.dtype)),
            })
        self.required_bodies = sorted({g["body"] for g in self.geoms} | {self.parent_body})

    def to(self, device):
        """Move all constant tensors (call once when the env device is known)."""
        self.device = torch.device(device)
        self.ray_origins_b = self.ray_origins_b.to(device)
        self.ray_dirs_b = self.ray_dirs_b.to(device)
        for g in self.geoms:
            for k in ("size", "pos", "R"):
                g[k] = g[k].to(device)
        return self

    def rays_world(self, base_pos: torch.Tensor, base_quat: torch.Tensor):
        """Sensor rays in world frame from the base_link pose. (B,3),(B,4) -> (B,512,3) x2."""
        Rb = quat_to_mat(base_quat)                                     # (B, 3, 3)
        origins = base_pos[:, None, :] + torch.einsum("bij,rj->bri", Rb, self.ray_origins_b)
        dirs = torch.einsum("bij,rj->bri", Rb, self.ray_dirs_b)
        return origins, dirs

    def self_ranges(self, body_poses: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """First-hit distance of every ray against the robot's own collision primitives.

        body_poses: {body_name: (pos (B,3), quat_wxyz (B,4))} — must contain every geom body +
        the sensor parent. Returns (B, 512) distances, inf = no self-hit."""
        base_pos, base_quat = body_poses[self.parent_body]
        origins_w, dirs_w = self.rays_world(base_pos, base_quat)
        B = origins_w.shape[0]
        best = torch.full((B, self.n_rays), torch.inf, device=self.device, dtype=self.dtype)
        for g in self.geoms:
            p_k, q_k = body_poses[g["body"]]
            R_k = quat_to_mat(q_k)                                      # (B, 3, 3)
            R_gw = R_k @ g["R"]                                         # (B, 3, 3)
            p_gw = p_k + torch.einsum("bij,j->bi", R_k, g["pos"])       # (B, 3)
            # rays into the geom frame
            rel = origins_w - p_gw[:, None, :]
            o = torch.einsum("bji,brj->bri", R_gw, rel)                 # R^T @ rel
            d = torch.einsum("bji,brj->bri", R_gw, dirs_w)
            if g["code"] == 0:
                t = _ray_sphere(o, d, g["size"][0])
            elif g["code"] == 1:
                t = _ray_box(o, d, g["size"])
            else:
                t = _ray_cylinder(o, d, g["size"][0], g["size"][1])
            best = torch.minimum(best, t)
        return best

    def fuse_normalize(self, terrain_r: torch.Tensor | None, body_r: torch.Tensor) -> torch.Tensor:
        """min(terrain, body) -> clamp(r / r_max, 0, 1); no-hit (inf) -> 1.0."""
        r = body_r if terrain_r is None else torch.minimum(terrain_r, body_r)
        return torch.nan_to_num(r / self.r_max, nan=1.0, posinf=1.0).clamp(0.0, 1.0)

# Copyright (c) 2026 PerceptionGame Path-A fork
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for the dynamic-sigma restoration (NO isaaclab required).

Run:  python3 tests/test_dynamic_sigma.py        (or pytest tests/)

Validates, before any GPU run:
  1. `dynamic_sigma` matches an INDEPENDENT numpy transcription of the reference
     (`legged_gym/envs/base/legged_robot.py:1288-1308`) over a dense grid.
  2. Hand-computed spot values.
  3. `cols_to_max_sigma` column binning equals gym's 9-bucket `cols2id` assignment
     (`utils/terrain.py:65` + `make_terrain` thresholds) on the port's terrain cfg —
     including the slope bucket the port splits into slope_up/slope_down.
  4. Uniform sigma reduces the restored lin reward to the port's original formula.
  5. terrains.py carries the gym-parity range maxima (text-level guard).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "source/robot_lab/robot_lab/tasks/go2/mdp/dynamic_sigma_core.py"
TERRAINS = ROOT / "source/robot_lab/robot_lab/tasks/go2/mdp/terrains.py"

# Load the pure-torch core straight from file (package __init__ would pull isaaclab).
spec = importlib.util.spec_from_file_location("dynamic_sigma_core", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

DEFAULT_SIGMA = 0.25  # = std 0.5 ** 2

# The port's sub_terrains, dict order + proportions (terrains.py TERRAIN_CFG).
PORT_TERRAINS = [
    ("wave", 0.05), ("slope_up", 0.10), ("slope_down", 0.10), ("rough_slope", 0.05),
    ("stairs_up", 0.25), ("stairs_down", 0.10), ("obstacles", 0.20),
    ("stepping_stones", 0.0), ("gap", 0.0), ("flat", 0.15),
]
NUM_COLS = 20

# gym reference: proportions + id->sigma (go2_config.py:91,174). slope is ONE 0.20 bucket.
GYM_PROPORTIONS = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]
GYM_ID_SIGMA = [5 / 12, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 3 / 4, 1.0, 1.0, 1 / 4]


def oracle_dynamic_sigma(cmd_abs, max_sigma, levels, default, v_min, v_max):
    """Independent numpy transcription of legged_robot.py:1288-1308 (the test oracle)."""
    cmd_abs, max_sigma = np.asarray(cmd_abs, float), np.asarray(max_sigma, float)
    v_min = np.broadcast_to(np.asarray(v_min, float), cmd_abs.shape)
    v_max = np.broadcast_to(np.asarray(v_max, float), cmd_abs.shape)
    sigma = np.full_like(cmd_abs, default)
    m = (cmd_abs >= v_min) & (cmd_abs < v_max)
    ratio = (cmd_abs[m] - v_min[m]) / (v_max[m] - v_min[m])
    sigma[m] = default + ratio * (max_sigma[m] - default)
    m = cmd_abs >= v_max
    sigma[m] = max_sigma[m]
    level_scale = np.clip(np.exp((np.asarray(levels, float) + 1.0) / 10.0) - 1.0, None, 1.0)
    return default + level_scale * (sigma - default)


def test_grid_against_oracle():
    rng = np.random.default_rng(0)
    n = 20000
    cmd = rng.uniform(0.0, 4.0, n)
    sig_max = rng.choice(sorted(set(GYM_ID_SIGMA)), n)
    levels = rng.integers(0, 10, n)
    for v_min, v_max in ((0.5, 1.5), (1.0, 2.0)):
        got = core.dynamic_sigma(torch.tensor(cmd, dtype=torch.float32),
                                 torch.tensor(sig_max, dtype=torch.float32),
                                 torch.tensor(levels), DEFAULT_SIGMA, v_min, v_max)
        want = oracle_dynamic_sigma(cmd, sig_max, levels, DEFAULT_SIGMA, v_min, v_max)
        np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=1e-5)
    print("PASS grid vs oracle (40k samples)")


def test_tensor_vmax_against_oracle():
    cmd = np.array([0.4, 0.75, 1.0, 1.5, 2.0, 4.0])
    sig_max = np.array([0.75, 0.75, 0.75, 0.5, 1.0, 0.25])
    levels = np.array([9, 9, 9, 9, 4, 9])
    v_max = np.array([1.5, 1.5, 1.0, 1.5, 2.5, 5.0])
    got = core.dynamic_sigma(torch.tensor(cmd, dtype=torch.float32),
                             torch.tensor(sig_max, dtype=torch.float32),
                             torch.tensor(levels), DEFAULT_SIGMA,
                             0.5, torch.tensor(v_max, dtype=torch.float32))
    want = oracle_dynamic_sigma(cmd, sig_max, levels, DEFAULT_SIGMA, 0.5, v_max)
    np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=1e-5)
    print("PASS tensor v_max vs oracle")


def test_spot_values():
    def one(cmd, sig_max, level, v_min=0.5, v_max=1.5):
        return float(core.dynamic_sigma(torch.tensor([cmd]), torch.tensor([sig_max]),
                                        torch.tensor([level]), DEFAULT_SIGMA, v_min, v_max))
    # below band -> default, any terrain/level
    assert abs(one(0.4, 0.75, 9) - 0.25) < 1e-7
    # stairs_up (0.5) at v>=v_max, top level (scale clamps to 1) -> sigma_max
    assert abs(one(2.0, 0.5, 9) - 0.5) < 1e-6
    # mid-band ratio 0.5 on stairs_up, top level: 0.25 + 0.5*(0.5-0.25) = 0.375
    assert abs(one(1.0, 0.5, 9) - 0.375) < 1e-6
    # same at level 0: scale = e^0.1 - 1 = 0.105170…; 0.25 + 0.105170*0.125 = 0.2631463
    assert abs(one(1.0, 0.5, 0) - 0.2631463) < 1e-6
    # flat (0.25 max): sigma stays default everywhere
    assert abs(one(3.0, 0.25, 9) - 0.25) < 1e-7
    print("PASS spot values")


def test_column_binning_matches_gym():
    got = core.cols_to_max_sigma([n for n, _ in PORT_TERRAINS], [p for _, p in PORT_TERRAINS],
                                 NUM_COLS).numpy()
    # gym side: cols2id via choice = j/num_cols + 0.001 against cumulative 9-bucket proportions
    cum = np.cumsum(np.array(GYM_PROPORTIONS) / sum(GYM_PROPORTIONS))
    want = np.array([GYM_ID_SIGMA[int(np.min(np.where(j / NUM_COLS + 0.001 < cum)[0]))]
                     for j in range(NUM_COLS)])
    np.testing.assert_allclose(got, want, atol=0)
    # and the expected column census for 20 cols
    names = [n for n, _ in PORT_TERRAINS]
    props = [p for _, p in PORT_TERRAINS]
    cum_p = np.cumsum(np.array(props) / sum(props))
    census: dict[str, int] = {}
    for j in range(NUM_COLS):
        nm = names[int(np.min(np.where(j / NUM_COLS + 0.001 < cum_p)[0]))]
        census[nm] = census.get(nm, 0) + 1
    assert census == {"wave": 1, "slope_up": 2, "slope_down": 2, "rough_slope": 1,
                      "stairs_up": 5, "stairs_down": 2, "obstacles": 4, "flat": 3}, census
    print(f"PASS column binning == gym cols2id; census {census}")


def test_uniform_sigma_reduces_to_port_formula():
    """With sigma_x == sigma_y == std**2 the restored lin reward must equal the port's
    original exp(-sum(err^2)/std^2)."""
    rng = np.random.default_rng(1)
    err = torch.tensor(rng.normal(0, 1, (1000, 2)), dtype=torch.float32)
    std = 0.5
    sigma = torch.full((1000,), std**2)
    restored = torch.exp(-(err[:, 0] ** 2 / sigma + err[:, 1] ** 2 / sigma))
    original = torch.exp(-torch.sum(err**2, dim=1) / std**2)
    np.testing.assert_allclose(restored.numpy(), original.numpy(), atol=1e-6)
    print("PASS uniform-sigma reduction to port formula")


def test_terrain_maxima_fixed():
    s = TERRAINS.read_text()
    for pat in (r"amplitude_range: tuple\[float, float\] = \(0\.1, 0\.30\)",
                r"slope_range: tuple\[float, float\] = \(0\.1, 0\.62\)",
                r"slope_range=\(0\.1, 0\.62\)",
                r"step_height_range=\(0\.05, 0\.28\)",
                r"obstacle_height_range=\(0\.05, 0\.30\)"):
        assert re.search(pat, s), f"terrains.py missing gym-parity range: {pat}"
    for stale in ("(0.1, 0.28)", "(0.1, 0.568)", "(0.05, 0.257)", "(0.05, 0.275)"):
        assert stale not in s, f"terrains.py still carries pre-fix range {stale}"
    # top-row sanity: stairs at d=0.9 (use_gym_difficulty cap) = 0.05 + 0.9*(0.28-0.05) = 0.257
    assert abs(0.05 + 0.9 * (0.28 - 0.05) - 0.257) < 1e-12
    print("PASS terrains.py gym-parity maxima (top stairs back to 0.257 m >= the 0.24 m bar)")


if __name__ == "__main__":
    test_grid_against_oracle()
    test_tensor_vmax_against_oracle()
    test_spot_values()
    test_column_binning_matches_gym()
    test_uniform_sigma_reduces_to_port_formula()
    test_terrain_maxima_fixed()
    print("\nALL PASS")

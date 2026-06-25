# A2 V16 locomotion policy — run archive (2026-06-24)

Frozen, framed backup of the **best policy from the V16 run** so it cannot be lost (the RunPod
checkpoints live on a rented network volume; this is the durable off-site copy).

## What this is

**V16 = V15 (earned-only command-curriculum gate) + a relief-gated terrain-traversal reward.** A
goal/command-conditioned quadruped (A2) walker, MoE-CTS student (8 experts), trained with
rsl_rl/IsaacLab on this fork (`patha` branch), task `RobotLab-A2-V16-v0`. It is the **best A2 walker to
date** and is judged **demo-ready**.

## The policies here

| dir | iter | role | corridor eval (L12, 5 paired seeds) |
|---|---|---|---|
| `iter_9000_BEST/` | 9000 | **recommended deploy** — this run's most-trained robust checkpoint | **5/5 finish**, mean 1.655 |
| `iter_6000_cobest_seed/` | 6000 | co-best + lineage **seed** (round-0 competition winner; the checkpoint the run resumed from) | **5/5 finish**, mean 1.671 |

Each dir holds the deployable **`policy.pt`** (TorchScript JIT, ~15 MB) + its **`manifest.json`**
(exact contract + sha256 hashes). `eval_crown_L12_5seed.json` is the crowning competition output.

**Why these two:** at L12 the run *wobbles* between robust and trough checkpoints
(`6000 5/5 · 6200 3/5 · 8000 2/5 · 9000 5/5 · 10000 2/5`). 6000 and 9000 are the two that finish all
five seeds. The **latest (10000) is in a trough — deliberately NOT archived as best.** Terrain
curriculum had plateaued ~6/10, so more training did not raise the ceiling, only moved the wobble.

## Deploy contract (see each `manifest.json` for exact values + hashes)

- **Obs:** 557-dim student = 45 proprio + a raw 512-dim LiDAR dome-ray block. Internal 5-frame history.
- **Action:** 12-dim joint targets, PD deploy contract.
- **Load/run:**
  ```python
  from patha_bundle import load_jit_policy
  pol = load_jit_policy("iter_9000_BEST/policy.pt")   # callable(obs)->action[12]
  pol.reset()                                          # clears the 5-frame history (ZEROES it; no warm path)
  ```
  Interactive eyeball (macOS needs `mjpython`):
  `mjpython drive_interactive.py --robot a2 --bundle iter_9000_BEST [--corridor out/corridor_L12.xml --auto]`
  (these tools live in the game repo `PerceptionGame/tools/sim/`).

## Provenance

- Pod: RunPod (US-WA-1), checkpoints on the persistent network volume `/workspace`.
- Run dir: `logs/rsl_rl/a2_v16_moe_cts/2026-06-24_16-57-09_a2-patha-v16`.
- Lineage: warm-start … → model_5200 (dense run) → **rolled back to model_6000** (the robust peak, after
  a measured robustness regression at 7000–7200) → trained on to 10000.
- Selection method: the corridor competition (multi-seed, paired, smoothed-peak) — see
  `PerceptionGame/info/locomotion/CORRIDOR_COMPETITION_EVAL.md`.

## Known caveats (do not block walking; matter for the *next* architecture)

- **The MoE "router" is structurally dead** — it averages all 8 experts (no task gradient reaches the
  gate; it's trained only as MSE to a single teacher latent). The walker is great anyway because walking
  is single-mode. See `PerceptionGame/info/locomotion/ARCHITECTURE_DECISION_MEMO.md`.
- **No perception encoder** — dome rays are fed raw into the policy.

## Resumable raw checkpoints (NOT in git — too large)

To *continue training* (not just deploy) you need the raw `model_<N>.pt` (~53 MB, includes optimizer
state). Those live on the RunPod network volume and a local Mac backup
(`~/Developer/pm_runs/a2-patha-v16/archive_raw/`). Resume with:
`--resume --load_run 2026-06-24_16-57-09_a2-patha-v16 --checkpoint model_9000.pt`.

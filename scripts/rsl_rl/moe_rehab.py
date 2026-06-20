"""MoE rehab — checkpoint surgery to revive a collapsed/frozen MoE-CTS gate.

WHY: the A2 student MoE gate saturated (logits ~±1000) → one-hot routing onto 2/8 experts →
exactly-zero gradient → frozen for ~all of training (see info/locomotion/HANDOFF.md ROOT CAUSE).
The post-softmax load-balance loss is powerless to recover a saturated gate, so we surgically
de-saturate the gate and revive the dead experts BEFORE resuming training (which must also add
logit-level anti-saturation — z-loss/temperature — or it just re-collapses; that's the algo change).

WHAT (operates on the checkpoint weights only; training restarts optimizers fresh):
  1. KEEP actor / teacher / critic / live experts — untouched (warm-start preserved).
  2. RESET the gate logit head (gating MLP final Linear) to small weights → logits ~O(0.1),
     softmax de-saturated, gradient can flow again. Earlier gate layers (feature extraction) kept.
  3. SEED each dead expert as a FUNCTION-PRESERVING CLONE of a live expert (backbone output slice
     + grouped-conv head slice) + small relative noise → sane latents from step 0, breaks symmetry
     so they can diverge. (Random reinit would emit garbage latents AND the gate wouldn't route to
     them — they'd stay dead or corrupt the actor.)
  4. STRIP optimizer state from the saved checkpoint → V11 starts Adam fresh (stale momentum on
     reset params is wrong; a rehab phase wants clean momentum anyway).

Live vs dead experts are AUTO-DETECTED from gate usage (no hand-coded indices).

USAGE
  python3 moe_rehab.py --in  /path/model_31000.pt  --out /path/model_31000_rehab.pt
  # then probe the output gate health (entropy/usage) before trusting it.
"""
from __future__ import annotations
import argparse, copy, math
import torch
import torch.nn as nn

GATE_PREFIX = "student_moe_encoder.moe.gating_network.0.network."   # MLP layers .0/.2/.4/.6
BB_LAST     = "student_moe_encoder.moe.experts.backbone.network.4"  # (expert_num*hid, in) shared-feat -> expert feats
CONV_HEAD   = "student_moe_encoder.moe.experts.experts"             # grouped Conv1d (expert_num*out, in/grp, 1)


def _gate_mlp_layers(sd):
    """Return the sorted integer indices of the gating MLP's Linear layers (e.g. [0,2,4,6])."""
    idx = set()
    for k in sd:
        if k.startswith(GATE_PREFIX) and k.endswith(".weight"):
            idx.add(int(k[len(GATE_PREFIX):].split(".")[0]))
    return sorted(idx)


def _detect_expert_usage(sd, n_samples=4096, seed=0):
    """Rebuild the gating MLP, run standard-normal (~normalized obs) inputs, return mean per-expert usage."""
    layers = _gate_mlp_layers(sd)
    in_dim = sd[f"{GATE_PREFIX}{layers[0]}.weight"].shape[1]
    mods = []
    for i in layers:
        w = sd[f"{GATE_PREFIX}{i}.weight"]; b = sd[f"{GATE_PREFIX}{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0]); lin.weight.data = w.float().clone(); lin.bias.data = b.float().clone()
        mods.append(lin)
        if i != layers[-1]:
            mods.append(nn.ELU())
    net = nn.Sequential(*mods).eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, in_dim, generator=g)
    with torch.no_grad():
        logits = net(x); probs = torch.softmax(logits, dim=-1)
    return probs.mean(0), logits, probs


def _stats(probs, logits):
    ent = -(probs.clamp_min(1e-9).log() * probs).sum(-1).mean().item()
    return dict(usage=[round(u, 4) for u in probs.mean(0).tolist()],
                entropy=round(ent, 4), max_entropy=round(math.log(probs.shape[-1]), 4),
                peak=round(probs.max(-1).values.mean().item(), 4),
                logit_absmax=round(logits.abs().max().item(), 1))


def rehab(sd, live_thresh=0.05, gate_head_std=0.02, seed_noise=0.05, seed=0):
    sd = copy.deepcopy(sd)
    usage, logits0, probs0 = _detect_expert_usage(sd, seed=seed)
    n_exp = usage.numel()
    live = [i for i in range(n_exp) if usage[i].item() >= live_thresh]
    dead = [i for i in range(n_exp) if usage[i].item() < live_thresh]
    print(f"[rehab] experts={n_exp}  live={live}  dead={dead}  (usage={[round(u,4) for u in usage.tolist()]})")
    if not live:
        raise SystemExit("[rehab] no live experts above threshold — refuse to seed from nothing.")

    gen = torch.Generator().manual_seed(seed)

    # --- (2) de-saturate the gate: reinit the logit head (final gate Linear) small ---
    layers = _gate_mlp_layers(sd)
    hk_w, hk_b = f"{GATE_PREFIX}{layers[-1]}.weight", f"{GATE_PREFIX}{layers[-1]}.bias"
    sd[hk_w] = torch.randn(sd[hk_w].shape, generator=gen) * gate_head_std
    sd[hk_b] = torch.zeros_like(sd[hk_b])

    # --- (3) seed dead experts as clones of live experts (+ noise) ---
    bb_w, bb_b = sd[f"{BB_LAST}.weight"], sd[f"{BB_LAST}.bias"]   # (n_exp*hid, in), (n_exp*hid)
    cv_w, cv_b = sd[f"{CONV_HEAD}.weight"], sd[f"{CONV_HEAD}.bias"]  # (n_exp*out, in/grp, 1), (n_exp*out)
    hid = bb_w.shape[0] // n_exp
    out = cv_w.shape[0] // n_exp

    def blk(t, k, sz):  # slice for expert k along dim0
        return slice(k * sz, (k + 1) * sz)

    def seed_slice(dst, src_k, dst_k, sz):
        s = dst[blk(dst, src_k, sz)]
        noise = torch.randn(s.shape, generator=gen) * seed_noise * (s.std() + 1e-8)
        dst[blk(dst, dst_k, sz)] = s + noise

    for n, dk in enumerate(dead):
        src = live[n % len(live)]   # round-robin over live donors
        seed_slice(bb_w, src, dk, hid)
        seed_slice(bb_b, src, dk, hid)
        seed_slice(cv_w, src, dk, out)
        seed_slice(cv_b, src, dk, out)
        print(f"[rehab] expert {dk} <- clone(expert {src}) + {seed_noise:.0%} noise")

    # report post-surgery gate health (head is random now, so this reflects de-saturation only)
    _, logits1, probs1 = _detect_expert_usage(sd, seed=seed + 1)
    print(f"[rehab] gate BEFORE: {_stats(probs0, logits0)}")
    print(f"[rehab] gate AFTER : {_stats(probs1, logits1)}")
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--live-thresh", type=float, default=0.05)
    ap.add_argument("--gate-head-std", type=float, default=0.02)
    ap.add_argument("--seed-noise", type=float, default=0.05)
    args = ap.parse_args()

    ck = torch.load(args.inp, map_location="cpu", weights_only=False)
    print(f"[rehab] checkpoint top keys: {list(ck.keys())}")
    ck["model_state_dict"] = rehab(ck["model_state_dict"],
                                   live_thresh=args.live_thresh,
                                   gate_head_std=args.gate_head_std,
                                   seed_noise=args.seed_noise)
    # (4) strip optimizer state so the resumed run starts Adam fresh on the surgically-changed params
    stripped = [k for k in list(ck.keys()) if "optim" in k.lower()]
    for k in stripped:
        ck[k] = {}
    print(f"[rehab] stripped optimizer state: {stripped}")
    torch.save(ck, args.out)
    print(f"[rehab] wrote {args.out}")


if __name__ == "__main__":
    main()

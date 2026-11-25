"""Give a trained policy a wider network without changing what it does.

    python python/train/widen.py runs/league/latest.pt runs/league-wide/start.pt --hidden 1024

Every hidden layer of the actor and the critic grows to ``--hidden`` units. The old units keep their weights;
the new ones get fresh incoming weights and outgoing weights of zero, so the wider network computes exactly
the same actions and values as before and training can simply go on (``ppo.py --resume``). The optimizer
state is dropped: its statistics do not fit the new shapes.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from policy import MLPPolicy, load_checkpoint, save_checkpoint


def widen_linear(old: nn.Linear, new: nn.Linear, grow_in: bool, grow_out: bool) -> None:
    """Copy ``old`` into the top-left corner of ``new``. Weights from new inputs to old outputs are zero (so old outputs
    are unchanged); weights into new outputs keep their fresh initialisation."""
    o, i = old.weight.shape
    with torch.no_grad():
        if grow_in:
            new.weight[:o, i:] = 0
        new.weight[:o, :i] = old.weight
        new.bias[:o] = old.bias
        if grow_out:
            new.bias[o:] = 0


def widen(policy: MLPPolicy, hidden: int) -> MLPPolicy:
    cfg = dict(policy.config)
    if hidden < cfg["hidden"]:
        raise ValueError("the new network must be at least as wide as the old one")
    cfg.pop("kind")
    cfg["hidden"] = hidden
    wide = MLPPolicy(**cfg)
    for name in ("body", "critic"):
        old_seq, new_seq = getattr(policy, name), getattr(wide, name)
        if old_seq is None:
            continue
        linears = [(a, b) for a, b in zip(old_seq, new_seq) if isinstance(a, nn.Linear)]
        for k, (a, b) in enumerate(linears):
            widen_linear(a, b, grow_in=k > 0, grow_out=True)
    for name in ("kind_head", "dir_head", "dist_head", "target_head", "value"):
        widen_linear(getattr(policy, name), getattr(wide, name), grow_in=True, grow_out=False)
    return wide


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--hidden", type=int, required=True)
    args = p.parse_args()
    policy, data = load_checkpoint(args.checkpoint, "cpu")
    wide = widen(policy, args.hidden)
    # The same inputs must give the same outputs.
    cfg = policy.config
    obs = torch.rand(256, cfg["obs_size"])
    obs[:, cfg["hero_offset"]:cfg["hero_offset"] + 5] = torch.eye(5)[torch.randint(0, 5, (256,))]
    mask = torch.ones(256, cfg["nvec"][0] + cfg["nvec"][3], dtype=torch.uint8)
    state = torch.rand(256, cfg["state_size"]) if cfg["state_size"] else None
    with torch.no_grad():
        act = policy(obs, mask, greedy=True, state=state)
        again = wide(obs, mask, action=act[0], state=state)
    err = max(float((act[1] - again[1]).abs().max()), float((act[3] - again[3]).abs().max()))
    assert err < 1e-4, f"the widened network differs from the original by {err}"
    save_checkpoint(args.out, wide, None, **{**data["meta"], "widened_from": cfg["hidden"]})
    n = sum(x.numel() for x in wide.parameters())
    print(f"wrote {args.out}: hidden {cfg['hidden']} -> {args.hidden}, {n / 1e6:.2f}M parameters, same outputs to {err:.1e}")

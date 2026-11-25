"""Policy networks and checkpoints shared by the trainers and the evaluation harness.

A policy maps a batch of per-hero observations and action masks to a MultiDiscrete action
[kind, direction, distance, target], its log-probability, the entropy and a value estimate.
One network plays every hero on either side (the hero is one-hot in the observation and
Red's view is mirrored by the environment).

The action is chosen in order: first the kind, then - given the kind - the target, direction and
distance. Only the components the kind uses count towards the log-probability (the environment's
``action_heads`` / ``action_targets`` tables say which, and which target slots fit).

The value estimate may additionally see ``state``, the environment's fog-free team state. It is
only needed for training: without it the policy still acts and simply reports a value of zero.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

NO_LABEL = 255  # in an expert label: this component does not matter; as the kind: no usable label
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


def mlp(width: int, hidden: int, layers: int, activation: str = "tanh") -> nn.Sequential:
    mods: list[nn.Module] = []
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), ACTIVATIONS[activation]()]
        width = hidden
    return nn.Sequential(*mods)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
    return F.log_softmax(logits, dim=-1)


class MLPPolicy(nn.Module):
    """An MLP body; a head for the action kind; target, direction and distance heads with one output set per kind.

    * ``action_heads`` (hero x kind x [direction, distance, target]) and ``action_targets`` (hero x kind x slot)
      are the environment's tables; ``hero_offset`` is where the observation holds the one-hot hero.
    * ``state_size`` > 0: the critic is a separate network over the observation plus the fog-free team state
      (a centralised critic); otherwise the value is a head on the actor's body.
    """

    def __init__(self, obs_size: int, nvec: list[int], hidden: int = 256, layers: int = 2, state_size: int = 0,
                 action_heads: list | None = None, action_targets: list | None = None, hero_offset: int = 0, activation: str = "tanh"):
        super().__init__()
        self.config = {"kind": "mlp", "obs_size": obs_size, "nvec": list(nvec), "hidden": hidden, "layers": layers, "state_size": state_size,
                       "action_heads": action_heads, "action_targets": action_targets, "hero_offset": hero_offset, "activation": activation}
        self.nvec = list(nvec)
        nk, nd, ns, nt = self.nvec
        self.state_size = state_size
        self.hero_offset = hero_offset
        self.body = mlp(obs_size, hidden, layers, activation)
        self.kind_head = nn.Linear(hidden, nk)
        self.dir_head = nn.Linear(hidden, nk * nd)
        self.dist_head = nn.Linear(hidden, nk * ns)
        self.target_head = nn.Linear(hidden, nk * nt)
        self.critic = mlp(obs_size + state_size, hidden, layers, activation) if state_size else None
        self.value = nn.Linear(hidden, 1)
        heads = torch.ones(5, nk, 3) if action_heads is None else torch.as_tensor(action_heads, dtype=torch.float32)
        targets = torch.ones(5, nk, nt) if action_targets is None else torch.as_tensor(action_targets, dtype=torch.float32)
        self.register_buffer("heads_used", heads, persistent=False)
        self.register_buffer("targets_ok", targets > 0, persistent=False)

    # -- pieces ---------------------------------------------------------------------------------
    def hero_of(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.hero_offset:self.hero_offset + self.heads_used.shape[0]].argmax(-1)

    def kind_log_probs(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_log_softmax(self.kind_head(h), mask[:, :self.nvec[0]].bool())

    def sub_log_probs(self, h: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, kind: torch.Tensor):
        """Log-probabilities of every direction, distance and target, given the action kind (B,)."""
        nk, nd, ns, nt = self.nvec
        pick = lambda head, n: head(h).view(-1, nk, n).gather(1, kind.view(-1, 1, 1).expand(-1, 1, n)).squeeze(1)  # noqa: E731
        ok = mask[:, nk:].bool() & self.targets_ok[self.hero_of(obs), kind]
        ok = ok | ~ok.any(dim=1, keepdim=True)  # the kind takes no target: the head is ignored
        return F.log_softmax(pick(self.dir_head, nd), -1), F.log_softmax(pick(self.dist_head, ns), -1), masked_log_softmax(pick(self.target_head, nt), ok)

    def used(self, obs: torch.Tensor, kind: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B, 3) floats: whether direction, distance and target count for this action."""
        u = self.heads_used[self.hero_of(obs), kind]
        aimed_at_unit = (u[:, 2] > 0) & (target != self.nvec[3] - 1)  # with a real target, direction and distance are not used
        free = (~aimed_at_unit).float()
        return torch.stack([u[:, 0] * free, u[:, 1] * free, u[:, 2]], dim=1)

    def value_of(self, obs: torch.Tensor, h: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if self.critic is None:
            return self.value(h).squeeze(-1)
        if state is None:
            return obs.new_zeros(obs.shape[0])  # acting only: the critic needs the privileged state
        return self.value(self.critic(torch.cat([obs, state], dim=-1))).squeeze(-1)

    # -- acting and scoring ------------------------------------------------------------------------
    def forward(self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None, greedy: bool = False, state: torch.Tensor | None = None):
        """obs (B, obs_size), mask (B, 8 + 22), state (B, state_size) or None -> action (B, 4), log-prob (B), entropy (B), value (B)."""
        h = self.body(obs)

        def choose(lp: torch.Tensor, i: int) -> torch.Tensor:
            if action is not None:
                return action[:, i]
            return lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1).squeeze(-1)

        lp_kind = self.kind_log_probs(h, mask)
        kind = choose(lp_kind, 0)
        lp_dir, lp_dist, lp_target = self.sub_log_probs(h, obs, mask, kind)
        target = choose(lp_target, 3)
        act = torch.stack([kind, choose(lp_dir, 1), choose(lp_dist, 2), target], dim=1)
        used = self.used(obs, kind, target)
        subs = (lp_dir, lp_dist, lp_target)
        at = lambda lp, a: lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)  # noqa: E731
        ent = lambda lp: -(lp.exp() * lp).sum(-1)  # noqa: E731
        logp = at(lp_kind, kind) + sum(used[:, j] * at(subs[j], act[:, j + 1]) for j in range(3))
        # Entropy of the kind plus, for the kind and target at hand, the entropy of the components they use.
        entropy = ent(lp_kind) + sum(used[:, j] * ent(subs[j]) for j in range(3))
        return act, logp, entropy, self.value_of(obs, h, state)


def imitation_loss(policy: MLPPolicy, obs: torch.Tensor, mask: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy between the policy and the expert's actions ``label`` (B, 4; the environment's ``env.expert``),
    over the components each action uses. Also returns how often the policy's most likely choice is the expert's;
    direction, distance and target are judged given the expert's kind."""
    h = policy.body(obs)
    label = label.long()
    kind = label[:, 0]
    has = kind != NO_LABEL
    lp_kind = policy.kind_log_probs(h, mask)
    subs = policy.sub_log_probs(h, obs, mask, kind.clamp(max=policy.nvec[0] - 1))
    loss, acc = 0.0, {}
    for i, (name, lp) in enumerate(zip(("kind", "dir", "dist", "target"), (lp_kind, *subs))):
        lab = label[:, i]
        picked = lp.gather(1, lab.clamp(max=policy.nvec[i] - 1).unsqueeze(1)).squeeze(1)
        valid = has & (lab != NO_LABEL) & (picked > -1e8)  # a label the action mask forbids cannot be learned
        n = valid.sum().clamp(min=1)
        loss = loss - (picked * valid).sum() / n
        guess = lp.argmax(-1)
        if name == "dir":  # also count a neighbouring direction as right
            off = (guess - lab).abs()
            acc["dir_near"] = (((off <= 1) | (off >= policy.nvec[i] - 1)) & valid).sum() / n
        acc[name] = ((guess == lab) & valid).sum() / n
    return loss, acc


class RandomPolicy(nn.Module):
    """Uniformly random valid actions: the floor every trained policy is measured against."""

    def __init__(self, obs_size: int, nvec: list[int]):
        super().__init__()
        self.config = {"kind": "random", "obs_size": obs_size, "nvec": list(nvec)}
        self.nvec = list(nvec)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None, greedy: bool = False, state: torch.Tensor | None = None):
        nk = self.nvec[0]
        masks = [mask[:, :nk].bool(), None, None, mask[:, nk:].bool()]
        acts = [torch.multinomial(masked_log_softmax(obs.new_zeros(obs.shape[0], n), masks[i]).exp(), 1).squeeze(-1) for i, n in enumerate(self.nvec)]
        zero = obs.new_zeros(obs.shape[0])
        return torch.stack(acts, dim=1), zero, zero, zero


def build_policy(config: dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    kind = cfg.pop("kind")
    if kind == "mlp":
        return MLPPolicy(**cfg)
    if kind == "random":
        return RandomPolicy(**cfg)
    raise ValueError(f"unknown policy kind {kind!r}")


def save_checkpoint(path: str | os.PathLike, policy: nn.Module, optimizer: torch.optim.Optimizer | None = None, **meta: Any) -> None:
    """Write a checkpoint atomically, so an interrupted save never leaves a broken file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": policy.config, "model": policy.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "meta": meta}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    # On Windows the old file can be held open for a moment by a virus scanner or an indexer, which makes the
    # replace fail. Try again for a few seconds; if it still fails keep training - the next save will try again.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    print(f"warning: could not replace {path} (it is in use); the new checkpoint is in {tmp}", flush=True)


def load_checkpoint(path: str | os.PathLike, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Returns the policy (on ``device``) and the whole checkpoint dict (optimizer state, meta)."""
    data = torch.load(path, map_location=device, weights_only=True)
    policy = build_policy(data["config"])
    policy.load_state_dict(data["model"])
    return policy.to(device), data


def load_policy(spec: str, obs_size: int, nvec: list[int], device: torch.device | str = "cpu") -> nn.Module:
    """A policy for playing: ``"random"`` or the path of a checkpoint."""
    if spec == "random":
        return RandomPolicy(obs_size, nvec).to(device).eval()
    policy, _ = load_checkpoint(spec, device)
    if policy.config["obs_size"] != obs_size or policy.config["nvec"] != list(nvec):
        raise ValueError(f"{spec} was trained for a different observation or action space than the current simulation")
    return policy.eval()

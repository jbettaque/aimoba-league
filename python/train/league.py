"""Bot games and self-play games behind one vector environment.

``LeagueEnv`` looks like ``AimobaParallelEnv`` with five controlled heroes per game. In some
processes the other team is a scripted bot, as usual. In the others ("self") both teams are
controlled from Python: the learner plays one side and an opponent policy - an earlier snapshot
of the learner, or the learner itself - plays the other. The opponent's actions are computed
here, so a trainer only ever sees the learner's five heroes.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from aimoba import AimobaVecEnv
from aimoba.parallel import _PROC_SEED_STRIDE

SELF = "self"


class LeagueEnv:
    """``opponents[p]`` is the scripted bot of process p ("easy", "basic", "expert") or ``"self"`` for a policy
    opponent; ``sides[p]`` is the team the learner plays there (0 Blue, 1 Red). Use :meth:`set_opponent` to choose
    the policy each self-play process faces (any module with the trainers' policy interface)."""

    def __init__(self, opponents: Sequence[str], sides: Sequence[int], envs_per_proc: int, device: torch.device | str, seed: int = 1, **kwargs: Any) -> None:
        self.num_procs = len(opponents)
        self.envs_per_proc = envs_per_proc
        self.opponents, self.sides = list(opponents), list(sides)
        self.device = device
        self.procs: list[AimobaVecEnv] = []
        try:
            for p, (opp, side) in enumerate(zip(opponents, sides)):
                who = dict(agents="all") if opp == SELF else dict(agents="red" if side else "blue", opponent=opp)
                self.procs.append(AimobaVecEnv(num_envs=envs_per_proc, seed=seed + p * _PROC_SEED_STRIDE, **who, **kwargs))
        except Exception:
            self.close()
            raise
        first = self.procs[0]
        e, n = envs_per_proc, self.num_procs * envs_per_proc
        self.num_envs, self.n_agents, self.n_teams = n, 5, 1
        for name in ("obs_size", "mask_size", "state_size", "action_nvec", "action_heads", "action_targets", "hero_offset", "frame_skip", "obs_blocks", "state_blocks"):
            setattr(self, name, getattr(first, name))
        self.agent_slots = np.repeat(np.array([[s * 5 + k for k in range(5)] for s in sides]), e, axis=0)
        self.team_index = np.zeros((n, 5), dtype=np.int64)
        self.obs = np.zeros((n, 5, self.obs_size), np.float32)
        self.rew = np.zeros((n, 5), np.float32)
        self.mask = np.zeros((n, 5, self.mask_size), np.uint8)
        self.done = np.zeros(n, np.bool_)
        self.state = np.zeros((n, 1, self.state_size), np.float32)
        self.expert = np.full((n, 5, 4), 255, np.uint8)
        # Self-play processes: everything for all ten heroes, and who plays the other side.
        self.full = {p: (np.zeros((e, 10, self.obs_size), np.float32), np.zeros((e, 10), np.float32), np.zeros((e, 10, self.mask_size), np.uint8),
                         np.zeros(e, np.bool_), np.zeros((e, 2, self.state_size), np.float32), np.zeros((e, 10, 4), np.uint8))
                     for p in range(self.num_procs) if opponents[p] == SELF}
        self.rival: dict[int, torch.nn.Module | None] = {p: None for p in self.full}
        self.rival_name: dict[int, str] = {p: "" for p in self.full}

    def set_opponent(self, proc: int, policy: torch.nn.Module, name: str) -> None:
        self.rival[proc], self.rival_name[proc] = policy, name

    def _rows(self, p: int) -> slice:
        return slice(p * self.envs_per_proc, (p + 1) * self.envs_per_proc)

    def _take(self, p: int) -> None:
        """Copy the learner's half of a self-play process into the combined arrays."""
        r, s = self._rows(p), self.sides[p]
        obs, rew, mask, done, state, expert = self.full[p]
        mine = slice(s * 5, s * 5 + 5)
        self.obs[r], self.rew[r], self.mask[r], self.done[r], self.state[r, 0], self.expert[r] = obs[:, mine], rew[:, mine], mask[:, mine], done, state[:, s], expert[:, mine]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        for p, env in enumerate(self.procs):
            sd = None if seed is None else seed + p * _PROC_SEED_STRIDE
            if p in self.full:
                f = self.full[p]
                env.reset(sd, out=(f[0], f[2], f[4], f[5]))
                self._take(p)
            else:
                r = self._rows(p)
                env.reset(sd, out=(self.obs[r], self.mask[r], self.state[r], self.expert[r]))
        return self.obs, self.mask

    @torch.no_grad()
    def step_send(self, actions: np.ndarray | Sequence, procs: range | None = None) -> None:
        """Hand the learner's actions to ``procs`` (default: all); in self-play processes the opponent policy moves the other team."""
        procs = procs if procs is not None else range(self.num_procs)
        a = np.asarray(actions).astype(np.uint8, copy=False).reshape(len(procs), self.envs_per_proc, 5, 4)
        # One forward pass per distinct opponent policy.
        by_policy: dict[int, list[int]] = {}
        for p in procs:
            if p in self.full:
                by_policy.setdefault(id(self.rival[p]), []).append(p)
        theirs: dict[int, np.ndarray] = {}
        for group in by_policy.values():
            policy = self.rival[group[0]]
            assert policy is not None, "set_opponent() must be called for every self-play process"
            rows = [slice((1 - self.sides[p]) * 5, (1 - self.sides[p]) * 5 + 5) for p in group]
            obs = torch.from_numpy(np.concatenate([self.full[p][0][:, r] for p, r in zip(group, rows)])).to(self.device).reshape(-1, self.obs_size)
            mask = torch.from_numpy(np.concatenate([self.full[p][2][:, r] for p, r in zip(group, rows)])).to(self.device).reshape(-1, self.mask_size)
            act = policy(obs, mask)[0].cpu().numpy().astype(np.uint8).reshape(len(group), self.envs_per_proc, 5, 4)
            for k, p in enumerate(group):
                theirs[p] = act[k]
        for k, p in enumerate(procs):
            if p in self.full:
                both = np.empty((self.envs_per_proc, 10, 4), np.uint8)
                s = self.sides[p]
                both[:, s * 5:s * 5 + 5], both[:, (1 - s) * 5:(1 - s) * 5 + 5] = a[k], theirs[p]
                self.procs[p].step_send(both)
            else:
                self.procs[p].step_send(a[k])

    def step_recv(self, procs: range | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict | None]]:
        """Wait for ``procs`` (default: all) and return the learner's rows of their games."""
        procs = procs if procs is not None else range(self.num_procs)
        infos: list[dict | None] = []
        for p in procs:
            if p in self.full:
                infos += self.procs[p].step_recv(out=self.full[p])[4]
                self._take(p)
            else:
                r = self._rows(p)
                infos += self.procs[p].step_recv(out=(self.obs[r], self.rew[r], self.mask[r], self.done[r], self.state[r], self.expert[r]))[4]
        rows = slice(procs.start * self.envs_per_proc, procs.stop * self.envs_per_proc)
        return self.obs[rows], self.rew[rows], self.done[rows], self.mask[rows], infos

    def step(self, actions: np.ndarray | Sequence):
        self.step_send(actions)
        return self.step_recv()

    def close(self) -> None:
        for env in self.procs:
            env.close()

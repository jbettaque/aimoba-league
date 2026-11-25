"""Many simulation processes stepped as one vector environment.

One ``AimobaVecEnv`` is a single Node.js process, so it uses a single CPU core. This class
runs ``num_procs`` of them: ``step`` hands every process its actions first and only then
collects the results, so all the games advance at the same time::

    from aimoba import AimobaParallelEnv
    env = AimobaParallelEnv(num_procs=20, envs_per_proc=16, agents="blue", opponent="expert")
    obs, mask = env.reset(seed=0)                        # obs: (320, 5, obs_size)
    obs, rew, done, mask, infos = env.step(actions)      # actions: (320, 5, 4) ints

The arrays returned by ``reset`` and ``step`` (and ``env.state``, the fog-free critic state) are
reused: they are overwritten by the next ``step``, so copy them (or move them to the GPU) before
stepping again.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from .env import AimobaVecEnv, sample_random_actions

_PROC_SEED_STRIDE = 15_485_863  # keeps the seed ranges of different processes apart


class AimobaParallelEnv:
    """``num_procs`` x ``envs_per_proc`` games. Takes the same options as ``AimobaVecEnv``.

    ``per_proc`` is an optional list of option overrides, one dict per process - for example
    half the processes playing Blue and half Red, or a different ``opponent`` in each. Every
    process must control the same number of heroes. ``agent_slots[env]`` lists the hero slots
    (0-4 Blue, 5-9 Red) behind each row of the observation, and ``team_index[env]`` the row of
    ``state[env]`` that belongs to each of them.
    """

    def __init__(
        self,
        num_procs: int,
        envs_per_proc: int = 16,
        seed: int = 1,
        per_proc: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if per_proc is not None and len(per_proc) != num_procs:
            raise ValueError(f"per_proc needs one entry per process ({num_procs}), got {len(per_proc)}")
        self.procs: list[AimobaVecEnv] = []
        try:
            for p in range(num_procs):
                opts = {**kwargs, **(per_proc[p] if per_proc else {})}
                opts.setdefault("seed", seed + p * _PROC_SEED_STRIDE)
                self.procs.append(AimobaVecEnv(num_envs=envs_per_proc, **opts))
        except Exception:
            self.close()
            raise
        first = self.procs[0]
        if any(e.n_agents != first.n_agents or e.n_teams != first.n_teams for e in self.procs):
            self.close()
            raise ValueError("every process must control the same number of heroes and teams")
        self.num_procs = num_procs
        self.envs_per_proc = envs_per_proc
        self.num_envs = num_procs * envs_per_proc
        self.n_agents = first.n_agents
        self.obs_size = first.obs_size
        self.mask_size = first.mask_size
        self.action_nvec = first.action_nvec
        self.frame_skip = first.frame_skip
        self.steps_per_second = first.steps_per_second
        self.agent_slots = np.repeat(np.array([e.agents for e in self.procs]), envs_per_proc, axis=0)
        self.state_size = first.state_size
        self.n_teams = first.n_teams
        self.team_index = np.repeat(np.array([e.team_index for e in self.procs]), envs_per_proc, axis=0)
        self.obs_blocks = first.obs_blocks
        self.state_blocks = first.state_blocks
        n, a = self.num_envs, self.n_agents
        self.state = np.zeros((n, self.n_teams, self.state_size), np.float32)
        self.action_heads = first.action_heads
        self.action_targets = first.action_targets
        self.hero_offset = first.hero_offset
        self.expert = np.full((n, a, 4), 255, np.uint8)  # what the shadow expert would do (see AimobaVecEnv)
        self.obs = np.zeros((n, a, self.obs_size), np.float32)
        self.rew = np.zeros((n, a), np.float32)
        self.mask = np.zeros((n, a, self.mask_size), np.uint8)
        self.done = np.zeros(n, np.bool_)

    def _rows(self, p: int) -> slice:
        return slice(p * self.envs_per_proc, (p + 1) * self.envs_per_proc)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        for p, env in enumerate(self.procs):
            r = self._rows(p)
            env.reset(None if seed is None else seed + p * _PROC_SEED_STRIDE, out=(self.obs[r], self.mask[r], self.state[r], self.expert[r]))
        return self.obs, self.mask

    def step(self, actions: np.ndarray | Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict | None]]:
        self.step_send(actions)
        return self.step_recv()

    def step_send(self, actions: np.ndarray | Sequence, procs: range | None = None) -> None:
        """First half of :meth:`step`: hand the actions to the processes ``procs`` (default: all) without waiting.

        ``actions`` holds the actions of just those processes' games (their rows are contiguous). Together with
        :meth:`step_recv` this lets a trainer pipeline the rollout: pick actions for one group of processes on the
        GPU while another group simulates.
        """
        procs = procs if procs is not None else range(self.num_procs)
        a = np.asarray(actions).astype(np.uint8, copy=False).reshape(len(procs), -1)
        for k, p in enumerate(procs):
            self.procs[p].step_send(a[k])

    def step_recv(self, procs: range | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict | None]]:
        """Second half of :meth:`step`: wait for ``procs`` (default: all) and return the rows of their games."""
        procs = procs if procs is not None else range(self.num_procs)
        infos: list[dict | None] = []
        for p in procs:
            r = self._rows(p)
            infos += self.procs[p].step_recv(out=(self.obs[r], self.rew[r], self.mask[r], self.done[r], self.state[r], self.expert[r]))[4]
        rows = slice(procs.start * self.envs_per_proc, procs.stop * self.envs_per_proc)
        return self.obs[rows], self.rew[rows], self.done[rows], self.mask[rows], infos

    def info(self) -> list[dict]:
        """Live summary of every running game."""
        return [i for env in self.procs for i in env.info()]

    def get_replay(self, env: int = 0, current: bool = False) -> dict | None:
        return self.procs[env // self.envs_per_proc].get_replay(env % self.envs_per_proc, current)

    def save_replay(self, path: str | os.PathLike, env: int = 0, current: bool = False) -> bool:
        return self.procs[env // self.envs_per_proc].save_replay(path, env % self.envs_per_proc, current)

    def sample_actions(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return sample_random_actions(mask, self.action_nvec, rng)

    def close(self) -> None:
        for env in self.procs:
            env.close()

    def __enter__(self) -> "AimobaParallelEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

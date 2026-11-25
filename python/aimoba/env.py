"""Vectorised multi-agent environment backed by the headless Node.js simulation.

The simulation itself is TypeScript (the exact same code the browser game runs);
this module launches it as a subprocess and exchanges length-prefixed binary frames
(protocol documented at the top of ``src/headless/server.ts``).

Quick start::

    from aimoba import AimobaVecEnv
    env = AimobaVecEnv(num_envs=8, agents="blue")       # learner plays blue, bots play red
    obs, mask = env.reset(seed=0)                        # obs: (8, 5, obs_size)
    obs, rew, done, mask, infos = env.step(actions)      # actions: (8, 5, 4) ints

One ``AimobaVecEnv`` is one simulation process (one CPU core). To use every core, see
``AimobaParallelEnv`` in ``aimoba.parallel``.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HEROES = ["korvath", "sable", "ignara", "vesper", "liora"]
ACTION_KINDS = ["noop", "move", "attack", "Q", "W", "E", "R", "recall"]
ACT_EXPERT = 255  # as an action kind: do exactly what the shadow expert would do (needs ``expert=``)
NO_LABEL = 255  # in ``env.expert``: this component does not matter; as the kind: no usable label

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SERVER = _REPO_ROOT / "src" / "headless" / "server.ts"


class AimobaError(RuntimeError):
    pass


def sample_random_actions(mask: np.ndarray, action_nvec: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Uniformly random valid actions for a mask of shape (..., 8 + 21); returns (..., 4) int64."""
    nk = action_nvec[0]
    acts = np.empty(mask.shape[:-1] + (4,), dtype=np.int64)
    # argmax of (uniform noise x mask) picks uniformly among the valid entries (index 0 when none is valid).
    acts[..., 0] = (rng.random(mask[..., :nk].shape) * mask[..., :nk]).argmax(-1)
    acts[..., 1] = rng.integers(action_nvec[1], size=mask.shape[:-1])
    acts[..., 2] = rng.integers(action_nvec[2], size=mask.shape[:-1])
    acts[..., 3] = (rng.random(mask[..., nk:].shape) * mask[..., nk:]).argmax(-1)
    return acts


def _resolve_agents(agents: str | Iterable[int]) -> list[int]:
    if isinstance(agents, str):
        table = {"blue": [0, 1, 2, 3, 4], "red": [5, 6, 7, 8, 9], "all": list(range(10))}
        if agents in table:
            return table[agents]
        if agents in HEROES:  # a single blue hero by name
            return [HEROES.index(agents)]
        raise ValueError(f"unknown agents spec {agents!r}")
    return [int(a) for a in agents]


class AimobaVecEnv:
    """N parallel 5v5 games; the learner controls ``agents``, scripted bots control the rest.

    Hero slots: 0-4 are Blue (korvath, sable, ignara, vesper, liora), 5-9 are Red in the
    same order. Observations are team-canonical (Red's view is mirrored), so a single
    policy can play either side - pass ``agents="all"`` for self-play.

    Action per agent is ``[kind, direction, distance, target]`` with sizes ``action_nvec``:

    * kind: 0 noop, 1 move, 2 attack, 3-6 cast Q/W/E/R, 7 recall
    * direction: 64 compass directions (movement / skillshot aim)
    * distance: 4 steps, fraction of the move or cast range
    * target: 0-4 enemy heroes, 5-12 nearest enemy creeps, 13-15 enemy structures, 16-20 allied heroes, 21 none.
      An aimed spell (skillshot, dash, ground effect) with an enemy as target is aimed at that unit, leading
      it if it moves; with "none" it is aimed by direction and distance.

    ``opponent`` picks the scripted AI for every slot you do not control: ``"easy"`` (a slow,
    sloppy beginner - the gentlest first curriculum stage), ``"basic"`` (simple,
    beatable - a good first curriculum stage) or ``"expert"`` (dodges skillshots, reacts in 0.1s,
    aims precisely; beats ``basic`` in roughly 95 games out of 100). Pass a list of ten names to mix
    them per slot, e.g. expert enemies with basic team-mates.

    ``mask`` has shape (num_envs, n_agents, 8 + 22): valid action kinds, then occupied target slots.

    Rewards are zero-sum by default: the enemy team's mean shaped reward is subtracted from every
    hero's reward (``zero_sum=1.0``; 0 turns it off). ``team_spirit`` blends a hero's own reward
    with its team's mean.

    ``state`` (num_envs, n_teams, state_size) is refreshed by every ``reset`` and ``step``: the whole
    game without fog of war, one row per controlled team, for training a critic. Never feed it to
    the policy. ``team_index[agent]`` is the row of each agent's team. ``obs_blocks`` and
    ``state_blocks`` describe both vectors as lists of ``[name, entities, features per entity]``.

    ``action_heads[hero][kind]`` says which of direction, distance and target an action kind can use
    (an attack only a target; an aimed spell a target, or a direction and distance when the target is
    "none"), ``action_targets[hero][kind]`` which target slots make sense for it, and ``hero_offset``
    is where the observation holds the one-hot hero index.

    Imitation learning: with ``expert="expert"`` (or ``"basic"`` / ``"easy"``) that scripted bot
    shadows every controlled hero. ``env.expert`` (num_envs, n_agents, 4), refreshed by every
    ``reset`` and ``step``, is the action it would take for the current observation; components
    that do not matter are ``NO_LABEL`` (255), and a kind of 255 means there is no usable label.
    Sending ``ACT_EXPERT`` (255) as an action's kind makes the hero do exactly what the bot wanted.
    Episodes auto-reset: when ``done[i]`` is true, ``infos[i]`` holds the final game summary and
    the returned observation is the first one of the next game.
    """

    def __init__(
        self,
        num_envs: int = 1,
        agents: str | Iterable[int] = "blue",
        frame_skip: int = 6,
        max_time: float = 1500.0,
        seed: int = 1,
        reward: dict[str, float] | None = None,
        team_spirit: float = 0.0,
        zero_sum: float = 1.0,
        record_replay: bool = False,
        expert: str | None = None,
        opponent: str | Sequence[str] = "basic",
        node: str | None = None,
        server: str | os.PathLike | None = None,
    ) -> None:
        node = node or os.environ.get("AIMOBA_NODE") or shutil.which("node")
        if not node:
            raise AimobaError("Node.js (>= 22.18) is required to run the simulation; `node` was not found on PATH")
        server_path = Path(server or os.environ.get("AIMOBA_SERVER") or _DEFAULT_SERVER)
        if not server_path.exists():
            raise AimobaError(f"simulation server not found at {server_path}")
        self._proc = subprocess.Popen(
            [node, str(server_path), "--binary"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 << 20
        )
        self.num_envs = num_envs
        self.agents = _resolve_agents(agents)
        config: dict[str, Any] = {
            "agents": self.agents,
            "frameSkip": frame_skip,
            "maxTime": max_time,
            "seed": seed,
            "teamSpirit": team_spirit,
            "zeroSum": zero_sum,
            "recordReplay": record_replay,
            "botKind": opponent if isinstance(opponent, str) else list(opponent),
        }
        if reward:
            config["reward"] = reward
        if expert:
            config["expert"] = expert
        spec = self._call({"cmd": "make", "num_envs": num_envs, "config": config})
        self.obs_size: int = spec["obs_size"]
        self.mask_size: int = spec["mask_size"]
        self.action_nvec: list[int] = spec["action_nvec"]
        self.n_agents: int = spec["n_agents"]
        self.agent_heroes: list[str] = spec["heroes"]
        self.frame_skip: int = spec["frame_skip"]
        self.steps_per_second = 30.0 / self.frame_skip
        self.state_size: int = spec["state_size"]
        self.n_teams: int = len(spec["teams"])
        self.team_index: list[int] = spec["team_index"]
        self.obs_blocks: list[list] = spec["obs_blocks"]
        self.state_blocks: list[list] = spec["state_blocks"]
        self.state = np.zeros((num_envs, self.n_teams, self.state_size), np.float32)
        self.action_heads = np.asarray(spec["action_heads"], dtype=np.uint8)
        self.action_targets = np.asarray(spec["action_targets"], dtype=np.uint8)
        self.hero_offset: int = spec["hero_offset"]
        self.expert = np.full((num_envs, self.n_agents, 4), NO_LABEL, np.uint8)

    # -- protocol -----------------------------------------------------------------
    def _send(self, kind: int, payload: bytes) -> None:
        if self._proc.poll() is not None:
            raise AimobaError("simulation process has exited")
        assert self._proc.stdin
        self._proc.stdin.write(struct.pack("<IB", len(payload), kind) + payload)
        self._proc.stdin.flush()

    def _read(self, n: int) -> bytes:
        assert self._proc.stdout
        data = self._proc.stdout.read(n)
        if len(data) != n:
            raise AimobaError("simulation process closed the connection")
        return data

    def _read_into(self, arr: np.ndarray) -> None:
        assert self._proc.stdout
        view = memoryview(arr).cast("B")
        got = 0
        while got < len(view):
            k = self._proc.stdout.readinto(view[got:])
            if not k:
                raise AimobaError("simulation process closed the connection")
            got += k

    def _head(self, expect: int) -> int:
        """Read a reply header; returns the payload size. A JSON reply where data was expected is an error."""
        size, kind = struct.unpack_from("<IB", self._read(8))
        if kind != expect:
            out = json.loads(self._read(size)) if kind == 0 else {}
            raise AimobaError(out.get("error", f"unexpected reply type {kind}"))
        return size

    def _call(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._send(0, json.dumps(msg, separators=(",", ":")).encode())
        out = json.loads(self._read(self._head(0)))
        if "error" in out:
            raise AimobaError(out["error"])
        return out

    def _alloc(self) -> tuple[np.ndarray, ...]:
        n, a = self.num_envs, self.n_agents
        return (np.empty((n, a, self.obs_size), np.float32), np.empty((n, a), np.float32),
                np.empty((n, a, self.mask_size), np.uint8), np.empty(n, np.bool_),
                np.empty((n, self.n_teams, self.state_size), np.float32), np.empty((n, a, 4), np.uint8))

    # -- API ------------------------------------------------------------------------
    def reset(self, seed: int | None = None, out: tuple[np.ndarray, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Start new games. ``out=(obs, mask, state, expert)`` receives the result in place (used by ``AimobaParallelEnv``)."""
        self._send(0, json.dumps({"cmd": "reset", "seed": seed}).encode())
        self._head(2)
        if out is None:
            bufs = self._alloc()
            out = (bufs[0], bufs[2], bufs[4], bufs[5])
            self.state, self.expert = out[2], out[3]
        for arr in out:
            self._read_into(arr)
        return out[0], out[1]

    def step_send(self, actions: np.ndarray | Sequence) -> None:
        """First half of :meth:`step`: hand the actions to the simulation without waiting for it."""
        a = np.asarray(actions)
        expected = self.num_envs * self.n_agents * 4
        if a.size != expected:
            raise ValueError(f"expected {expected} action ints (num_envs x n_agents x 4), got {a.size}")
        self._send(1, a.astype(np.uint8, copy=False).tobytes())

    def step_recv(self, out: tuple[np.ndarray, ...] | None = None):
        """Second half of :meth:`step`. ``out=(obs, rew, mask, done, state, expert)`` receives the result in place."""
        size = self._head(1)
        if out is None:
            out = self._alloc()
            self.state, self.expert = out[4], out[5]
        obs, rew, mask, done = out[:4]
        for arr in out:
            self._read_into(arr)
            size -= arr.nbytes
        infos: list[dict | None] = [None] * self.num_envs
        if size:
            for e, info in json.loads(self._read(size)):
                infos[e] = info
        return obs, rew, done, mask, infos

    def step(self, actions: np.ndarray | Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict | None]]:
        self.step_send(actions)
        return self.step_recv()

    def info(self) -> list[dict]:
        """Live summary of every running game."""
        return self._call({"cmd": "info"})["info"]

    def get_replay(self, env: int = 0, current: bool = False) -> dict | None:
        """Replay of the last finished game of ``env`` (or the in-progress one with ``current=True``).

        Requires ``record_replay=True``. Save it with :meth:`save_replay` and load the file in the
        browser client ("Watch replay") to see the game with full visuals.
        """
        return self._call({"cmd": "replay", "env": env, "current": current})["replay"]

    def save_replay(self, path: str | os.PathLike, env: int = 0, current: bool = False) -> bool:
        replay = self.get_replay(env, current)
        if replay is None:
            return False
        Path(path).write_text(json.dumps(replay, separators=(",", ":")))
        return True

    def sample_actions(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Uniformly random valid actions (handy for smoke tests and exploration)."""
        return sample_random_actions(mask, self.action_nvec, rng)

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send(0, b'{"cmd":"close"}')
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()

    def __enter__(self) -> "AimobaVecEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_single_agent_env(hero: str | int = "ignara", **kwargs: Any):
    """A Gymnasium ``Env`` where you control one hero and bots play the other nine.

    Observation: ``Box(obs_size)``; action: ``MultiDiscrete([8, 64, 4, 21])``. The current action
    mask is available as ``info["action_mask"]``. Requires ``pip install gymnasium``.
    """
    import gymnasium as gym

    slot = HEROES.index(hero) if isinstance(hero, str) else int(hero)

    class AimobaSingleEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.vec = AimobaVecEnv(num_envs=1, agents=[slot], **kwargs)
            self.observation_space = gym.spaces.Box(-4.0, 4.0, (self.vec.obs_size,), np.float32)
            self.action_space = gym.spaces.MultiDiscrete(self.vec.action_nvec)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            obs, mask = self.vec.reset(seed)
            return obs[0, 0].copy(), {"action_mask": mask[0, 0].copy()}

        def step(self, action):
            obs, rew, done, mask, infos = self.vec.step(np.asarray(action).reshape(1, 1, 4))
            info: dict[str, Any] = {"action_mask": mask[0, 0].copy()}
            truncated = False
            if done[0] and infos[0]:
                info["episode"] = infos[0]
                truncated = bool(infos[0].get("truncated"))
            return obs[0, 0].copy(), float(rew[0, 0]), bool(done[0]) and not truncated, truncated, info

        def close(self) -> None:
            self.vec.close()

    return AimobaSingleEnv()

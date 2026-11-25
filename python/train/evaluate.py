"""Measure a policy against the scripted bots (or another checkpoint) on a fixed set of games.

    python python/train/evaluate.py runs/ppo/latest.pt                          # 100 games each vs easy, basic, expert
    python python/train/evaluate.py runs/ppo/latest.pt --opponents expert --games 200 --replays 4
    python python/train/evaluate.py runs/b/latest.pt --opponents runs/a/latest.pt   # checkpoint vs checkpoint
    python python/train/evaluate.py random --opponents easy --games 20          # the floor

Every game is a full 5v5: the policy plays all five heroes of one team. It plays Blue in half
the games and Red in the other half, and the seeds are fixed, so two checkpoints evaluated with
the same options face the same games. A game that hits the time limit counts as a draw.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import AimobaParallelEnv  # noqa: E402

from policy import load_policy  # noqa: E402

BOT_KINDS = ("easy", "basic", "expert")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n))


@torch.no_grad()
def evaluate(
    policy: str | torch.nn.Module,
    opponent: str = "expert",
    games: int = 100,
    procs: int = 20,
    seed: int = 100_000,
    max_time: float = 1800.0,
    frame_skip: int = 6,
    device: torch.device | str = "cpu",
    greedy: bool = False,
    replays: int = 0,
    replay_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Play ``games`` full games (rounded up to fill every process evenly) and summarise them.

    ``policy`` is a module, a checkpoint path or ``"random"``; ``opponent`` is a scripted bot kind
    or a checkpoint path. Each env plays exactly one game, so the set of games is fixed by ``seed``.
    """
    procs = max(2, min(procs, games) // 2 * 2)
    per_env = math.ceil(games / procs)
    versus_bot = opponent in BOT_KINDS
    record = replays > 0 and replay_dir is not None
    # Even processes play Blue, odd ones Red. Replays come from the first process of each side.
    side_of_proc = [p % 2 for p in range(procs)]
    per_proc = [{"record_replay": record and p < 2} for p in range(procs)]
    if versus_bot:
        for p in range(procs):
            per_proc[p]["agents"] = "red" if side_of_proc[p] else "blue"
    common = dict(max_time=max_time, frame_skip=frame_skip, seed=seed, per_proc=per_proc)
    env = AimobaParallelEnv(procs, per_env, opponent=opponent, **common) if versus_bot else AimobaParallelEnv(procs, per_env, agents="all", **common)
    try:
        if isinstance(policy, str):
            policy = load_policy(policy, env.obs_size, env.action_nvec, device)
        was_training = policy.training
        policy.eval()
        rival = None if versus_bot else load_policy(opponent, env.obs_size, env.action_nvec, device)
        n = env.num_envs
        side = np.repeat(side_of_proc, per_env)  # the team the policy plays in each env
        mine = torch.as_tensor(env.agent_slots // 5 == side[:, None], device=device).reshape(-1)  # rows the policy controls
        obs, mask = env.reset(seed=seed)
        finished = np.zeros(n, dtype=bool)
        results: list[tuple[int, dict]] = []
        saved = 0
        t0 = time.time()
        while not finished.all():
            o = torch.from_numpy(obs).to(device).reshape(n * env.n_agents, -1)
            m = torch.from_numpy(mask).to(device).reshape(n * env.n_agents, -1)
            if rival is None:
                act = policy(o, m, greedy=greedy)[0]
            else:
                act = torch.empty(o.shape[0], 4, dtype=torch.long, device=device)
                act[mine] = policy(o[mine], m[mine], greedy=greedy)[0]
                act[~mine] = rival(o[~mine], m[~mine], greedy=greedy)[0]
            obs, _, done, mask, infos = env.step(act.cpu().numpy())
            for i in np.flatnonzero(done & ~finished):
                finished[i] = True
                results.append((int(side[i]), infos[i]))
                if record and saved < replays and i < 2 * per_env:
                    name = f"{'red' if side[i] else 'blue'}_vs_{Path(opponent).stem}_{'win' if infos[i]['winner'] == side[i] else 'loss' if infos[i]['winner'] >= 0 else 'draw'}_{saved}.replay.json"
                    Path(replay_dir).mkdir(parents=True, exist_ok=True)
                    saved += env.save_replay(Path(replay_dir) / name, env=int(i))
        policy.train(was_training)
    finally:
        env.close()

    wins = sum(info["winner"] == s for s, info in results)
    draws = sum(info["winner"] < 0 for s, info in results)
    lo, hi = wilson(wins, n)
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")  # noqa: E731
    by_side = [[info["winner"] == s for s, info in results if s == t] for t in (0, 1)]
    team_gold = [sum(h["gold"] for h in info["heroes"] if h["team"] == s) for s, info in results]
    return {
        "opponent": opponent, "games": n, "wins": int(wins), "losses": int(n - wins - draws), "draws": int(draws),
        "win_rate": wins / n, "ci_low": lo, "ci_high": hi,
        "win_rate_as_blue": mean(by_side[0]), "win_rate_as_red": mean(by_side[1]),
        "minutes": mean([info["time"] / 60 for _, info in results]),
        "kills": mean([info["kills"][s] for s, info in results]), "deaths": mean([info["kills"][1 - s] for s, info in results]),
        "towers_taken": mean([info["towers_lost"][1 - s] for s, info in results]), "towers_lost": mean([info["towers_lost"][s] for s, info in results]),
        "team_gold": mean(team_gold), "replays_saved": saved, "seconds": time.time() - t0,
    }


def format_result(r: dict[str, Any]) -> str:
    return (f"vs {Path(r['opponent']).stem:8s} {r['wins']:3d}W {r['losses']:3d}L {r['draws']:3d}D of {r['games']:3d} | win rate {100 * r['win_rate']:5.1f}% "
            f"(95% CI {100 * r['ci_low']:.0f}-{100 * r['ci_high']:.0f}%, blue {100 * r['win_rate_as_blue']:.0f}% / red {100 * r['win_rate_as_red']:.0f}%) | "
            f"{r['minutes']:4.1f} min | kills {r['kills']:4.1f}-{r['deaths']:4.1f} | towers {r['towers_taken']:.1f}-{r['towers_lost']:.1f} | gold {r['team_gold'] / 1000:5.1f}k")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help='checkpoint path, or "random"')
    p.add_argument("--opponents", default="easy,basic,expert", help="comma-separated bot kinds and/or checkpoint paths")
    p.add_argument("--games", type=int, default=100, help="games per opponent")
    p.add_argument("--procs", type=int, default=20, help="simulation processes")
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--max-time", type=float, default=1800.0, help="game time limit in seconds; hitting it is a draw")
    p.add_argument("--frame-skip", type=int, default=6)
    p.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    p.add_argument("--replays", type=int, default=0, help="save this many replays per opponent")
    p.add_argument("--replay-dir", default="replays")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", default=None, help="also write the results to this file")
    args = p.parse_args()

    out = []
    for opponent in args.opponents.split(","):
        r = evaluate(args.policy, opponent, args.games, args.procs, args.seed, args.max_time, args.frame_skip, args.device, args.greedy, args.replays, args.replay_dir)
        out.append(r)
        print(format_result(r), f"| {r['seconds']:.0f}s", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))

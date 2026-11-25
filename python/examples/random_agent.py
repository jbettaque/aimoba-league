"""Smoke test / throughput benchmark: random valid actions for Blue against the scripted bots.

    python python/examples/random_agent.py --envs 8 --steps 2000 --replay my_game.replay.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import AimobaVecEnv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=4)
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--agents", default="blue")
p.add_argument("--replay", default=None, help="write the first finished (or in-progress) game to this file")
args = p.parse_args()

rng = np.random.default_rng(0)
with AimobaVecEnv(num_envs=args.envs, agents=args.agents, record_replay=args.replay is not None) as env:
    obs, mask = env.reset(seed=0)
    print(f"obs {obs.shape} mask {mask.shape} action_nvec {env.action_nvec} heroes {env.agent_heroes}")
    t0 = time.time()
    total = np.zeros(env.n_agents)
    saved = False
    for step in range(args.steps):
        obs, rew, done, mask, infos = env.step(env.sample_actions(mask, rng))
        total += rew.sum(axis=0)
        for i in np.flatnonzero(done):
            info = infos[i]
            print(f"env {i} finished at {info['time'] / 60:.1f} min: winner={info['winner']} kills={info['kills']}")
            if args.replay and not saved:
                saved = env.save_replay(args.replay, env=int(i))
    dt = time.time() - t0
    n = args.steps * args.envs
    print(f"{n} env steps in {dt:.2f}s = {n / dt:.0f} steps/s ({n * env.n_agents / dt:.0f} agent-steps/s, {n / dt / env.steps_per_second:.0f}x realtime)")
    print("mean reward per agent per step:", np.round(total / n, 4))
    assert np.isfinite(obs).all()
    if args.replay and not saved:
        saved = env.save_replay(args.replay, current=True)
    if args.replay:
        print("replay saved:" if saved else "no replay available:", args.replay)

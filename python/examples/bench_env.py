"""Throughput of the multi-process environment with random actions.

    python python/examples/bench_env.py --procs 20 --envs 16 --steps 300
    python python/examples/bench_env.py --sweep 1,4,8,12,16,20,24      # find the best process count for this machine
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import AimobaParallelEnv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--procs", type=int, default=8, help="simulation processes (one core each)")
p.add_argument("--envs", type=int, default=16, help="games per process")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--warmup", type=int, default=50, help="untimed steps first, so the JIT is warm and minions are on the map")
p.add_argument("--agents", default="blue")
p.add_argument("--opponent", default="basic")
p.add_argument("--sweep", default=None, help="comma-separated process counts to compare")
args = p.parse_args()

for procs in [int(x) for x in args.sweep.split(",")] if args.sweep else [args.procs]:
    rng = np.random.default_rng(0)
    with AimobaParallelEnv(num_procs=procs, envs_per_proc=args.envs, agents=args.agents, opponent=args.opponent) as env:
        obs, mask = env.reset(seed=0)
        t_env = 0.0
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                t0, t_env = time.perf_counter(), 0.0
            acts = env.sample_actions(mask, rng)
            t1 = time.perf_counter()
            obs, rew, done, mask, infos = env.step(acts)
            t_env += time.perf_counter() - t1
        dt = time.perf_counter() - t0
        n = args.steps * env.num_envs
        assert np.isfinite(obs).all()
        print(f"{procs:3d} procs x {args.envs} envs, {env.n_agents} heroes each: {n / dt:7.0f} env steps/s, {n * env.n_agents / dt:8.0f} agent-steps/s, "
              f"{n / dt / env.steps_per_second:6.0f}x realtime, {1000 * t_env / args.steps:5.1f} ms per step call ({100 * t_env / dt:.0f}% of wall time in the sim)", flush=True)

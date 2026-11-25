"""Minimal PPO baseline (PyTorch): one shared policy controls all five Blue heroes vs. the scripted bots.

    pip install torch numpy
    python python/examples/train_ppo.py --envs 16 --updates 200
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db       # browse runs (pip install mlflow)

This is a compact, readable starting point (CleanRL style), not a tuned agent. The
policy uses the action masks from the environment, and each hero is treated as an
independent sample of the same network (parameter sharing).

When MLflow is installed, every run is tracked (hyperparameters, losses, episode stats and
checkpoints) in the experiment given by --experiment; pass --no-mlflow to turn it off.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import AimobaVecEnv  # noqa: E402
from aimoba.tracking import start_run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--steps", type=int, default=128, help="rollout length per update")
p.add_argument("--updates", type=int, default=200)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.995)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--minibatches", type=int, default=4)
p.add_argument("--max-time", type=float, default=900.0)
p.add_argument("--team-spirit", type=float, default=0.3)
p.add_argument("--save", default="ppo_aimoba.pt")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--experiment", default="aimoba-ppo", help="MLflow experiment name")
p.add_argument("--run-name", default=None, help="MLflow run name (default: generated)")
p.add_argument("--no-mlflow", action="store_true", help="do not track this run in MLflow")
args = p.parse_args()

torch.manual_seed(args.seed)
env = AimobaVecEnv(num_envs=args.envs, agents="blue", max_time=args.max_time, team_spirit=args.team_spirit, seed=args.seed)
NVEC = env.action_nvec
NK = NVEC[0]
B = args.envs * env.n_agents  # every hero in every env is one sample


class Policy(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_size, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(256, n) for n in NVEC])
        self.value = nn.Linear(256, 1)

    def forward(self, obs, mask, action=None):
        h = self.body(obs)
        kind_mask = mask[:, :NK].bool()
        target_mask = mask[:, NK:].bool()
        target_mask = target_mask | ~target_mask.any(dim=1, keepdim=True)  # nothing targetable -> ignore head
        masks = [kind_mask, None, None, target_mask]
        logp, ent, acts = 0.0, 0.0, []
        for i, head in enumerate(self.heads):
            logits = head(h)
            if masks[i] is not None:
                logits = logits.masked_fill(~masks[i], -1e9)
            dist = Categorical(logits=logits)
            a = dist.sample() if action is None else action[:, i]
            logp = logp + dist.log_prob(a)
            ent = ent + dist.entropy()
            acts.append(a)
        return torch.stack(acts, dim=1), logp, ent, self.value(h).squeeze(-1)


policy = Policy(env.obs_size)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
flat = lambda a: torch.as_tensor(np.array(a)).reshape(B, -1)  # noqa: E731

obs_np, mask_np = env.reset(seed=args.seed)
obs, mask = flat(obs_np).float(), flat(mask_np)
ep_returns = np.zeros(args.envs)
recent: list[tuple[float, int, float]] = []
run = start_run(
    args.experiment,
    run_name=args.run_name,
    params={**vars(args), "obs_size": env.obs_size, "n_agents": env.n_agents, "action_nvec": list(NVEC)},
    tags={"algo": "ppo", "torch": torch.__version__},
    enabled=not args.no_mlflow,
)
t0 = time.time()
with run:
    for update in range(1, args.updates + 1):
        buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
        for _ in range(args.steps):
            with torch.no_grad():
                act, logp, _, val = policy(obs, mask)
            obs_np, rew_np, done_np, mask_np, infos = env.step(act.numpy().reshape(args.envs, env.n_agents, 4))
            for k, v in zip(buf, (obs, mask, act, logp, val, torch.as_tensor(rew_np.reshape(B).copy()), torch.as_tensor(np.repeat(done_np, env.n_agents)).float())):
                buf[k].append(v)
            ep_returns += rew_np.mean(axis=1)
            for i in np.flatnonzero(done_np):
                recent.append((ep_returns[i], infos[i]["winner"], infos[i]["time"]))
                ep_returns[i] = 0
            obs, mask = flat(obs_np).float(), flat(mask_np)

        # Generalised advantage estimation.
        with torch.no_grad():
            next_val = policy(obs, mask)[3]
        T = args.steps
        vals, rews, dones = torch.stack(buf["val"]), torch.stack(buf["rew"]), torch.stack(buf["done"])
        adv = torch.zeros(T, B)
        last = torch.zeros(B)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else vals[t + 1]
            delta = rews[t] + args.gamma * nv * (1 - dones[t]) - vals[t]
            last = delta + args.gamma * args.lam * (1 - dones[t]) * last
            adv[t] = last
        ret = adv + vals

        b_obs, b_mask, b_act = torch.cat(buf["obs"]), torch.cat(buf["mask"]), torch.cat(buf["act"])
        b_logp, b_adv, b_ret = torch.cat(buf["logp"]), adv.reshape(-1), ret.reshape(-1)
        n = T * B
        stats = {k: [] for k in ("policy", "value", "entropy", "approx_kl", "clip_frac")}
        for _ in range(args.epochs):
            for idx in torch.randperm(n).split(n // args.minibatches):
                _, logp, ent, val = policy(b_obs[idx], b_mask[idx], b_act[idx])
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - b_logp[idx]).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                loss = pg + 0.5 * (val - b_ret[idx]).pow(2).mean() - args.ent * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    log_ratio = logp - b_logp[idx]
                    for k, v in zip(stats, (pg, (val - b_ret[idx]).pow(2).mean(), ent.mean(), ((ratio - 1) - log_ratio).mean(), ((ratio - 1).abs() > args.clip).float().mean())):
                        stats[k].append(v.item())

        last_eps = recent[-20:]
        sps = update * T * args.envs / (time.time() - t0)
        msg = f"update {update:4d} | env-steps/s {sps:6.0f} | mean step reward {rews.mean().item():+.4f}"
        if last_eps:
            msg += f" | episodes {len(recent)} | return {np.mean([e[0] for e in last_eps]):7.2f} | win rate {np.mean([e[1] == 0 for e in last_eps]):.2f} | length {np.mean([e[2] for e in last_eps]) / 60:.1f}m"
        print(msg, flush=True)
        metrics = {"perf/env_steps_per_sec": sps, "rollout/mean_step_reward": rews.mean().item(), "episode/count": len(recent)}
        metrics.update({f"loss/{k}": float(np.mean(v)) for k, v in stats.items()})
        if last_eps:
            metrics.update({
                "episode/return": float(np.mean([e[0] for e in last_eps])),
                "episode/win_rate": float(np.mean([e[1] == 0 for e in last_eps])),
                "episode/length_min": float(np.mean([e[2] for e in last_eps]) / 60),
            })
        run.log_metrics(metrics, step=update * T * args.envs)  # x axis: environment steps
        if update % 20 == 0 or update == args.updates:
            torch.save(policy.state_dict(), args.save)
            run.log_artifact(args.save, "checkpoints")

env.close()

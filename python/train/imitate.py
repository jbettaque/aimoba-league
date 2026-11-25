"""Imitation learning from a scripted bot (DAgger): the policy learns to do what the expert bot would do.

    python python/train/imitate.py --run imitate --updates 400
    python python/train/ppo.py --run finetune --resume runs/imitate/latest.pt --opponent expert    # then improve on it with PPO

Every step the environment reports, for each hero, the action the expert bot would take in its
place (``expert=`` in the Python env). The policy is trained to predict that action. Who actually
moves the heroes changes over the run:

* first the expert itself (``--bc-updates``): plain behaviour cloning on the expert's own games;
* then, over ``--decay-updates``, more and more often the policy (each hero, each step, follows the
  expert with probability beta), down to ``--beta-end``. The expert still labels every step, so the
  policy learns what to do in the situations its own mistakes lead to.

Training samples come from a replay buffer of the last ``--replay-updates`` rollouts, so the many
games that start together do not make the policy forget the early game while it sees the late game.
The critic is trained alongside on the returns of the games played, as a warm start for PPO.

Output goes to ``runs/<run>/`` like ``ppo.py`` (latest.pt, snapshots, metrics.jsonl, config.json), and
to MLflow (experiment ``aimoba-imitation``). The checkpoints load in ``ppo.py`` and ``evaluate.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import ACT_EXPERT, ACTION_KINDS, NO_LABEL, AimobaParallelEnv  # noqa: E402
from aimoba.tracking import start_run  # noqa: E402

from evaluate import evaluate, format_result  # noqa: E402
from policy import MLPPolicy, imitation_loss, load_checkpoint, save_checkpoint  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--run", default="imitate", help="name of the run; output goes to runs/<run>/")
p.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[2] / "runs"))
p.add_argument("--resume", default=None, help="checkpoint to continue from")
p.add_argument("--updates", type=int, default=400, help="updates to run in this invocation")
# Environment.
p.add_argument("--procs", type=int, default=22, help="simulation processes (one CPU core each)")
p.add_argument("--envs", type=int, default=16, help="games per process")
p.add_argument("--expert", default="expert", help="scripted bot to imitate: expert, basic or easy")
p.add_argument("--agents", default="all", choices=["all", "sides"], help="all: both teams learn in every game; sides: one team per game, against --opponent bots")
p.add_argument("--opponent", default="expert", help="with --agents sides: the scripted bot on the other team")
p.add_argument("--frame-skip", type=int, default=6, help="sim ticks per decision (30 ticks = 1 s)")
p.add_argument("--max-time", type=float, default=1500.0, help="game time limit in seconds")
p.add_argument("--team-spirit", type=float, default=0.3)
p.add_argument("--zero-sum", type=float, default=1.0)
# Who plays.
p.add_argument("--bc-updates", type=int, default=100, help="updates in which the expert plays every step (beta = 1)")
p.add_argument("--decay-updates", type=int, default=150, help="updates over which beta then falls to --beta-end")
p.add_argument("--beta-end", type=float, default=0.0, help="final probability that a hero follows the expert in a step")
# Learning.
p.add_argument("--steps", type=int, default=64, help="rollout length per update")
p.add_argument("--replay-updates", type=int, default=16, help="rollouts kept in the replay buffer")
p.add_argument("--sgd-steps", type=int, default=96, help="minibatches per update")
p.add_argument("--minibatch", type=int, default=8192)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--vf", type=float, default=0.5, help="weight of the critic's loss")
p.add_argument("--gamma", type=float, default=0.995)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--hidden", type=int, default=512)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--activation", default="relu", choices=["tanh", "relu", "gelu"])
# Bookkeeping.
p.add_argument("--seed", type=int, default=1)
p.add_argument("--save-every", type=int, default=10)
p.add_argument("--snapshot-every", type=int, default=100)
p.add_argument("--eval-every", type=int, default=50, help="run the evaluation harness every this many updates (0 = never)")
p.add_argument("--eval-games", type=int, default=40)
p.add_argument("--eval-opponents", default="easy,basic,expert")
p.add_argument("--eval-greedy", action=argparse.BooleanOptionalAction, default=True, help="evaluate with the most likely action instead of sampling")
p.add_argument("--experiment", default="aimoba-imitation", help="MLflow experiment name")
p.add_argument("--no-mlflow", action="store_true")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device(args.device)
run_dir = Path(args.runs_dir) / args.run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

common = dict(seed=args.seed, expert=args.expert, frame_skip=args.frame_skip, max_time=args.max_time, team_spirit=args.team_spirit, zero_sum=args.zero_sum)
if args.agents == "all":
    env = AimobaParallelEnv(args.procs, args.envs, agents="all", **common)
else:
    env = AimobaParallelEnv(args.procs, args.envs, per_proc=[{"agents": "red" if i % 2 else "blue"} for i in range(args.procs)], opponent=args.opponent, **common)
N, A, O, M, T = env.num_envs, env.n_agents, env.obs_size, env.mask_size, args.steps
S, R = env.state_size, env.num_envs * env.n_teams
B = N * A
NVEC = env.action_nvec

start_update, env_steps = 0, 0
if args.resume:
    policy, ckpt = load_checkpoint(args.resume, device)
    start_update, env_steps = ckpt["meta"].get("update", 0), ckpt["meta"].get("env_steps", 0)
else:
    policy, ckpt = MLPPolicy(O, NVEC, args.hidden, args.layers, S, env.action_heads.tolist(), env.action_targets.tolist(), env.hero_offset, args.activation).to(device), None
opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
if ckpt and ckpt.get("optimizer"):
    opt.load_state_dict(ckpt["optimizer"])
    for group in opt.param_groups:
        group["lr"] = args.lr

print(f"run {run_dir} | {device} | {args.procs} procs x {args.envs} games = {N} games, {B} heroes | {T * B} samples per update, "
      f"replay {args.replay_updates * T * B / 1e6:.1f}M | {sum(x.numel() for x in policy.parameters()) / 1e6:.2f}M parameters | imitating {args.expert}", flush=True)

# Fresh rollout (also used for the critic's returns).
buf_obs = torch.zeros(T, B, O, device=device)
buf_mask = torch.zeros(T, B, M, dtype=torch.uint8, device=device)
buf_label = torch.zeros(T, B, 4, dtype=torch.uint8, device=device)
buf_val, buf_rew, buf_done = (torch.zeros(T, B, device=device) for _ in range(3))
buf_state = torch.zeros(T, R, S, device=device)
state_row = torch.as_tensor(np.arange(N)[:, None] * env.n_teams + env.team_index, device=device).reshape(B)
# Replay of what the expert said, in half precision.
K = args.replay_updates * T * B
rep_obs = torch.zeros(K, O, dtype=torch.float16, device=device)
rep_mask = torch.zeros(K, M, dtype=torch.uint8, device=device)
rep_label = torch.zeros(K, 4, dtype=torch.uint8, device=device)
rep_filled, rep_at = 0, 0


def to_device(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(device)  # copies: the env reuses its arrays


def save(name: str, update: int) -> None:
    save_checkpoint(run_dir / name, policy, opt, update=update, env_steps=env_steps, trainer="imitate", imitated=args.expert, frame_skip=args.frame_skip)


def beta_at(update: int) -> float:
    if update <= args.bc_updates:
        return 1.0
    frac = min(1.0, (update - args.bc_updates) / max(1, args.decay_updates))
    return 1.0 + frac * (args.beta_end - 1.0)


obs_np, mask_np = env.reset(seed=args.seed + start_update)
obs, mask = to_device(obs_np).reshape(B, O), to_device(mask_np).reshape(B, M)
state, label = to_device(env.state).reshape(R, S), to_device(env.expert).reshape(B, 4)
follow_expert = torch.tensor([ACT_EXPERT, 0, 0, 0], device=device)
episodes: deque[float] = deque(maxlen=200)  # length in minutes of recent games
n_episodes = 0
tracker = start_run(
    args.experiment,
    run_name=args.run,
    params={**vars(args), "obs_size": O, "n_agents": A, "num_envs": N, "action_nvec": list(NVEC), "parameters": sum(x.numel() for x in policy.parameters())},
    tags={"algo": "dagger", "trainer": "python/train/imitate.py", "torch": torch.__version__, "policy": policy.config["kind"], "start_update": start_update},
    enabled=not args.no_mlflow,
)
status = "FINISHED"
metrics = (run_dir / "metrics.jsonl").open("a")
update = start_update
t_start = time.time()
try:
    for update in range(start_update + 1, start_update + args.updates + 1):
        beta = beta_at(update)
        # --- Rollout: the expert labels every step; each hero follows it with probability beta ------------
        t0 = time.time()
        t_sim = 0.0
        kinds = torch.zeros(len(ACTION_KINDS) + 1, device=device)
        for t in range(T):
            with torch.no_grad():
                act, _, _, val = policy(obs, mask, state=state[state_row])
            buf_obs[t], buf_mask[t], buf_label[t], buf_val[t], buf_state[t] = obs, mask, label, val, state
            kinds += torch.bincount(label[:, 0].long().clamp(max=len(ACTION_KINDS)), minlength=len(ACTION_KINDS) + 1)
            if beta > 0:
                act = torch.where((torch.rand(B, device=device) < beta).unsqueeze(1), follow_expert, act)
            act_np = act.cpu().numpy()
            t1 = time.time()
            obs_np, rew_np, done_np, mask_np, infos = env.step(act_np)
            t_sim += time.time() - t1
            buf_rew[t] = to_device(rew_np).reshape(B)
            buf_done[t] = to_device(done_np).repeat_interleave(A).float()
            for i in np.flatnonzero(done_np):
                episodes.append(infos[i]["time"] / 60)
                n_episodes += 1
            obs, mask = to_device(obs_np).reshape(B, O), to_device(mask_np).reshape(B, M)
            state, label = to_device(env.state).reshape(R, S), to_device(env.expert).reshape(B, 4)
        env_steps += T * N
        t_rollout = time.time() - t0

        # --- Returns for the critic, and the rollout into the replay buffer ---------------------------------
        t0 = time.time()
        with torch.no_grad():
            next_val = policy(obs, mask, state=state[state_row])[3]
            adv = torch.zeros(T, B, device=device)
            last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else buf_val[t + 1]
                alive = 1 - buf_done[t]
                delta = buf_rew[t] + args.gamma * nv * alive - buf_val[t]
                last = delta + args.gamma * args.lam * alive * last
                adv[t] = last
            b_ret = (adv + buf_val).reshape(-1)
        b_obs, b_state = buf_obs.reshape(T * B, O), buf_state.reshape(T * R, S)
        rows = torch.arange(rep_at, rep_at + T * B, device=device) % K
        rep_obs[rows], rep_mask[rows], rep_label[rows] = b_obs.half(), buf_mask.reshape(T * B, M), buf_label.reshape(T * B, 4)
        rep_at = (rep_at + T * B) % K
        rep_filled = min(K, rep_filled + T * B)

        # How well the policy predicts the expert on these new games, before it has trained on them.
        with torch.no_grad():
            every = max(1, T * B // 65536)
            fresh_loss, fresh_acc = imitation_loss(policy, b_obs[::every], buf_mask.reshape(T * B, M)[::every], buf_label.reshape(T * B, 4)[::every])

        # --- Learn ------------------------------------------------------------------------------------------
        stats: dict[str, list[torch.Tensor]] = {}
        for _ in range(args.sgd_steps):
            idx = torch.randint(rep_filled, (args.minibatch,), device=device)
            loss_im, acc = imitation_loss(policy, rep_obs[idx].float(), rep_mask[idx], rep_label[idx])
            fresh = torch.randint(T * B, (args.minibatch,), device=device)
            val = policy(b_obs[fresh], buf_mask.reshape(T * B, M)[fresh], state=b_state[(fresh // B) * R + state_row[fresh % B]])[3]
            loss_v = (val - b_ret[fresh]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            (loss_im + args.vf * loss_v).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            for k, v in {"imitation_loss": loss_im, "value_loss": loss_v, **{f"acc_{n}": a for n, a in acc.items()}}.items():
                stats.setdefault(k, []).append(v.detach())
        t_learn = time.time() - t0

        # --- Report -----------------------------------------------------------------------------------------
        kinds = (kinds / kinds.sum()).tolist()
        row = {
            "update": update, "env_steps": env_steps, "agent_steps": env_steps * A, "hours": (time.time() - t_start) / 3600, "beta": beta,
            "env_steps_per_s": T * N / (t_rollout + t_learn), "rollout_s": t_rollout, "sim_s": t_sim, "learn_s": t_learn,
            "reward": buf_rew.mean().item(), "episodes": n_episodes, "no_label": kinds[-1],
            "explained_var": (1 - (b_ret - buf_val.reshape(-1)).var() / (b_ret.var() + 1e-8)).item(),
            "fresh_loss": fresh_loss.item(), "fresh_acc_kind": fresh_acc["kind"].item(),
            **{k: torch.stack(v).mean().item() for k, v in stats.items()},
        }
        msg = (f"update {update:5d} | beta {beta:.2f} | {row['env_steps_per_s']:6.0f} env steps/s (rollout {t_rollout:4.1f}s, sim {t_sim:4.1f}s; learn {t_learn:4.1f}s) | "
               f"loss {row['imitation_loss']:.3f} (new games {row['fresh_loss']:.3f}) | acc kind {100 * row['acc_kind']:.0f}% dir {100 * row['acc_dir']:.0f}% (near {100 * row['acc_dir_near']:.0f}%) "
               f"dist {100 * row['acc_dist']:.0f}% target {100 * row['acc_target']:.0f}% | ev {row['explained_var']:+.2f}")
        if episodes:
            row["game_minutes"] = float(np.mean(episodes))
            msg += f" | games {n_episodes}, {row['game_minutes']:.1f} min"
        print(msg, flush=True)
        tracked = {
            "perf/env_steps_per_sec": row["env_steps_per_s"], "perf/learn_s": t_learn, "imitation/beta": beta, "imitation/no_label": row["no_label"],
            "imitation/loss": row["imitation_loss"], "imitation/loss_new_games": row["fresh_loss"], "imitation/acc_kind_new_games": row["fresh_acc_kind"],
            **{f"imitation/{k}": row[k] for k in row if k.startswith("acc_")},
            **{f"labels/{name}": share for name, share in zip(ACTION_KINDS, kinds)},
            "loss/value": row["value_loss"], "loss/explained_var": row["explained_var"], "rollout/mean_step_reward": row["reward"], "episode/count": n_episodes,
        }
        if episodes:
            tracked["episode/length_min"] = row["game_minutes"]
        if args.eval_every and update % args.eval_every == 0:
            row["eval"] = [evaluate(policy, opp, args.eval_games, args.procs, frame_skip=args.frame_skip, device=device, greedy=args.eval_greedy) for opp in args.eval_opponents.split(",")]
            for r in row["eval"]:
                print("  eval", format_result(r), flush=True)
                tracked.update({f"eval/{Path(r['opponent']).stem}/{k}": r[k] for k in ("win_rate", "minutes", "kills", "deaths", "towers_taken", "towers_lost")})
        tracker.log_metrics(tracked, step=env_steps)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if update % args.save_every == 0:
            save("latest.pt", update)
        if args.snapshot_every and update % args.snapshot_every == 0:
            save(f"update_{update:06d}.pt", update)
            tracker.log_artifact(str(run_dir / f"update_{update:06d}.pt"), "checkpoints")
except KeyboardInterrupt:
    status = "KILLED"
    print("interrupted", flush=True)
except BaseException:
    status = "FAILED"
    raise
finally:
    save("latest.pt", update)
    tracker.log_artifact(str(run_dir / "latest.pt"), "checkpoints")
    tracker.end(status)
    metrics.close()
    env.close()
    print(f"saved {run_dir / 'latest.pt'} at update {update} ({env_steps} env steps)", flush=True)

"""PPO on the GPU over the multi-process environment: one shared policy plays all five heroes of a team.

    python python/train/ppo.py --run first --opponent basic --updates 500
    python python/train/ppo.py --run first --resume runs/first/latest.pt --updates 500     # carry on

The simulation runs on the CPU cores (``--procs`` processes x ``--envs`` games each), the policy
and the PPO update on the GPU. Everything a run produces goes to ``runs/<run>/``:

    latest.pt            newest checkpoint (also written on Ctrl+C)
    update_000100.pt     snapshots every --snapshot-every updates
    metrics.jsonl        one JSON object per update (and per evaluation)
    config.json          the options of the run

With MLflow installed the run is also tracked in the experiment --experiment (see aimoba.tracking):
the options, per-update metrics against environment steps, evaluation results and the snapshots.
Browse with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`; --no-mlflow turns it off.

Measure a checkpoint with ``python python/train/evaluate.py runs/<run>/latest.pt``.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimoba import AimobaParallelEnv  # noqa: E402
from aimoba.tracking import start_run  # noqa: E402

from evaluate import evaluate, format_result  # noqa: E402
from league import SELF, LeagueEnv  # noqa: E402
from policy import MLPPolicy, imitation_loss, load_checkpoint, save_checkpoint  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--run", default="ppo", help="name of the run; output goes to runs/<run>/")
p.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[2] / "runs"))
p.add_argument("--resume", default=None, help="checkpoint to continue from")
p.add_argument("--updates", type=int, default=200, help="updates to run in this invocation")
# Environment.
p.add_argument("--procs", type=int, default=20, help="simulation processes (one CPU core each)")
p.add_argument("--envs", type=int, default=16, help="games per process")
p.add_argument("--groups", type=int, default=1, help="pipeline the rollout over this many groups of processes: the policy picks actions for "
               "one group while the others simulate (1 = every process steps in lockstep). Each group costs a policy call per "
               "step, so 2 only pays off when the sim is slow: about +10%% vs expert bots, -7%% vs easy ones")
p.add_argument("--opponent", default="basic", help="scripted bot the learner plays against: easy, basic or expert; several, comma-separated, "
               "are spread over the processes")
p.add_argument("--expert", default=None, help="scripted bot whose choices are an extra imitation target (see --bc-coef), e.g. expert")
p.add_argument("--self-play-procs", type=int, default=0, help="of --procs, how many play against policies instead of scripted bots: "
               "earlier snapshots of this run (the pool) or the current policy")
p.add_argument("--pool-every", type=int, default=50, help="add a snapshot to the pool, and redraw the self-play opponents, every this many updates")
p.add_argument("--pool-size", type=int, default=10, help="snapshots kept in the pool (oldest dropped first)")
p.add_argument("--pool-latest", type=float, default=0.25, help="share of self-play processes that face the current policy rather than a snapshot")
p.add_argument("--pool-draw", type=int, default=3, help="distinct snapshots faced at a time, redrawn every --pool-every updates "
               "(each one costs a forward pass per step)")
p.add_argument("--sides", default="both", choices=["both", "blue", "red"], help="which team the learner plays")
p.add_argument("--frame-skip", type=int, default=6, help="sim ticks per decision (30 ticks = 1 s)")
p.add_argument("--max-time", type=float, default=1500.0, help="game time limit in seconds")
p.add_argument("--team-spirit", type=float, default=0.3, help="0 = every hero for itself, 1 = fully shared team reward")
p.add_argument("--zero-sum", type=float, default=1.0, help="share of the enemy team's mean reward subtracted from every hero's reward")
p.add_argument("--reward", default=None, help="reward weights to change, e.g. kill=3,assist=1.5 (see DEFAULT_REWARD in src/sim/ml.ts); "
               "the win rate is what tells whether a differently paid policy plays better")
# PPO.
p.add_argument("--steps", type=int, default=128, help="rollout length per update")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.995)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--vf", type=float, default=0.5)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--minibatch", type=int, default=16384)
p.add_argument("--bc-coef", type=float, default=0.0, help="weight of the imitation loss towards --expert at the start of this invocation")
p.add_argument("--bc-updates", type=int, default=0, help="updates over which that weight falls to --bc-end (0 = keep it constant)")
p.add_argument("--bc-end", type=float, default=0.0)
p.add_argument("--critic-updates", type=int, default=0, help="at the start of this invocation, train only the critic for this many updates")
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--activation", default="tanh", choices=["tanh", "relu", "gelu"])
p.add_argument("--critic-state", action=argparse.BooleanOptionalAction, default=True, help="let the critic see the fog-free team state")
# Bookkeeping.
p.add_argument("--seed", type=int, default=1)
p.add_argument("--save-every", type=int, default=10, help="write latest.pt every this many updates")
p.add_argument("--snapshot-every", type=int, default=100, help="keep a numbered checkpoint every this many updates (0 = never)")
p.add_argument("--eval-every", type=int, default=0, help="run the evaluation harness every this many updates (0 = never)")
p.add_argument("--eval-games", type=int, default=40)
p.add_argument("--eval-opponents", default="basic,expert")
p.add_argument("--eval-greedy", action="store_true", help="evaluate with the most likely action instead of sampling")
p.add_argument("--experiment", default="aimoba-ppo", help="MLflow experiment name")
p.add_argument("--no-mlflow", action="store_true", help="do not track this run in MLflow")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device(args.device)
run_dir = Path(args.runs_dir) / args.run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

# Opponents are dealt out over the processes; with --sides both, each opponent is met from both sides.
# The last --self-play-procs processes play against policies instead.
opponents = args.opponent.split(",")
n_self = max(0, min(args.self_play_procs, args.procs))
proc_opponent = [opponents[(i // 2 if args.sides == "both" else i) % len(opponents)] for i in range(args.procs - n_self)] + [SELF] * n_self
proc_side = [(i % 2) if args.sides == "both" else int(args.sides == "red") for i in range(args.procs)]
reward = {k: float(v) for k, v in (kv.split("=") for kv in args.reward.split(","))} if args.reward else None
common = dict(seed=args.seed, expert=args.expert, frame_skip=args.frame_skip, max_time=args.max_time, team_spirit=args.team_spirit, zero_sum=args.zero_sum, reward=reward)
if n_self:
    env = LeagueEnv(proc_opponent, proc_side, args.envs, device, **common)
else:
    env = AimobaParallelEnv(args.procs, args.envs, per_proc=[{"opponent": o, "agents": "red" if sd else "blue"} for o, sd in zip(proc_opponent, proc_side)], **common)
env_opponent = np.repeat(proc_opponent, args.envs)
N, A, O, M, T = env.num_envs, env.n_agents, env.obs_size, env.mask_size, args.steps
S, R = env.state_size, env.num_envs * env.n_teams  # the critic state comes as one row per team per game
B = N * A  # every hero in every game is one sample
learner_team = env.agent_slots[:, 0] // 5

start_update, env_steps = 0, 0
if args.resume:
    policy, ckpt = load_checkpoint(args.resume, device)
    start_update, env_steps = ckpt["meta"].get("update", 0), ckpt["meta"].get("env_steps", 0)
else:
    policy, ckpt = MLPPolicy(O, env.action_nvec, args.hidden, args.layers, S if args.critic_state else 0, env.action_heads.tolist(), env.action_targets.tolist(), env.hero_offset, args.activation).to(device), None
opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
if ckpt and ckpt.get("optimizer") and ckpt["meta"].get("trainer", "ppo") == "ppo" and "imitated" not in ckpt["meta"]:
    opt.load_state_dict(ckpt["optimizer"])  # only when continuing a PPO run: another trainer's Adam statistics do not fit
    for group in opt.param_groups:
        group["lr"] = args.lr

print(f"run {run_dir} | {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) | {args.procs} procs x {args.envs} games = {N} games in {max(1, min(args.groups, args.procs))} pipelined groups, "
      f"{B} heroes | {T * B} samples per update | {sum(x.numel() for x in policy.parameters()) / 1e6:.2f}M parameters | vs {args.opponent}", flush=True)

buf_obs = torch.zeros(T, B, O, device=device)
buf_mask = torch.zeros(T, B, M, dtype=torch.uint8, device=device)
buf_act = torch.zeros(T, B, 4, dtype=torch.long, device=device)
buf_logp, buf_val, buf_rew, buf_done = (torch.zeros(T, B, device=device) for _ in range(4))
buf_state = torch.zeros(T, R, S, device=device)
buf_label = torch.full((T, B, 4), 255, dtype=torch.uint8, device=device)  # what --expert would have done
state_row = torch.as_tensor(np.arange(N)[:, None] * env.n_teams + env.team_index, device=device).reshape(B)  # each hero's row of the state
# Pipelining: the processes are split into contiguous groups; while one group waits for its actions, the others simulate.
G = max(1, min(args.groups, args.procs))
bounds = np.linspace(0, args.procs, G + 1).round().astype(int)
proc_groups = [range(bounds[g], bounds[g + 1]) for g in range(G)]
env_rows = [slice(r.start * args.envs, r.stop * args.envs) for r in proc_groups]
hero_rows = [slice(e.start * A, e.stop * A) for e in env_rows]
state_rows = [slice(e.start * env.n_teams, e.stop * env.n_teams) for e in env_rows]


def to_device(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(device)  # copies: the env reuses its arrays


def act(t: int, g: int) -> None:
    """Pick the actions of group g for rollout step t and send them off."""
    hs, ss = hero_rows[g], state_rows[g]
    with torch.no_grad():
        a, logp, _, val = policy(obs[hs], mask[hs], state=state[state_row[hs]])
    buf_obs[t, hs], buf_mask[t, hs], buf_act[t, hs], buf_logp[t, hs], buf_val[t, hs] = obs[hs], mask[hs], a, logp, val
    buf_state[t, ss], buf_label[t, hs] = state[ss], label[hs]
    env.step_send(a.cpu().numpy(), proc_groups[g])


def collect(t: int, g: int) -> float:
    """Wait for group g's step t; returns the seconds spent waiting for the simulation."""
    global n_episodes
    es, hs, ss = env_rows[g], hero_rows[g], state_rows[g]
    t1 = time.time()
    obs_np, rew_np, done_np, mask_np, infos = env.step_recv(proc_groups[g])
    waited = time.time() - t1
    buf_rew[t, hs] = to_device(rew_np).reshape(-1)
    buf_done[t, hs] = to_device(done_np).repeat_interleave(A).float()
    ep_return[es] += rew_np.mean(axis=1)
    for i in np.flatnonzero(done_np):
        j = es.start + i
        w = infos[i]["winner"]
        episodes.append((float(w == learner_team[j]), float(w < 0), infos[i]["time"] / 60, ep_return[j]))
        wins_vs[env_opponent[j]].append(float(w == learner_team[j]))
        ep_return[j] = 0
        n_episodes += 1
    obs[hs], mask[hs], state[ss] = to_device(obs_np).reshape(-1, O), to_device(mask_np).reshape(-1, M), to_device(env.state[es]).reshape(-1, S)
    label[hs] = to_device(env.expert[es]).reshape(-1, 4)
    return waited


def save(name: str, update: int) -> None:
    save_checkpoint(run_dir / name, policy, opt, update=update, env_steps=env_steps, trainer="ppo", opponent=args.opponent, frame_skip=args.frame_skip)


obs_np, mask_np = env.reset(seed=args.seed + start_update)
obs, mask, state = to_device(obs_np).reshape(B, O), to_device(mask_np).reshape(B, M), to_device(env.state).reshape(R, S)
label = to_device(env.expert).reshape(B, 4)
wins_vs: dict[str, deque[float]] = {o: deque(maxlen=200) for o in dict.fromkeys(proc_opponent)}  # recent results against each opponent

# Self-play: a pool of frozen snapshots of this run; each self-play process faces one of them or the current policy.
pool: list[tuple[str, torch.nn.Module]] = []
pool_dir = run_dir / "pool"
rng = random.Random(args.seed)


def draw_opponents() -> None:
    faced = rng.sample(pool, min(max(1, args.pool_draw), len(pool)))
    for i in range(args.procs - n_self, args.procs):
        if not faced or rng.random() < args.pool_latest:
            env.set_opponent(i, policy, "latest")
        else:
            env.set_opponent(i, *reversed(rng.choice(faced)))


if n_self:
    for f in sorted(pool_dir.glob("update_*.pt"))[-args.pool_size:]:
        pool.append((f.stem, load_checkpoint(f, device)[0].eval()))
    draw_opponents()
episodes: deque[tuple[float, float, float, float]] = deque(maxlen=200)  # (win, draw, minutes, return) of recent games
ep_return = np.zeros(N)  # mean per-hero return of the game in progress
n_episodes = 0
tracker = start_run(
    args.experiment,
    run_name=args.run,
    params={**vars(args), "obs_size": O, "n_agents": A, "num_envs": N, "action_nvec": list(env.action_nvec), "parameters": sum(x.numel() for x in policy.parameters())},
    tags={"algo": "ppo", "trainer": "python/train/ppo.py", "torch": torch.__version__, "policy": policy.config["kind"], "start_update": start_update},
    enabled=not args.no_mlflow,
)
status = "FINISHED"
metrics = (run_dir / "metrics.jsonl").open("a")
update = start_update
t_start = time.time()
try:
    for update in range(start_update + 1, start_update + args.updates + 1):
        # --- Rollout -------------------------------------------------------------------------
        # Step k of the pipeline is rollout step k // G of group k % G. Every group always has its next step in
        # flight except the one whose actions are being picked, so with G > 1 the GPU work hides behind the simulation.
        t0 = time.time()
        t_sim = 0.0  # time spent waiting for the simulation
        for k in range(G):
            act(0, k)
        for k in range(T * G):
            t_sim += collect(k // G, k % G)
            if k + G < T * G:
                act((k + G) // G, k % G)
        env_steps += T * N
        t_rollout = time.time() - t0

        # --- Generalised advantage estimation ------------------------------------------------------
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
            ret = adv + buf_val

        # --- PPO update -------------------------------------------------------------------------
        b_obs, b_mask, b_act = buf_obs.reshape(T * B, O), buf_mask.reshape(T * B, M), buf_act.reshape(T * B, 4)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_state, b_label = buf_state.reshape(T * R, S), buf_label.reshape(T * B, 4)
        done_here = update - start_update  # updates so far in this invocation
        bc = args.bc_coef if not args.bc_updates else args.bc_end + (args.bc_coef - args.bc_end) * max(0.0, 1 - (done_here - 1) / args.bc_updates)
        bc = bc if args.expert else 0.0
        critic_only = done_here <= args.critic_updates
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "imitation_loss", "acc_kind")}
        for _ in range(args.epochs):
            for idx in torch.randperm(T * B, device=device).split(args.minibatch):
                _, logp, ent, val = policy(b_obs[idx], b_mask[idx], b_act[idx], state=b_state[(idx // B) * R + state_row[idx % B]])
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                log_ratio = logp - b_logp[idx]
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = (val - b_ret[idx]).pow(2).mean()
                loss = args.vf * vl if critic_only else pg + args.vf * vl - args.ent * ent.mean()
                im, acc_kind = pg.new_zeros(()), pg.new_zeros(())
                if args.expert:  # staying close to what the scripted expert would do
                    with torch.set_grad_enabled(bc > 0 and not critic_only):
                        im, acc = imitation_loss(policy, b_obs[idx], b_mask[idx], b_label[idx])
                    acc_kind = acc["kind"]
                    if bc > 0 and not critic_only:
                        loss = loss + bc * im
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    for k, v in zip(stats, (pg, vl, ent.mean(), ((ratio - 1) - log_ratio).mean(), ((ratio - 1).abs() > args.clip).float().mean(), im, acc_kind)):
                        stats[k].append(v.detach())
        t_learn = time.time() - t0

        # --- Report -------------------------------------------------------------------------------
        row = {
            "update": update, "env_steps": env_steps, "agent_steps": env_steps * A, "hours": (time.time() - t_start) / 3600,
            "env_steps_per_s": T * N / (t_rollout + t_learn), "rollout_s": t_rollout, "sim_s": t_sim, "learn_s": t_learn,
            "reward": buf_rew.mean().item(), "episodes": n_episodes, "bc_coef": bc, "critic_only": critic_only,
            "explained_var": (1 - (b_ret - buf_val.reshape(-1)).var() / (b_ret.var() + 1e-8)).item(),
            **{k: torch.stack(v).mean().item() for k, v in stats.items()},
        }
        msg = (f"update {update:5d} | {row['env_steps_per_s']:6.0f} env steps/s (rollout {t_rollout:4.1f}s, of which waiting for the sim {t_sim:4.1f}s; learn {t_learn:4.1f}s) | "
               f"reward {row['reward']:+.4f} | entropy {row['entropy']:.2f} | kl {row['approx_kl']:.4f} | ev {row['explained_var']:+.2f}")
        if episodes:
            wins, draws, minutes, returns = (float(np.mean(x)) for x in zip(*episodes))
            row.update(win_rate=wins, draw_rate=draws, game_minutes=minutes, episode_return=returns)
            msg += f" | last {len(episodes)} games: win {100 * wins:.0f}% draw {100 * draws:.0f}% {minutes:.1f} min"
            if len(wins_vs) > 1:
                msg += " (" + ", ".join(f"{o} {100 * np.mean(w):.0f}%" for o, w in wins_vs.items() if w) + ")"
        if args.expert:
            msg += f" | like {args.expert} {100 * row['acc_kind']:.0f}% (bc {bc:.2f})"
        print(msg, flush=True)
        # Same metric names as python/examples/train_ppo.py, so the two trainers share charts.
        tracked = {
            "perf/env_steps_per_sec": row["env_steps_per_s"], "perf/sim_share_of_rollout": t_sim / t_rollout, "perf/learn_s": t_learn,
            "rollout/mean_step_reward": row["reward"], "episode/count": n_episodes, "loss/explained_var": row["explained_var"],
            **{f"loss/{k.replace('_loss', '')}": row[k] for k in stats},
        }
        if episodes:
            tracked.update({"episode/win_rate": wins, "episode/draw_rate": draws, "episode/length_min": minutes, "episode/return": returns})
            tracked.update({f"episode/win_rate_vs_{o}": float(np.mean(w)) for o, w in wins_vs.items() if w})
        if args.expert:
            tracked.update({"imitation/loss": row["imitation_loss"], "imitation/acc_kind": row["acc_kind"], "imitation/coef": bc})
        if args.eval_every and update % args.eval_every == 0:
            row["eval"] = [evaluate(policy, opp, args.eval_games, args.procs, frame_skip=args.frame_skip, device=device, greedy=args.eval_greedy) for opp in args.eval_opponents.split(",")]
            for r in row["eval"]:
                print("  eval", format_result(r), flush=True)
                tracked.update({f"eval/{Path(r['opponent']).stem}/{k}": r[k] for k in ("win_rate", "minutes", "kills", "deaths", "towers_taken", "towers_lost")})
        tracker.log_metrics(tracked, step=env_steps)  # x axis: environment steps
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if update % args.save_every == 0:
            save("latest.pt", update)
        if n_self and update % args.pool_every == 0:
            save_checkpoint(pool_dir / f"update_{update:06d}.pt", policy, None, update=update, env_steps=env_steps, trainer="ppo")
            frozen = copy.deepcopy(policy).eval()
            for w in frozen.parameters():
                w.requires_grad_(False)
            pool.append((f"update_{update:06d}", frozen))
            del pool[:-args.pool_size]
            draw_opponents()
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

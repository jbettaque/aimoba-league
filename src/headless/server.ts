/**
 * Headless environment server. Speaks newline-delimited JSON over stdin/stdout so
 * any language (see python/aimoba) can drive vectorised games.
 *
 *   {"cmd":"make","num_envs":4,"config":{"agents":[0,1,2,3,4],"frameSkip":6}}
 *   {"cmd":"reset","seed":123}
 *   {"cmd":"step","actions":[...num_envs*n_agents*4 ints]}
 *   {"cmd":"replay","env":0}
 *   {"cmd":"close"}
 *
 * Float/byte arrays are returned base64-encoded (little-endian float32 / uint8). `reset` and `step`
 * return `obs`, `mask` and `state`: the fog-free critic state, one row of `state_size` floats per
 * controlled team per env (`teams` and `team_index` in the `make` reply say which row is whose).
 * They also return `expert`: per agent the action a shadowing scripted bot would take (uint8 x4,
 * 255 = "does not matter"; all 255 unless the config sets `expert`). Action kind 255 executes it.
 *
 * `--binary` switches to length-prefixed frames, which is what the Python bindings use
 * (no JSON or base64 on the hot path; all integers little-endian):
 *
 *   request   [u32 payload bytes][u8 type][payload]
 *               type 0: a JSON command as above (utf8)
 *               type 1: step; payload = num_envs*n_agents*4 action bytes (uint8)
 *   response  [u32 payload bytes][u8 type][3 pad bytes][payload]
 *               type 0: JSON (utf8)
 *               type 1: step  = obs f32 | rew f32 | mask u8 | done u8 | state f32 | expert u8 | JSON [[env, info], ...] for finished envs
 *               type 2: reset = obs f32 | mask u8 | state f32 | expert u8
 */
import { createInterface } from 'node:readline';
import { ACTION_HEADS, ACTION_NVEC, ACTION_TARGETS, MASK_SIZE, MobaEnv, OBS_BLOCKS, OBS_HERO_OFFSET, OBS_SIZE, STATE_BLOCKS, STATE_SIZE, makeReplay, type EnvConfig } from '../sim/ml.ts';
import { HEROES } from '../sim/heroes.ts';

let envs: MobaEnv[] = [];
let nAgents = 0, nTeams = 0;
let obs = new Float32Array(0), mask = new Uint8Array(0), rew = new Float32Array(0), state = new Float32Array(0), expert = new Uint8Array(0);

const b64 = (a: Float32Array | Uint8Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');

function gather() {
  for (let e = 0; e < envs.length; e++) {
    obs.set(envs[e].obs, e * nAgents * OBS_SIZE);
    mask.set(envs[e].mask, e * nAgents * MASK_SIZE);
    state.set(envs[e].state, e * nTeams * STATE_SIZE);
    expert.set(envs[e].labels, e * nAgents * 4);
  }
}

function resetAll(seed: number | null | undefined) {
  envs.forEach((env, e) => env.reset(seed !== undefined && seed !== null ? seed + e * 100003 : undefined));
  gather();
}

/** Step every env with a flat [kind, dir, dist, target] per agent; finished envs auto-reset. */
function stepAll(acts: ArrayLike<number> & { subarray?: (a: number, b: number) => ArrayLike<number>; slice: (a: number, b: number) => ArrayLike<number> }) {
  const per = nAgents * 4;
  if (acts.length !== envs.length * per) throw new Error(`expected ${envs.length * per} action ints, got ${acts.length}`);
  const done: boolean[] = [];
  const infos: any[] = [];
  for (let e = 0; e < envs.length; e++) {
    const env = envs[e];
    const over = env.step(acts.subarray ? acts.subarray(e * per, (e + 1) * per) : acts.slice(e * per, (e + 1) * per));
    rew.set(env.rewards, e * nAgents);
    done.push(over);
    if (over) {
      const info: any = env.info();
      info.truncated = env.game.winner < 0;
      infos.push(info);
      env.reset(); // auto-reset: the returned obs belongs to the new episode
    } else infos.push(null);
  }
  gather();
  return { done, infos };
}

function handle(msg: any): any {
  switch (msg.cmd) {
    case 'make': {
      const cfg: EnvConfig = msg.config ?? {};
      const n = msg.num_envs ?? 1;
      const seed = cfg.seed ?? 1;
      envs = [];
      for (let e = 0; e < n; e++) envs.push(new MobaEnv({ ...cfg, seed: seed + e * 100003 }));
      nAgents = envs[0].agents.length;
      nTeams = envs[0].teams.length;
      state = new Float32Array(n * nTeams * STATE_SIZE);
      expert = new Uint8Array(n * nAgents * 4);
      obs = new Float32Array(n * nAgents * OBS_SIZE);
      mask = new Uint8Array(n * nAgents * MASK_SIZE);
      rew = new Float32Array(n * nAgents);
      return {
        ok: true, obs_size: OBS_SIZE, mask_size: MASK_SIZE, action_nvec: ACTION_NVEC, n_agents: nAgents, num_envs: n,
        agents: envs[0].agents, heroes: envs[0].agents.map((a) => HEROES[a % 5].id), frame_skip: envs[0].frameSkip,
        state_size: STATE_SIZE, teams: envs[0].teams, team_index: envs[0].teamIndex, obs_blocks: OBS_BLOCKS, state_blocks: STATE_BLOCKS,
        action_heads: ACTION_HEADS, action_targets: ACTION_TARGETS, hero_offset: OBS_HERO_OFFSET,
      };
    }
    case 'reset': {
      resetAll(msg.seed);
      return { obs: b64(obs), mask: b64(mask), state: b64(state), expert: b64(expert) };
    }
    case 'step': {
      const { done, infos } = stepAll(msg.actions);
      return { obs: b64(obs), mask: b64(mask), rew: b64(rew), state: b64(state), expert: b64(expert), done, info: infos };
    }
    case 'info':
      return { info: envs.map((e) => e.info()) };
    case 'replay': {
      const env = envs[msg.env ?? 0];
      const current = msg.current && env.game.cmdLog ? makeReplay(env.game) : env.lastReplay;
      return { replay: current };
    }
    case 'close':
      process.exit(0);
    // eslint-disable-next-line no-fallthrough
    default:
      return { error: `unknown cmd ${msg.cmd}` };
  }
}

const bytes = (a: Float32Array | Uint8Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);

function sendFrame(type: number, parts: Buffer[]) {
  const head = Buffer.alloc(8);
  head.writeUInt32LE(parts.reduce((n, p) => n + p.length, 0), 0);
  head[4] = type;
  // One copy per reply: the typed arrays are reused by the next step.
  process.stdout.write(Buffer.concat([head, ...parts]));
}

function handleFrame(type: number, payload: Buffer) {
  try {
    if (type === 1) {
      const { done, infos } = stepAll(payload);
      const finished: [number, any][] = [];
      infos.forEach((info, e) => { if (info) finished.push([e, info]); });
      sendFrame(1, [bytes(obs), bytes(rew), bytes(mask), Buffer.from(done.map(Number)), bytes(state), bytes(expert), Buffer.from(finished.length ? JSON.stringify(finished) : '')]);
      return;
    }
    const msg = JSON.parse(payload.toString('utf8'));
    if (msg.cmd === 'reset') {
      resetAll(msg.seed);
      sendFrame(2, [bytes(obs), bytes(mask), bytes(state), bytes(expert)]);
    } else sendFrame(0, [Buffer.from(JSON.stringify(handle(msg)))]);
  } catch (err) {
    sendFrame(0, [Buffer.from(JSON.stringify({ error: String((err as Error).stack ?? err) }))]);
  }
}

if (process.argv.includes('--binary')) {
  let pending: Buffer = Buffer.alloc(0);
  process.stdin.on('data', (chunk: Buffer) => {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    while (pending.length >= 5) {
      const len = pending.readUInt32LE(0);
      if (pending.length < 5 + len) break;
      const type = pending[4];
      const payload = pending.subarray(5, 5 + len);
      pending = pending.subarray(5 + len);
      handleFrame(type, payload);
    }
  });
  process.stdin.on('end', () => process.exit(0));
} else {
  const rl = createInterface({ input: process.stdin });
  rl.on('line', (line) => {
    if (!line.trim()) return;
    let out: any;
    try {
      out = handle(JSON.parse(line));
    } catch (err) {
      out = { error: String((err as Error).stack ?? err) };
    }
    process.stdout.write(JSON.stringify(out) + '\n');
  });
  rl.on('close', () => process.exit(0));
}

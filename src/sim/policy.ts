/**
 * Trained policies in the game: loads a network exported by `python/train/export.py` and lets it
 * play heroes of a running `Game`, the same way the training environment (`MobaEnv` in ml.ts) does:
 * one decision every `frameSkip` ticks from the hero's own fog-limited observation.
 *
 * The heroes a policy plays must not be scripted bots (`GameConfig.bots[slot] = false`) and not
 * `manual`, so they level up and shop automatically as in training. Their orders go through
 * `game.command`, so a recorded game replays without the network.
 *
 *   const net = loadPolicy(await (await fetch('policies/selfplay.bin')).arrayBuffer());
 *   const driver = new PolicyDriver(game, net, [5, 6, 7, 8, 9]);
 *   while (!game.over) { driver.tick(); game.step(); }
 */
import type { Game } from './game.ts';
import { ACTION_NVEC, ACTION_TARGETS, FogMemory, MASK_SIZE, N_KINDS, N_TARGETS, OBS_SIZE, decodeAction, observe } from './ml.ts';
import { Rng } from './rng.ts';

interface Layer { w: Float32Array; b: Float32Array; rows: number; cols: number }

export interface PolicyHeader {
  obs_size: number;
  nvec: number[];
  hidden: number;
  layers: number;
  activation: string;
  hero_offset: number;
  dtype: string;
  tensors: [string, ...number[]][];
  check?: { obs: number[]; mask: number[]; kind: number[]; sub_kind: number; dir: number[]; dist: number[]; target: number[] };
  meta?: Record<string, unknown>;
}

function half(h: number): number {
  const e = (h >> 10) & 31, f = h & 1023, s = h & 0x8000 ? -1 : 1;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** y = W x + b for the output rows [from, from + n). */
function affine(l: Layer, x: Float32Array, y: Float32Array, from = 0, n = l.rows) {
  const { w, b, cols } = l;
  for (let i = 0; i < n; i++) {
    let o = (from + i) * cols, s = b[from + i], j = 0;
    for (; j + 3 < cols; j += 4, o += 4) s += w[o] * x[j] + w[o + 1] * x[j + 1] + w[o + 2] * x[j + 2] + w[o + 3] * x[j + 3];
    for (; j < cols; j++, o++) s += w[o] * x[j];
    y[i] = s;
  }
}

/** In place: logits -> log-probabilities over the allowed entries (the others become -Infinity). */
function logSoftmax(v: Float32Array, n: number, allowed?: (i: number) => boolean) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (allowed && !allowed(i)) v[i] = -Infinity;
    else if (v[i] > max) max = v[i];
  }
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(v[i] - max);
  const z = max + Math.log(sum);
  for (let i = 0; i < n; i++) v[i] -= z;
}

/** The most likely entry, or with `rng` one drawn from the distribution. */
function choose(lp: Float32Array, n: number, rng: Rng | null): number {
  let best = 0;
  if (!rng) {
    for (let i = 1; i < n; i++) if (lp[i] > lp[best]) best = i;
    return best;
  }
  let r = rng.next();
  for (let i = 0; i < n; i++) {
    const p = Math.exp(lp[i]);
    if (p <= 0) continue;
    best = i;
    r -= p;
    if (r < 0) break;
  }
  return best;
}

/** The actor of an exported policy: an MLP body, a head for the action kind, and target / direction / distance heads with one output set per kind. */
export class PolicyNet {
  header: PolicyHeader;
  private body: Layer[] = [];
  private heads: Record<string, Layer> = {};
  private a: Float32Array;
  private b: Float32Array;
  private relu: boolean;
  /** Log-probabilities of the last `act` / `logProbs` call: kinds, then for the chosen kind directions, distances and targets. */
  kind = new Float32Array(N_KINDS);
  dir: Float32Array;
  dist: Float32Array;
  target = new Float32Array(N_TARGETS);

  constructor(header: PolicyHeader, tensors: Map<string, Layer>) {
    this.header = header;
    if (header.obs_size !== OBS_SIZE || header.nvec.join() !== ACTION_NVEC.join()) throw new Error('this policy was trained for a different observation or action space than the current game');
    if (header.activation !== 'relu' && header.activation !== 'tanh') throw new Error(`unsupported activation ${header.activation}`);
    this.relu = header.activation === 'relu';
    const layer = (name: string): Layer => {
      const w = tensors.get(`${name}.weight`), b = tensors.get(`${name}.bias`);
      if (!w || !b) throw new Error(`policy file has no ${name}`);
      return { w: w.w, b: b.w, rows: w.rows, cols: w.cols };
    };
    for (let i = 0; i < header.layers; i++) this.body.push(layer(`body.${i * 2}`));
    for (const h of ['kind_head', 'dir_head', 'dist_head', 'target_head']) this.heads[h] = layer(h);
    this.a = new Float32Array(header.hidden);
    this.b = new Float32Array(header.hidden);
    this.dir = new Float32Array(header.nvec[1]);
    this.dist = new Float32Array(header.nvec[2]);
  }

  private hidden(obs: Float32Array, off: number): Float32Array {
    let x = off === 0 && obs.length === OBS_SIZE ? obs : obs.subarray(off, off + OBS_SIZE);
    let y = this.a;
    for (const l of this.body) {
      affine(l, x, y);
      if (this.relu) { for (let i = 0; i < l.rows; i++) if (y[i] < 0) y[i] = 0; } else for (let i = 0; i < l.rows; i++) y[i] = Math.tanh(y[i]);
      x = y;
      y = y === this.a ? this.b : this.a;
    }
    return x;
  }

  /** Fills `kind`, and `dir` / `dist` / `target` for the kind `pick(kind log-probabilities)` returns; returns that kind. */
  logProbs(obs: Float32Array, off: number, mask: Uint8Array, moff: number, pick: (kind: Float32Array) => number): number {
    const h = this.hidden(obs, off);
    const [nk, nd, ns, nt] = this.header.nvec;
    affine(this.heads.kind_head, h, this.kind);
    logSoftmax(this.kind, nk, (i) => mask[moff + i] > 0);
    const k = pick(this.kind);
    affine(this.heads.dir_head, h, this.dir, k * nd, nd);
    logSoftmax(this.dir, nd);
    affine(this.heads.dist_head, h, this.dist, k * ns, ns);
    logSoftmax(this.dist, ns);
    affine(this.heads.target_head, h, this.target, k * nt, nt);
    // Target slots that are occupied and fit the kind; a kind that takes no target ignores the head.
    let hero = 0;
    const ho = off + this.header.hero_offset;
    for (let i = 1; i < ACTION_TARGETS.length; i++) if (obs[ho + i] > obs[ho + hero]) hero = i;
    const fits = ACTION_TARGETS[hero][k];
    const ok = (i: number) => mask[moff + nk + i] > 0 && fits[i] > 0;
    let any = false;
    for (let i = 0; i < nt; i++) any ||= ok(i);
    logSoftmax(this.target, nt, any ? ok : undefined);
    return k;
  }

  /** Chooses an action [kind, direction, distance, target] for one hero: the most likely one, or a sampled one with `rng`. */
  act(obs: Float32Array, off: number, mask: Uint8Array, moff: number, rng: Rng | null, out: Int32Array | number[]) {
    const [nk, nd, ns, nt] = this.header.nvec;
    out[0] = this.logProbs(obs, off, mask, moff, (kind) => choose(kind, nk, rng));
    out[3] = choose(this.target, nt, rng);
    out[1] = choose(this.dir, nd, rng);
    out[2] = choose(this.dist, ns, rng);
  }

  /** Largest difference between this implementation and the exporter's on the worked example in the file (should be ~1e-5). */
  check(): number {
    const c = this.header.check;
    if (!c) return 0;
    this.logProbs(Float32Array.from(c.obs), 0, Uint8Array.from(c.mask), 0, () => c.sub_kind);
    let worst = 0;
    const cmp = (got: Float32Array, want: number[]) => {
      for (let i = 0; i < want.length; i++) {
        const d = want[i] < -1e8 ? (got[i] < -1e8 ? 0 : Infinity) : Math.abs(got[i] - want[i]);
        if (d > worst) worst = d;
      }
    };
    cmp(this.kind, c.kind);
    cmp(this.dir, c.dir);
    cmp(this.dist, c.dist);
    cmp(this.target, c.target);
    return worst;
  }
}

/** Parse a policy file written by `python/train/export.py`. */
export function loadPolicy(data: ArrayBuffer | Uint8Array): PolicyNet {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 12 || String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'AIMP') throw new Error('not an AIMOBA policy file');
  if (view.getUint32(4, true) !== 1) throw new Error('unsupported policy file version');
  const hlen = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + hlen))) as PolicyHeader;
  if (header.dtype !== 'f16') throw new Error(`unsupported policy dtype ${header.dtype}`);
  const tensors = new Map<string, Layer>();
  let off = 12 + hlen;
  for (const [name, rows, cols = 1] of header.tensors) {
    const n = rows * cols;
    if (off + n * 2 > bytes.length) throw new Error('policy file is truncated');
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++, off += 2) w[i] = half(view.getUint16(off, true));
    tensors.set(name, { w, b: w, rows, cols });
  }
  return new PolicyNet(header, tensors);
}

/** Lets a policy play some heroes of a game. Call `tick()` before every `game.step()`. */
export class PolicyDriver {
  game: Game;
  net: PolicyNet;
  slots: number[];
  frameSkip: number;
  private memory = new FogMemory();
  private rng: Rng | null;
  private start: number;
  private obs = new Float32Array(OBS_SIZE);
  private mask = new Uint8Array(MASK_SIZE);
  private targets = new Int32Array(N_TARGETS);
  private action = new Int32Array(4);

  /**
   * `slots`: the hero slots to play (0-4 Blue, 5-9 Red). `greedy`: always take the most likely action instead of
   * sampling one (sampling is what the policies trained with; it is seeded from the game's seed, so a game is reproducible).
   */
  constructor(game: Game, net: PolicyNet, slots: number[], opts: { greedy?: boolean; frameSkip?: number; seed?: number } = {}) {
    this.game = game;
    this.net = net;
    this.slots = slots;
    this.frameSkip = opts.frameSkip ?? 6;
    this.rng = opts.greedy ? null : new Rng((opts.seed ?? game.seed) ^ 0x51ed270b ^ (slots[0] << 16));
    this.start = game.tick;
    this.memory.reset(game);
  }

  tick() {
    const g = this.game;
    if (g.over) return;
    if (g.tick > this.start) this.memory.update(g);
    if ((g.tick - this.start) % this.frameSkip !== 0) return;
    for (const slot of this.slots) {
      const u = g.heroes[slot];
      observe(g, u, this.obs, 0, this.mask, 0, this.targets, this.memory);
      this.net.act(this.obs, 0, this.mask, 0, this.rng, this.action);
      const c = decodeAction(g, u, this.action[0], this.action[1], this.action[2], this.action[3], this.targets);
      if (c) g.command(slot, c);
    }
  }
}

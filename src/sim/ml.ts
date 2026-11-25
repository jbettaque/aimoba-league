/**
 * Machine-learning interface: fixed-size observations, a MultiDiscrete action
 * space with masks, shaped rewards and a gym-style multi-agent environment.
 *
 * Observations are egocentric and team-canonical: Red's view is mirrored across
 * the river so "own base" is always bottom-left. One policy can play both sides.
 *
 * Besides the per-hero observation (which respects fog of war) there is a per-team
 * state vector with no fog, meant only for a critic during training.
 *
 * For imitation learning the environment can also say, every step, what a scripted bot would
 * do in each controlled hero's place (EnvConfig.expert), as an action label.
 */
import { MAX_LEVEL, TICK_RATE, TOWER_RANGE, xpForLevel, type Command, type Projectile, type Unit, type Zone } from './defs.ts';
import { BOT_KINDS, Game, type GameConfig } from './game.ts';
import { HEROES } from './heroes.ts';
import { CAMPS, FOUNTAIN_POS, LANE_LENGTH, MAP_SIZE, findPath, laneProgress } from './map.ts';
import { Rng } from './rng.ts';
import { atan2, cos, hypot, sin, sq } from './dmath.ts';

export const ACT_NOOP = 0, ACT_MOVE = 1, ACT_ATTACK = 2, ACT_Q = 3, ACT_W = 4, ACT_E = 5, ACT_R = 6, ACT_RECALL = 7;
export const N_KINDS = 8;
/** Compass directions for movement and aim. 64 keeps a skillshot's aim error under 50 units at 1000 range. */
export const N_DIRS = 64;
export const N_DISTS = 4;
/**
 * Target slots: 0-4 enemy heroes, 5-12 nearest enemy creeps, 13-15 nearest enemy structures, 16-20 allied heroes,
 * 21 "none". An aimed spell (skillshot, dash, ground effect) with an enemy as target is aimed at that unit, leading
 * it if it moves; with "none" it is aimed by direction and distance.
 */
export const N_TARGETS = 22;
export const ACTION_NVEC = [N_KINDS, N_DIRS, N_DISTS, N_TARGETS];
export const MASK_SIZE = N_KINDS + N_TARGETS;
/** As an action kind: do exactly what the shadow expert would do (needs EnvConfig.expert). */
export const ACT_EXPERT = 255;
/** In an expert label: this component does not matter (or, as the kind, there is no usable label). */
export const NO_LABEL = 255;
/**
 * Which components an action kind can use, per hero: ACTION_HEADS[hero][kind] = [direction, distance, target].
 * Where a kind lists the target together with direction or distance (aimed spells), the direction and distance
 * only count when the target is "none".
 */
export const ACTION_HEADS: number[][][] = HEROES.map((def) => {
  const rows: number[][] = [];
  for (let kind = 0; kind < N_KINDS; kind++) {
    if (kind === ACT_MOVE) rows.push([1, 1, 0]);
    else if (kind === ACT_ATTACK) rows.push([0, 0, 1]);
    else if (kind >= ACT_Q && kind <= ACT_R) {
      const tg = def.abilities[kind - ACT_Q].targeting;
      rows.push(tg === 'self' ? [0, 0, 0] : tg === 'dir' ? [1, 0, 1] : tg === 'point' ? [1, 1, 1] : [0, 0, 1]);
    } else rows.push([0, 0, 0]);
  }
  return rows;
});
/** Which target slots make sense for an action kind, per hero: ACTION_TARGETS[hero][kind][slot] (the mask says which are occupied). */
export const ACTION_TARGETS: number[][][] = HEROES.map((def) => {
  const span = (from: number, to: number, none = false) => Array.from({ length: N_TARGETS }, (_, s) => ((s >= from && s < to) || (none && s === 21) ? 1 : 0));
  const rows: number[][] = [];
  for (let kind = 0; kind < N_KINDS; kind++) {
    if (kind === ACT_ATTACK) rows.push(span(0, 16));
    else if (kind >= ACT_Q && kind <= ACT_R) {
      const tg = def.abilities[kind - ACT_Q].targeting;
      rows.push(tg === 'self' ? span(0, 0) : tg === 'ally' ? span(16, 21) : tg === 'enemyHero' ? span(0, 5) : tg === 'enemy' ? span(0, 13) : span(0, 13, true));
    } else rows.push(span(0, 0));
  }
  return rows;
});
/**
 * How each aimed spell travels, by hero id and ability slot: [speed, delay]. Speed 0 = lands after `delay` wherever
 * it was aimed. Used to lead a moving target.
 */
const AIM: Record<string, ([number, number] | null)[]> = {
  korvath: [[1300, 0], null, null, [1500, 0]],
  sable: [null, null, null, null],
  ignara: [[1500, 0], [0, 0.65], [1150, 0], [0, 1.0]],
  vesper: [[1700, 0], null, [1400, 0], [2200, 0]],
  liora: [null, [1300, 0], null, null],
};

/** Where to aim a spell cast by `u` so that it meets `t` if `t` keeps its course. */
function aimPoint(u: Unit, t: Unit, slot: number): [number, number] {
  const spec = AIM[HEROES[u.hero!.def].id]?.[slot];
  const still = t.stunned || t.rooted || t.dash !== null;
  if (!spec || still) return [t.x, t.y];
  let px = t.x, py = t.y;
  for (let i = 0; i < 3; i++) {
    const tm = spec[1] + (spec[0] > 0 ? hypot(px - u.x, py - u.y) / spec[0] : 0);
    px = t.x + t.vx * tm;
    py = t.y + t.vy * tm;
  }
  return [px, py];
}

const N_CREEPS = 8, N_ESTRUCT = 3, N_AMINIONS = 4, N_ASTRUCT = 2, N_PROJ = 3, N_ZONES = 2;
const T_CREEP = 5, T_STRUCT = 13, T_ALLY = 16;
export const T_NONE = 21;
const F_ALLY = 9, F_ENEMY = 20, F_CREEP = 9, F_AMINION = 4, F_ESTRUCT = 8, F_ASTRUCT = 4, F_PROJ = 12, F_ZONE = 7;

/** The observation as a list of blocks, in order: [name, entities, features per entity]. */
export const OBS_BLOCKS: [string, number, number][] = [
  ['global', 1, 12],
  ['self', 1, 48],
  ['allies', 5, F_ALLY],
  ['enemies', 5, F_ENEMY],
  ['creeps', N_CREEPS, F_CREEP],
  ['allyMinions', N_AMINIONS, F_AMINION],
  ['enemyStructures', N_ESTRUCT, F_ESTRUCT],
  ['allyStructures', N_ASTRUCT, F_ASTRUCT],
  ['lanes', 1, 6],
  ['projectiles', N_PROJ, F_PROJ],
  ['zones', N_ZONES, F_ZONE],
];
/** Floats per block. */
export const OBS_LAYOUT: Record<string, number> = Object.fromEntries(OBS_BLOCKS.map(([name, n, f]) => [name, n * f]));
/** Index of the first float of each block. */
export const OBS_OFFSET: Record<string, number> = {};
export const OBS_SIZE = OBS_BLOCKS.reduce((off, [name, n, f]) => { OBS_OFFSET[name] = off; return off + n * f; }, 0);
/** Index of the five floats that say which hero the observer is (one-hot, roster order). */
export const OBS_HERO_OFFSET = OBS_OFFSET.self + 36;

const F_SHERO = 19, N_STRUCT_TEAM = 9, N_CAMP_TEAM = 4;
/** The fog-free team state as blocks, like OBS_BLOCKS. Own team first, then the enemy. */
export const STATE_BLOCKS: [string, number, number][] = [
  ['global', 1, 8],
  ['camps', 2, N_CAMP_TEAM],
  ['heroes', 10, F_SHERO],
  ['structures', 2, N_STRUCT_TEAM],
  ['lanes', 1, 6],
  ['minions', 1, 6],
];
export const STATE_SIZE = STATE_BLOCKS.reduce((n, b) => n + b[1] * b[2], 0);

const clamp = (v: number, lo = -1, hi = 1) => (v < lo ? lo : v > hi ? hi : v);
const REL = 1 / 1600;

function nearest(list: Unit[], u: Unit, n: number): Unit[] {
  list.sort((a, b) => sq(a.x - u.x) + sq(a.y - u.y) - sq(b.x - u.x) - sq(b.y - u.y) || a.id - b.id);
  if (list.length > n) list.length = n;
  return list;
}

/** Whether a creep (minion or monster that is not on `u`'s side) stands in the line of fire from `u` to `t`: it would stop a plain skillshot. */
function lineBlocked(near: Unit[], u: Unit, t: Unit): boolean {
  const dx = t.x - u.x, dy = t.y - u.y;
  const len = hypot(dx, dy);
  if (len < 1 || len > 1300) return false;
  for (const m of near) {
    if (m.team === u.team || !m.alive || (m.kind !== 'minion' && m.kind !== 'monster')) continue;
    const along = ((m.x - u.x) * dx + (m.y - u.y) * dy) / len;
    if (along <= 0 || along >= len - 30) continue;
    if (Math.abs((m.x - u.x) * -dy + (m.y - u.y) * dx) / len < 70 + m.radius) return true;
  }
  return false;
}

/**
 * What each team remembers about enemy heroes it cannot currently see: where and when it last
 * saw them and how healthy they were. A dead hero is known to be at its fountain when it respawns.
 * Indexed by hero slot (each hero is only ever "remembered" by the other team).
 */
export class FogMemory {
  x = new Float64Array(10);
  y = new Float64Array(10);
  hp = new Float64Array(10);
  hpFrac = new Float64Array(10);
  tick = new Int32Array(10);

  reset(g: Game) {
    for (let s = 0; s < 10; s++) this.atFountain(g, s);
  }

  private atFountain(g: Game, s: number) {
    const u = g.heroes[s];
    this.x[s] = FOUNTAIN_POS[u.team][0];
    this.y[s] = FOUNTAIN_POS[u.team][1];
    this.hp[s] = u.maxHp;
    this.hpFrac[s] = 1;
    this.tick[s] = g.tick;
  }

  /** Call once per sim tick (or at least once per decision). */
  update(g: Game) {
    for (let s = 0; s < 10; s++) {
      const u = g.heroes[s];
      if (!u.alive) this.atFountain(g, s);
      else if (g.canSee(1 - u.team, u)) {
        this.x[s] = u.x;
        this.y[s] = u.y;
        this.hp[s] = u.hp;
        this.hpFrac[s] = u.hp / u.maxHp;
        this.tick[s] = g.tick;
      }
    }
  }
}

// The lane fronts are the same for every hero of a team, so they are computed once per team and tick.
const frontsCache = { game: null as Game | null, tick: -1, team: -1, v: new Float64Array(6) };

/** [own top, enemy top, own mid, enemy mid, own bot, enemy bot] lane fronts as seen by `team`. */
function laneFronts(g: Game, team: number): Float64Array {
  const c = frontsCache, v = c.v;
  if (c.game === g && c.tick === g.tick && c.team === team) return v;
  v.fill(0);
  for (const m of g.units) {
    if (m.kind !== 'minion' || !m.alive) continue;
    if (m.team !== team && !g.canSee(team, m)) continue;
    let p = laneProgress(m.lane, m.x, m.y) / LANE_LENGTH[m.lane];
    if (m.team === 1) p = 1 - p;
    const k = m.lane * 2 + (m.team === team ? 0 : 1);
    if (p > v[k]) v[k] = p;
  }
  c.game = g;
  c.tick = g.tick;
  c.team = team;
  return v;
}

/**
 * Write the observation for hero `u` into `obs` at `off`, the action mask into `mask` at `moff`,
 * and the unit ids behind each target slot into `targets` (-1 = empty). With `mem`, enemy heroes
 * in the fog are reported at their last known position together with how long ago that was.
 */
export function observe(g: Game, u: Unit, obs: Float32Array, off: number, mask: Uint8Array, moff: number, targets: Int32Array, mem?: FogMemory) {
  const h = u.hero!;
  const team = u.team;
  const flip = team === 1;
  const end = off + OBS_SIZE;
  obs.fill(0, off, end);
  mask.fill(0, moff, moff + MASK_SIZE);
  targets.fill(-1);
  let i = off;
  // Position of a map point relative to the hero, in the canonical (mirrored for Red) frame.
  const relpx = (x: number, y: number) => clamp((flip ? y - u.y : x - u.x) * REL, -2, 2);
  const relpy = (x: number, y: number) => clamp((flip ? x - u.x : y - u.y) * REL, -2, 2);
  const relx = (o: Unit) => relpx(o.x, o.y);
  const rely = (o: Unit) => relpy(o.x, o.y);
  const d = (o: Unit) => hypot(o.x - u.x, o.y - u.y);
  const inRange = (o: Unit) => (d(o) <= u.range + o.radius ? 1 : 0);

  // Global.
  obs[i++] = g.time / 1800;
  obs[i++] = clamp((g.teamKills[team] - g.teamKills[1 - team]) / 20);
  obs[i++] = g.towersLost[team] / 8;
  obs[i++] = g.towersLost[1 - team] / 8;
  obs[i++] = g.wyrmStacks[team] / 4;
  obs[i++] = g.wyrmStacks[1 - team] / 4;
  obs[i++] = g.campAlive[CAMPS.length - 2] > 0 ? 1 : 0;
  obs[i++] = g.campAlive[CAMPS.length - 1] > 0 ? 1 : 0;
  // The four camps of the own jungle (insight, stalkers, might, golem).
  for (let c = 0; c < N_CAMP_TEAM; c++) obs[i++] = g.campAlive[team * N_CAMP_TEAM + c] > 0 ? 1 : 0;

  // Self.
  const def = HEROES[h.def];
  obs[i++] = (flip ? u.y : u.x) / MAP_SIZE;
  obs[i++] = (flip ? u.x : u.y) / MAP_SIZE;
  obs[i++] = u.hp / u.maxHp;
  obs[i++] = u.maxMana > 0 ? u.mana / u.maxMana : 0;
  obs[i++] = u.maxHp / 4000;
  obs[i++] = h.level / MAX_LEVEL;
  obs[i++] = h.level >= MAX_LEVEL ? 1 : h.xp / xpForLevel(h.level);
  obs[i++] = clamp(h.gold / 3000, 0, 2);
  obs[i++] = u.ad / 300;
  obs[i++] = u.ap / 400;
  obs[i++] = u.armor / 200;
  obs[i++] = u.mr / 200;
  obs[i++] = u.as / 2.5;
  obs[i++] = u.range / 1000;
  obs[i++] = u.ms / 600;
  obs[i++] = u.alive ? 1 : 0;
  obs[i++] = u.alive ? 0 : clamp(h.respawn / 60, 0, 1);
  for (let s = 0; s < 4; s++) {
    const rank = h.ranks[s];
    obs[i++] = rank / (s === 3 ? 3 : 4);
    obs[i++] = rank > 0 ? clamp(h.cds[s] / def.abilities[s].cd[rank - 1], 0, 1) : 1;
    obs[i++] = g.castError(u, s) === '' ? 1 : 0;
  }
  obs[i++] = h.items.length / 6;
  obs[i++] = u.stunned ? 1 : 0;
  obs[i++] = u.rooted ? 1 : 0;
  obs[i++] = u.stealthed ? 1 : 0;
  obs[i++] = clamp(u.shield / 500, 0, 2);
  obs[i++] = h.order.type === 'recall' ? h.recallT / 5 : 0;
  obs[i++] = g.inFountain(u) ? 1 : 0;
  obs[i + h.def] = 1;
  i += 5;
  obs[i++] = g.hasBuff(u, 'might') ? 1 : 0;
  obs[i++] = g.hasBuff(u, 'insight') ? 1 : 0;
  obs[i++] = g.hasBuff(u, 'titan') ? 1 : 0;
  // Attack timing: seconds until the next attack can start, and whether one is winding up right now.
  obs[i++] = clamp(u.attackCd, 0, 2) / 2;
  obs[i++] = u.windup > 0 ? 1 : 0;
  // What the hero is already doing: an order keeps running until another action replaces it.
  obs[i++] = h.order.type === 'move' || h.order.type === 'amove' ? 1 : 0;
  obs[i++] = h.order.type === 'attack' ? 1 : 0;

  // Allied heroes (slot order = hero roster order; includes self).
  for (let k = 0; k < 5; k++) {
    const a = g.heroes[team * 5 + k];
    const ah = a.hero!;
    obs[i++] = a.alive ? 1 : 0;
    obs[i++] = relx(a);
    obs[i++] = rely(a);
    obs[i++] = a.hp / a.maxHp;
    obs[i++] = a.maxMana > 0 ? a.mana / a.maxMana : 0;
    obs[i++] = ah.level / MAX_LEVEL;
    obs[i++] = a === u ? 1 : 0;
    obs[i++] = a.stunned || a.rooted ? 1 : 0;
    obs[i++] = a.alive ? 0 : clamp(ah.respawn / 60, 0, 1);
    if (a.alive && u.alive) { targets[T_ALLY + k] = a.id; mask[moff + N_KINDS + T_ALLY + k] = 1; }
  }

  const near = u.alive ? g.unitsIn(u.x, u.y, 1400) : [];

  // Enemy heroes. Live details only for those our team can see; position and health fall back to
  // what the team last saw (`mem`), with the age of that sighting.
  for (let k = 0; k < 5; k++, i += F_ENEMY) {
    const e = g.heroes[(1 - team) * 5 + k];
    const eh = e.hero!;
    const vis = e.alive && g.canSee(team, e);
    obs[i] = vis ? 1 : 0;
    obs[i + 1] = e.alive ? 1 : 0;
    obs[i + 6] = eh.level / MAX_LEVEL;
    obs[i + 17] = vis ? 0 : 1;
    obs[i + 18] = e.alive ? 0 : clamp(eh.respawn / 60, 0, 1);
    if (vis || mem) {
      const px = vis ? e.x : mem!.x[eh.slot], py = vis ? e.y : mem!.y[eh.slot];
      obs[i + 2] = relpx(px, py);
      obs[i + 3] = relpy(px, py);
      obs[i + 4] = clamp(hypot(px - u.x, py - u.y) / 2000, 0, 2);
      obs[i + 5] = vis ? e.hp / e.maxHp : mem!.hpFrac[eh.slot];
      obs[i + 10] = (vis ? e.hp : mem!.hp[eh.slot]) / 4000;
      if (!vis) obs[i + 17] = clamp((g.tick - mem!.tick[eh.slot]) / (30 * TICK_RATE), 0, 1);
    }
    if (vis) {
      obs[i + 7] = e.stunned || e.rooted ? 1 : 0;
      obs[i + 8] = e.range / 1000;
      obs[i + 9] = inRange(e);
      obs[i + 11] = clamp((flip ? e.vy : e.vx) / 600, -3, 3);
      obs[i + 12] = clamp((flip ? e.vx : e.vy) / 600, -3, 3);
      obs[i + 13] = e.targetId === u.id || (eh.order.type === 'attack' && eh.order.targetId === u.id) ? 1 : 0;
      obs[i + 19] = u.alive && lineBlocked(near, u, e) ? 1 : 0;
      if (e.dash) {
        // A charge or leap in progress and where it will end.
        obs[i + 14] = 1;
        obs[i + 15] = relpx(e.dash.x, e.dash.y);
        obs[i + 16] = relpy(e.dash.x, e.dash.y);
      }
      if (u.alive) { targets[k] = e.id; mask[moff + N_KINDS + k] = 1; }
    }
  }

  // Nearby units.
  const creeps: Unit[] = [], friends: Unit[] = [];
  if (u.alive) {
    for (const o of near) {
      if (o.kind === 'minion') {
        if (o.team === team) friends.push(o);
        else if (g.canSee(team, o)) creeps.push(o);
      } else if (o.kind === 'monster' && g.canSee(team, o)) creeps.push(o);
    }
  }
  nearest(creeps, u, N_CREEPS);
  const hit = u.ad;
  for (let k = 0; k < N_CREEPS; k++) {
    const c = creeps[k];
    if (c) {
      obs[i] = 1;
      obs[i + 1] = relx(c);
      obs[i + 2] = rely(c);
      obs[i + 3] = c.hp / c.maxHp;
      obs[i + 4] = clamp(c.hp / 1500, 0, 4);
      obs[i + 5] = c.kind === 'monster' ? 1 : 0;
      obs[i + 6] = c.hp <= (hit * 100) / (100 + c.armor) ? 1 : 0;
      obs[i + 7] = inRange(c);
      obs[i + 8] = c.targetId === u.id ? 1 : 0;
      targets[T_CREEP + k] = c.id;
      mask[moff + N_KINDS + T_CREEP + k] = 1;
    }
    i += F_CREEP;
  }
  nearest(friends, u, N_AMINIONS);
  for (let k = 0; k < N_AMINIONS; k++) {
    const c = friends[k];
    if (c) {
      obs[i] = 1;
      obs[i + 1] = relx(c);
      obs[i + 2] = rely(c);
      obs[i + 3] = c.hp / c.maxHp;
    }
    i += F_AMINION;
  }
  const es: Unit[] = [], as: Unit[] = [];
  for (const s of g.structures) if (s.alive) (s.team === team ? as : es).push(s);
  nearest(es, u, N_ESTRUCT);
  for (let k = 0; k < N_ESTRUCT; k++) {
    const s = es[k];
    if (s) {
      obs[i] = 1;
      obs[i + 1] = relx(s);
      obs[i + 2] = rely(s);
      obs[i + 3] = s.hp / s.maxHp;
      obs[i + 4] = s.vulnerable ? 1 : 0;
      obs[i + 5] = s.kind === 'tower' && d(s) <= TOWER_RANGE + u.radius ? 1 : 0;
      obs[i + 6] = s.tier / 3;
      obs[i + 7] = u.alive && s.targetId === u.id ? 1 : 0;
      if (u.alive && s.vulnerable) { targets[T_STRUCT + k] = s.id; mask[moff + N_KINDS + T_STRUCT + k] = 1; }
    }
    i += F_ESTRUCT;
  }
  nearest(as, u, N_ASTRUCT);
  for (let k = 0; k < N_ASTRUCT; k++) {
    const s = as[k];
    if (s) {
      obs[i] = 1;
      obs[i + 1] = relx(s);
      obs[i + 2] = rely(s);
      obs[i + 3] = s.hp / s.maxHp;
    }
    i += F_ASTRUCT;
  }

  // Lane fronts: how far each team's foremost minion has pushed (0 = own base, 1 = enemy base).
  const fronts = laneFronts(g, team);
  for (let l = 0; l < 6; l++) obs[i++] = fronts[l];

  // Enemy skillshots in flight that could still reach the hero, soonest first. Homing attack
  // projectiles are left out: they cannot be dodged.
  if (u.alive && g.projectiles.length) {
    let shots: Projectile[] | null = null;
    const along = (p: Projectile) => (u.x - p.x) * p.dx + (u.y - p.y) * p.dy;
    const perp = (p: Projectile) => (u.x - p.x) * -p.dy + (u.y - p.y) * p.dx; // signed distance to the left of travel
    for (const p of g.projectiles) {
      if (p.team === team || p.targetId >= 0) continue;
      const a = along(p);
      if (a < -100 || a > Math.min(p.dist + 100, 2000) || Math.abs(perp(p)) > 500) continue;
      (shots ??= []).push(p);
    }
    if (shots) {
      shots.sort((a, b) => along(a) / a.speed - along(b) / b.speed || a.id - b.id);
      for (let k = 0, j = i; k < N_PROJ && k < shots.length; k++, j += F_PROJ) {
        const p = shots[k];
        const a = along(p), s = perp(p), reach = p.radius + u.radius;
        obs[j] = 1;
        obs[j + 1] = relpx(p.x, p.y);
        obs[j + 2] = relpy(p.x, p.y);
        obs[j + 3] = flip ? p.dy : p.dx;
        obs[j + 4] = flip ? p.dx : p.dy;
        obs[j + 5] = p.speed / 2500;
        obs[j + 6] = p.radius / 200;
        obs[j + 7] = clamp(p.dist / 1500, 0, 2);
        obs[j + 8] = (flip ? -s : s) / 500; // mirroring reverses left and right
        obs[j + 9] = clamp(Math.max(0, a) / p.speed, 0, 2); // seconds until it is level with the hero
        obs[j + 10] = Math.abs(s) < reach && a > -reach && a < p.dist + reach ? 1 : 0; // hits if the hero stands still
        obs[j + 11] = !p.pierce && !p.heroOnly ? 1 : 0; // a creep in the way would stop it
      }
    }
  }
  i += N_PROJ * F_PROJ;

  // Enemy ground effects about to go off nearby, soonest first.
  if (u.alive && g.zones.length) {
    let zones: Zone[] | null = null;
    const zd = (z: Zone) => hypot(z.x - u.x, z.y - u.y);
    for (const z of g.zones) if (z.team !== team && zd(z) <= z.radius + 700) (zones ??= []).push(z);
    if (zones) {
      zones.sort((a, b) => a.delay - b.delay || a.id - b.id);
      for (let k = 0, j = i; k < N_ZONES && k < zones.length; k++, j += F_ZONE) {
        const z = zones[k];
        const dist = zd(z);
        obs[j] = 1;
        obs[j + 1] = relpx(z.x, z.y);
        obs[j + 2] = relpy(z.x, z.y);
        obs[j + 3] = z.radius / 500;
        obs[j + 4] = clamp(z.delay, 0, 2); // seconds until it goes off
        obs[j + 5] = dist <= z.radius + u.radius * 0.6 ? 1 : 0; // hits if the hero stands still
        obs[j + 6] = clamp((dist - z.radius) / 500, -1, 2); // distance to the edge, negative inside
      }
    }
  }
  i += N_ZONES * F_ZONE;

  // Action-kind mask.
  mask[moff + N_KINDS + T_NONE] = 1;
  mask[moff + ACT_NOOP] = 1;
  if (u.alive) {
    const canAct = !u.stunned && !u.dash;
    mask[moff + ACT_MOVE] = canAct && !u.rooted ? 1 : 0;
    let anyEnemy = 0;
    for (let k = 0; k < T_ALLY; k++) anyEnemy |= mask[moff + N_KINDS + k];
    mask[moff + ACT_ATTACK] = canAct && anyEnemy ? 1 : 0;
    for (let s = 0; s < 4; s++) {
      let ok = g.castError(u, s) === '';
      const tg = def.abilities[s].targeting;
      if (ok && tg === 'enemy') { ok = false; for (let k = 0; k < T_STRUCT; k++) if (mask[moff + N_KINDS + k]) ok = true; }
      if (ok && tg === 'enemyHero') { ok = false; for (let k = 0; k < 5; k++) if (mask[moff + N_KINDS + k]) ok = true; }
      mask[moff + ACT_Q + s] = ok ? 1 : 0;
    }
    mask[moff + ACT_RECALL] = canAct && !g.inFountain(u) ? 1 : 0;
  }
}

/**
 * Write the full, fog-free state of the game as seen from `team` into `out` at `off` (STATE_SIZE
 * floats, mirrored for Red like the observations). This is privileged information: use it for a
 * critic during training, never as policy input.
 */
export function observeState(g: Game, team: number, out: Float32Array, off: number) {
  const flip = team === 1;
  let i = off;
  out[i++] = g.time / 1800;
  out[i++] = clamp((g.teamKills[team] - g.teamKills[1 - team]) / 20);
  out[i++] = g.towersLost[team] / 8;
  out[i++] = g.towersLost[1 - team] / 8;
  out[i++] = g.wyrmStacks[team] / 4;
  out[i++] = g.wyrmStacks[1 - team] / 4;
  out[i++] = g.campAlive[CAMPS.length - 2] > 0 ? 1 : 0;
  out[i++] = g.campAlive[CAMPS.length - 1] > 0 ? 1 : 0;
  for (const t of [team, 1 - team]) for (let c = 0; c < N_CAMP_TEAM; c++) out[i++] = g.campAlive[t * N_CAMP_TEAM + c] > 0 ? 1 : 0;

  for (const t of [team, 1 - team]) {
    for (let k = 0; k < 5; k++) {
      const u = g.heroes[t * 5 + k];
      const h = u.hero!;
      const def = HEROES[h.def];
      out[i++] = u.alive ? 1 : 0;
      out[i++] = (flip ? u.y : u.x) / MAP_SIZE;
      out[i++] = (flip ? u.x : u.y) / MAP_SIZE;
      out[i++] = clamp((flip ? u.vy : u.vx) / 600, -3, 3);
      out[i++] = clamp((flip ? u.vx : u.vy) / 600, -3, 3);
      out[i++] = u.alive ? u.hp / u.maxHp : 0;
      out[i++] = u.maxHp / 4000;
      out[i++] = u.maxMana > 0 ? u.mana / u.maxMana : 0;
      out[i++] = h.level / MAX_LEVEL;
      out[i++] = clamp(h.gold / 3000, 0, 2);
      out[i++] = h.itemGold / 15000;
      out[i++] = u.alive ? 0 : clamp(h.respawn / 60, 0, 1);
      out[i++] = u.alive && g.canSee(1 - t, u) ? 1 : 0;
      out[i++] = u.stunned || u.rooted ? 1 : 0;
      out[i++] = u.stealthed ? 1 : 0;
      for (let s = 0; s < 4; s++) {
        const rank = h.ranks[s];
        out[i++] = rank > 0 ? clamp(h.cds[s] / def.abilities[s].cd[rank - 1], 0, 1) : 1;
      }
    }
  }

  // Structures come in the same order for both teams: six lane towers, two guardians, the core.
  for (const t of [team, 1 - team]) {
    for (let k = 0; k < N_STRUCT_TEAM; k++) {
      const s = g.structures[t * N_STRUCT_TEAM + k];
      out[i++] = s.alive ? s.hp / s.maxHp : 0;
    }
  }

  // True lane fronts and minion counts per lane (own, enemy).
  const front = [0, 0, 0, 0, 0, 0], count = [0, 0, 0, 0, 0, 0];
  for (const m of g.units) {
    if (m.kind !== 'minion' || !m.alive || m.lane < 0) continue;
    let p = laneProgress(m.lane, m.x, m.y) / LANE_LENGTH[m.lane];
    if (m.team === 1) p = 1 - p;
    const j = m.lane * 2 + (m.team === team ? 0 : 1);
    if (p > front[j]) front[j] = p;
    count[j]++;
  }
  for (let j = 0; j < 6; j++) out[i++] = front[j];
  for (let j = 0; j < 6; j++) out[i++] = clamp(count[j] / 12, 0, 2);
}

/** Translate a MultiDiscrete action [kind, dir, dist, target] into a sim command (null = no-op). */
export function decodeAction(g: Game, u: Unit, kind: number, dir: number, dist: number, target: number, targets: Int32Array): Command | null {
  if (!u.alive || kind <= ACT_NOOP || kind >= N_KINDS) return null;
  const ang = ((dir % N_DIRS) / N_DIRS) * Math.PI * 2;
  let dx = cos(ang), dy = sin(ang);
  if (u.team === 1) [dx, dy] = [dy, dx];
  const tid = target >= 0 && target < N_TARGETS ? targets[target] : -1;
  const frac = (Math.min(N_DISTS - 1, Math.max(0, dist)) + 1) / N_DISTS;
  switch (kind) {
    case ACT_MOVE: {
      const r = 600 * frac;
      return { t: 'move', x: u.x + dx * r, y: u.y + dy * r };
    }
    case ACT_ATTACK:
      return tid >= 0 && target < T_ALLY ? { t: 'attack', id: tid } : null;
    case ACT_RECALL:
      return u.hero!.order.type === 'recall' ? null : { t: 'recall' };
    default: {
      const slot = kind - ACT_Q;
      const ab = HEROES[u.hero!.def].abilities[slot];
      switch (ab.targeting) {
        case 'self': return { t: 'cast', slot, x: u.x, y: u.y };
        case 'dir':
        case 'point': {
          // Aimed at a unit when one is named, otherwise by direction and distance.
          const t = tid >= 0 && target < T_STRUCT ? g.byId.get(tid) : undefined;
          if (t && t.alive) {
            const [px, py] = aimPoint(u, t, slot);
            return { t: 'cast', slot, x: px, y: py };
          }
          const r = Math.min(ab.range, 1200) * frac;
          return { t: 'cast', slot, x: u.x + dx * r, y: u.y + dy * r };
        }
        case 'ally': return { t: 'cast', slot, x: u.x, y: u.y, id: target >= T_ALLY && tid >= 0 ? tid : u.id };
        default: return tid >= 0 && target < T_STRUCT ? { t: 'cast', slot, x: u.x, y: u.y, id: tid } : null;
      }
    }
  }
}

const clampInt = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

function dirBin(u: Unit, dx: number, dy: number): number {
  if (u.team === 1) [dx, dy] = [dy, dx];
  const b = Math.round((atan2(dy, dx) / (2 * Math.PI)) * N_DIRS);
  return ((b % N_DIRS) + N_DIRS) % N_DIRS;
}

/** The point `look` units along the hero's way to (x, y), following the path around walls. */
function wayPoint(u: Unit, x: number, y: number, look: number): [number, number] {
  const o = u.hero!.order;
  // Reuse the path the hero is already walking when it leads to the same place.
  const same = (o.type === 'move' || o.type === 'amove') && u.path !== null && Math.abs(o.x - x) + Math.abs(o.y - y) < 50;
  const path = same ? u.path! : findPath(u.x, u.y, x, y);
  let px = u.x, py = u.y;
  for (let i = same ? u.pathIdx : 0; i < path.length; i += 2) {
    const d = hypot(path[i] - px, path[i + 1] - py);
    if (d >= look) return [px + ((path[i] - px) * look) / d, py + ((path[i + 1] - py) * look) / d];
    look -= d;
    px = path[i];
    py = path[i + 1];
  }
  return [px, py];
}

/**
 * The inverse of decodeAction: write the action [kind, dir, dist, target] that comes closest to
 * command `c` (null = let the current order run) into `out` at `off`. Components the action does
 * not use are NO_LABEL. If the command cannot be expressed, or the action mask would not allow
 * it right now, the kind itself is NO_LABEL.
 */
export function encodeAction(g: Game, u: Unit, c: Command | null, targets: Int32Array, mask: Uint8Array, moff: number, out: Uint8Array, off: number) {
  out[off] = ACT_NOOP;
  out[off + 1] = out[off + 2] = out[off + 3] = NO_LABEL;
  if (!c || !u.alive) return;
  const slotOf = (id: number, from: number, to: number) => {
    for (let s = from; s < to; s++) if (targets[s] === id) return s;
    return -1;
  };
  const move = (x: number, y: number) => {
    // A far destination becomes "walk 450 units along the path"; a near one keeps its distance.
    const [px, py] = wayPoint(u, x, y, 450);
    const d = hypot(px - u.x, py - u.y);
    if (d < 20) return;
    out[off] = ACT_MOVE;
    out[off + 1] = dirBin(u, px - u.x, py - u.y);
    out[off + 2] = clampInt(Math.round(d / (600 / N_DISTS)) - 1, 0, N_DISTS - 1);
  };
  switch (c.t) {
    case 'move':
    case 'amove':
      move(c.x, c.y);
      break;
    case 'attack': {
      const s = slotOf(c.id, 0, T_ALLY);
      if (s >= 0) {
        out[off] = ACT_ATTACK;
        out[off + 3] = s;
      } else {
        // Not among the hero's target slots (too far, or not one of the nearest): walk towards it.
        const t = g.byId.get(c.id);
        if (t && t.alive) move(t.x, t.y);
      }
      break;
    }
    case 'cast': {
      const ab = HEROES[u.hero!.def].abilities[c.slot];
      out[off] = ACT_Q + c.slot;
      let dx = c.x - u.x, dy = c.y - u.y;
      const d = hypot(dx, dy);
      if (d < 1) { dx = cos(u.facing); dy = sin(u.facing); }
      switch (ab.targeting) {
        case 'self': break;
        case 'dir':
        case 'point': {
          // If the aim matches where one of the hero's enemy targets will be, name that target instead of a direction.
          let best = -1, bestErr = ab.targeting === 'dir' ? 60 : 130;
          for (let s = 0; s < T_STRUCT; s++) {
            const t = targets[s] >= 0 ? g.byId.get(targets[s]) : undefined;
            if (!t || !t.alive) continue;
            const [px, py] = aimPoint(u, t, c.slot);
            let err: number;
            if (ab.targeting === 'point') err = hypot(px - c.x, py - c.y);
            else {
              // Sideways miss at the target's distance, for a shot along the commanded direction.
              const tx = px - u.x, ty = py - u.y;
              err = (tx * dx + ty * dy > 0 ? Math.abs(tx * -dy + ty * dx) : Infinity) / (d < 1 ? 1 : d);
            }
            if (err < bestErr) { bestErr = err; best = s; }
          }
          if (best >= 0) out[off + 3] = best;
          else {
            out[off + 3] = T_NONE;
            out[off + 1] = dirBin(u, dx, dy);
            if (ab.targeting === 'point') out[off + 2] = clampInt(Math.round(d / (Math.min(ab.range, 1200) / N_DISTS)) - 1, 0, N_DISTS - 1);
          }
          break;
        }
        case 'ally': {
          const s = slotOf(c.id ?? u.id, T_ALLY, N_TARGETS);
          out[off + 3] = s >= 0 ? s : T_ALLY + (u.hero!.slot % 5);
          break;
        }
        default: {
          const s = slotOf(c.id ?? -1, 0, ab.targeting === 'enemyHero' ? 5 : T_STRUCT);
          if (s >= 0) out[off + 3] = s;
          else out[off] = NO_LABEL;
        }
      }
      break;
    }
    case 'recall':
      out[off] = ACT_RECALL;
      break;
    default:
      break; // 'stop' has no action of its own; doing nothing is the closest
  }
  if (out[off] !== NO_LABEL && !mask[moff + out[off]]) out[off] = NO_LABEL;
}

/**
 * Runs a scripted bot "in shadow" for a hero that someone else controls: asks what the bot would
 * do right now without letting it do anything. The bot is handed a stand-in for the game that
 * reads everything from the real one but intercepts whatever the bot could change (commands,
 * casts, level-ups, purchases, the random stream). The bots' own per-game caches attach to the
 * stand-in too, so a game with a shadow plays out exactly as it would without one.
 */
export class Shadow {
  private view!: Game;
  /** The commands the bot issued during the last `ask`, in order. */
  readonly cmds: Command[] = [];

  /** Build the stand-in for a game. Call once per Game. */
  attach(g: Game) {
    const view: Game = Object.create(g);
    const cmds = this.cmds;
    view.rng = new Rng(0x5eed);
    view.applyCommand = (u, c) => { if (c.t !== 'level' && c.t !== 'buy') cmds.push(c); };
    view.tryCast = (u, slot, x, y, id) => { cmds.push({ t: 'cast', slot, x, y, id }); return true; };
    view.levelUp = () => false;
    view.buy = () => false;
    view.autoLevel = () => {};
    view.autoBuy = () => {};
    this.view = view;
  }

  /** What bot `kind` would do for hero `u` now. The returned array is reused by the next call. */
  ask(u: Unit, kind: string): Command[] {
    const h = u.hero!;
    this.cmds.length = 0;
    const autoLevel = h.autoLevel, autoBuy = h.autoBuy;
    try {
      (BOT_KINDS[kind] ?? BOT_KINDS.expert)(this.view, u);
    } finally {
      h.autoLevel = autoLevel;
      h.autoBuy = autoBuy;
    }
    return this.cmds;
  }
}

// --- Rewards -------------------------------------------------------------------------
export interface RewardWeights {
  gold: number; xp: number; kill: number; death: number; assist: number; hp: number;
  towerDamage: number; towerKill: number; towerLost: number; win: number;
}
export const DEFAULT_REWARD: RewardWeights = {
  gold: 0.006, xp: 0.002, kill: 1, death: -1, assist: 0.5, hp: 0.5, towerDamage: 0.002, towerKill: 2, towerLost: -1, win: 10,
};

function totalXp(u: Unit): number {
  const h = u.hero!;
  let x = h.xp;
  for (let l = 1; l < h.level; l++) x += xpForLevel(l);
  return x;
}

const N_STATS = 9;
/** The running totals the shaped reward is the change of, in the order of the first nine reward weights. */
function writeStats(g: Game, u: Unit, out: Float64Array, off: number) {
  const h = u.hero!;
  out[off] = h.earnedGold;
  out[off + 1] = totalXp(u);
  out[off + 2] = h.kills;
  out[off + 3] = h.deaths;
  out[off + 4] = h.assists;
  out[off + 5] = u.alive ? u.hp / u.maxHp : 0;
  out[off + 6] = h.towerDmg;
  out[off + 7] = g.towersLost[1 - u.team];
  out[off + 8] = g.towersLost[u.team];
}

// --- Environment -----------------------------------------------------------------------
export interface EnvConfig {
  /** Hero slots controlled by the learner (0-4 blue, 5-9 red). Everything else is a scripted bot. */
  agents?: number[];
  /** Sim ticks per env step (30 ticks = 1 second). */
  frameSkip?: number;
  /** Episode time limit in game seconds. */
  maxTime?: number;
  seed?: number;
  reward?: Partial<RewardWeights>;
  /** Blend each hero's reward with its team's mean: 0 = selfish, 1 = fully shared. */
  teamSpirit?: number;
  /**
   * How much of the enemy team's mean shaped reward is subtracted from every hero's reward (default 1).
   * At 1 the shaped rewards of the two teams sum to zero, so only doing better than the enemy pays.
   * The win/loss reward is added afterwards and is not affected.
   */
  zeroSum?: number;
  recordReplay?: boolean;
  /**
   * Scripted AI ('expert', 'basic' or 'easy') that shadows every controlled hero: each step the env
   * reports the action it would take as a label (`labels`), and action kind ACT_EXPERT executes its
   * exact commands. For imitation learning; costs one bot decision per controlled hero per step.
   */
  expert?: string;
  /** Scripted AI for every slot the learner does not control: 'easy', 'basic' (default) or 'expert', or one kind per slot. */
  botKind?: string | string[];
}

/** Who played a recorded game and how it was decided. Display only: it never changes how a replay plays out. */
export interface ReplayMeta {
  /** What to call the two teams, e.g. the names of the competing bots: [blue, red]. */
  teams?: [string, string];
  /** The winning team as the match was scored (0 blue, 1 red, -1 draw). Can differ from `winner`: a game stopped at the time limit or by a forfeit is still awarded. */
  result?: number;
  /** How the match was decided: 'core', 'time', 'timeout', 'crash', ... */
  reason?: string;
  [key: string]: unknown;
}

export interface Replay {
  version: 1;
  seed: number;
  bots: boolean[];
  /** Scripted AI per slot (absent in old replays = 'basic'). */
  botKind?: string[];
  manual: boolean[];
  /** Per slot: automatic level-ups / shopping / idle attacks, where they differ from the default (see GameConfig). */
  autoLevel?: (boolean | null)[];
  autoBuy?: (boolean | null)[];
  autoAttack?: (boolean | null)[];
  maxTime: number;
  endTick: number;
  winner: number;
  commands: [number, number, Command][];
  meta?: ReplayMeta;
}

export function makeReplay(g: Game, meta?: ReplayMeta): Replay {
  const r: Replay = {
    version: 1, seed: g.seed, bots: g.heroes.map((h) => h.hero!.isBot), botKind: g.heroes.map((h) => h.hero!.botKind), manual: g.cfg.manual ?? new Array(10).fill(false),
    maxTime: g.cfg.maxTime ?? 0, endTick: g.tick, winner: g.winner, commands: g.cmdLog ?? [],
  };
  for (const k of ['autoLevel', 'autoBuy', 'autoAttack'] as const) {
    const v = g.cfg[k];
    if (v) r[k] = Array.from({ length: 10 }, (_, s) => v[s] ?? null);
  }
  if (meta) r.meta = meta;
  return r;
}

/** The game configuration that plays a replay back: feed it the replay's commands at their ticks and stop at `endTick`. */
export function replayConfig(r: Replay): GameConfig {
  return { seed: r.seed, bots: r.bots, botKind: r.botKind, manual: r.manual, autoLevel: r.autoLevel, autoBuy: r.autoBuy, autoAttack: r.autoAttack, maxTime: r.maxTime };
}

/**
 * Plays a replay to its end without drawing anything and returns the finished game. A replay that was cut short
 * (a forfeit) ends at its last recorded tick with the game still undecided.
 */
export function runReplay(r: Replay, onTick?: (g: Game) => void): Game {
  const g = new Game(replayConfig(r));
  let i = 0;
  while (!g.over && g.tick < r.endTick) {
    while (i < r.commands.length && r.commands[i][0] <= g.tick + 1) { g.command(r.commands[i][1], r.commands[i][2]); i++; }
    g.step();
    onTick?.(g);
  }
  return g;
}

export class MobaEnv {
  cfg: EnvConfig;
  agents: number[];
  /** The teams that have at least one controlled hero, and for each agent the index of its team in that list. */
  teams: number[];
  teamIndex: number[];
  frameSkip: number;
  game!: Game;
  obs: Float32Array;
  mask: Uint8Array;
  rewards: Float32Array;
  /** Fog-free state per entry of `teams` (STATE_SIZE floats each), for a critic. */
  state: Float32Array;
  /** Per agent, the action [kind, dir, dist, target] the shadow expert would take now (all NO_LABEL without EnvConfig.expert). */
  labels: Uint8Array;
  memory = new FogMemory();
  private shadow = new Shadow();
  /** Per agent, the commands behind `labels`, for ACT_EXPERT. */
  private want: Command[][];
  private targets: Int32Array[];
  private prev = new Float64Array(10 * N_STATS);
  private cur = new Float64Array(10 * N_STATS);
  private shaped = new Float64Array(10);
  private w: RewardWeights;
  private nextSeed: number;
  lastReplay: Replay | null = null;

  constructor(cfg: EnvConfig = {}) {
    this.cfg = cfg;
    this.agents = cfg.agents ?? [0, 1, 2, 3, 4];
    this.teams = [0, 1].filter((t) => this.agents.some((a) => (a < 5 ? 0 : 1) === t));
    this.teamIndex = this.agents.map((a) => this.teams.indexOf(a < 5 ? 0 : 1));
    this.frameSkip = cfg.frameSkip ?? 6;
    this.w = { ...DEFAULT_REWARD, ...(cfg.reward ?? {}) };
    const n = this.agents.length;
    this.obs = new Float32Array(n * OBS_SIZE);
    this.mask = new Uint8Array(n * MASK_SIZE);
    this.rewards = new Float32Array(n);
    this.state = new Float32Array(this.teams.length * STATE_SIZE);
    this.labels = new Uint8Array(n * 4).fill(NO_LABEL);
    this.want = this.agents.map(() => []);
    this.targets = this.agents.map(() => new Int32Array(N_TARGETS));
    this.nextSeed = cfg.seed ?? 1;
    this.reset();
  }

  reset(seed?: number) {
    if (seed !== undefined) this.nextSeed = seed;
    const bots = new Array(10).fill(true);
    for (const a of this.agents) bots[a] = false;
    const gc: GameConfig = { seed: this.nextSeed++, bots, botKind: this.cfg.botKind, maxTime: this.cfg.maxTime ?? 1500, recordCommands: this.cfg.recordReplay ?? false };
    this.game = new Game(gc);
    if (this.cfg.expert) this.shadow.attach(this.game);
    this.memory.reset(this.game);
    for (let s = 0; s < 10; s++) writeStats(this.game, this.game.heroes[s], this.prev, s * N_STATS);
    this.writeObs();
  }

  private writeObs() {
    const g = this.game;
    for (let k = 0; k < this.agents.length; k++) {
      const u = g.heroes[this.agents[k]];
      observe(g, u, this.obs, k * OBS_SIZE, this.mask, k * MASK_SIZE, this.targets[k], this.memory);
      if (this.cfg.expert) {
        const cmds = this.shadow.ask(u, this.cfg.expert);
        this.want[k] = cmds.slice();
        // One action per step: a cast if there was one, otherwise the bot's last word.
        let c: Command | null = cmds.find((x) => x.t === 'cast') ?? cmds[cmds.length - 1] ?? null;
        const content = !c; // the bot is content with the running order
        if (!c) {
          // Name that order, so the label does not depend on history.
          const o = u.hero!.order;
          if (o.type === 'attack') c = { t: 'attack', id: o.targetId };
          else if (o.type === 'recall') c = { t: 'recall' };
        }
        encodeAction(g, u, c, this.targets[k], this.mask, k * MASK_SIZE, this.labels, k * 4);
        if (content && this.labels[k * 4] === NO_LABEL) { // cannot be named: "do nothing" says the same
          this.labels[k * 4] = ACT_NOOP;
          this.labels[k * 4 + 1] = this.labels[k * 4 + 2] = this.labels[k * 4 + 3] = NO_LABEL;
        }
      }
    }
    for (let t = 0; t < this.teams.length; t++) observeState(g, this.teams[t], this.state, t * STATE_SIZE);
  }

  /** actions: flat [kind, dir, dist, target] per agent. Returns true when the episode ended. */
  step(actions: ArrayLike<number>): boolean {
    const g = this.game;
    const n = this.agents.length;
    for (let k = 0; k < n; k++) {
      const u = g.heroes[this.agents[k]];
      if (actions[k * 4] === ACT_EXPERT) {
        for (const c of this.want[k]) g.command(this.agents[k], c);
        continue;
      }
      const c = decodeAction(g, u, actions[k * 4] | 0, actions[k * 4 + 1] | 0, actions[k * 4 + 2] | 0, actions[k * 4 + 3] | 0, this.targets[k]);
      if (c) g.command(this.agents[k], c);
    }
    for (let f = 0; f < this.frameSkip && !g.over; f++) {
      g.step();
      this.memory.update(g);
    }

    // Shaped reward of all ten heroes (bots included, so the enemy's progress can be subtracted).
    const w = this.w;
    const cur = this.cur, prev = this.prev, shaped = this.shaped;
    const teamMean = [0, 0];
    for (let s = 0; s < 10; s++) {
      const o = s * N_STATS;
      writeStats(g, g.heroes[s], cur, o);
      const r = w.gold * (cur[o] - prev[o]) + w.xp * (cur[o + 1] - prev[o + 1]) + w.kill * (cur[o + 2] - prev[o + 2]) + w.death * (cur[o + 3] - prev[o + 3]) +
        w.assist * (cur[o + 4] - prev[o + 4]) + w.hp * (cur[o + 5] - prev[o + 5]) + w.towerDamage * (cur[o + 6] - prev[o + 6]) +
        w.towerKill * (cur[o + 7] - prev[o + 7]) + w.towerLost * (cur[o + 8] - prev[o + 8]);
      shaped[s] = r;
      teamMean[s < 5 ? 0 : 1] += r / 5;
    }
    this.cur = prev;
    this.prev = cur;
    const ts = this.cfg.teamSpirit ?? 0, zs = this.cfg.zeroSum ?? 1;
    for (let k = 0; k < n; k++) {
      const slot = this.agents[k], team = slot < 5 ? 0 : 1;
      let r = (1 - ts) * shaped[slot] + ts * teamMean[team] - zs * teamMean[1 - team];
      if (g.over && g.winner >= 0) r += g.winner === team ? w.win : -w.win;
      this.rewards[k] = r;
    }
    this.writeObs();
    if (g.over && g.cmdLog) this.lastReplay = makeReplay(g);
    return g.over;
  }

  info() {
    const g = this.game;
    return {
      time: g.time, tick: g.tick, winner: g.winner, over: g.over, kills: g.teamKills, towers_lost: g.towersLost,
      heroes: this.agents.map((a) => {
        const u = g.heroes[a], h = u.hero!;
        return { slot: a, hero: HEROES[h.def].id, team: u.team, level: h.level, kills: h.kills, deaths: h.deaths, assists: h.assists, cs: h.cs, gold: Math.round(h.totalGold), damage: Math.round(h.dmgDealt) };
      }),
    };
  }
}

export const STEPS_PER_SECOND = (frameSkip: number) => TICK_RATE / frameSkip;

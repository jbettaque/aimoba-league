/**
 * Expert scripted AI: the strong hand-tuned opponent.
 *
 * It is built on the basic bot's proven decision skeleton and layers on the features
 * that measurably win games in head-to-head arena tests (see src/headless/arena.ts):
 * fast reactions, dodging of telegraphed abilities and skillshots, and more.
 * Every feature sits behind a tunable so it can be A/B tested per team.
 * Only uses information its team can legitimately see; fully deterministic.
 */
import type { Game } from './game.ts';
import { HERO_VS_MINION, MINION_VS_MINION, TOWER_RANGE, nextPurchase, planShopping, type Command, type Unit } from './defs.ts';
import { HEROES, HERO_IGNARA, HERO_KORVATH, HERO_LIORA, HERO_SABLE, HERO_VESPER } from './heroes.ts';
import { CAMPS, FOUNTAIN_POS, LANE_LENGTH, LANE_MID, lanePoint, laneProgress, rayClamp } from './map.ts';
import { hypot, sq } from './dmath.ts';

// --- Tunables --------------------------------------------------------------------------
export const TUNE_DEFAULT = {
  thinkTicks: 3,
  dodge: 1,
  aim: 1,
  kite: 0,
  lasthit: 0,
  lhWait: 0.5,
  // Decision thresholds (inherited from the basic bot, now tunable).
  engage: 0.75,
  engageLow: 0.6,
  engageHurt: 1.25,
  lowHp: 0.3,
  lowHpTank: 0.24,
  fleeRatio: 1.4,
  backoffRatio: 1.0,
  retreatExit: 0.6,
  shopExtra: 350,
  /** Minimum component spend worth a recall when no build item can be finished. */
  shopMin: 1200,
  /** When the team gathers in the middle lane. Under the pack-punishing economy 20:00 beats 15:50 (57% of 1,280 games) and 11:40 loses two of three. */
  groupTime: 1200,
  /** Send the nearest free heroes home when a wave is at an inner tower, a guardian or the core with nobody there (structures of this tier and up). */
  baseDef: 1,
  engageRange: 850,
  engageExtra: 380,
  farmMana: 0.5,
  pokeMana: 0.6,
  wyrmLevel: 5,
  healAt: 0.68,
  baseHp: 0.92,
  baseMana: 0.8,
  towerAlly: 5,
  towerFoe: 7,
  // Experiments that did not win their A/B tests. Kept switchable (off) so they can be re-tested
  // against different opponents; see README "Scripted bots".
  spells: 0,
  react: 0,
  gank: 0,
  snipe: 0,
  fountain: 1,
  tempo: 0,
  dodgeDash: 0,
  dodgeLead: 0.3,
  build: 1,
  skills: 0,
  /**
   * Answer to enemies moving as a pack (three or more together). 1: nobody faces them alone. 2: also, the laners of the
   * tower they are at wait behind it. 3: also, once towers are no longer fortified the whole team gathers behind the
   * threatened one. Off: against a trained model that plays as a pack none of the levels won more games than carrying
   * on farming did (and level 3 lost every game), and against the unchanged bot levels 1-2 were even.
   */
  pack: 0,
  /** With `pack`: how many fewer heroes than the pack we still stand our ground with. */
  packOdds: 1,
  /** With `pack`: when the team gathers to defend, the laner farthest from the pack keeps pushing its own lane instead. */
  packSplit: 0,
};
export type Tune = typeof TUNE_DEFAULT;
/** Per-team tunables (index = team) so the arena can A/B test parameter sets. */
export const TUNES: Tune[] = [{ ...TUNE_DEFAULT }, { ...TUNE_DEFAULT }];
let TUNE: Tune = TUNES[0];
/** Optional development hook: receives decision traces. */
export const DEBUG: { log: null | ((u: Unit, msg: string) => void) } = { log: null };

// ============================================================================
// Baseline behaviours (shared skeleton with the basic bot)
// ============================================================================

function b_act(g: Game, u: Unit, c: Command) {
  g.applyCommand(u, c);
}

function b_cast(g: Game, u: Unit, slot: number, x: number, y: number, id = -1): boolean {
  if (g.castError(u, slot) !== '') return false;
  return g.tryCast(u, slot, x, y, id);
}

/** Cast a skillshot with simple target leading. */
function b_castLead(g: Game, u: Unit, slot: number, t: Unit, speed: number): boolean {
  if (TUNE.aim) return castAim(g, u, slot, t, speed);
  const d = dist(u, t);
  const lead = Math.min(0.6, d / speed) * g.rng.range(0.5, 1);
  const err = d * 0.07;
  return b_cast(g, u, slot, t.x + g.botVx(t) * lead + g.rng.range(-err, err), t.y + g.botVy(t) * lead + g.rng.range(-err, err), t.id);
}

/** Skillshot geometry by hero*4+slot: [projectile radius, can be body-blocked by creeps]. */
const SHOT: Record<number, [number, boolean]> = { 0: [100, true], 8: [68, true], 10: [78, true], 12: [65, false], 15: [110, false], 17: [78, true] };

/** Precise skillshot: lead the target exactly and do not fire into a creep wall. */
function castAim(g: Game, u: Unit, slot: number, t: Unit, speed: number): boolean {
  if (g.castError(u, slot) !== '') return false;
  const [px, py] = intercept(g, u, t, speed);
  const info = SHOT[u.hero!.def * 4 + slot];
  if (info && info[1] && shotBlocked(g, u, px, py, info[0], t.id)) return false;
  return g.tryCast(u, slot, px, py, t.id);
}

function b_power(x: Unit): number {
  return (0.3 + 0.7 * (x.hp / x.maxHp)) * (x.hero!.level + 3) * (1 + x.hero!.itemGold * 0.00005);
}

// --- Lane front cache ------------------------------------------------------------
interface BFronts { tick: number; s: number[] }
const b_frontCache = new WeakMap<Game, BFronts>();

/** Lane progress (blue-oriented) of each team's foremost minion per lane; index = team*3+lane. */
function b_laneFronts(g: Game): number[] {
  let f: BFronts | undefined = b_frontCache.get(g);
  if (f && g.tick - f.tick < 6) return f.s;
  const s = [-1, -1, -1, 1e9, 1e9, 1e9];
  for (const m of g.units) {
    if (m.kind !== 'minion' || !m.alive) continue;
    const p = laneProgress(m.lane, m.x, m.y);
    const i = m.team * 3 + m.lane;
    if (m.team === 0) { if (p > s[i]) s[i] = p; } else if (p < s[i]) s[i] = p;
  }
  f = { tick: g.tick, s };
  b_frontCache.set(g, f);
  return s;
}

/** Where on the lane our team currently holds the line. */
function b_holdProgress(g: Game, team: number, lane: number): number {
  const fronts = b_laneFronts(g);
  const dir = team === 0 ? 1 : -1;
  // Outermost living allied tower in this lane (fallback: base).
  let towerS = team === 0 ? 300 : LANE_LENGTH[lane] - 300;
  for (const s of g.structures) {
    if (s.team !== team || !s.alive || s.kind !== 'tower' || s.lane !== lane) continue;
    const p = laneProgress(lane, s.x, s.y);
    if ((p - towerS) * dir > 0) towerS = p;
  }
  const mine = fronts[team * 3 + lane];
  const hasMinions = team === 0 ? mine >= 0 : mine < 1e8;
  let s = hasMinions && (mine - towerS) * dir > -200 ? mine : towerS;
  // Never plan to stand inside an un-escorted enemy tower.
  for (const st of g.structures) {
    if (st.team === team || !st.alive || st.lane !== lane || st.kind !== 'tower') continue;
    const p = laneProgress(lane, st.x, st.y);
    const edge = p - dir * (TOWER_RANGE + 150);
    if ((s - edge) * dir > 0 && !(hasMinions && (mine - edge) * dir > 100)) s = edge;
  }
  return s;
}

function b_enemyTowerThreat(g: Game, u: Unit, x: number, y: number): Unit | null {
  for (const s of g.structures) {
    if (s.team === u.team || !s.alive || s.kind !== 'tower') continue;
    if (sq(s.x - x) + sq(s.y - y) > sq(TOWER_RANGE + 120)) continue;
    let escort = 0;
    for (const m of g.unitsIn(s.x, s.y, TOWER_RANGE)) if (m.kind === 'minion' && m.team === u.team) escort++;
    if (escort < 1 || s.targetId === u.id || g.tick - u.hero!.aggroTick < 75) return s;
  }
  return null;
}

function b_retreat(g: Game, u: Unit) {
  const f = FOUNTAIN_POS[u.team];
  b_act(g, u, { t: 'move', x: f[0], y: f[1] });
}

// --- Ability usage -----------------------------------------------------------------
function b_fightSpells(g: Game, u: Unit, t: Unit, enemies: Unit[], allies: Unit[]): boolean {
  const h = u.hero!;
  const d = dist(u, t);
  const tPct = t.hp / t.maxHp;
  switch (h.def) {
    case HERO_KORVATH: {
      const clump = enemies.filter((e) => dist(e, t) < 380).length;
      if (d < 750 && (clump >= 2 || tPct < 0.55 || allies.length >= 2) && b_cast(g, u, 3, t.x, t.y)) return true;
      if (d < 500 && b_castLead(g, u, 0, t, 1300)) return true;
      if (d < 300 && b_cast(g, u, 2, u.x, u.y)) return true;
      if (d < 450 && (u.hp < u.maxHp * 0.8 || g.tick - u.lastDamageTick < 30) && b_cast(g, u, 1, u.x, u.y)) return true;
      return false;
    }
    case HERO_SABLE:
      if (d < 620 && tPct < 0.5 && b_cast(g, u, 3, t.x, t.y, t.id)) return true;
      if (d > 350 && d < 1000 && !u.stealthed && b_cast(g, u, 2, u.x, u.y)) return true;
      if (d < 520 && b_cast(g, u, 0, t.x, t.y, t.id)) return true;
      if (d < 260 && b_cast(g, u, 1, u.x, u.y)) return true;
      return false;
    case HERO_IGNARA: {
      const held = t.stunned || t.rooted;
      const clump = enemies.filter((e) => dist(e, t) < 380).length;
      if (d < 930 && (clump >= 2 || (tPct < 0.6 && tPct > 0.12) || held) && b_cast(g, u, 3, t.x + g.botVx(t) * 0.5, t.y + g.botVy(t) * 0.5)) return true;
      if (d < 800 && b_castLead(g, u, 2, t, 1150)) return true;
      if (d < 900 && b_castLead(g, u, 0, t, 1500)) return true;
      if (d < 850 && b_cast(g, u, 1, t.x + (held ? 0 : g.botVx(t) * 0.5), t.y + (held ? 0 : g.botVy(t) * 0.5))) return true;
      return false;
    }
    case HERO_VESPER:
      if (d < 2200 && (tPct < 0.4 || (d < 900 && allies.length >= 2)) && b_castLead(g, u, 3, t, 2200)) return true;
      if (d < 950 && b_castLead(g, u, 0, t, 1700)) return true;
      if (d < u.range + 150 && b_cast(g, u, 1, u.x, u.y)) return true;
      if (d < 280 && t.range < 250 && b_cast(g, u, 2, u.x + (u.x - t.x), u.y + (u.y - t.y))) return true;
      return false;
    case HERO_LIORA:
      if (d < 880 && b_castLead(g, u, 1, t, 1300)) return true;
      return false;
  }
  return false;
}

function b_supportSpells(g: Game, u: Unit, allies: Unit[], enemies: Unit[]): boolean {
  let low: Unit | null = null;
  let lowPct = TUNE.healAt;
  let pressured: Unit | null = null;
  for (const a of allies) {
    if (dist(a, u) > 750) continue;
    const p = a.hp / a.maxHp;
    if (p < lowPct) { lowPct = p; low = a; }
    if (g.tick - a.lastDamageTick < 20 && (!pressured || a.hp < pressured.hp) && (g.tick - a.lastHeroDamageTick < 45 || p < 0.6)) pressured = a;
  }
  const hurt = allies.filter((a) => dist(a, u) < 700 && a.hp < a.maxHp * 0.5).length;
  if (enemies.length > 0 && (hurt >= 2 || (low !== null && lowPct < 0.25)) && b_cast(g, u, 3, u.x, u.y)) return true;
  if (low && b_cast(g, u, 0, low.x, low.y, low.id)) return true;
  if (pressured && b_cast(g, u, 2, pressured.x, pressured.y, pressured.id)) return true;
  return false;
}

function b_escapeSpells(g: Game, u: Unit, threat: Unit | null) {
  const h = u.hero!;
  const f = FOUNTAIN_POS[u.team];
  switch (h.def) {
    case HERO_KORVATH: if (threat) b_cast(g, u, 1, u.x, u.y); break;
    case HERO_SABLE: if (threat) b_cast(g, u, 2, u.x, u.y); break;
    case HERO_IGNARA: if (threat && dist(u, threat) < 700) b_castLead(g, u, 2, threat, 1150); break;
    case HERO_VESPER:
      if (threat && dist(u, threat) < 600) {
        const p = u.path && u.pathIdx < u.path.length ? [u.path[u.pathIdx], u.path[u.pathIdx + 1]] : f;
        b_cast(g, u, 2, p[0], p[1]);
      }
      break;
    case HERO_LIORA:
      if (threat && dist(u, threat) < 800) { if (!b_cast(g, u, 2, u.x, u.y, u.id)) b_castLead(g, u, 1, threat, 1300); }
      if (u.hp < u.maxHp * 0.6) b_cast(g, u, 0, u.x, u.y, u.id);
      break;
  }
}

function b_farmSpells(g: Game, u: Unit, creeps: Unit[]): boolean {
  const h = u.hero!;
  if (creeps.length < 3 || u.mana < u.maxMana * TUNE.farmMana) return false;
  // Aim at the creep with the most neighbours.
  let best = creeps[0], bestN = 0;
  for (const c of creeps) {
    let n = 0;
    for (const o of creeps) if (dist(c, o) < 230) n++;
    if (n > bestN) { bestN = n; best = c; }
  }
  if (bestN < 3) return false;
  const d = dist(u, best);
  switch (h.def) {
    case HERO_KORVATH: return d < 260 && b_cast(g, u, 2, u.x, u.y);
    case HERO_IGNARA: return d < 850 && b_cast(g, u, 1, best.x, best.y);
    case HERO_VESPER: return d < 900 && h.level >= 3 && b_cast(g, u, 0, best.x, best.y);
    default: return false;
  }
}

// --- Objectives ---------------------------------------------------------------------
function b_attackDamage(u: Unit, t: Unit): number {
  const vsMinion = u.hero && t.kind === 'minion' ? HERO_VS_MINION : 1;
  return (u.ad * vsMinion * 100) / (100 + Math.max(0, t.armor - u.armorPen)) + (u.hero ? u.hero.it.onHit * 0.8 : 0);
}

function b_laneThink(g: Game, u: Unit, lane: number, near: Unit[], isSupport: boolean) {
  const h = u.hero!;
  const team = u.team;
  const dir = team === 0 ? 1 : -1;
  const ranged = u.range > 250;
  const reach = u.range + 420;

  // Structures first: hit towers / the core whenever our minions are tanking.
  for (const s of g.structures) {
    if (s.team === team || !s.alive || !s.vulnerable) continue;
    if (dist(s, u) > reach + s.radius) continue;
    if (s.kind === 'tower' && b_enemyTowerThreat(g, u, u.x, u.y) === s) continue;
    let escort = 0;
    for (const m of g.unitsIn(s.x, s.y, 900)) if (m.kind === 'minion' && m.team === team) escort++;
    if (escort >= 1) { b_act(g, u, { t: 'attack', id: s.id }); h.botState = 'siege'; return; }
  }

  const creeps: Unit[] = [];
  for (const m of near) {
    if (m.kind === 'minion' && m.team !== team && g.botSees(team, m) && dist(m, u) < reach + 150) creeps.push(m);
  }
  if (creeps.length > 0) {
    const hit = b_attackDamage(u, creeps[0]);
    let lastHit: Unit | null = null, lowest: Unit | null = null;
    for (const m of creeps) {
      if (b_enemyTowerThreat(g, u, m.x, m.y)) continue;
      if (m.hp <= hit * 1.05 && (!lastHit || dist(m, u) < dist(lastHit, u))) lastHit = m;
      if (!lowest || m.hp < lowest.hp) lowest = m;
    }
    if (!isSupport && b_farmSpells(g, u, creeps)) return;
    if (TUNE.lasthit && !isSupport) {
      const lh = pickLastHit(g, u, creeps, near);
      if (lh === 'wait') { if (h.order.type === 'attack' && u.windup <= 0) b_act(g, u, { t: 'stop' }); h.botState = 'wait'; return; }
      if (lh) { b_act(g, u, { t: 'attack', id: lh.id }); h.botState = 'farm'; return; }
      // Nothing to last-hit: push with the healthiest creep so we do not spoil upcoming last hits.
      let top: Unit | null = null;
      for (const m of creeps) if (!b_enemyTowerThreat(g, u, m.x, m.y) && (!top || m.hp > top.hp)) top = m;
      if (top) { b_act(g, u, { t: 'attack', id: top.id }); h.botState = 'farm'; return; }
    }
    const pick = lastHit ?? (isSupport ? null : lowest);
    if (pick) { b_act(g, u, { t: 'attack', id: pick.id }); h.botState = 'farm'; return; }
  }

  // Hold the line just behind our front minion.
  const s = b_holdProgress(g, team, lane) - dir * (ranged ? 330 : 120) - (isSupport ? dir * 120 : 0);
  const p = lanePoint(lane, Math.max(200, Math.min(LANE_LENGTH[lane] - 200, s)));
  const off = (h.slot % 5) * 37 - 70;
  h.botState = 'lane';
  b_act(g, u, { t: 'move', x: p[0] + off, y: p[1] + off * 0.6 });
}

function b_jungleThink(g: Game, u: Unit, near: Unit[]): boolean {
  const h = u.hero!;
  let ci = h.botCamp;
  if (ci < 0 || g.campAlive[ci] <= 0) {
    ci = -1;
    let bestD = Infinity;
    for (let i = 0; i < CAMPS.length; i++) {
      const c = CAMPS[i];
      if (g.campAlive[i] <= 0) continue;
      const mine = c.side === u.team;
      const wyrm = c.type === 'wyrm' && h.level >= TUNE.wyrmLevel && u.hp > u.maxHp * 0.7;
      if (!mine && !wyrm) continue;
      const d = hypot(c.x - u.x, c.y - u.y) + (wyrm ? 2500 : 0);
      if (d < bestD) { bestD = d; ci = i; }
    }
    h.botCamp = ci;
  }
  if (ci < 0) return false;
  const c = CAMPS[ci];
  h.botState = 'jungle';
  let target: Unit | null = null;
  for (const m of near) {
    if (m.kind === 'monster' && m.camp === ci && g.botSees(u.team, m) && (!target || m.hp < target.hp)) target = m;
  }
  if (!target) { b_act(g, u, { t: 'move', x: c.x, y: c.y }); return true; }
  const d = dist(u, target);
  if (d < 500) b_cast(g, u, 0, target.x, target.y, target.id);
  if (d < 250) b_cast(g, u, 1, u.x, u.y);
  b_act(g, u, { t: 'attack', id: target.id });
  return true;
}


// ============================================================================
// Expert modules
// ============================================================================
// --- Small helpers ------------------------------------------------------------------------
const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => hypot(a.x - b.x, a.y - b.y);
const d2p = (a: { x: number; y: number }, x: number, y: number) => hypot(a.x - x, a.y - y);

function attack(g: Game, u: Unit, t: Unit) {
  const o = u.hero!.order;
  if (o.type === 'attack' && o.targetId === t.id) return;
  g.applyCommand(u, { t: 'attack', id: t.id });
}

/** Move without cancelling an attack that is about to land. */
function moveTo(g: Game, u: Unit, x: number, y: number, force = false) {
  if (u.windup > 0 && !force) return;
  g.applyCommand(u, { t: 'move', x, y });
}

function cast(g: Game, u: Unit, slot: number, x: number, y: number, id = -1): boolean {
  if (g.castError(u, slot) !== '') return false;
  return g.tryCast(u, slot, x, y, id);
}
const ready = (g: Game, u: Unit, slot: number) => g.castError(u, slot) === '';

function buffTime(u: Unit, kind: string): number {
  let t = 0;
  for (const b of u.buffs) if (b.kind === kind && b.t > t) t = b.t;
  return t;
}
const heldTime = (u: Unit) => Math.max(buffTime(u, 'stun'), buffTime(u, 'root'));

/** Where a projectile of `speed` (after `delay`) meets the target if it keeps its course. */
function intercept(g: Game, u: Unit, t: Unit, speed: number, delay = 0): [number, number, number] {
  const held = heldTime(t);
  let vx = g.botVx(t), vy = g.botVy(t);
  if (t.stunned || t.rooted) { vx = 0; vy = 0; }
  let tm = delay + dist(u, t) / speed;
  let px = t.x, py = t.y;
  for (let i = 0; i < 3; i++) {
    const mv = Math.max(0, tm - held);
    px = t.x + vx * mv;
    py = t.y + vy * mv;
    tm = delay + hypot(px - u.x, py - u.y) / speed;
  }
  return [px, py, tm];
}

/** True when an enemy creep would body-block a non-piercing shot from u to (x,y). */
function shotBlocked(g: Game, u: Unit, x: number, y: number, radius: number, ignoreId: number): boolean {
  const dx = x - u.x, dy = y - u.y;
  const len = hypot(dx, dy);
  if (len < 1) return false;
  for (const m of g.unitsIn(u.x + dx / 2, u.y + dy / 2, len / 2 + 160)) {
    if (m.team === u.team || m.id === ignoreId || m.kind === 'tower' || m.kind === 'core' || m.kind === 'hero') continue;
    const t = ((m.x - u.x) * dx + (m.y - u.y) * dy) / (len * len);
    if (t < 0 || t > 1) continue;
    const px = u.x + dx * t - m.x, py = u.y + dy * t - m.y;
    const rr = radius + m.radius;
    if (px * px + py * py < rr * rr) return true;
  }
  return false;
}

// --- Strength model -----------------------------------------------------------------------
const R1 = (a: number[], r: number) => a[Math.max(1, Math.min(a.length, r)) - 1];
/** Rough burst damage `u` can put on `t` right now with abilities (after resistances). */
function burst(_g: Game, u: Unit, t: Unit): number {
  const h = u.hero!;
  const r = h.ranks;
  const pm = 100 / (100 + Math.max(0, t.armor - u.armorPen)), mm = 100 / (100 + t.mr * (1 - u.magicPen));
  let d = 0;
  const up = (s: number) => r[s] > 0 && h.cds[s] <= 0.3;
  switch (h.def) {
    case HERO_KORVATH:
      if (up(0)) d += (R1([60, 95, 130, 165], r[0]) + 0.5 * u.ad) * pm;
      if (up(2)) d += (R1([70, 110, 150, 190], r[2]) + 0.3 * u.ap + 0.03 * u.maxHp) * mm;
      if (up(3)) d += (R1([200, 325, 450], r[3]) + 0.5 * u.ap + 0.06 * u.maxHp) * mm;
      break;
    case HERO_SABLE:
      if (up(0)) d += (R1([65, 100, 135, 170], r[0]) + 0.7 * u.ad) * pm;
      if (up(3)) d += (R1([150, 250, 350], r[3]) + 0.8 * u.ad) * (2 - t.hp / t.maxHp) * pm;
      break;
    case HERO_IGNARA:
      if (up(0)) d += (R1([80, 125, 170, 215], r[0]) + 0.75 * u.ap) * mm;
      if (up(1)) d += (R1([90, 140, 190, 240], r[1]) + 0.7 * u.ap) * mm;
      if (up(2)) d += (R1([50, 75, 100, 125], r[2]) + 0.4 * u.ap) * mm;
      if (up(3)) d += (R1([300, 450, 600], r[3]) + 0.9 * u.ap) * mm;
      break;
    case HERO_VESPER:
      if (up(0)) d += (R1([50, 85, 120, 155], r[0]) + u.ad) * pm;
      if (up(3)) d += (R1([250, 375, 500], r[3]) + u.ad) * mm;
      break;
    default:
      if (up(1)) d += (R1([60, 100, 140, 180], r[1]) + 0.5 * u.ap) * mm;
  }
  return d * u.dmgAmp * (u.mana > 90 ? 1 : 0.4);
}
function hitDamage(u: Unit, t: Unit): number {
  const vsMinion = u.hero && t.kind === 'minion' ? HERO_VS_MINION : 1;
  return (u.ad * vsMinion * 100) / (100 + Math.max(0, t.armor - u.armorPen)) + (u.hero ? u.hero.it.onHit * 0.75 + (u.hero.def === HERO_KORVATH ? 0.012 * u.maxHp : 0) : 0);
}

/** Damage that allied minions / towers / shots already in flight will deal to `m` within `t` seconds. */
function incomingDamage(g: Game, m: Unit, attackers: Unit[], t: number): number {
  let dmg = 0;
  const k = 100 / (100 + m.armor);
  for (const f of attackers) {
    const r = f.range + m.radius;
    const d = dist(f, m);
    if (d > r + 30) continue; // still walking up
    const wind = Math.min(0.28, 0.22 / f.as);
    const flight = f.range >= 250 ? d / (f.kind === 'tower' ? 1700 : 1500) : 0;
    let first = (f.windup > 0 && f.windupTarget === m.id ? f.windup : Math.max(0, f.attackCd) + wind) + flight;
    const period = 1 / f.as;
    const each = f.ad * k * (f.kind === 'minion' && m.kind === 'minion' ? MINION_VS_MINION : 1);
    while (first <= t) { dmg += each; first += period; }
  }
  for (const p of g.projectiles) {
    if (p.targetId !== m.id || p.team === m.team) continue;
    const fromMinion = m.kind === 'minion' && g.unit(p.srcId)?.kind === 'minion';
    if (hypot(p.x - m.x, p.y - m.y) / p.speed <= t) dmg += p.dmg * k * (fromMinion ? MINION_VS_MINION : 1);
  }
  return dmg;
}

/** Predictive last-hit pick. Returns the creep to hit now, 'wait' to hold fire for an imminent one, or null. */
function pickLastHit(g: Game, u: Unit, creeps: Unit[], near: Unit[]): Unit | 'wait' | null {
  const ranged = u.range > 250;
  const windup = Math.min(0.28, 0.22 / u.as) + 0.07;
  const friends: Unit[] = [];
  for (const o of near) if (o.team === u.team && (o.kind === 'minion' || o.kind === 'tower')) friends.push(o);
  const atk: Unit[] = [];
  let best: Unit | null = null, bestScore = -Infinity, wait = false;
  for (const m of creeps) {
    const dm = dist(m, u);
    const hit = hitDamage(u, m);
    atk.length = 0;
    for (const f of friends) if (f.targetId === m.id) atk.push(f);
    const gap = Math.max(0, dm - u.range - m.radius);
    const tHit = gap / u.ms + windup + (ranged ? Math.min(dm, u.range + m.radius) / 1500 : 0) + Math.max(0, u.attackCd);
    const pred = m.hp - incomingDamage(g, m, atk, tHit);
    if (pred <= 0) continue;
    if (pred <= hit) {
      const score = m.goldValue - gap * 0.03;
      if (score > bestScore) { bestScore = score; best = m; }
    } else if (atk.length > 0 && m.hp - incomingDamage(g, m, atk, tHit + TUNE.lhWait) <= hit) wait = true;
  }
  return best ?? (wait ? 'wait' : null);
}

function x_fightSpells(g: Game, u: Unit, t: Unit, enemies: Unit[], allies: Unit[], ratio: number): boolean {
  const h = u.hero!;
  const d = dist(u, t);
  const tPct = t.hp / t.maxHp;
  const held = heldTime(t);
  const clumpAt = (x: number, y: number, r: number) => { let n = 0; for (const e of enemies) if (d2p(e, x, y) < r) n++; return n; };
  const alliesNear = allies.length - 1;
  switch (h.def) {
    case HERO_KORVATH: {
      if (ready(g, u, 3) && d < 800) {
        const [px, py] = intercept(g, u, t, 1500);
        const n = clumpAt(px, py, 380);
        if (d2p(u, px, py) < 760 && (n >= 2 || (ratio > 1 && (alliesNear >= 1 || tPct < 0.5)) || burst(g, u, t) > t.hp) && !b_enemyTowerThreat(g, u, px, py) && cast(g, u, 3, px, py)) return true;
      }
      if (ready(g, u, 0) && d < 540 && held < 0.5) {
        const [px, py] = intercept(g, u, t, 1300);
        if (d2p(u, px, py) < 560 && !shotBlocked(g, u, px, py, 100, t.id) && cast(g, u, 0, px, py)) return true;
      }
      if (d < 310 && cast(g, u, 2, u.x, u.y)) return true;
      if (d < 500 && (g.tick - u.lastHeroDamageTick < 30 || u.hp < u.maxHp * 0.7) && cast(g, u, 1, u.x, u.y)) return true;
      return false;
    }
    case HERO_SABLE: {
      if (ready(g, u, 3) && d < 640) {
        const exec = burst(g, u, t) + hitDamage(u, t) * 2 > t.hp;
        if ((exec || (tPct < 0.5 && ratio > 0.9)) && cast(g, u, 3, t.x, t.y, t.id)) return true;
      }
      if (d > 320 && d < 1100 && !u.stealthed && cast(g, u, 2, u.x, u.y)) return true;
      if (d < 540 && cast(g, u, 0, t.x, t.y, t.id)) return true;
      if (d < 260 && cast(g, u, 1, u.x, u.y)) return true;
      return false;
    }
    case HERO_IGNARA: {
      if (ready(g, u, 2) && held < 0.25 && d < 840) {
        const [px, py, tm] = intercept(g, u, t, 1150);
        if (d2p(u, px, py) < 840 && (tm < 0.62 || t.ms < 260) && !shotBlocked(g, u, px, py, 78, t.id) && cast(g, u, 2, px, py)) return true;
      }
      if (ready(g, u, 3) && d < 1000) {
        const lead = held >= 0.7 ? 0 : 0.55;
        const px = t.x + g.botVx(t) * lead, py = t.y + g.botVy(t) * lead;
        const n = clumpAt(px, py, 360);
        const kill = burst(g, u, t) > t.hp * 0.95;
        if (d2p(u, px, py) < 960 && (held >= 0.7 || n >= 3 || (n >= 2 && ratio > 0.8) || (kill && (t.ms < 300 || tPct < 0.5))) && tPct > 0.1 && cast(g, u, 3, px, py)) return true;
      }
      if (ready(g, u, 1) && d < 880) {
        const lead = held >= 0.5 ? 0 : 0.5;
        const px = t.x + g.botVx(t) * lead, py = t.y + g.botVy(t) * lead;
        if (d2p(u, px, py) < 850 && cast(g, u, 1, px, py)) return true;
      }
      if (ready(g, u, 0) && d < 940) {
        const [px, py, tm] = intercept(g, u, t, 1500);
        if (d2p(u, px, py) < 950 && (tm < 0.5 || held > 0.2) && !shotBlocked(g, u, px, py, 68, t.id) && cast(g, u, 0, px, py)) return true;
      }
      return false;
    }
    case HERO_VESPER: {
      if (ready(g, u, 3)) {
        const [px, py] = intercept(g, u, t, 2200);
        const rd = (R1([250, 375, 500], h.ranks[3]) + u.ad) * (100 / (100 + t.mr)) * u.dmgAmp;
        if (((rd > t.hp && d > u.range * 0.8) || (d < 1200 && alliesNear >= 1 && ratio > 0.9 && held < 0.3 && tPct > 0.3)) && cast(g, u, 3, px, py)) return true;
      }
      if (d < u.range + 160 && cast(g, u, 1, u.x, u.y)) return true;
      if (ready(g, u, 0) && d < 980) {
        const [px, py, tm] = intercept(g, u, t, 1700);
        if ((tm < 0.5 || held > 0.2) && cast(g, u, 0, px, py)) return true;
      }
      if (ready(g, u, 2)) {
        if (d < 300 && t.range < 250) { if (cast(g, u, 2, u.x + (u.x - t.x), u.y + (u.y - t.y))) return true; }
        else if (tPct < 0.25 && d > u.range + 40 && d < u.range + 400 && ratio > 1 && !b_enemyTowerThreat(g, u, t.x, t.y) && cast(g, u, 2, t.x, t.y)) return true;
      }
      return false;
    }
    default: {
      if (ready(g, u, 1) && d < 880) {
        const [px, py, tm] = intercept(g, u, t, 1300);
        if (d2p(u, px, py) < 900 && (tm < 0.6 || held > 0.2) && !shotBlocked(g, u, px, py, 78, t.id) && cast(g, u, 1, px, py)) return true;
      }
      return false;
    }
  }
}


// --- Team memory --------------------------------------------------------------------------
interface TeamMem {
  tick: number; seenX: number[]; seenY: number[]; seenTick: number[]; packLane: number; packUntil: number; myAlive: number; enemyAlive: number; defLane: number[]; defX: number[]; defY: number[];
  /** The enemy pack: how many, and where it was last seen. */
  packN: number; packX: number; packY: number;
  /** Whether the pack is at (or closing in on) one of our structures, where to wait behind it, and whether the whole team should come. */
  threat: boolean; rallyX: number; rallyY: number; rallyAll: boolean;
}
const teamMems = new WeakMap<Game, TeamMem[]>();
function teamMem(g: Game, team: number): TeamMem {
  let t = teamMems.get(g);
  if (!t) {
    const mk = (): TeamMem => ({ tick: -99, seenX: [0, 0, 0, 0, 0], seenY: [0, 0, 0, 0, 0], seenTick: [-9999, -9999, -9999, -9999, -9999], packLane: -1, packUntil: 0, myAlive: 5, enemyAlive: 5, defLane: [-1, -1, -1, -1, -1], defX: [0, 0, 0, 0, 0], defY: [0, 0, 0, 0, 0], packN: 0, packX: 0, packY: 0, threat: false, rallyX: 0, rallyY: 0, rallyAll: false });
    t = [mk(), mk()];
    teamMems.set(g, t);
  }
  const m = t[team];
  if (g.tick - m.tick >= 10) {
    m.tick = g.tick;
    const eb = (1 - team) * 5, mb = team * 5;
    m.myAlive = 0;
    m.enemyAlive = 0;
    for (let k = 0; k < 5; k++) {
      const e = g.heroes[eb + k];
      if (e.alive) { m.enemyAlive++; if (g.botSees(team, e)) { m.seenX[k] = e.x; m.seenY[k] = e.y; m.seenTick[k] = g.tick; } } else m.seenTick[k] = -9999;
      if (g.heroes[mb + k].alive) m.myAlive++;
    }
    // Enemy pack: three or more heroes seen close together recently.
    let bestN = 0, bx = 0, by = 0;
    for (let k = 0; k < 5; k++) {
      if (g.tick - m.seenTick[k] > 240) continue;
      let n = 0;
      for (let j = 0; j < 5; j++) if (g.tick - m.seenTick[j] <= 240 && hypot(m.seenX[k] - m.seenX[j], m.seenY[k] - m.seenY[j]) < 1700) n++;
      if (n > bestN) { bestN = n; bx = m.seenX[k]; by = m.seenY[k]; }
    }
    if (bestN >= 3) {
      let bd = Infinity, lane = LANE_MID;
      for (let l = 0; l < 3; l++) {
        const p = lanePoint(l, laneProgress(l, bx, by));
        const d = hypot(p[0] - bx, p[1] - by);
        if (d < bd) { bd = d; lane = l; }
      }
      m.packLane = lane;
      m.packUntil = g.time + 20;
      m.packN = bestN;
      m.packX = bx;
      m.packY = by;
      // Which of our structures is the pack at? The nearest one that can be attacked, if they are close to it.
      let st: Unit | null = null, sd = 2300;
      for (const s of g.structures) {
        if (s.team !== team || !s.alive || !s.vulnerable) continue;
        const d = hypot(s.x - bx, s.y - by);
        if (d < sd) { sd = d; st = s; }
      }
      m.threat = st !== null;
      if (st) {
        // Wait a little behind it, on the side of our base.
        const f = FOUNTAIN_POS[team];
        const dx = f[0] - st.x, dy = f[1] - st.y, dl = hypot(dx, dy) || 1;
        m.rallyX = st.x + (dx / dl) * 380;
        m.rallyY = st.y + (dy / dl) * 380;
        m.rallyAll = !st.fortified;
      }
    }
    // Base defence: a guardian or the core is being hit by a wave with nobody home.
    m.defLane.fill(-1);
    if (TUNE.baseDef) {
      const free = [0, 1, 2, 3, 4].filter((k) => g.heroes[mb + k].alive);
      for (const s of g.structures) {
        if (s.team !== team || !s.alive || !s.vulnerable || s.tier < TUNE.baseDef) continue;
        let em = 0, am = 0, lane = s.lane, bd = Infinity, eh = 0;
        for (const o of g.unitsIn(s.x, s.y, 1100)) {
          if (o.kind === 'hero') { if (o.team !== team) eh++; continue; }
          if (o.kind !== 'minion') continue;
          const w = o.subtype === 'super' ? 3 : o.subtype === 'siege' ? 2 : 1;
          if (o.team === team) am += w; else { em += w; const d = dist(o, s); if (d < bd) { bd = d; if (s.lane < 0) lane = o.lane; } }
        }
        if (em - am < 3) continue;
        let home = 0;
        for (const k of [0, 1, 2, 3, 4]) { const a = g.heroes[mb + k]; if (a.alive && dist(a, s) < 1200) home++; }
        const need = (em >= 9 ? 2 : 1) + (eh > 0 ? 1 : 0) - home;
        if (need <= 0 || free.length === 0) continue;
        free.sort((p, q) => dist(g.heroes[mb + p], s) - dist(g.heroes[mb + q], s) || p - q);
        for (const k of free.splice(0, need)) { m.defLane[k] = lane < 0 ? LANE_MID : lane; m.defX[k] = s.x; m.defY[k] = s.y; }
      }
    }
  }
  return m;
}

function inEnemyFountain(team: number, x: number, y: number): boolean {
  const f = FOUNTAIN_POS[1 - team];
  return sq(x - f[0]) + sq(y - f[1]) < 850 * 850;
}

// --- Builds and skill orders ---------------------------------------------------------------
/** Item build variants per hero (index 0 = the default build from the hero definition). */
const BUILDS: string[][][] = [
  [],
  // 1: damage first, boots third
  [['vest', 'plate', 'boots', 'mantle', 'cleaver', 'colossus', 'chalice'], ['blade', 'edge', 'boots', 'fang', 'cleaver', 'mantle', 'plate'], ['tome', 'orb', 'boots', 'staff', 'prism', 'scepter', 'mantle'], ['blade', 'stormbow', 'boots', 'edge', 'fang', 'mantle', 'plate'], ['tome', 'chalice', 'boots', 'orb', 'mantle', 'scepter', 'plate']],
  // 2: straight to the big item (no starter component)
  [['boots', 'plate', 'mantle', 'cleaver', 'colossus', 'chalice'], ['boots', 'edge', 'fang', 'cleaver', 'mantle', 'plate'], ['boots', 'orb', 'staff', 'prism', 'scepter', 'mantle'], ['boots', 'stormbow', 'edge', 'fang', 'mantle', 'plate'], ['boots', 'chalice', 'orb', 'mantle', 'scepter', 'plate']],
  // 3: bruiser tank, raw-damage carries
  [['blade', 'boots', 'cleaver', 'plate', 'mantle', 'colossus', 'chalice'], ['blade', 'boots', 'fang', 'edge', 'cleaver', 'mantle', 'plate'], ['tome', 'boots', 'staff', 'orb', 'prism', 'scepter', 'mantle'], ['blade', 'boots', 'edge', 'stormbow', 'fang', 'mantle', 'plate'], ['tome', 'boots', 'orb', 'chalice', 'mantle', 'scepter', 'plate']],
  // 4: survivability earlier on the squishies
  [['vest', 'boots', 'plate', 'mantle', 'colossus', 'cleaver', 'chalice'], ['blade', 'boots', 'cleaver', 'edge', 'fang', 'mantle', 'plate'], ['tome', 'boots', 'scepter', 'staff', 'orb', 'prism', 'mantle'], ['blade', 'boots', 'stormbow', 'mantle', 'edge', 'fang', 'plate'], ['charm', 'boots', 'chalice', 'mantle', 'orb', 'scepter', 'plate']],
];
const SKILLS: number[][][] = [
  [],
  [[0, 2, 1], [1, 0, 2], [1, 0, 2], [1, 0, 2], [2, 0, 1]],
  [[2, 1, 0], [0, 2, 1], [0, 2, 1], [0, 2, 1], [1, 0, 2]],
];
function buildOf(u: Unit): string[] {
  const h = u.hero!;
  // The tuned build stands in for the hero's default; the alternative builds are played as they are.
  return h.buildIdx === 0 && TUNE.build > 0 ? BUILDS[TUNE.build][h.def] : h.build;
}
function expertBuy(g: Game, u: Unit) {
  const h = u.hero!, build = buildOf(u);
  for (let id = nextPurchase(h.items, h.gold, build); id && g.buy(u, id); id = nextPurchase(h.items, h.gold, build));
}
function expertLevel(g: Game, u: Unit) {
  const h = u.hero!;
  if (TUNE.skills <= 0) { g.autoLevel(u); return; }
  const order = SKILLS[TUNE.skills][h.def];
  while (h.skillPoints > 0) {
    if (g.levelUp(u, 3)) continue;
    let done = false;
    if (h.level >= 3) for (const sl of order) if (h.ranks[sl] === 0 && g.levelUp(u, sl)) { done = true; break; }
    if (!done) for (const sl of order) if (g.levelUp(u, sl)) { done = true; break; }
    if (!done) break;
  }
}

// --- Dodging ----------------------------------------------------------------------------------------
function tryStep(g: Game, u: Unit, dx: number, dy: number, len: number): boolean {
  const n = hypot(dx, dy) || 1;
  const tx = u.x + (dx / n) * len, ty = u.y + (dy / n) * len;
  const [ex, ey] = rayClamp(u.x, u.y, tx, ty);
  if (hypot(ex - u.x, ey - u.y) < len * 0.6) return false;
  moveTo(g, u, ex, ey, true);
  return true;
}

function dodge(g: Game, u: Unit): boolean {
  if (!TUNE.dodge || u.rooted) return false;
  const team = u.team;
  // Telegraphed ground effects.
  for (const z of g.zones) {
    if (z.team === team || !g.botNoticed(z.born)) continue;
    const dx = u.x - z.x, dy = u.y - z.y;
    const d = hypot(dx, dy);
    const need = z.radius + u.radius * 0.6 + 40 - d;
    if (need <= 0) continue;
    if (need / u.ms > z.delay + 0.35) {
      // Cannot walk out: dash if we have one.
      if (u.hero!.def === HERO_VESPER && ready(g, u, 2) && need < 420) { cast(g, u, 2, u.x + (d > 1 ? dx : 1) * 10, u.y + (d > 1 ? dy : 0) * 10); return true; }
      continue;
    }
    const f = FOUNTAIN_POS[team];
    const ax = d > 20 ? dx : f[0] - u.x, ay = d > 20 ? dy : f[1] - u.y;
    if (tryStep(g, u, ax, ay, need + 70) || tryStep(g, u, -ay, ax, z.radius) || tryStep(g, u, ay, -ax, z.radius)) return true;
  }
  if (TUNE.dodgeDash) {
    for (let k = 0; k < 5; k++) {
      const e = g.heroes[(1 - team) * 5 + k];
      const d = e.dash;
      if (!d || !e.alive || !g.botSees(team, e)) continue;
      if (d.kind === 'korvR') {
        // Leap: get out of the landing circle.
        const dx = u.x - d.x, dy = u.y - d.y;
        const dd = hypot(dx, dy);
        const need = 400 + u.radius * 0.6 + 40 - dd;
        if (need <= 0) continue;
        const eta = hypot(e.x - d.x, e.y - d.y) / d.speed;
        if (need / u.ms > eta + 0.35) {
          if (u.hero!.def === HERO_VESPER && ready(g, u, 2) && need < 420) { cast(g, u, 2, u.x + (dd > 1 ? dx : 1) * 10, u.y + (dd > 1 ? dy : 0) * 10); return true; }
          continue;
        }
        const f = FOUNTAIN_POS[team];
        const ax = dd > 20 ? dx : f[0] - u.x, ay = dd > 20 ? dy : f[1] - u.y;
        if (tryStep(g, u, ax, ay, need + 70) || tryStep(g, u, -ay, ax, 400) || tryStep(g, u, ay, -ax, 400)) return true;
      } else if (d.kind === 'korvQ') {
        // Charge: sidestep its path.
        let vx = d.x - e.x, vy = d.y - e.y;
        const len = hypot(vx, vy);
        if (len < 1) continue;
        vx /= len; vy /= len;
        const rx = u.x - e.x, ry = u.y - e.y;
        const along = rx * vx + ry * vy;
        if (along < -40 || along > len + 120) continue;
        const perp = rx * -vy + ry * vx;
        const need = 110 + u.radius + 30 - Math.abs(perp);
        if (need <= 0) continue;
        const side = perp >= 0 ? 1 : -1;
        if (tryStep(g, u, -vy * side, vx * side, need + 50)) return true;
      }
    }
  }
  // Skillshots.
  for (const p of g.projectiles) {
    if (p.team === team || p.targetId >= 0 || !g.botNoticed(p.born)) continue;
    const rx = u.x - p.x, ry = u.y - p.y;
    const along = rx * p.dx + ry * p.dy;
    if (along < -40 || along > p.dist + 60 || along > 1500) continue;
    const perp = rx * -p.dy + ry * p.dx; // signed distance to the left of travel
    const need = p.radius + u.radius + 35 - Math.abs(perp);
    if (need <= 0) continue;
    const tHit = along / p.speed;
    if (tHit > need / u.ms + TUNE.dodgeLead) continue; // plenty of time, keep doing what we were doing
    if (!p.pierce && !p.heroOnly) {
      // A creep in front of us will eat the shot.
      let blocked = false;
      for (const m of g.unitsIn(p.x + p.dx * along * 0.5, p.y + p.dy * along * 0.5, along * 0.5 + 120)) {
        if (m.team === p.team || m === u || m.kind === 'tower' || m.kind === 'core') continue;
        const ma = (m.x - p.x) * p.dx + (m.y - p.y) * p.dy;
        if (ma < 0 || ma > along - 30) continue;
        const mp = Math.abs((m.x - p.x) * -p.dy + (m.y - p.y) * p.dx);
        if (mp < p.radius + m.radius - 8) { blocked = true; break; }
      }
      if (blocked) continue;
    }
    const side = perp >= 0 ? 1 : -1;
    const nx = -p.dy * side, ny = p.dx * side;
    if (tryStep(g, u, nx, ny, need + 50)) return true;
    if (tryStep(g, u, -nx, -ny, p.radius * 2 + u.radius + 80 - need)) return true;
  }
  return false;
}

// --- Movement helpers -------------------------------------------------------------------------------
function stepAway(g: Game, u: Unit, from: Unit, len: number): boolean {
  const f = FOUNTAIN_POS[u.team];
  let dx = u.x - from.x, dy = u.y - from.y;
  const n = hypot(dx, dy) || 1;
  const fx = f[0] - u.x, fy = f[1] - u.y;
  const fn = hypot(fx, fy) || 1;
  dx = dx / n + (fx / fn) * 0.4;
  dy = dy / n + (fy / fn) * 0.4;
  if (tryStep(g, u, dx, dy, len)) return true;
  return tryStep(g, u, fx, fy, len);
}

/** Attack `t`, repositioning between shots: ranged heroes kite anything that out-brawls them up close. */
function fightMicro(g: Game, u: Unit, t: Unit, enemies: Unit[]) {
  if (u.windup > 0) { attack(g, u, t); return; }
  if (u.range > 250 && u.attackCd > 0.14) {
    let k: Unit | null = null, kd = Infinity;
    for (const e of enemies) {
      if (e.range > u.range - 90 || e.stunned || e.rooted) continue;
      const d = dist(e, u);
      if (d < kd) { kd = d; k = e; }
    }
    if (k && kd < Math.min(u.range - 60, k.range + 280) && stepAway(g, u, k, 200)) return;
  }
  attack(g, u, t);
}

// --- Main entry ------------------------------------------------------------------------
export function expertThink(g: Game, u: Unit) {
  const h = u.hero!;
  TUNE = TUNES[u.team];
  if (TUNE.thinkTicks > 0) h.botNextThink = g.tick + TUNE.thinkTicks;
  if (TUNE.skills > 0) h.autoLevel = false;
  if (h.skillPoints > 0) expertLevel(g, u);
  if (!u.alive) { expertBuy(g, u); return; }
  const team = u.team;
  const def = HEROES[h.def];
  const hpPct = u.hp / u.maxHp;
  const manaPct = u.maxMana > 0 ? u.mana / u.maxMana : 1;
  const inBase = g.inFountain(u);
  if (inBase) {
    expertBuy(g, u);
    if (h.botState === 'retreat') h.botState = '';
    if (hpPct < TUNE.baseHp || manaPct < TUNE.baseMana) { if (h.order.type !== 'idle') b_act(g, u, { t: 'stop' }); return; }
  }
  if (u.dash || u.stunned) return;
  if (TUNE.dodge && dodge(g, u)) return;
  const tm = teamMem(g, team);

  const near = g.unitsIn(u.x, u.y, 1600);
  const enemies: Unit[] = [], allies: Unit[] = [];
  let closest: Unit | null = null;
  for (const o of near) {
    if (o.kind !== 'hero') continue;
    if (o.team === team) { allies.push(o); continue; }
    if (!g.botSees(team, o)) continue;
    enemies.push(o);
    if (!closest || dist(o, u) < dist(closest, u)) closest = o;
  }
  const closestD = closest ? dist(closest, u) : Infinity;

  if (h.def === HERO_LIORA && b_supportSpells(g, u, allies, enemies)) return;

  // --- Survival -----------------------------------------------------------
  // Head home once a trip would finish a build item or buy a worthwhile chunk of components.
  const trip = planShopping(h.items, h.gold - (hpPct < 0.65 || manaPct < 0.3 ? 0 : TUNE.shopExtra), buildOf(u));
  const wantShop = trip.completes || trip.spent >= TUNE.shopMin;
  let myPow = 0, theirPow = 0;
  for (const a of allies) if (dist(a, u) < 1100) myPow += b_power(a);
  for (const e of enemies) if (dist(e, u) < 1100) theirPow += b_power(e);
  const lowThreshold = h.def === HERO_KORVATH ? TUNE.lowHpTank : TUNE.lowHp;
  if (hpPct < lowThreshold || (hpPct < 0.5 && theirPow > myPow * TUNE.fleeRatio && closestD < 800)) h.botState = 'retreat';
  else if (h.botState === 'retreat' && hpPct > TUNE.retreatExit && !wantShop) h.botState = '';
  if (h.botState !== 'retreat' && wantShop && enemies.length === 0 && !(TUNE.tempo && tm.myAlive - tm.enemyAlive >= 2 && hpPct > 0.5)) h.botState = 'retreat';
  if (h.botState === 'retreat') {
    let creepNear = false;
    for (const o of near) {
      if ((o.kind === 'minion' && o.team !== team && dist(o, u) < 550) || (o.kind === 'monster' && o.targetId === u.id)) { creepNear = true; break; }
    }
    const safe = closestD > 1250 && !creepNear && !b_enemyTowerThreat(g, u, u.x, u.y) && g.tick - u.lastDamageTick > 45;
    if (h.order.type === 'recall') { if (safe) return; }
    b_escapeSpells(g, u, closest);
    if (u.dash) return;
    if (safe) b_act(g, u, { t: 'recall' });
    else b_retreat(g, u);
    return;
  }
  if (h.order.type === 'recall') return;

  // --- Enemy pack ----------------------------------------------------------
  let holdAtRally = false;
  if (TUNE.pack && g.time < tm.packUntil) {
    let foes = 0, mates = 0;
    for (const e of enemies) if (dist(e, u) < 1400) foes++;
    for (const a of allies) if (dist(a, u) < 900) mates++; // counts this hero too
    // Never face three or more alone: back off before they arrive, towards the rest of the team.
    if (foes >= 3 && mates < foes - TUNE.packOdds) {
      h.botState = 'regroup';
      if (closestD < 750) { b_escapeSpells(g, u, closest); if (u.dash) return; }
      b_retreat(g, u);
      return;
    }
    if (tm.threat && TUNE.pack >= 2) {
      // While the tower is fortified it will not fall soon: only its own laners wait behind it, everyone else keeps farming.
      let comes = (tm.rallyAll && TUNE.pack >= 3) || def.lane === tm.packLane;
      if (comes && tm.rallyAll && TUNE.packSplit > 0 && def.lane >= 0 && def.lane !== tm.packLane) {
        // The laner farthest from the pack stays out and pushes.
        let far = -1, fd = -1;
        for (let k = 0; k < 5; k++) {
          const a = g.heroes[team * 5 + k], lane = HEROES[k].lane;
          if (!a.alive || lane < 0 || lane === tm.packLane) continue;
          const d = hypot(a.x - tm.packX, a.y - tm.packY);
          if (d > fd) { fd = d; far = k; }
        }
        if (far === h.def) comes = false;
      }
      if (comes) {
        const rd = hypot(u.x - tm.rallyX, u.y - tm.rallyY);
        if (rd > 500) {
          if (closestD > 800) { h.botState = 'rally'; b_act(g, u, { t: 'move', x: tm.rallyX, y: tm.rallyY }); return; }
        } else holdAtRally = true;
      }
    }
  }

  // --- Tower safety --------------------------------------------------------
  if (TUNE.fountain && inEnemyFountain(team, u.x, u.y)) { h.botState = 'backoff'; b_retreat(g, u); return; }
  const tower = b_enemyTowerThreat(g, u, u.x, u.y);
  if (tower) {
    const killable = closest && closest.hp < closest.maxHp * 0.15 && closestD < u.range + 100 && hpPct > 0.5;
    if (!killable) { h.botState = 'backoff'; b_retreat(g, u); return; }
  }

  // --- Fighting ---------------------------------------------------------------
  const engageR = Math.max(TUNE.engageRange, u.range + TUNE.engageExtra) + (h.def === HERO_SABLE ? 350 : 0);
  let target: Unit | null = null, bestScore = Infinity;
  for (const e of enemies) {
    const d = dist(e, u);
    if (d > engageR + (e.hp < e.maxHp * 0.3 ? 400 : 0)) continue;
    const score = e.hp + d * 0.6 + (b_enemyTowerThreat(g, u, e.x, e.y) ? 900 : 0);
    if (score < bestScore) { bestScore = score; target = e; }
  }
  if (target) {
    let ally = 0, foe = 0;
    for (const a of allies) if (dist(a, target) < 1200) ally += b_power(a);
    for (const e of enemies) if (dist(e, target) < 1000) foe += b_power(e);
    for (const s of g.structures) {
      if (!s.alive || s.kind !== 'tower') continue;
      if (dist(s, target) < TOWER_RANGE + 100) { if (s.team === team) ally += TUNE.towerAlly; else foe += TUNE.towerFoe; }
    }
    const tPct = target.hp / target.maxHp;
    const need = (tPct < 0.3 ? TUNE.engageLow : hpPct < 0.45 ? TUNE.engageHurt : TUNE.engage) / h.botAggro;
    const diving = (b_enemyTowerThreat(g, u, target.x, target.y) !== null && !(tPct < 0.2 && hpPct > 0.55)) || (TUNE.fountain > 0 && inEnemyFountain(team, target.x, target.y));
    if (ally >= foe * need && !diving) {
      h.botState = 'fight';
      if (TUNE.spells ? x_fightSpells(g, u, target, enemies, allies, ally / Math.max(0.01, foe)) : b_fightSpells(g, u, target, enemies, allies)) return;
      if (TUNE.kite) fightMicro(g, u, target, enemies);
      else b_act(g, u, { t: 'attack', id: target.id });
      return;
    }
    if (dist(target, u) < 700 && ally < foe * TUNE.backoffRatio) { h.botState = 'backoff'; b_escapeSpells(g, u, target); if (!u.dash) b_retreat(g, u); return; }
    // Otherwise poke with long-range spells while farming.
    if (manaPct > TUNE.pokeMana && (h.def === HERO_IGNARA || h.def === HERO_VESPER || h.def === HERO_LIORA) && !diving) {
      const d = dist(target, u);
      if (h.def === HERO_IGNARA && d < 900 && b_castLead(g, u, 0, target, 1500)) return;
      if (h.def === HERO_VESPER && d < 950 && b_castLead(g, u, 0, target, 1700)) return;
      if (h.def === HERO_LIORA && d < 850 && b_castLead(g, u, 1, target, 1300)) return;
    }
  }

  // Vesper: cross-map finisher on anything visible and nearly dead.
  if (TUNE.snipe && h.def === HERO_VESPER && g.castError(u, 3) === '') {
    const rd = ([250, 375, 500][h.ranks[3] - 1] + u.ad) * u.dmgAmp;
    for (let k = 0; k < 5; k++) {
      const e = g.heroes[(1 - team) * 5 + k];
      if (!e.alive || !g.botSees(team, e)) continue;
      const d = dist(e, u);
      if (d < 3500 && d > 650 && (rd * 100) / (100 + e.mr) > e.hp + 20 + d * 0.03) {
        const [px, py] = intercept(g, u, e, 2200);
        if (g.tryCast(u, 3, px, py, -1)) return;
      }
    }
  }

  // --- Objectives -----------------------------------------------------------------
  if (holdAtRally) {
    // Waiting behind the threatened tower: take the creeps that come within reach, otherwise stay put.
    h.botState = 'hold';
    let pick: Unit | null = null;
    for (const m of near) {
      if (m.kind !== 'minion' || m.team === team || !g.botSees(team, m) || dist(m, u) > u.range + 220 || hypot(m.x - tm.rallyX, m.y - tm.rallyY) > 900) continue;
      if (!pick || m.hp < pick.hp) pick = m;
    }
    if (pick && (pick.hp <= hitDamage(u, pick) * 1.05 || h.def !== HERO_LIORA)) b_act(g, u, { t: 'attack', id: pick.id });
    else if (hypot(u.x - tm.rallyX, u.y - tm.rallyY) > 160) b_act(g, u, { t: 'move', x: tm.rallyX, y: tm.rallyY });
    else if (h.order.type !== 'idle' && h.order.type !== 'attack') b_act(g, u, { t: 'stop' });
    return;
  }
  if (TUNE.baseDef && tm.defLane[h.def] >= 0) {
    h.botState = 'defend';
    const dx = tm.defX[h.def], dy = tm.defY[h.def];
    const far = hypot(u.x - dx, u.y - dy);
    const f = FOUNTAIN_POS[team];
    if (far > 4000 && hypot(f[0] - dx, f[1] - dy) < far * 0.5 && enemies.length === 0) { b_act(g, u, { t: 'recall' }); return; }
    if (far > 1100) { b_act(g, u, { t: 'move', x: dx, y: dy }); return; }
    b_laneThink(g, u, tm.defLane[h.def], near, false);
    return;
  }
  const packed = TUNE.react > 0 && g.time < tm.packUntil;
  const grouping = g.time > TUNE.groupTime || packed;
  const groupLane = packed ? tm.packLane : LANE_MID;
  let lane = def.lane;
  if (h.def === HERO_SABLE) {
    if (!grouping && b_jungleThink(g, u, near)) return;
    lane = grouping ? groupLane : [1, 0, 2][Math.floor(g.time / 60) % 3];
  } else if (grouping) lane = groupLane;
  const carry = g.heroes[team * 5 + HERO_VESPER];
  b_laneThink(g, u, lane, near, h.def === HERO_LIORA && carry.alive && dist(carry, u) < 1500);
}

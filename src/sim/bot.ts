/**
 * Scripted hero AI. Used for bot opponents/teammates in human games and as the
 * built-in baseline opponent for ML training. Fully deterministic.
 */
import type { Game } from './game.ts';
import { HERO_VS_MINION, TOWER_RANGE, planShopping, type Command, type Unit } from './defs.ts';
import { HEROES, HERO_IGNARA, HERO_KORVATH, HERO_LIORA, HERO_SABLE, HERO_VESPER } from './heroes.ts';
import { CAMPS, FOUNTAIN_POS, LANE_LENGTH, LANE_MID, lanePoint, laneProgress } from './map.ts';
import { hypot, sq } from './dmath.ts';

const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => hypot(a.x - b.x, a.y - b.y);

// --- Skill profiles ------------------------------------------------------------------
/** Knobs that make the same decision logic play better or worse. */
interface Skill {
  /** Extra ticks between decisions (slower reactions). */
  thinkExtra: number;
  thinkJitter: number;
  /** Skillshot aim error as a fraction of distance, and how much of the target's motion is led. */
  aimErr: number;
  lead: number;
  /** Chance per decision to use abilities at all (fights, pokes, escapes, heals, farming). */
  spells: number;
  /** Multiplier on the strength advantage needed before taking a fight. */
  engage: number;
  /** Extra range at which enemy heroes are considered for a fight. */
  engageRange: number;
  /** Retreat below this fraction of health (tank / others). */
  lowHpTank: number;
  lowHp: number;
  /** Last-hit when the creep is below this many hits (late values miss last hits). */
  lastHit: number;
  /** Gold kept in reserve before recalling, and the component spend worth a recall. */
  shopExtra: number;
  shopMin: number;
}
const NORMAL: Skill = { thinkExtra: 0, thinkJitter: 0, aimErr: 0.07, lead: 1, spells: 1, engage: 1, engageRange: 0, lowHpTank: 0.24, lowHp: 0.3, lastHit: 1.05, shopExtra: 350, shopMin: 1200 };
/** A beginner: reacts in ~0.5 s, misses skillshots, forgets abilities, fights cautiously, misses last hits and shops late. */
const EASY: Skill = { thinkExtra: 9, thinkJitter: 6, aimErr: 0.2, lead: 0.4, spells: 0.4, engage: 1.35, engageRange: -200, lowHpTank: 0.18, lowHp: 0.22, lastHit: 0.55, shopExtra: 700, shopMin: Infinity };
let SK = NORMAL;
/** Skips ability use this decision (only ever rolls the RNG for weaker profiles, so `basic` stays bit-identical). */
const noSpells = (g: Game) => SK.spells < 1 && g.rng.next() >= SK.spells;

function act(g: Game, u: Unit, c: Command) {
  g.applyCommand(u, c);
}

function cast(g: Game, u: Unit, slot: number, x: number, y: number, id = -1): boolean {
  if (g.castError(u, slot) !== '') return false;
  return g.tryCast(u, slot, x, y, id);
}

/** Cast a skillshot with simple target leading. */
function castLead(g: Game, u: Unit, slot: number, t: Unit, speed: number): boolean {
  const d = dist(u, t);
  const lead = Math.min(0.6, d / speed) * g.rng.range(0.5, 1) * SK.lead;
  const err = d * SK.aimErr;
  return cast(g, u, slot, t.x + g.botVx(t) * lead + g.rng.range(-err, err), t.y + g.botVy(t) * lead + g.rng.range(-err, err), t.id);
}

function power(x: Unit): number {
  return (0.3 + 0.7 * (x.hp / x.maxHp)) * (x.hero!.level + 3) * (1 + x.hero!.itemGold * 0.00005);
}

// --- Lane front cache ------------------------------------------------------------
interface Fronts { tick: number; s: number[] }
const frontCache = new WeakMap<Game, Fronts>();

/** Lane progress (blue-oriented) of each team's foremost minion per lane; index = team*3+lane. */
function laneFronts(g: Game): number[] {
  let f = frontCache.get(g);
  if (f && g.tick - f.tick < 6) return f.s;
  const s = [-1, -1, -1, 1e9, 1e9, 1e9];
  for (const m of g.units) {
    if (m.kind !== 'minion' || !m.alive) continue;
    const p = laneProgress(m.lane, m.x, m.y);
    const i = m.team * 3 + m.lane;
    if (m.team === 0) { if (p > s[i]) s[i] = p; } else if (p < s[i]) s[i] = p;
  }
  f = { tick: g.tick, s };
  frontCache.set(g, f);
  return s;
}

/** Where on the lane our team currently holds the line. */
function holdProgress(g: Game, team: number, lane: number): number {
  const fronts = laneFronts(g);
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

function enemyTowerThreat(g: Game, u: Unit, x: number, y: number): Unit | null {
  for (const s of g.structures) {
    if (s.team === u.team || !s.alive || s.kind !== 'tower') continue;
    if (sq(s.x - x) + sq(s.y - y) > sq(TOWER_RANGE + 120)) continue;
    let escort = 0;
    for (const m of g.unitsIn(s.x, s.y, TOWER_RANGE)) if (m.kind === 'minion' && m.team === u.team) escort++;
    if (escort < 1 || s.targetId === u.id || g.tick - u.hero!.aggroTick < 75) return s;
  }
  return null;
}

function retreat(g: Game, u: Unit) {
  const f = FOUNTAIN_POS[u.team];
  act(g, u, { t: 'move', x: f[0], y: f[1] });
}

// --- Ability usage -----------------------------------------------------------------
function fightSpells(g: Game, u: Unit, t: Unit, enemies: Unit[], allies: Unit[]): boolean {
  const h = u.hero!;
  const d = dist(u, t);
  const tPct = t.hp / t.maxHp;
  switch (h.def) {
    case HERO_KORVATH: {
      const clump = enemies.filter((e) => dist(e, t) < 380).length;
      if (d < 750 && (clump >= 2 || tPct < 0.55 || allies.length >= 2) && cast(g, u, 3, t.x, t.y)) return true;
      if (d < 500 && castLead(g, u, 0, t, 1300)) return true;
      if (d < 300 && cast(g, u, 2, u.x, u.y)) return true;
      if (d < 450 && (u.hp < u.maxHp * 0.8 || g.tick - u.lastDamageTick < 30) && cast(g, u, 1, u.x, u.y)) return true;
      return false;
    }
    case HERO_SABLE:
      if (d < 620 && tPct < 0.5 && cast(g, u, 3, t.x, t.y, t.id)) return true;
      if (d > 350 && d < 1000 && !u.stealthed && cast(g, u, 2, u.x, u.y)) return true;
      if (d < 520 && cast(g, u, 0, t.x, t.y, t.id)) return true;
      if (d < 260 && cast(g, u, 1, u.x, u.y)) return true;
      return false;
    case HERO_IGNARA: {
      const held = t.stunned || t.rooted;
      const clump = enemies.filter((e) => dist(e, t) < 380).length;
      if (d < 930 && (clump >= 2 || (tPct < 0.6 && tPct > 0.12) || held) && cast(g, u, 3, t.x + g.botVx(t) * 0.5, t.y + g.botVy(t) * 0.5)) return true;
      if (d < 800 && castLead(g, u, 2, t, 1150)) return true;
      if (d < 900 && castLead(g, u, 0, t, 1500)) return true;
      if (d < 850 && cast(g, u, 1, t.x + (held ? 0 : g.botVx(t) * 0.5), t.y + (held ? 0 : g.botVy(t) * 0.5))) return true;
      return false;
    }
    case HERO_VESPER:
      if (d < 2200 && (tPct < 0.4 || (d < 900 && allies.length >= 2)) && castLead(g, u, 3, t, 2200)) return true;
      if (d < 950 && castLead(g, u, 0, t, 1700)) return true;
      if (d < u.range + 150 && cast(g, u, 1, u.x, u.y)) return true;
      if (d < 280 && t.range < 250 && cast(g, u, 2, u.x + (u.x - t.x), u.y + (u.y - t.y))) return true;
      return false;
    case HERO_LIORA:
      if (d < 880 && castLead(g, u, 1, t, 1300)) return true;
      return false;
  }
  return false;
}

function supportSpells(g: Game, u: Unit, allies: Unit[], enemies: Unit[]): boolean {
  let low: Unit | null = null;
  let lowPct = 0.68;
  let pressured: Unit | null = null;
  for (const a of allies) {
    if (dist(a, u) > 750) continue;
    const p = a.hp / a.maxHp;
    if (p < lowPct) { lowPct = p; low = a; }
    if (g.tick - a.lastDamageTick < 20 && (!pressured || a.hp < pressured.hp) && (g.tick - a.lastHeroDamageTick < 45 || p < 0.6)) pressured = a;
  }
  const hurt = allies.filter((a) => dist(a, u) < 700 && a.hp < a.maxHp * 0.5).length;
  if (enemies.length > 0 && (hurt >= 2 || (low !== null && lowPct < 0.25)) && cast(g, u, 3, u.x, u.y)) return true;
  if (low && cast(g, u, 0, low.x, low.y, low.id)) return true;
  if (pressured && cast(g, u, 2, pressured.x, pressured.y, pressured.id)) return true;
  return false;
}

function escapeSpells(g: Game, u: Unit, threat: Unit | null) {
  const h = u.hero!;
  const f = FOUNTAIN_POS[u.team];
  switch (h.def) {
    case HERO_KORVATH: if (threat) cast(g, u, 1, u.x, u.y); break;
    case HERO_SABLE: if (threat) cast(g, u, 2, u.x, u.y); break;
    case HERO_IGNARA: if (threat && dist(u, threat) < 700) castLead(g, u, 2, threat, 1150); break;
    case HERO_VESPER:
      if (threat && dist(u, threat) < 600) {
        const p = u.path && u.pathIdx < u.path.length ? [u.path[u.pathIdx], u.path[u.pathIdx + 1]] : f;
        cast(g, u, 2, p[0], p[1]);
      }
      break;
    case HERO_LIORA:
      if (threat && dist(u, threat) < 800) { if (!cast(g, u, 2, u.x, u.y, u.id)) castLead(g, u, 1, threat, 1300); }
      if (u.hp < u.maxHp * 0.6) cast(g, u, 0, u.x, u.y, u.id);
      break;
  }
}

function farmSpells(g: Game, u: Unit, creeps: Unit[]): boolean {
  const h = u.hero!;
  if (creeps.length < 3 || u.mana < u.maxMana * 0.5 || noSpells(g)) return false;
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
    case HERO_KORVATH: return d < 260 && cast(g, u, 2, u.x, u.y);
    case HERO_IGNARA: return d < 850 && cast(g, u, 1, best.x, best.y);
    case HERO_VESPER: return d < 900 && h.level >= 3 && cast(g, u, 0, best.x, best.y);
    default: return false;
  }
}

// --- Objectives ---------------------------------------------------------------------
function attackDamage(u: Unit, t: Unit): number {
  const vsMinion = u.hero && t.kind === 'minion' ? HERO_VS_MINION : 1;
  return (u.ad * vsMinion * 100) / (100 + Math.max(0, t.armor - u.armorPen)) + (u.hero ? u.hero.it.onHit * 0.8 : 0);
}

function laneThink(g: Game, u: Unit, lane: number, near: Unit[], isSupport: boolean) {
  const h = u.hero!;
  const team = u.team;
  const dir = team === 0 ? 1 : -1;
  const ranged = u.range > 250;
  const reach = u.range + 420;

  // Structures first: hit towers / the core whenever our minions are tanking.
  for (const s of g.structures) {
    if (s.team === team || !s.alive || !s.vulnerable) continue;
    if (dist(s, u) > reach + s.radius) continue;
    if (s.kind === 'tower' && enemyTowerThreat(g, u, u.x, u.y) === s) continue;
    let escort = 0;
    for (const m of g.unitsIn(s.x, s.y, 900)) if (m.kind === 'minion' && m.team === team) escort++;
    if (escort >= 1) { act(g, u, { t: 'attack', id: s.id }); h.botState = 'siege'; return; }
  }

  const creeps: Unit[] = [];
  for (const m of near) {
    if (m.kind === 'minion' && m.team !== team && g.botSees(team, m) && dist(m, u) < reach + 150) creeps.push(m);
  }
  if (creeps.length > 0) {
    const hit = attackDamage(u, creeps[0]);
    let lastHit: Unit | null = null, lowest: Unit | null = null;
    for (const m of creeps) {
      if (enemyTowerThreat(g, u, m.x, m.y)) continue;
      if (m.hp <= hit * SK.lastHit && (!lastHit || dist(m, u) < dist(lastHit, u))) lastHit = m;
      if (!lowest || m.hp < lowest.hp) lowest = m;
    }
    if (!isSupport && farmSpells(g, u, creeps)) return;
    const pick = lastHit ?? (isSupport ? null : lowest);
    if (pick) { act(g, u, { t: 'attack', id: pick.id }); h.botState = 'farm'; return; }
  }

  // Hold the line just behind our front minion.
  const s = holdProgress(g, team, lane) - dir * (ranged ? 330 : 120) - (isSupport ? dir * 120 : 0);
  const p = lanePoint(lane, Math.max(200, Math.min(LANE_LENGTH[lane] - 200, s)));
  const off = (h.slot % 5) * 37 - 70;
  h.botState = 'lane';
  act(g, u, { t: 'move', x: p[0] + off, y: p[1] + off * 0.6 });
}

function jungleThink(g: Game, u: Unit, near: Unit[]): boolean {
  const h = u.hero!;
  let ci = h.botCamp;
  if (ci < 0 || g.campAlive[ci] <= 0) {
    ci = -1;
    let bestD = Infinity;
    for (let i = 0; i < CAMPS.length; i++) {
      const c = CAMPS[i];
      if (g.campAlive[i] <= 0) continue;
      const mine = c.side === u.team;
      const wyrm = c.type === 'wyrm' && h.level >= 8 && u.hp > u.maxHp * 0.7;
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
  if (!target) { act(g, u, { t: 'move', x: c.x, y: c.y }); return true; }
  const d = dist(u, target);
  if (d < 500) cast(g, u, 0, target.x, target.y, target.id);
  if (d < 250) cast(g, u, 1, u.x, u.y);
  act(g, u, { t: 'attack', id: target.id });
  return true;
}

// --- Main entry ------------------------------------------------------------------------
/** Beginner-level bot: the basic logic played with the EASY skill profile. */
export function easyThink(g: Game, u: Unit) {
  SK = EASY;
  try { botThink(g, u); } finally { SK = NORMAL; }
  u.hero!.botNextThink += EASY.thinkExtra + g.rng.int(EASY.thinkJitter);
}

export function botThink(g: Game, u: Unit) {
  const h = u.hero!;
  if (h.skillPoints > 0) g.autoLevel(u);
  if (!u.alive) { g.autoBuy(u); return; }
  const team = u.team;
  const def = HEROES[h.def];
  const hpPct = u.hp / u.maxHp;
  const manaPct = u.maxMana > 0 ? u.mana / u.maxMana : 1;
  const inBase = g.inFountain(u);
  if (inBase) {
    g.autoBuy(u);
    if (h.botState === 'retreat') h.botState = '';
    if (hpPct < 0.92 || manaPct < 0.8) { if (h.order.type !== 'idle') act(g, u, { t: 'stop' }); return; }
  }
  if (u.dash || u.stunned) return;

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

  const spells = !noSpells(g);
  if (h.def === HERO_LIORA && spells && supportSpells(g, u, allies, enemies)) return;

  // --- Survival -----------------------------------------------------------
  // Head home once a trip would finish a build item or buy a worthwhile chunk of components.
  const trip = planShopping(h.items, h.gold - (hpPct < 0.65 || manaPct < 0.3 ? 0 : SK.shopExtra), h.build);
  const wantShop = trip.completes || trip.spent >= SK.shopMin;
  let myPow = 0, theirPow = 0;
  for (const a of allies) if (dist(a, u) < 1100) myPow += power(a);
  for (const e of enemies) if (dist(e, u) < 1100) theirPow += power(e);
  const lowThreshold = h.def === HERO_KORVATH ? SK.lowHpTank : SK.lowHp;
  if (hpPct < lowThreshold || (hpPct < 0.5 && theirPow > myPow * 1.4 && closestD < 800)) h.botState = 'retreat';
  else if (h.botState === 'retreat' && hpPct > 0.6 && !wantShop) h.botState = '';
  if (h.botState !== 'retreat' && wantShop && enemies.length === 0) h.botState = 'retreat';
  if (h.botState === 'retreat') {
    let creepNear = false;
    for (const o of near) {
      if ((o.kind === 'minion' && o.team !== team && dist(o, u) < 550) || (o.kind === 'monster' && o.targetId === u.id)) { creepNear = true; break; }
    }
    const safe = closestD > 1250 && !creepNear && !enemyTowerThreat(g, u, u.x, u.y) && g.tick - u.lastDamageTick > 45;
    if (h.order.type === 'recall') { if (safe) return; }
    if (spells) escapeSpells(g, u, closest);
    if (u.dash) return;
    if (safe) act(g, u, { t: 'recall' });
    else retreat(g, u);
    return;
  }
  if (h.order.type === 'recall') return;

  // --- Tower safety --------------------------------------------------------
  const tower = enemyTowerThreat(g, u, u.x, u.y);
  if (tower) {
    const killable = closest && closest.hp < closest.maxHp * 0.15 && closestD < u.range + 100 && hpPct > 0.5;
    if (!killable) { h.botState = 'backoff'; retreat(g, u); return; }
  }

  // --- Fighting ---------------------------------------------------------------
  const engageR = Math.max(850, u.range + 380) + (h.def === HERO_SABLE ? 350 : 0) + SK.engageRange;
  let target: Unit | null = null, bestScore = Infinity;
  for (const e of enemies) {
    const d = dist(e, u);
    if (d > engageR + (e.hp < e.maxHp * 0.3 ? 400 : 0)) continue;
    const score = e.hp + d * 0.6 + (enemyTowerThreat(g, u, e.x, e.y) ? 900 : 0);
    if (score < bestScore) { bestScore = score; target = e; }
  }
  if (target) {
    let ally = 0, foe = 0;
    for (const a of allies) if (dist(a, target) < 1200) ally += power(a);
    for (const e of enemies) if (dist(e, target) < 1000) foe += power(e);
    for (const s of g.structures) {
      if (!s.alive || s.kind !== 'tower') continue;
      if (dist(s, target) < TOWER_RANGE + 100) { if (s.team === team) ally += 5; else foe += 7; }
    }
    const tPct = target.hp / target.maxHp;
    const need = ((tPct < 0.3 ? 0.6 : hpPct < 0.45 ? 1.25 : 0.9) * SK.engage) / h.botAggro;
    const diving = enemyTowerThreat(g, u, target.x, target.y) !== null && !(tPct < 0.2 && hpPct > 0.55);
    if (ally >= foe * need && !diving) {
      h.botState = 'fight';
      if (spells && fightSpells(g, u, target, enemies, allies)) return;
      act(g, u, { t: 'attack', id: target.id });
      return;
    }
    if (dist(target, u) < 700 && ally < foe * 0.8) { h.botState = 'backoff'; if (spells) escapeSpells(g, u, target); if (!u.dash) retreat(g, u); return; }
    // Otherwise poke with long-range spells while farming.
    if (spells && manaPct > 0.4 && (h.def === HERO_IGNARA || h.def === HERO_VESPER || h.def === HERO_LIORA) && !diving) {
      const d = dist(target, u);
      if (h.def === HERO_IGNARA && d < 900 && castLead(g, u, 0, target, 1500)) return;
      if (h.def === HERO_VESPER && d < 950 && castLead(g, u, 0, target, 1700)) return;
      if (h.def === HERO_LIORA && d < 850 && castLead(g, u, 1, target, 1300)) return;
    }
  }

  // --- Objectives -----------------------------------------------------------------
  const grouping = g.time > 540;
  let lane = def.lane;
  if (h.def === HERO_SABLE) {
    if (!grouping && jungleThink(g, u, near)) return;
    lane = grouping ? LANE_MID : [1, 0, 2][Math.floor(g.time / 60) % 3];
  } else if (grouping) lane = LANE_MID;
  const carry = g.heroes[team * 5 + HERO_VESPER];
  laneThink(g, u, lane, near, h.def === HERO_LIORA && carry.alive && dist(carry, u) < 1500);
}

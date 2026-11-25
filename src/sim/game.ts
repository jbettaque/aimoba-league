/**
 * The deterministic game simulation. No DOM, no wall-clock, no Math.random:
 * the same seed and the same command stream always produce the same game.
 */
import {
  BUFF_DURATION, DT, ITEM_BY_ID, MAGIC, applyPurchase, canAfford, itemPrice, nextPurchase, MAX_LEVEL, MINION_STATS, MONSTER_STATS, PHYS, SIGHT_RANGE,
  TEAM_NEUTRAL, TICK_RATE, TITAN_BUFF_DURATION, TOWER_AD, TOWER_HP, TOWER_RANGE, TRUE_DMG, XP_RANGE,
  HeroState, Unit, emptyItemStats, xpForLevel,
  CHAIN_RANGE, CLEAVE_RADIUS, CRIT_DAMAGE, EXECUTE_BELOW, FORTIFY_REDUCTION, FORTIFY_UNTIL, HERO_VS_MINION, MINION_VS_MINION, PASSIVE_GOLD, PROXIMITY_GOLD,
  PROXIMITY_GOLD_RANGE, SPELLBLADE_AP, SPELLBLADE_WINDOW, THORNS_ARMOR, TOWER_GOLD_GLOBAL, TOWER_GOLD_LOCAL, TOWER_GOLD_RANGE, TOWER_PLATES, TOWER_PLATE_GOLD,
  VIGOR_DELAY, WARD_COOLDOWN, WARD_DURATION, XP_SHARE_GROUP, XP_SHARE_TWO,
  type BuffKind, type Command, type GameEvent, type ItemStats, type Projectile, type UnitStats, type Zone,
} from './defs.ts';
import { DASH_END, DASH_TICK, HEROES, HERO_IGNARA, HERO_KORVATH, HERO_SABLE, HERO_VESPER, PROJ_HIT, ZONE_HIT } from './heroes.ts';
import {
  CAMPS, CORE_POS, FOUNTAIN_POS, FOUNTAIN_RADIUS, MAP_SIZE, STRUCTURES, findPath, isBlocked, laneWaypoints, lineClear,
  nearestWalkable,
} from './map.ts';
import { Rng } from './rng.ts';
import { botThink, easyThink } from './bot.ts';
import { expertThink } from './bot2.ts';
import { atan2, cos, hypot, sin, sq } from './dmath.ts';

/** Registry of scripted AIs selectable through GameConfig.botKind. */
export const BOT_KINDS: Record<string, (g: Game, u: Unit) => void> = { easy: easyThink, basic: botThink, expert: expertThink };

export interface GameConfig {
  seed?: number;
  /** Per hero slot (0-4 blue, 5-9 red): true = driven by the built-in scripted AI. */
  bots?: boolean[];
  /** Collect visual/audio events (disable for headless training). */
  recordEvents?: boolean;
  /** Game ends in a draw after this many seconds (0 = unlimited). */
  maxTime?: number;
  /** Scripted AI per hero slot ('easy' | 'basic' | 'expert'), or one kind for everyone. Default 'basic'. */
  botKind?: string | string[];
  /** Per hero slot: true = fully manual (no auto level-up / auto shopping; auto-attacks when idle). Used for human players. */
  manual?: boolean[];
  /**
   * Per hero slot, for heroes that are not scripted bots: level abilities up automatically / shop automatically along
   * the hero's build / attack enemies in range while idle. Entries left out keep the default (level and shop on, attack
   * off; the other way round for a `manual` slot). A controller that turns one off sends the `level` / `buy` commands itself.
   */
  autoLevel?: (boolean | null | undefined)[];
  autoBuy?: (boolean | null | undefined)[];
  autoAttack?: (boolean | null | undefined)[];
  /** Record every external command so the game can be replayed. */
  recordCommands?: boolean;
  /** Minimum reaction time of scripted bots in seconds (default 0.2, human-like). 0 = instant. */
  botReaction?: number;
}

/** Velocity history length per hero, in ticks (must exceed the largest reaction time). */
const VEL_HIST = 32;

const GC = 400;
const GN = MAP_SIZE / GC;
const WAVE_FIRST = 12;
const WAVE_INTERVAL = 28;
const ASSIST_TICKS = 10 * TICK_RATE;

/**
 * What a basic attack of `u` does to `t` right now, leaving out critical strikes and one-off bonuses:
 * if this is at least `t.hp`, the attack is a last hit.
 */
export function attackDamage(u: Unit, t: Unit): number {
  const h = u.hero;
  let phys = u.ad * u.dmgAmp, magic = 0;
  if (h) {
    magic = (h.it.onHit + (h.def === HERO_KORVATH ? 0.015 * u.maxHp : 0)) * u.dmgAmp;
    if (t.kind === 'minion') phys *= HERO_VS_MINION;
  } else if (u.kind === 'minion' && t.kind === 'minion') phys *= MINION_VS_MINION;
  return phys * (100 / (100 + Math.max(0, t.armor - u.armorPen))) + magic * (100 / (100 + t.mr * (1 - u.magicPen)));
}

export class Game {
  cfg: GameConfig;
  rng: Rng;
  seed: number;
  tick = 0;
  time = 0;
  units: Unit[] = [];
  byId = new Map<number, Unit>();
  heroes: Unit[] = [];
  structures: Unit[] = [];
  projectiles: Projectile[] = [];
  zones: Zone[] = [];
  events: GameEvent[] = [];
  recordEvents: boolean;
  winner = -1;
  over = false;
  /** Bot reaction time in ticks: new threats are ignored and enemy motion is read this many ticks late. */
  botReact: number;
  /** Per team, per hero slot: tick an enemy hero became visible to that team (-1 = not visible). */
  private seenSince = [new Array(10).fill(-1), new Array(10).fill(-1)];
  /** Ring buffer of hero velocities: [(tick % VEL_HIST) * 10 + slot] * 2 -> vx, vy. */
  private velHist = new Float64Array(VEL_HIST * 10 * 2);
  nextId = 1;
  nextWave = WAVE_FIRST;
  waveCount = 0;
  campAlive: number[] = CAMPS.map(() => 0);
  campRespawnAt: number[] = CAMPS.map((c) => c.firstSpawn);
  wyrmStacks = [0, 0];
  teamKills = [0, 0];
  towersLost = [0, 0];
  firstBlood = false;
  pending: [number, Command][] = [];
  cmdLog: [number, number, Command][] | null = null;
  /**
   * Spatial grid of the living units, rebuilt every tick by a stable counting sort: the units of cell c
   * are gridUnits[cellStart[c] .. cellStart[c + 1]), in the order of this.units.
   */
  private cellStart = new Int32Array(GN * GN + 1);
  private gridUnits: Unit[] = [];
  private gridCell = new Int32Array(64);
  private cellFill = new Int32Array(GN * GN + 1);

  constructor(cfg: GameConfig = {}) {
    this.cfg = cfg;
    this.botReact = Math.min(VEL_HIST - 1, Math.round((cfg.botReaction ?? 0.2) * TICK_RATE));
    this.seed = cfg.seed ?? 1;
    this.rng = new Rng(this.seed);
    this.recordEvents = cfg.recordEvents ?? false;
    if (cfg.recordCommands) this.cmdLog = [];
    for (const s of STRUCTURES) {
      const u = new Unit();
      u.kind = s.kind;
      u.team = s.team;
      u.x = u.px = s.x;
      u.y = u.py = s.y;
      u.lane = s.lane;
      u.tier = s.tier;
      u.radius = s.kind === 'core' ? 130 : 75;
      u.maxHp = u.hp = TOWER_HP[s.tier];
      u.baseAd = TOWER_AD[s.tier];
      u.baseArmor = 40;
      u.baseMr = 40;
      u.baseAs = 0.85;
      u.baseRange = TOWER_RANGE;
      u.baseMs = 0;
      u.vulnerable = s.tier === 0;
      u.fortified = s.kind === 'tower' && s.tier === 0;
      u.hpRegen = s.kind === 'core' ? 4 : 0;
      this.addUnit(u);
      this.structures.push(u);
      this.recalcUnit(u);
    }
    for (let slot = 0; slot < 10; slot++) {
      const team = slot < 5 ? 0 : 1;
      const u = new Unit();
      u.kind = 'hero';
      u.team = team;
      u.radius = 45;
      u.hero = new HeroState(slot % 5, slot);
      u.hero.isBot = cfg.bots ? cfg.bots[slot] : true;
      if (cfg.botKind) u.hero.botKind = typeof cfg.botKind === 'string' ? cfg.botKind : (cfg.botKind[slot] ?? 'basic');
      if (cfg.manual && cfg.manual[slot]) {
        u.hero.autoLevel = u.hero.autoBuy = false;
        u.hero.autoAttack = true;
      }
      u.hero.autoLevel = cfg.autoLevel?.[slot] ?? u.hero.autoLevel;
      u.hero.autoBuy = cfg.autoBuy?.[slot] ?? u.hero.autoBuy;
      u.hero.autoAttack = cfg.autoAttack?.[slot] ?? u.hero.autoAttack;
      u.hero.botNextThink = this.rng.int(8);
      u.hero.botAggro = this.rng.range(0.85, 1.15);
      const hd = HEROES[u.hero.def];
      u.hero.buildIdx = this.rng.int(1 + hd.altBuilds.length);
      u.hero.build = u.hero.buildIdx === 0 ? hd.build : hd.altBuilds[u.hero.buildIdx - 1];
      this.addUnit(u);
      this.heroes.push(u);
      this.placeAtFountain(u);
      this.recalcHero(u);
      u.hp = u.maxHp;
      u.mana = u.maxMana;
      if (u.hero.autoLevel) this.autoLevel(u);
    }
    this.updateVision();
  }

  // --- Basic helpers -------------------------------------------------------
  private addUnit(u: Unit) {
    u.id = this.nextId++;
    u.thinkOffset = u.id % 8;
    this.units.push(u);
    this.byId.set(u.id, u);
  }

  unit(id: number): Unit | undefined {
    return this.byId.get(id);
  }

  canSee(team: number, u: Unit): boolean {
    return u.team === team || (team === 0 ? u.vis0 : u.vis1);
  }

  // --- Bot perception (human reaction time) --------------------------------------
  /** Like canSee, but an enemy hero only registers with bots once it has been visible for their reaction time. */
  botSees(team: number, u: Unit): boolean {
    if (!this.canSee(team, u)) return false;
    if (u.kind !== 'hero' || u.team === team) return true;
    const s = this.seenSince[team][u.hero!.slot];
    return s >= 0 && this.tick - s >= this.botReact;
  }

  /** Whether a projectile or zone spawned at tick `born` has existed long enough for a bot to react to it. */
  botNoticed(born: number): boolean {
    return this.tick - born >= this.botReact;
  }

  /** A hero's velocity as a bot perceives it: what it was one reaction time ago (changes of course register late). */
  botVx(u: Unit): number {
    return this.pastVel(u, 0);
  }
  botVy(u: Unit): number {
    return this.pastVel(u, 1);
  }
  private pastVel(u: Unit, axis: number): number {
    if (!u.hero || this.botReact <= 0) return axis ? u.vy : u.vx;
    const back = Math.min(this.botReact, this.tick);
    const t = (this.tick - back) % VEL_HIST;
    return this.velHist[(t * 10 + u.hero.slot) * 2 + axis];
  }

  fx(name: string, x: number, y: number, r: number, team: number, id = -1, x2 = 0, y2 = 0) {
    if (this.recordEvents) this.events.push({ type: 'fx', name, x, y, r, team, id, x2, y2 });
  }

  private placeAtFountain(u: Unit) {
    const f = FOUNTAIN_POS[u.team];
    const a = (u.hero!.slot % 5) * ((Math.PI * 2) / 5);
    u.x = u.px = f[0] + cos(a) * 150;
    u.y = u.py = f[1] + sin(a) * 150;
    u.facing = u.team === 0 ? -Math.PI / 4 : (Math.PI * 3) / 4;
  }

  inFountain(u: Unit): boolean {
    const f = FOUNTAIN_POS[u.team];
    return sq(u.x - f[0]) + sq(u.y - f[1]) < FOUNTAIN_RADIUS * FOUNTAIN_RADIUS;
  }

  private rebuildGrid() {
    const units = this.units, n = units.length, start = this.cellStart;
    if (this.gridCell.length < n) this.gridCell = new Int32Array(Math.max(n, this.gridCell.length * 2));
    const cellOf = this.gridCell;
    start.fill(0);
    let alive = 0;
    for (let i = 0; i < n; i++) {
      const u = units[i];
      if (!u.alive) { cellOf[i] = -1; continue; }
      const cx = Math.min(GN - 1, Math.max(0, (u.x / GC) | 0));
      const cy = Math.min(GN - 1, Math.max(0, (u.y / GC) | 0));
      const c = cy * GN + cx;
      cellOf[i] = c;
      start[c + 1]++;
      alive++;
    }
    for (let c = 0; c < GN * GN; c++) start[c + 1] += start[c];
    const out = this.gridUnits, fill = this.cellFill;
    if (out.length !== alive) out.length = alive;
    fill.set(start);
    for (let i = 0; i < n; i++) {
      const c = cellOf[i];
      if (c >= 0) out[fill[c]++] = units[i];
    }
  }

  /** All living units whose centre lies within r of (x,y). */
  unitsIn(x: number, y: number, r: number, out: Unit[] = []): Unit[] {
    const x0 = Math.max(0, ((x - r) / GC) | 0), x1 = Math.min(GN - 1, ((x + r) / GC) | 0);
    const y0 = Math.max(0, ((y - r) / GC) | 0), y1 = Math.min(GN - 1, ((y + r) / GC) | 0);
    const r2 = r * r;
    const start = this.cellStart, grid = this.gridUnits;
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) {
        const c = cy * GN + cx;
        for (let i = start[c], end = start[c + 1]; i < end; i++) {
          const u = grid[i];
          const dx = u.x - x, dy = u.y - y;
          if (u.alive && dx * dx + dy * dy <= r2) out.push(u);
        }
      }
    }
    return out;
  }

  /** Enemy (and neutral) non-structure units touching the circle. */
  enemiesIn(team: number, x: number, y: number, r: number): Unit[] {
    const res: Unit[] = [];
    for (const u of this.unitsIn(x, y, r + 110)) {
      if (u.team === team || u.kind === 'tower' || u.kind === 'core') continue;
      const dx = u.x - x, dy = u.y - y;
      const rr = r + u.radius * 0.6;
      if (dx * dx + dy * dy <= rr * rr) res.push(u);
    }
    return res;
  }

  // --- Buffs / stats ---------------------------------------------------------
  addBuff(t: Unit, kind: BuffKind, dur: number, val: number, src: number) {
    if (!t.alive || t.kind === 'tower' || t.kind === 'core') return;
    if ((kind === 'stun' || kind === 'root') && t.kind === 'monster' && (t.subtype === 'titan' || t.subtype === 'wyrm')) dur *= 0.4;
    for (const b of t.buffs) {
      if (b.kind === kind && b.src === src && kind !== 'shield') {
        b.t = Math.max(b.t, dur);
        b.val = Math.max(b.val, val);
        t.statsDirty = true;
        return;
      }
    }
    t.buffs.push({ kind, t: dur, val, src });
    t.statsDirty = true;
    if (kind === 'stun') {
      t.stunned = true;
      t.windup = 0;
      if (t.hero && t.hero.order.type === 'recall') t.hero.order.type = 'idle';
    } else if (kind === 'root') t.rooted = true;
    else if (kind === 'stealth') t.stealthed = true;
  }

  addShield(src: Unit, t: Unit, amount: number, dur: number) {
    amount *= 1 + src.healPower;
    this.addBuff(t, 'shield', dur, amount, src.id);
    t.shield += amount;
    if (src.hero) src.hero.healDone += amount * 0.5;
  }

  hasBuff(u: Unit, kind: BuffKind): boolean {
    for (const b of u.buffs) if (b.kind === kind) return true;
    return false;
  }

  private removeBuff(u: Unit, kind: BuffKind) {
    if (u.buffs.length === 0) return;
    u.buffs = u.buffs.filter((b) => b.kind !== kind);
    u.statsDirty = true;
    if (kind === 'stealth') u.stealthed = false;
  }

  heal(src: Unit | null, t: Unit, amount: number) {
    if (!t.alive) return;
    if (src) amount *= 1 + src.healPower;
    const before = t.hp;
    t.hp = Math.min(t.maxHp, t.hp + amount);
    const done = t.hp - before;
    if (src && src.hero) src.hero.healDone += done;
    if (this.recordEvents && done >= 1) this.events.push({ type: 'heal', id: t.id, x: t.x, y: t.y, amount: done });
  }

  recalcHero(u: Unit) {
    const h = u.hero!, d = HEROES[h.def], L = h.level - 1, it = h.it;
    let ad = d.ad + d.adL * L + it.ad, ap = it.ap;
    let armor = d.armor + d.armorL * L + it.armor, mr = d.mr + d.mrL * L + it.mr;
    let asBonus = d.asL * L + it.as, haste = it.haste, ls = it.lifesteal, range = d.range;
    let hpRegen = d.hpRegen + 0.12 * L + it.hpRegen, manaRegen = d.manaRegen + 0.1 * L + it.manaRegen;
    let slow = 0, msBonus = 0, shield = 0;
    let stunned = false, rooted = false, stealthed = false;
    const maxMana = d.mana + d.manaL * L + it.mana;
    for (const b of u.buffs) {
      switch (b.kind) {
        case 'stun': stunned = true; break;
        case 'root': rooted = true; break;
        case 'slow': if (b.val > slow) slow = b.val; break;
        case 'haste': msBonus += b.val; break;
        case 'shield': shield += b.val; break;
        case 'as': asBonus += b.val; break;
        case 'ad': ad += b.val; break;
        case 'ap': ap += b.val; break;
        case 'resist': armor += b.val; mr += b.val; break;
        case 'lifesteal': ls += b.val; break;
        case 'range': range += b.val; break;
        case 'stealth': stealthed = true; break;
        case 'insight': haste += 20; manaRegen += 0.015 * maxMana; break;
        case 'might': ad += 8 + 2 * h.level; hpRegen += 0.006 * u.maxHp; break;
        case 'titan': ad += 30; ap += 50; break;
        default: break;
      }
    }
    if (h.def === HERO_VESPER) asBonus += 0.06 * h.focusStacks;
    ap *= 1 + it.apAmp; // Archmage Staff
    const maxHp = d.hp + d.hpL * L + it.hp;
    if (maxHp > u.maxHp && u.alive) u.hp += maxHp - u.maxHp;
    u.maxHp = maxHp;
    if (u.hp > maxHp) u.hp = maxHp;
    if (maxMana > u.maxMana && u.alive) u.mana += maxMana - u.maxMana;
    u.maxMana = maxMana;
    u.ad = ad; u.ap = ap; u.armor = armor; u.mr = mr;
    u.as = Math.min(2.5, d.as * (1 + asBonus));
    u.range = range;
    u.ms = Math.max(110, (d.ms + it.ms) * (1 + msBonus) * (1 - slow));
    u.hpRegen = hpRegen; u.manaRegen = manaRegen;
    u.lifesteal = ls; u.haste = haste;
    u.armorPen = it.armorPen; u.magicPen = it.magicPen; u.healPower = it.healPower;
    u.dmgAmp = 1 + 0.05 * this.wyrmStacks[u.team];
    u.stunned = stunned; u.rooted = rooted; u.stealthed = stealthed; u.shield = shield;
  }

  private recalcUnit(u: Unit) {
    let slow = 0, armor = u.baseArmor, mr = u.baseMr, shield = 0;
    let stunned = false, rooted = false;
    for (const b of u.buffs) {
      if (b.kind === 'stun') stunned = true;
      else if (b.kind === 'root') rooted = true;
      else if (b.kind === 'slow') { if (b.val > slow) slow = b.val; }
      else if (b.kind === 'resist') { armor += b.val; mr += b.val; }
      else if (b.kind === 'shield') shield += b.val;
    }
    u.ad = u.baseAd; u.armor = armor; u.mr = mr; u.as = u.baseAs; u.range = u.baseRange;
    u.ms = u.baseMs * (1 - slow);
    u.stunned = stunned; u.rooted = rooted; u.shield = shield;
    u.statsDirty = false;
  }

  private recalcItems(h: HeroState) {
    const s = emptyItemStats();
    h.itemGold = 0;
    for (const id of h.items) {
      h.itemGold += ITEM_BY_ID[id].cost;
      const st = ITEM_BY_ID[id].stats;
      for (const k in st) s[k as keyof ItemStats] += st[k as keyof ItemStats]!;
    }
    h.it = s;
  }

  // --- Damage / death ----------------------------------------------------------
  /** how: 0 = other (dots), 1 = basic attack, 2 = ability, 3 = on-hit bonus. Returns damage dealt. */
  damage(src: Unit | undefined | null, t: Unit, amount: number, dtype: number, how: number): number {
    if (!t.alive || amount <= 0) return 0;
    const structure = t.kind === 'tower' || t.kind === 'core';
    if (structure) {
      if (!t.vulnerable || how === 2 || how === 0) return 0;
      if (src) {
        let escorted = false;
        for (const m of this.unitsIn(t.x, t.y, 1000)) {
          if (m.kind === 'minion' && m.team === src.team) { escorted = true; break; }
        }
        if (!escorted) amount *= 0.35;
        if (t.fortified) amount *= 1 - FORTIFY_REDUCTION;
        if (src.kind === 'minion') amount *= 0.6;
        else if (src.hero && this.hasBuff(src, 'titan')) amount *= 1.4;
      }
    }
    if (src) {
      amount *= src.dmgAmp;
      if (src.hero && src.hero.def === HERO_SABLE && t.hp < 0.4 * t.maxHp) amount *= 1.12;
      if (t.kind === 'minion') {
        // Minions wear each other down slowly; a hero's attack hits them hard. Together that leaves time for the last hit.
        if (src.kind === 'minion') amount *= MINION_VS_MINION;
        else if (src.hero && how === 1) amount *= HERO_VS_MINION;
      }
      // Shadow Fang.
      if (src.hero && src.hero.it.execute > 0 && how !== 0 && !structure && t.hp < EXECUTE_BELOW * t.maxHp) amount *= 1 + src.hero.it.execute;
    }
    if (dtype === PHYS) amount *= 100 / (100 + Math.max(0, t.armor - (src ? src.armorPen : 0)));
    else if (dtype === MAGIC) amount *= 100 / (100 + t.mr * (1 - (src ? src.magicPen : 0)));
    // Void Prism: ability damage to a hero also takes a share of the hero's current health (as magic damage).
    if (how === 2 && src && src.hero && src.hero.it.burn > 0 && t.hero) amount += src.hero.it.burn * t.hp * (100 / (100 + t.mr * (1 - src.magicPen)));
    let dealt = amount;
    if (t.shield > 0) {
      for (const b of t.buffs) {
        if (b.kind !== 'shield' || amount <= 0) continue;
        const a = Math.min(b.val, amount);
        b.val -= a;
        amount -= a;
        if (b.val <= 0) b.t = 0;
      }
      t.shield = Math.max(0, t.shield - (dealt - amount));
    }
    if (t.kind === 'tower' && src && src.team !== t.team) {
      // Plating: every fifth of a tower's health pays the attackers' heroes who are there.
      const before = Math.ceil((t.hp / t.maxHp) * TOWER_PLATES), after = Math.ceil((Math.max(0, t.hp - amount) / t.maxHp) * TOWER_PLATES);
      if (after < before) this.towerGold(t, src.team, TOWER_PLATE_GOLD * (Math.min(before, TOWER_PLATES) - after));
    }
    t.hp -= amount;
    t.lastDamageTick = this.tick;
    const sh = src ? src.hero : null;
    if (sh) {
      sh.dmgDealt += dealt;
      if (structure) sh.towerDmg += dealt;
    }
    if (t.hero) {
      const th = t.hero;
      th.dmgTaken += amount;
      if (th.order.type === 'recall') th.order.type = 'idle';
      if (sh) {
        th.dmgBy[sh.slot] = this.tick;
        sh.aggroTick = this.tick;
        t.lastHeroDamageTick = this.tick;
      }
    } else if (t.kind === 'monster' && src && src.alive && src.team !== TEAM_NEUTRAL) {
      if (t.targetId === -1) {
        for (const m of this.unitsIn(t.homeX, t.homeY, 500)) if (m.kind === 'monster' && m.camp === t.camp) m.targetId = src.id;
      }
      t.targetId = src.id;
    }
    if (src && !structure) {
      if (how === 1 && src.lifesteal > 0 && src.alive) src.hp = Math.min(src.maxHp, src.hp + amount * src.lifesteal);
      if (how === 2 && sh) {
        if (sh.def === HERO_IGNARA) this.addBuff(t, 'dot', 3, 10 + 2.5 * sh.level + 0.08 * src.ap, src.id);
        if (sh.it.spellSlow > 0) this.addBuff(t, 'slow', 1, sh.it.spellSlow, src.id);
      }
    }
    if (t.hero && t.hp > 0 && src && src !== t) {
      const worn = t.hero.it;
      // Spirit Mantle: ability damage from a hero raises a shield.
      if (worn.ward > 0 && how === 2 && sh && this.tick >= t.hero.wardAt) {
        t.hero.wardAt = this.tick + WARD_COOLDOWN * TICK_RATE;
        this.addShield(t, t, worn.ward * t.maxHp, WARD_DURATION);
        if (this.recordEvents) this.events.push({ type: 'fx', name: 'shield', x: t.x, y: t.y, r: 90, team: t.team, id: t.id });
      }
      // Bulwark Plate: whoever attacks the wearer is hurt in return (how 0, so it cannot echo back and forth).
      if (worn.thorns > 0 && how === 1 && src.alive && src.kind !== 'tower') this.damage(t, src, worn.thorns + THORNS_ARMOR * t.armor, MAGIC, 0);
    }
    if (this.recordEvents && dealt >= 1) this.events.push({ type: 'dmg', id: t.id, x: t.x, y: t.y, amount: dealt, dtype, other: src ? src.id : -1, slot: how });
    if (t.hp <= 0 && t.alive) this.kill(t, src ?? null);
    return dealt;
  }

  giveGold(u: Unit, amount: number, show = true) {
    const h = u.hero!;
    h.gold += amount;
    h.totalGold += amount;
    h.earnedGold += amount;
    if (show && this.recordEvents) this.events.push({ type: 'gold', id: u.id, x: u.x, y: u.y, amount });
  }

  giveXp(u: Unit, amount: number) {
    const h = u.hero!;
    if (h.level >= MAX_LEVEL) return;
    h.xp += amount;
    while (h.level < MAX_LEVEL && h.xp >= xpForLevel(h.level)) {
      h.xp -= xpForLevel(h.level);
      h.level++;
      h.skillPoints++;
      if (u.alive) {
        this.recalcHero(u);
        if (this.recordEvents) this.events.push({ type: 'levelup', id: u.id, x: u.x, y: u.y });
      }
    }
    if (h.level >= MAX_LEVEL) h.xp = 0;
    if (h.autoLevel) this.autoLevel(u);
  }

  private shareXp(x: number, y: number, team: number, amount: number, killer: Unit | null) {
    const near: Unit[] = [];
    for (const h of this.heroes) {
      if (h.team !== team || !h.alive) continue;
      if (h === killer || sq(h.x - x) + sq(h.y - y) < XP_RANGE * XP_RANGE) near.push(h);
    }
    if (near.length === 0) return;
    const each = amount * (near.length === 1 ? 1 : near.length === 2 ? XP_SHARE_TWO : XP_SHARE_GROUP / near.length);
    for (const h of near) this.giveXp(h, each);
  }

  private kill(t: Unit, src: Unit | null) {
    t.alive = false;
    t.hp = 0;
    t.buffs.length = 0;
    t.dash = null;
    t.windup = 0;
    t.path = null;
    t.targetId = -1;
    const killerHero = src && src.hero ? src : null;
    if (this.recordEvents) this.events.push({ type: 'death', id: t.id, x: t.x, y: t.y, other: src ? src.id : -1, team: t.team, name: t.kind, sub: t.subtype });
    if (t.kind === 'hero') {
      const th = t.hero!;
      th.deaths++;
      th.respawn = 6 + 2.2 * th.level;
      th.order.type = 'idle';
      th.focusStacks = 0;
      // Credit the most recent hero damager when a tower or minion lands the last hit.
      let killer = killerHero;
      const assists: Unit[] = [];
      let latest = -1;
      for (const h of this.heroes) {
        if (h.team === t.team) continue;
        const at = th.dmgBy[h.hero!.slot];
        if (this.tick - at <= ASSIST_TICKS) {
          assists.push(h);
          if (!killerHero && at > latest) { latest = at; killer = h; }
        }
      }
      th.dmgBy.fill(-9999);
      const xpPool = 140 + 40 * th.level;
      if (killer) {
        const kh = killer.hero!;
        kh.kills++;
        this.teamKills[killer.team]++;
        let gold = 300;
        if (!this.firstBlood) { this.firstBlood = true; gold += 100; if (this.recordEvents) this.events.push({ type: 'announce', name: 'firstblood', text: 'First Blood!', team: killer.team }); }
        this.giveGold(killer, gold);
        const helpers = assists.filter((a) => a !== killer);
        for (const a of helpers) {
          a.hero!.assists++;
          this.giveGold(a, 150 / helpers.length);
        }
        const parts = helpers.length + 1;
        this.giveXp(killer, xpPool / parts);
        for (const a of helpers) this.giveXp(a, xpPool / parts);
        if (kh.def === HERO_SABLE) { kh.cds[0] *= 0.4; kh.cds[3] *= 0.4; }
        for (const a of helpers) if (a.hero!.def === HERO_SABLE) { a.hero!.cds[0] *= 0.4; a.hero!.cds[3] *= 0.4; }
      } else if (src && src.team !== TEAM_NEUTRAL) {
        this.teamKills[src.team]++;
      }
      if (this.recordEvents) this.events.push({ type: 'kill', id: t.id, other: killer ? killer.id : src ? src.id : -1, team: t.team });
      return;
    }
    if (killerHero) {
      const kh = killerHero.hero!;
      if (kh.def === HERO_SABLE && kh.refund[2] === t.id && this.tick <= kh.refund[1]) kh.cds[0] *= 0.5;
    }
    if (t.kind === 'minion') {
      if (killerHero) { this.giveGold(killerHero, t.goldValue); killerHero.hero!.cs++; }
      else this.proximityGold(t);
      this.shareXp(t.x, t.y, 1 - t.team, t.xpValue, killerHero);
    } else if (t.kind === 'monster') {
      const camp = CAMPS[t.camp];
      const team = src ? src.team : TEAM_NEUTRAL;
      if (--this.campAlive[t.camp] <= 0) this.campRespawnAt[t.camp] = this.time + camp.respawn;
      if (team !== TEAM_NEUTRAL) {
        if (killerHero) { this.giveGold(killerHero, t.goldValue); killerHero.hero!.cs++; }
        this.shareXp(t.x, t.y, team, t.xpValue, killerHero);
        if ((t.subtype === 'might' || t.subtype === 'insight') && killerHero) {
          this.addBuff(killerHero, t.subtype, BUFF_DURATION, 1, -1);
        } else if (t.subtype === 'wyrm') {
          this.wyrmStacks[team] = Math.min(4, this.wyrmStacks[team] + 1);
          for (const h of this.heroes) if (h.team === team) this.giveGold(h, 80, h.alive);
          if (this.recordEvents) this.events.push({ type: 'announce', name: 'objective', text: `${team === 0 ? 'Blue' : 'Red'} team has slain the Wyrm!`, team });
        } else if (t.subtype === 'titan') {
          for (const h of this.heroes) if (h.team === team) { this.giveGold(h, 150, h.alive); if (h.alive) this.addBuff(h, 'titan', TITAN_BUFF_DURATION, 1, -1); }
          if (this.recordEvents) this.events.push({ type: 'announce', name: 'objective', text: `${team === 0 ? 'Blue' : 'Red'} team has slain the Titan!`, team });
        }
      }
    } else if (t.kind === 'tower') {
      const team = 1 - t.team;
      this.towersLost[t.team]++;
      this.towerGold(t, team, TOWER_GOLD_LOCAL);
      for (const h of this.heroes) if (h.team === team) this.giveGold(h, TOWER_GOLD_GLOBAL, h.alive);
      this.updateVulnerability();
      if (this.recordEvents) this.events.push({ type: 'announce', name: 'tower', text: `${team === 0 ? 'Blue' : 'Red'} team destroyed a tower!`, team });
    } else if (t.kind === 'core') {
      this.winner = 1 - t.team;
      this.over = true;
      if (this.recordEvents) this.events.push({ type: 'gameover', team: this.winner });
    }
  }

  /** Tower gold is shared by the attacking team's heroes at the tower; with nobody there it is lost. */
  private towerGold(t: Unit, team: number, amount: number) {
    const near: Unit[] = [];
    for (const h of this.heroes) {
      if (h.alive && h.team === team && sq(h.x - t.x) + sq(h.y - t.y) < TOWER_GOLD_RANGE * TOWER_GOLD_RANGE) near.push(h);
    }
    for (const h of near) this.giveGold(h, amount / near.length);
  }

  /** A minion nobody last-hit still pays the enemy heroes standing by it: a share of its gold, split between them. */
  private proximityGold(t: Unit) {
    const near: Unit[] = [];
    for (const h of this.heroes) {
      if (h.alive && h.team !== t.team && sq(h.x - t.x) + sq(h.y - t.y) < PROXIMITY_GOLD_RANGE * PROXIMITY_GOLD_RANGE) near.push(h);
    }
    for (const h of near) this.giveGold(h, (PROXIMITY_GOLD * t.goldValue) / near.length);
  }

  private updateVulnerability() {
    for (let team = 0; team < 2; team++) {
      const mine = this.structures.filter((s) => s.team === team);
      const alive = (lane: number, tier: number) => mine.some((s) => s.alive && s.lane === lane && s.tier === tier && s.kind === 'tower');
      let innerDown = false;
      for (let lane = 0; lane < 3; lane++) if (!alive(lane, 1)) innerDown = true;
      const guardiansUp = mine.some((s) => s.alive && s.tier === 2);
      for (const s of mine) {
        if (s.tier === 0) s.vulnerable = true;
        else if (s.tier === 1) s.vulnerable = !alive(s.lane, 0);
        else if (s.tier === 2) s.vulnerable = innerDown;
        else s.vulnerable = !guardiansUp;
      }
    }
  }

  // --- Commands ----------------------------------------------------------------
  /** Queue a command for a hero slot; it is applied at the start of the next tick. */
  command(slot: number, cmd: Command) {
    this.pending.push([slot, cmd]);
  }

  applyCommand(u: Unit, c: Command) {
    const h = u.hero!;
    if (c.t === 'level') { this.levelUp(u, c.slot); return; }
    if (c.t === 'buy') { this.buy(u, c.item); return; }
    if (!u.alive) return;
    const o = h.order;
    switch (c.t) {
      case 'move':
      case 'amove': {
        if (!Number.isFinite(c.x) || !Number.isFinite(c.y)) return;
        const same = (o.type === 'move' || o.type === 'amove') && u.path !== null && Math.abs(o.x - c.x) + Math.abs(o.y - c.y) < 50;
        o.type = c.t;
        if (!same) {
          o.x = c.x;
          o.y = c.y;
          this.setPath(u, c.x, c.y);
        }
        if (c.t === 'move') { u.windup = 0; u.targetId = -1; }
        break;
      }
      case 'attack': {
        const t = this.byId.get(c.id);
        if (t && t.alive && t.team !== u.team) this.orderAttack(u, t);
        break;
      }
      case 'stop':
        o.type = 'idle';
        u.path = null;
        u.windup = 0;
        u.targetId = -1;
        break;
      case 'cast':
        this.tryCast(u, c.slot, c.x, c.y, c.id ?? -1);
        break;
      case 'recall':
        if (o.type !== 'recall' && !u.dash && !u.stunned) {
          o.type = 'recall';
          h.recallT = 5;
          u.path = null;
          u.windup = 0;
          this.fx('recall', u.x, u.y, 80, u.team, u.id);
        }
        break;
    }
  }

  orderAttack(u: Unit, t: Unit) {
    const o = u.hero!.order;
    if (o.type !== 'attack' || o.targetId !== t.id) u.path = null;
    o.type = 'attack';
    o.targetId = t.id;
  }

  private setPath(u: Unit, x: number, y: number) {
    u.path = findPath(u.x, u.y, x, y);
    u.pathIdx = 0;
    u.repathAt = this.tick + 12;
  }

  canLevel(u: Unit, slot: number): boolean {
    const h = u.hero!;
    if (h.skillPoints <= 0 || slot < 0 || slot > 3) return false;
    const r = h.ranks[slot];
    if (slot === 3) return r < 3 && h.level >= [6, 11, 15][r];
    return r < 4 && r < Math.ceil(h.level / 2);
  }

  levelUp(u: Unit, slot: number): boolean {
    if (!this.canLevel(u, slot)) return false;
    const h = u.hero!;
    h.ranks[slot]++;
    h.skillPoints--;
    return true;
  }

  autoLevel(u: Unit) {
    const h = u.hero!;
    const order = HEROES[h.def].skillOrder;
    while (h.skillPoints > 0) {
      if (this.levelUp(u, 3)) continue;
      let done = false;
      if (h.level >= 3) for (const s of order) if (h.ranks[s] === 0 && this.levelUp(u, s)) { done = true; break; }
      if (!done) for (const s of order) if (this.levelUp(u, s)) { done = true; break; }
      if (!done) break;
    }
  }

  canBuy(u: Unit, item: string): boolean {
    return canAfford(u.hero!.items, u.hero!.gold, item) && (!u.alive || this.inFountain(u));
  }

  /** Buys an item; owned components of its recipe are consumed and discounted from the price. */
  buy(u: Unit, item: string): boolean {
    if (!this.canBuy(u, item)) return false;
    const h = u.hero!;
    h.gold -= itemPrice(h.items, item);
    h.items = applyPurchase(h.items, item);
    this.recalcItems(h);
    this.recalcHero(u);
    return true;
  }

  /** Spends gold along the recommended build, buying components when a full item is out of reach. */
  autoBuy(u: Unit) {
    const h = u.hero!, build = h.build;
    for (let id = nextPurchase(h.items, h.gold, build); id && this.buy(u, id); id = nextPurchase(h.items, h.gold, build));
  }

  /** Why a cast would fail right now ('' when castable). */
  castError(u: Unit, slot: number): string {
    const h = u.hero!;
    const rank = h.ranks[slot];
    if (!u.alive) return 'dead';
    if (rank <= 0) return 'not learned';
    if (h.cds[slot] > 0) return 'on cooldown';
    if (u.mana < HEROES[h.def].abilities[slot].cost[rank - 1]) return 'not enough mana';
    if (u.stunned) return 'stunned';
    if (u.dash) return 'busy';
    return '';
  }

  tryCast(u: Unit, slot: number, x: number, y: number, id: number): boolean {
    if (slot < 0 || slot > 3 || this.castError(u, slot) !== '') return false;
    const h = u.hero!;
    const ab = HEROES[h.def].abilities[slot];
    let target: Unit | null = null;
    if (ab.targeting === 'enemy' || ab.targeting === 'enemyHero') {
      const t = this.byId.get(id);
      if (!t || !t.alive || t.team === u.team || t.kind === 'tower' || t.kind === 'core' || !this.canSee(u.team, t)) return false;
      if (ab.targeting === 'enemyHero' && t.kind !== 'hero') return false;
      target = t;
    } else if (ab.targeting === 'ally') {
      const t = this.byId.get(id);
      target = t && t.alive && t.team === u.team && t.kind === 'hero' ? t : u;
    } else if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    if (target && target !== u) {
      const d = hypot(target.x - u.x, target.y - u.y);
      if (d > ab.range + target.radius) {
        const o = h.order;
        o.type = 'cast';
        o.slot = slot;
        o.targetId = target.id;
        u.path = null;
        return true;
      }
    }
    this.executeCast(u, slot, x, y, target);
    return true;
  }

  private executeCast(u: Unit, slot: number, x: number, y: number, target: Unit | null) {
    const h = u.hero!;
    const ab = HEROES[h.def].abilities[slot];
    const rank = h.ranks[slot];
    u.mana -= ab.cost[rank - 1];
    h.cds[slot] = (ab.cd[rank - 1] * 100) / (100 + u.haste);
    if (u.stealthed) this.removeBuff(u, 'stealth');
    if (h.order.type === 'recall') h.order.type = 'idle';
    u.windup = 0;
    h.castLock = 0.12;
    if (target && target !== u) u.facing = atan2(target.y - u.y, target.x - u.x);
    if (this.recordEvents) this.events.push({ type: 'cast', id: u.id, slot, x: u.x, y: u.y });
    if (h.it.spellblade > 0) h.spellbladeUntil = this.tick + SPELLBLADE_WINDOW * TICK_RATE; // Sage's Orb
    ab.cast(this, u, rank, x, y, target);
  }

  // --- Spawning ------------------------------------------------------------------
  spawnProjectile(p: Partial<Projectile>): Projectile {
    const proj: Projectile = {
      id: this.nextId++, kind: 'aa', srcId: -1, team: 0, x: 0, y: 0, px: 0, py: 0, dx: 1, dy: 0, speed: 1500, dist: 2000,
      radius: 40, targetId: -1, pierce: false, heroOnly: false, dmg: 0, dmg2: 0, rank: 1, hits: [], born: this.tick, ...p,
    };
    proj.px = proj.x;
    proj.py = proj.y;
    this.projectiles.push(proj);
    return proj;
  }

  spawnZone(z: Partial<Zone>) {
    const zone: Zone = { id: this.nextId++, kind: '', srcId: -1, team: 0, x: 0, y: 0, radius: 100, delay: 1, total: 1, rank: 1, born: this.tick, ...z };
    zone.total = zone.delay;
    this.zones.push(zone);
  }

  startDash(u: Unit, x: number, y: number, speed: number, kind: string, rank: number, targetId: number) {
    u.dash = { x, y, speed, kind, rank, targetId };
    u.path = null;
    u.windup = 0;
    const o = u.hero ? u.hero.order : null;
    if (o && (o.type === 'move' || o.type === 'amove')) o.type = 'idle';
  }

  private spawnUnit(kind: 'minion' | 'monster', subtype: string, st: UnitStats, team: number, x: number, y: number): Unit {
    const u = new Unit();
    const min = this.time / 60;
    u.kind = kind;
    u.subtype = subtype;
    u.team = team;
    u.x = u.px = x;
    u.y = u.py = y;
    u.radius = st.radius;
    u.maxHp = u.hp = st.hp + st.hpPerMin * min;
    u.baseAd = st.ad + st.adPerMin * min;
    u.baseArmor = st.armor;
    u.baseMr = st.mr;
    u.baseAs = st.as;
    u.baseRange = st.range;
    u.baseMs = st.ms;
    u.goldValue = st.gold + (kind === 'minion' ? Math.floor(min * 0.5) : Math.floor(min * 2));
    u.xpValue = st.xp * (1 + 0.03 * min);
    this.addUnit(u);
    this.recalcUnit(u);
    return u;
  }

  private spawnWave() {
    this.waveCount++;
    for (let team = 0; team < 2; team++) {
      const enemyStructs = this.structures.filter((s) => s.team !== team);
      for (let lane = 0; lane < 3; lane++) {
        const wps = laneWaypoints(team, lane);
        const c = CORE_POS[team];
        const dx = wps[0][0] - c[0], dy = wps[0][1] - c[1];
        const d = hypot(dx, dy);
        const types = ['melee', 'melee', 'melee', 'ranged', 'ranged', 'ranged'];
        if (this.waveCount % 3 === 0) types.splice(3, 0, 'siege');
        if (!enemyStructs.some((s) => s.alive && s.lane === lane && s.tier === 1)) types.unshift('super');
        types.forEach((type, i) => {
          const off = 420 - i * 55;
          const side = ((i % 3) - 1) * 40;
          const m = this.spawnUnit('minion', type, MINION_STATS[type], team, c[0] + (dx / d) * off - (dy / d) * side + this.rng.range(-12, 12), c[1] + (dy / d) * off + (dx / d) * side + this.rng.range(-12, 12));
          m.lane = lane;
          m.wp = 0;
          m.facing = atan2(dy, dx);
        });
      }
    }
  }

  private spawnCamp(ci: number) {
    const c = CAMPS[ci];
    const n = c.type === 'stalkers' ? 3 : 1;
    const sub = c.type === 'stalkers' ? 'stalker' : c.type;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + 0.5;
      const ox = n > 1 ? cos(a) * 80 : 0, oy = n > 1 ? sin(a) * 80 : 0;
      const m = this.spawnUnit('monster', sub, MONSTER_STATS[sub], TEAM_NEUTRAL, c.x + ox, c.y + oy);
      m.camp = ci;
      m.homeX = m.x;
      m.homeY = m.y;
      m.facing = atan2(MAP_SIZE / 2 - c.y, MAP_SIZE / 2 - c.x);
    }
    this.campAlive[ci] = n;
    if ((c.type === 'wyrm' || c.type === 'titan') && this.recordEvents) {
      this.events.push({ type: 'announce', name: 'awaken', text: c.type === 'wyrm' ? 'The Wyrm has awakened' : 'The Titan has awakened', team: 2 });
    }
  }

  // --- Movement ------------------------------------------------------------------
  private moveStep(u: Unit, tx: number, ty: number, step: number): boolean {
    const dx = tx - u.x, dy = ty - u.y;
    const d = hypot(dx, dy);
    if (d <= step) {
      if (!isBlocked(tx, ty)) { u.x = tx; u.y = ty; }
      return true;
    }
    u.facing = atan2(dy, dx);
    const nx = u.x + (dx / d) * step, ny = u.y + (dy / d) * step;
    if (!isBlocked(nx, ny)) { u.x = nx; u.y = ny; }
    else if (!isBlocked(nx, u.y)) u.x = nx;
    else if (!isBlocked(u.x, ny)) u.y = ny;
    return false;
  }

  /** Advance along u.path. Returns true when the end is reached. */
  private followPath(u: Unit): boolean {
    const p = u.path;
    if (!p) return true;
    let step = u.ms * DT;
    while (u.pathIdx < p.length) {
      const tx = p[u.pathIdx], ty = p[u.pathIdx + 1];
      const d = hypot(tx - u.x, ty - u.y);
      if (this.moveStep(u, tx, ty, step)) {
        u.pathIdx += 2;
        step -= d;
        if (step <= 0.01) break;
      } else return false;
    }
    if (u.pathIdx >= p.length) { u.path = null; return true; }
    return false;
  }

  /** Walk toward a (possibly moving) point, using A* only when the straight line is blocked. */
  private approach(u: Unit, x: number, y: number) {
    if (lineClear(u.x, u.y, x, y)) {
      u.path = null;
      this.moveStep(u, x, y, u.ms * DT);
      return;
    }
    if (!u.path || this.tick >= u.repathAt) this.setPath(u, x, y);
    this.followPath(u);
  }

  private updateDash(u: Unit) {
    const d = u.dash!;
    let t: Unit | undefined;
    let stopAt = 0;
    if (d.targetId >= 0) {
      t = this.byId.get(d.targetId);
      if (!t || !t.alive) { u.dash = null; this.fixPosition(u); return; }
      d.x = t.x;
      d.y = t.y;
      stopAt = u.radius + t.radius;
    }
    const dx = d.x - u.x, dy = d.y - u.y;
    const dist = hypot(dx, dy);
    const step = d.speed * DT;
    if (dist > 1) u.facing = atan2(dy, dx);
    if (dist - stopAt <= step) {
      const k = dist > 1 ? Math.max(0, dist - stopAt) / dist : 0;
      u.x += dx * k;
      u.y += dy * k;
      u.dash = null;
      this.fixPosition(u);
      const end = DASH_END[d.kind];
      if (end) end(this, u, d, t);
      return;
    }
    u.x += (dx / dist) * step;
    u.y += (dy / dist) * step;
    const tick = DASH_TICK[d.kind];
    if (tick && tick(this, u, d)) { u.dash = null; this.fixPosition(u); }
  }

  private fixPosition(u: Unit) {
    if (isBlocked(u.x, u.y)) [u.x, u.y] = nearestWalkable(u.x, u.y);
  }

  // --- Attacks -------------------------------------------------------------------
  private inRange(u: Unit, t: Unit): boolean {
    const r = u.range + t.radius;
    return sq(t.x - u.x) + sq(t.y - u.y) <= r * r;
  }

  private attackTick(u: Unit, t: Unit) {
    if (u.windup > 0) {
      if (u.windupTarget !== t.id) { u.windup = 0; return; }
      u.windup -= DT;
      if (u.windup <= 0) this.fireAttack(u, t);
      return;
    }
    if (u.attackCd <= 0) {
      u.windup = Math.min(0.28, 0.22 / u.as);
      u.windupTarget = t.id;
      u.attackCd = 1 / u.as;
      u.attackAnim = u.windup + 0.15;
      u.facing = atan2(t.y - u.y, t.x - u.x);
      if (u.stealthed) this.removeBuff(u, 'stealth');
    }
  }

  private fireAttack(u: Unit, t: Unit) {
    let phys = u.ad, magic = 0;
    const h = u.hero;
    if (h) {
      for (const b of u.buffs) {
        if (b.kind === 'empower') { phys += b.val; b.t = 0; break; }
      }
      if (h.it.crit > 0 && this.rng.next() < h.it.crit) phys *= CRIT_DAMAGE; // Executioner's Edge
      if (h.it.spellblade > 0 && h.spellbladeUntil >= this.tick) { magic += h.it.spellblade + SPELLBLADE_AP * u.ap; h.spellbladeUntil = -1; }
      magic += h.it.onHit;
      if (h.def === HERO_KORVATH) magic += 0.015 * u.maxHp;
      if (h.def === HERO_VESPER) {
        if (h.focusId === t.id) h.focusStacks = Math.min(5, h.focusStacks + 1);
        else { h.focusId = t.id; h.focusStacks = 0; }
      }
    } else if (u.kind === 'tower') {
      if (t.kind === 'hero') { phys *= 1 + 0.3 * u.towerHits; u.towerHits = Math.min(4, u.towerHits + 1); }
    }
    if (this.recordEvents) this.events.push({ type: 'attack', id: u.id, other: t.id, x: u.x, y: u.y });
    if (u.range < 250) this.applyAttack(u, t, phys, magic);
    else {
      this.spawnProjectile({
        kind: u.kind === 'tower' ? 'tower' : 'aa', srcId: u.id, team: u.team, x: u.x, y: u.y, targetId: t.id,
        speed: u.kind === 'tower' ? 1700 : 1500, dmg: phys, dmg2: magic, radius: u.kind === 'tower' ? 26 : u.hero ? 16 : 10,
      });
    }
  }

  private applyAttack(src: Unit | undefined, t: Unit, phys: number, magic: number) {
    this.damage(src, t, phys, PHYS, 1);
    if (magic > 0 && t.alive) this.damage(src, t, magic, MAGIC, 3);
    const worn = src && src.alive && src.hero ? src.hero.it : null;
    if (src && worn && t.kind !== 'tower' && t.kind !== 'core') {
      const foe = (o: Unit) => o !== t && o.alive && o.team !== src.team && o.kind !== 'tower' && o.kind !== 'core';
      // Titan Cleaver: the blow carries on into the enemies around the target.
      if (worn.cleave > 0) for (const o of this.unitsIn(t.x, t.y, CLEAVE_RADIUS)) if (foe(o)) this.damage(src, o, phys * worn.cleave, PHYS, 3);
      // Stormbow: every third attack arcs lightning to two more enemies.
      if (worn.chain > 0 && ++src.hero!.hitCount % 3 === 0) {
        let arcs = 0;
        for (const o of this.unitsIn(t.x, t.y, CHAIN_RANGE)) {
          if (arcs >= 2) break;
          if (!foe(o) || !this.canSee(src.team, o)) continue;
          if (this.recordEvents) this.events.push({ type: 'fx', name: 'chain', x: t.x, y: t.y, x2: o.x, y2: o.y, team: src.team });
          this.damage(src, o, worn.chain, MAGIC, 3);
          arcs++;
        }
      }
    }
    if (src && src.hero && t.alive && t.kind !== 'tower' && t.kind !== 'core' && this.hasBuff(src, 'might')) {
      this.addBuff(t, 'dot', 3, 6 + 2 * src.hero.level, src.id);
      this.addBuff(t, 'slow', 1.5, 0.12, src.id);
    }
  }

  private validTarget(u: Unit, t: Unit | undefined): t is Unit {
    if (!t || !t.alive || t.team === u.team) return false;
    if ((t.kind === 'tower' || t.kind === 'core') && !t.vulnerable) return false;
    if (u.team !== TEAM_NEUTRAL && !this.canSee(u.team, t)) return false;
    return true;
  }

  /** Nearest attackable enemy within r (heroes/minions/structures of the other team, or monsters that are fighting us). */
  private acquire(u: Unit, r: number, heroBias: number): number {
    let best = -1, bestD = Infinity;
    for (const t of this.unitsIn(u.x, u.y, r + 130)) {
      if (t.team === u.team) continue;
      if (t.team === TEAM_NEUTRAL && t.targetId !== u.id) continue;
      if (!this.validTarget(u, t)) continue;
      let d = hypot(t.x - u.x, t.y - u.y) - t.radius;
      if (d > r) continue;
      if (t.kind === 'hero') d += heroBias;
      if (d < bestD) { bestD = d; best = t.id; }
    }
    return best;
  }

  // --- Per-unit updates ----------------------------------------------------------
  private updateHero(u: Unit) {
    const h = u.hero!;
    if (h.castLock > 0) h.castLock -= DT;
    if (u.dash) { this.updateDash(u); return; }
    if (u.stunned) { u.windup = 0; return; }
    if (h.castLock > 0) return;
    const o = h.order;
    switch (o.type) {
      case 'recall':
        h.recallT -= DT;
        if (h.recallT <= 0) {
          this.fx('teleport', u.x, u.y, 100, u.team);
          this.placeAtFountain(u);
          u.px = u.x;
          u.py = u.y;
          o.type = 'idle';
          this.fx('teleport', u.x, u.y, 100, u.team);
        }
        return;
      case 'move':
        if (!u.rooted && this.followPath(u)) o.type = 'idle';
        return;
      case 'attack': {
        const t = this.byId.get(o.targetId);
        if (!this.validTarget(u, t)) { o.type = 'idle'; u.windup = 0; return; }
        this.engage(u, t);
        return;
      }
      case 'amove': {
        let t = this.byId.get(u.targetId);
        if (!this.validTarget(u, t) || hypot(t.x - u.x, t.y - u.y) > u.range + 500) {
          t = undefined;
          if ((this.tick + u.id) % 3 === 0) {
            u.targetId = this.acquire(u, u.range + 250, 0);
            t = this.byId.get(u.targetId);
          }
        }
        if (t) { this.engage(u, t); u.path = null; }
        else {
          u.windup = 0;
          if (!u.path) this.setPath(u, o.x, o.y);
          if (!u.rooted && this.followPath(u)) o.type = 'idle';
        }
        return;
      }
      case 'cast': {
        const t = this.byId.get(o.targetId);
        const ab = HEROES[h.def].abilities[o.slot];
        if (!t || !t.alive || (t.team !== u.team && !this.canSee(u.team, t)) || this.castError(u, o.slot) !== '') { o.type = 'idle'; return; }
        if (hypot(t.x - u.x, t.y - u.y) <= ab.range + t.radius) {
          this.executeCast(u, o.slot, t.x, t.y, t);
          if (o.type === 'cast') {
            if (t.team !== u.team && t.alive) { o.type = 'attack'; o.targetId = t.id; }
            else o.type = 'idle';
          }
        } else if (!u.rooted) this.approach(u, t.x, t.y);
        return;
      }
      default: {
        // Idle: defend ourselves / auto-attack whatever wanders into range.
        let t = this.byId.get(u.targetId);
        if (!this.validTarget(u, t) || !this.inRange(u, t)) {
          t = undefined;
          u.targetId = -1;
          if (h.autoAttack && !u.stealthed && (this.tick + u.id) % 4 === 0) {
            u.targetId = this.acquire(u, u.range, 0);
            t = this.byId.get(u.targetId);
          }
        }
        if (t) this.attackTick(u, t);
        else u.windup = 0;
      }
    }
  }

  private engage(u: Unit, t: Unit) {
    if (this.inRange(u, t)) { u.path = null; this.attackTick(u, t); }
    else {
      u.windup = 0;
      if (!u.rooted) this.approach(u, t.x, t.y);
    }
  }

  private updateMinion(u: Unit) {
    if (u.stunned) { u.windup = 0; return; }
    if ((this.tick + u.thinkOffset) % 8 === 0) u.targetId = this.acquire(u, 650, 120);
    const t = this.byId.get(u.targetId);
    if (this.validTarget(u, t)) {
      if (this.inRange(u, t)) this.attackTick(u, t);
      else { u.windup = 0; if (!u.rooted) this.approach(u, t.x, t.y); }
      return;
    }
    u.windup = 0;
    if (u.rooted) return;
    const wps = laneWaypoints(u.team, u.lane);
    const wp = wps[Math.min(u.wp, wps.length - 1)];
    if (u.wp < wps.length - 1 && sq(wp[0] - u.x) + sq(wp[1] - u.y) < 220 * 220) u.wp++;
    this.approach(u, wp[0], wp[1]);
  }

  private updateMonster(u: Unit) {
    if (u.stunned) { u.windup = 0; return; }
    const t = u.targetId >= 0 ? this.byId.get(u.targetId) : undefined;
    const leash = u.subtype === 'titan' || u.subtype === 'wyrm' ? 700 : 850;
    if (t && t.alive && !t.stealthed && hypot(t.x - u.homeX, t.y - u.homeY) < leash) {
      if (this.inRange(u, t)) this.attackTick(u, t);
      else { u.windup = 0; if (!u.rooted) this.approach(u, t.x, t.y); }
      return;
    }
    // Lost the target: look for another attacker nearby, otherwise reset.
    u.windup = 0;
    if (u.targetId >= 0) {
      u.targetId = -1;
      for (const h of this.heroes) {
        if (h.alive && !h.stealthed && hypot(h.x - u.homeX, h.y - u.homeY) < leash * 0.7 && this.tick - h.lastDamageTick < 90) { u.targetId = h.id; return; }
      }
    }
    const d = hypot(u.homeX - u.x, u.homeY - u.y);
    if (d > 10) this.moveStep(u, u.homeX, u.homeY, u.ms * 1.5 * DT);
    if (u.hp < u.maxHp && this.tick - u.lastDamageTick > 60) u.hp = Math.min(u.maxHp, u.hp + u.maxHp * 0.15 * DT);
  }

  private updateTower(u: Unit) {
    let t = this.byId.get(u.targetId);
    const r = u.range;
    const ok = (x: Unit | undefined): x is Unit => !!x && x.alive && x.team === 1 - u.team && sq(x.x - u.x) + sq(x.y - u.y) <= sq(r + x.radius);
    if (!ok(t)) t = undefined;
    if ((this.tick + u.thinkOffset) % 4 === 0) {
      // Protect allied heroes: switch to an enemy hero who just attacked one.
      if (!t || t.kind !== 'hero') {
        for (const h of this.heroes) {
          if (h.team !== u.team && ok(h) && this.tick - h.hero!.aggroTick < 60) { t = h; break; }
        }
      }
      if (!t) {
        let bestD = Infinity;
        for (const c of this.unitsIn(u.x, u.y, r + 60)) {
          if (!ok(c)) continue;
          const d = hypot(c.x - u.x, c.y - u.y) + (c.kind === 'hero' ? 2000 : 0);
          if (d < bestD) { bestD = d; t = c; }
        }
      }
    }
    const id = t ? t.id : -1;
    if (id !== u.targetId) { u.targetId = id; u.towerHits = 0; u.windup = 0; }
    if (t) this.attackTick(u, t);
  }

  // --- Projectiles / zones ---------------------------------------------------------
  private updateProjectiles() {
    const ps = this.projectiles;
    for (let i = ps.length - 1; i >= 0; i--) {
      const p = ps[i];
      p.px = p.x;
      p.py = p.y;
      const step = p.speed * DT;
      let done = false;
      if (p.targetId >= 0) {
        const t = this.byId.get(p.targetId);
        if (!t || !t.alive) done = true;
        else {
          const dx = t.x - p.x, dy = t.y - p.y;
          const d = hypot(dx, dy);
          if (d <= step + t.radius * 0.5) {
            this.applyAttack(this.byId.get(p.srcId), t, p.dmg, p.dmg2);
            done = true;
          } else {
            p.dx = dx / d;
            p.dy = dy / d;
            p.x += p.dx * step;
            p.y += p.dy * step;
          }
        }
      } else {
        p.x += p.dx * step;
        p.y += p.dy * step;
        p.dist -= step;
        const handler = PROJ_HIT[p.kind];
        const src = this.byId.get(p.srcId);
        const cands = this.unitsIn(p.x, p.y, p.radius + 120);
        // Nearest first so non-piercing shots hit the closest unit.
        if (cands.length > 1) cands.sort((a, b) => sq(a.x - p.px) + sq(a.y - p.py) - sq(b.x - p.px) - sq(b.y - p.py) || a.id - b.id);
        for (const t of cands) {
          if (t.team === p.team || t.kind === 'tower' || t.kind === 'core' || !t.alive) continue;
          if (p.heroOnly && t.kind !== 'hero') continue;
          const rr = p.radius + t.radius;
          if (sq(t.x - p.x) + sq(t.y - p.y) > rr * rr || p.hits.includes(t.id)) continue;
          handler(this, p, src, t);
          p.hits.push(t.id);
          if (!p.pierce) { done = true; break; }
        }
        if (p.dist <= 0 || p.x < 0 || p.y < 0 || p.x > MAP_SIZE || p.y > MAP_SIZE) done = true;
      }
      if (done) {
        ps[i] = ps[ps.length - 1];
        ps.pop();
      }
    }
  }

  private updateZones() {
    const zs = this.zones;
    for (let i = zs.length - 1; i >= 0; i--) {
      const z = zs[i];
      z.delay -= DT;
      if (z.delay <= 0) {
        ZONE_HIT[z.kind](this, z, this.byId.get(z.srcId));
        zs.splice(i, 1);
      }
    }
  }

  // --- Vision ----------------------------------------------------------------------
  updateVision() {
    for (const u of this.units) {
      if (!u.alive) { u.vis0 = u.vis1 = false; continue; }
      if (u.kind === 'tower' || u.kind === 'core') { u.vis0 = u.vis1 = true; continue; }
      let v0 = u.team === 0, v1 = u.team === 1;
      const hidden = u.stealthed && !this.hasBuff(u, 'reveal');
      // Is u within sight of a living unit of the other team(s)? Walks the grid cells like unitsIn(u, SIGHT_RANGE)
      // and stops as soon as both teams see it; the answer does not depend on the order.
      const x = u.x, y = u.y, R2 = SIGHT_RANGE * SIGHT_RANGE;
      const gx0 = Math.max(0, ((x - SIGHT_RANGE) / GC) | 0), gx1 = Math.min(GN - 1, ((x + SIGHT_RANGE) / GC) | 0);
      const gy0 = Math.max(0, ((y - SIGHT_RANGE) / GC) | 0), gy1 = Math.min(GN - 1, ((y + SIGHT_RANGE) / GC) | 0);
      const start = this.cellStart, grid = this.gridUnits;
      scan: for (let cy = gy0; cy <= gy1; cy++) {
        for (let cx = gx0; cx <= gx1; cx++) {
          const c = cy * GN + cx;
          for (let i = start[c], end = start[c + 1]; i < end; i++) {
            const o = grid[i];
            if (o.team > 1 || o.team === u.team || (o.team === 0 ? v0 : v1) || !o.alive) continue;
            const ox = o.x - x, oy = o.y - y, d2 = ox * ox + oy * oy;
            if (d2 > R2) continue;
            if (hidden) {
              if (!((o.kind === 'tower' && d2 < 800 * 800) || (o.kind === 'hero' && d2 < 230 * 230))) continue;
            } else if (o.kind === 'minion' && d2 > 800 * 800) continue;
            if (o.team === 0) v0 = true;
            else v1 = true;
            if (v0 && v1) break scan;
          }
        }
      }
      u.vis0 = v0;
      u.vis1 = v1;
      if (u.hero) {
        const seen = this.seenSince, sl = u.hero.slot;
        if (!v0) seen[0][sl] = -1; else if (seen[0][sl] < 0) seen[0][sl] = this.tick;
        if (!v1) seen[1][sl] = -1; else if (seen[1][sl] < 0) seen[1][sl] = this.tick;
      }
    }
  }

  // --- Main step ---------------------------------------------------------------------
  step() {
    if (this.over) return;
    this.tick++;
    this.time = this.tick * DT;
    const tick = this.tick;

    if (this.time >= this.nextWave) { this.spawnWave(); this.nextWave += WAVE_INTERVAL; }
    if (tick === FORTIFY_UNTIL * TICK_RATE) {
      for (const s of this.structures) s.fortified = false;
      if (this.recordEvents) this.events.push({ type: 'announce', name: 'fortify', text: 'The outer towers have lost their fortification', team: 2 });
    }
    for (let c = 0; c < CAMPS.length; c++) {
      if (this.campAlive[c] <= 0 && this.time >= this.campRespawnAt[c]) this.spawnCamp(c);
    }

    this.rebuildGrid();

    if (this.pending.length) {
      for (const [slot, cmd] of this.pending) {
        const u = this.heroes[slot];
        if (!u) continue;
        if (this.cmdLog) this.cmdLog.push([tick, slot, cmd]);
        this.applyCommand(u, cmd);
      }
      this.pending.length = 0;
    }
    for (const u of this.heroes) {
      const h = u.hero!;
      if (h.isBot && tick >= h.botNextThink) {
        h.botNextThink = tick + 5 + this.rng.int(3);
        (BOT_KINDS[h.botKind] ?? botThink)(this, u);
      }
    }

    const units = this.units;
    const n = units.length;
    for (let i = 0; i < n; i++) {
      const u = units[i];
      const h = u.hero;
      if (h) {
        for (let s = 0; s < 4; s++) if (h.cds[s] > 0) h.cds[s] = Math.max(0, h.cds[s] - DT);
        if (!u.alive) {
          h.respawn -= DT;
          if (h.respawn <= 0) this.respawn(u);
          continue;
        }
      } else if (!u.alive) continue;
      u.px = u.x;
      u.py = u.y;
      if (u.attackCd > 0) u.attackCd -= DT;
      if (u.attackAnim > 0) u.attackAnim -= DT;
      // Buff timers and damage / heal over time.
      const bs = u.buffs;
      if (bs.length) {
        let expired = false;
        for (let b = 0; b < bs.length; b++) {
          const buff = bs[b];
          if (buff.kind === 'dot' && Math.floor(buff.t * 2) !== Math.floor((buff.t - DT) * 2)) {
            this.damage(this.byId.get(buff.src), u, buff.val * 0.5, MAGIC, 0);
            if (!u.alive) break;
          }
          buff.t -= DT;
          if (buff.t <= 0) expired = true;
        }
        if (!u.alive) continue;
        if (expired) {
          u.buffs = bs.filter((b) => b.t > 0);
          u.statsDirty = true;
        }
      }
      if (h) this.recalcHero(u);
      else if (u.statsDirty) this.recalcUnit(u);
      if (u.hpRegen > 0 && u.hp < u.maxHp) u.hp = Math.min(u.maxHp, u.hp + u.hpRegen * DT);
      // Heart of the Colossus: fast recovery once out of the fight.
      if (h && h.it.vigor > 0 && u.alive && u.hp < u.maxHp && tick - u.lastDamageTick > VIGOR_DELAY * TICK_RATE) u.hp = Math.min(u.maxHp, u.hp + h.it.vigor * u.maxHp * DT);
      if (u.manaRegen > 0 && u.mana < u.maxMana) u.mana = Math.min(u.maxMana, u.mana + u.manaRegen * DT);
      switch (u.kind) {
        case 'hero': this.updateHero(u); break;
        case 'minion': this.updateMinion(u); break;
        case 'monster': this.updateMonster(u); break;
        case 'tower': this.updateTower(u); break;
        default: break;
      }
      u.vx = (u.x - u.px) * TICK_RATE;
      u.vy = (u.y - u.py) * TICK_RATE;
      if (h) {
        const k = ((tick % VEL_HIST) * 10 + h.slot) * 2;
        this.velHist[k] = u.vx;
        this.velHist[k + 1] = u.vy;
      }
    }

    this.updateProjectiles();
    this.updateZones();
    this.separate();
    this.fountains();
    if (tick % 3 === 0) this.updateVision();

    // Passive income and auto-shopping for non-bot heroes that opted in.
    if (tick % 10 === 0) {
      const income = (PASSIVE_GOLD * 10) / TICK_RATE;
      for (const u of this.heroes) {
        const h = u.hero!;
        h.gold += income;
        h.totalGold += income;
        if (!h.isBot && h.autoBuy && (!u.alive || this.inFountain(u))) this.autoBuy(u);
      }
    }

    // Remove dead non-hero, non-structure units.
    if (tick % 5 === 0) {
      let w = 0;
      for (let i = 0; i < units.length; i++) {
        const u = units[i];
        if (!u.alive && (u.kind === 'minion' || u.kind === 'monster')) { this.byId.delete(u.id); continue; }
        units[w++] = u;
      }
      units.length = w;
    }

    if (this.cfg.maxTime && this.time >= this.cfg.maxTime && !this.over) {
      this.over = true;
      if (this.recordEvents) this.events.push({ type: 'gameover', team: -1 });
    }
  }

  private respawn(u: Unit) {
    const h = u.hero!;
    u.alive = true;
    this.placeAtFountain(u);
    this.recalcHero(u);
    u.hp = u.maxHp;
    u.mana = u.maxMana;
    h.order.type = 'idle';
    u.attackCd = 0;
    if (h.autoLevel) this.autoLevel(u);
    if (this.recordEvents) this.events.push({ type: 'respawn', id: u.id, x: u.x, y: u.y });
  }

  private separate() {
    const start = this.cellStart, grid = this.gridUnits;
    for (let c = 0; c < GN * GN; c++) {
      const s0 = start[c], end = start[c + 1];
      if (end - s0 < 2) continue;
      for (let i = s0; i < end; i++) {
        const a = grid[i];
        if (!a.alive || a.dash) continue;
        for (let j = i + 1; j < end; j++) {
          const b = grid[j];
          if (!b.alive || b.dash) continue;
          const dx = b.x - a.x, dy = b.y - a.y;
          const rr = (a.radius + b.radius) * 0.85;
          const d2 = dx * dx + dy * dy;
          if (d2 >= rr * rr) continue;
          const aStatic = a.kind === 'tower' || a.kind === 'core', bStatic = b.kind === 'tower' || b.kind === 'core';
          if (aStatic && bStatic) continue;
          let d = Math.sqrt(d2), nx: number, ny: number;
          if (d < 0.01) { const ang = ((a.id * 7 + b.id * 13) % 628) / 100; nx = cos(ang); ny = sin(ang); d = 0; }
          else { nx = dx / d; ny = dy / d; }
          const push = (rr - d) * 0.5;
          const wa = aStatic ? 0 : bStatic ? 1 : 0.5, wb = bStatic ? 0 : aStatic ? 1 : 0.5;
          if (wa > 0) { const x = a.x - nx * push * wa * 2, y = a.y - ny * push * wa * 2; if (!isBlocked(x, y)) { a.x = x; a.y = y; } }
          if (wb > 0) { const x = b.x + nx * push * wb * 2, y = b.y + ny * push * wb * 2; if (!isBlocked(x, y)) { b.x = x; b.y = y; } }
        }
      }
    }
  }

  private fountains() {
    for (const u of this.heroes) {
      if (!u.alive) continue;
      if (this.inFountain(u)) {
        if (u.hp < u.maxHp) u.hp = Math.min(u.maxHp, u.hp + u.maxHp * 0.15 * DT);
        if (u.mana < u.maxMana) u.mana = Math.min(u.maxMana, u.mana + u.maxMana * 0.15 * DT);
      } else {
        const f = FOUNTAIN_POS[1 - u.team];
        if (sq(u.x - f[0]) + sq(u.y - f[1]) < FOUNTAIN_RADIUS * FOUNTAIN_RADIUS) {
          this.damage(null, u, u.maxHp * 0.3 * DT + 20, TRUE_DMG, 3);
        }
      }
    }
  }

  /** Cheap state digest used by determinism tests. */
  hash(): number {
    let h = this.tick;
    for (const u of this.units) h = (Math.imul(h, 31) + ((u.x * 16) | 0) + ((u.y * 16) | 0) * 7 + ((u.hp * 4) | 0) * 13) | 0;
    for (const u of this.heroes) h = (Math.imul(h, 31) + ((u.hero!.totalGold * 4) | 0) + u.hero!.level * 101) | 0;
    return h >>> 0;
  }
}

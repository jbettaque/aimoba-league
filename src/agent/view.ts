/**
 * What one team's agent is told about a running game, and how its orders get into it.
 *
 * A `TeamGateway` sits between the simulation and the agent that controls one team's five heroes. Each decision
 * it can produce
 *
 *   - the `view`: everything the team can see right now as plain data (own heroes in full, enemies and creeps
 *     only while visible, with the last sighting of enemies in the fog), and / or
 *   - the fixed-size observation and action mask of the ML interface (ml.ts), one row per hero,
 *
 * and it turns the agent's orders into simulation commands. Nothing the fog of war hides leaves through here:
 * unit ids are per-team aliases handed out when a unit is first seen, and an order naming a unit the team cannot
 * see is dropped.
 *
 * With `mirror` (the default) Red's coordinates are swapped (x, y) -> (y, x), so every agent sees its own base
 * bottom-left at (480, 7520) and one piece of code plays both sides. Lanes keep their names under the mirror, and
 * the indexes into `info.map.structures` and `info.map.camps` are those of the mirrored map too: for Red as for
 * Blue, structures 0-8 and camps 0-3 are its own.
 */
import { ITEM_BY_ID, MAX_LEVEL, SIGHT_RANGE, TICK_RATE, xpForLevel, type Command, type Unit } from '../sim/defs.ts';
import type { Game } from '../sim/game.ts';
import { HEROES } from '../sim/heroes.ts';
import { CAMPS, STRUCTURES } from '../sim/map.ts';
import { FogMemory, MASK_SIZE, N_KINDS, N_TARGETS, OBS_SIZE, decodeAction, observe } from '../sim/ml.ts';

// --- The view --------------------------------------------------------------------------------------------------
export interface BuffView { kind: string; time: number; value: number }

export interface OwnHeroView {
  /** Unit id, for orders that target this hero (heals, shields). */
  id: number;
  /** Index within the team, 0-4 in roster order (korvath, sable, ignara, vesper, liora). This is the `hero` of an order. */
  index: number;
  hero: string;
  x: number; y: number; vx: number; vy: number; facing: number;
  alive: boolean;
  /** Seconds until respawn (0 while alive). */
  respawn: number;
  hp: number; maxHp: number; mana: number; maxMana: number; shield: number;
  level: number; xp: number; xpToNext: number;
  gold: number;
  skillPoints: number;
  /** Per ability slot (Q, W, E, R): rank, seconds of cooldown left, and whether it can be cast right now. */
  ranks: number[]; cooldowns: number[]; ready: boolean[];
  /** Which ability slots a skill point could go to right now. */
  canLevel: boolean[];
  items: string[];
  /** The item build the automatic shopper follows this game (one of the hero's builds). */
  build: string[];
  /** At the fountain or dead: `buy` orders work now. */
  canShop: boolean;
  ad: number; ap: number; armor: number; mr: number; attackSpeed: number; range: number; moveSpeed: number;
  stunned: boolean; rooted: boolean; stealthed: boolean;
  buffs: BuffView[];
  /** The order the hero is carrying out. It keeps running until another order replaces it. */
  order: { type: string; x: number; y: number; target: number; slot: number };
  /** Seconds left on a recall in progress (0 = not recalling). */
  recall: number;
  /** Seconds until the next basic attack can start, and whether one is winding up. */
  attackCooldown: number; attacking: boolean;
  /** Id of the unit being attacked (-1 = none). */
  target: number;
  inFountain: boolean;
  kills: number; deaths: number; assists: number; lastHits: number;
}

export interface EnemyHeroView {
  id: number;
  index: number;
  hero: string;
  /** Whether your team sees it right now. The fields from `x` to `dash` are only present while it does. */
  visible: boolean;
  alive: boolean;
  respawn: number;
  level: number;
  kills: number; deaths: number; assists: number;
  /** Where and how your team last saw it (at its fountain when it is dead or has not been seen yet). `age` in seconds. */
  lastSeen: { x: number; y: number; hp: number; age: number };
  x?: number; y?: number; vx?: number; vy?: number; facing?: number;
  hp?: number; maxHp?: number; mana?: number; maxMana?: number; shield?: number;
  ad?: number; ap?: number; armor?: number; mr?: number; attackSpeed?: number; range?: number; moveSpeed?: number;
  stunned?: boolean; rooted?: boolean;
  buffs?: BuffView[];
  items?: string[];
  target?: number;
  /** A charge or leap in progress: where it ends. */
  dash?: { x: number; y: number } | null;
}

export interface CreepView {
  id: number;
  /** 'minion' or 'monster'. */
  kind: string;
  /** melee, ranged, siege, super for minions; the camp type (stalker, golem, insight, might, wyrm, titan) for monsters. */
  type: string;
  /** True for your team's minions. Monsters are nobody's. */
  own: boolean;
  /** Lane index for minions (0 top, 1 mid, 2 bot), camp index for monsters. */
  lane: number;
  x: number; y: number;
  hp: number; maxHp: number;
  ad: number; armor: number; range: number;
  target: number;
}

export interface StructureView {
  id: number;
  /** Index into `info.map.structures`. */
  index: number;
  kind: string;
  own: boolean;
  lane: number;
  /** 0 outer tower, 1 inner tower, 2 guardian, 3 core. */
  tier: number;
  x: number; y: number;
  hp: number; maxHp: number;
  alive: boolean;
  /** Can be damaged now (the structures in front of it are down). */
  vulnerable: boolean;
  /** Outer tower in the opening minutes: takes much less damage. */
  fortified: boolean;
  target: number;
}

export interface ProjectileView {
  id: number;
  own: boolean;
  kind: string;
  x: number; y: number;
  /** Unit direction of travel and speed in units per second. */
  dx: number; dy: number; speed: number;
  radius: number;
  /** Distance it can still travel. */
  range: number;
  /** Id of the unit it homes in on (-1 for a skillshot, which flies straight). */
  target: number;
  /** Passes through units / only hits heroes. A shot that does neither is stopped by the first creep in its way. */
  pierce: boolean; heroOnly: boolean;
}

export interface ZoneView {
  id: number;
  own: boolean;
  kind: string;
  x: number; y: number; radius: number;
  /** Seconds until it goes off. */
  delay: number;
}

export interface CampView {
  /** Index into `info.map.camps`. */
  index: number;
  type: string;
  /** 'own', 'enemy' or 'river'. */
  side: string;
  x: number; y: number;
  /** Whether monsters are up. Known for your own jungle and the river; null for the enemy's jungle unless you see into it. */
  alive: boolean | null;
}

export interface TeamView {
  tick: number;
  /** Game time in seconds. */
  time: number;
  /** 0 = Blue, 1 = Red. With `mirror` the coordinates look the same either way. */
  team: number;
  /** [yours, theirs] */
  kills: [number, number];
  towersLost: [number, number];
  wyrmStacks: [number, number];
  heroes: OwnHeroView[];
  enemies: EnemyHeroView[];
  /** Your minions, and the enemy minions and monsters your team can see. */
  creeps: CreepView[];
  structures: StructureView[];
  projectiles: ProjectileView[];
  zones: ZoneView[];
  camps: CampView[];
}

// --- Orders ----------------------------------------------------------------------------------------------------
/**
 * An order for one hero. `hero` is its index within the team (0-4). Coordinates are in the frame of the view.
 *
 *   { hero, type: 'move', x, y }                 walk there (the game finds the path)
 *   { hero, type: 'attack_move', x, y }          walk there, attacking what comes into range
 *   { hero, type: 'attack', target }             attack a unit until it dies or is lost from sight
 *   { hero, type: 'cast', slot, x, y, target }   slot 0-3 or 'Q' 'W' 'E' 'R'; x/y for aimed spells, target for unit-targeted ones
 *   { hero, type: 'stop' } / { hero, type: 'recall' }
 *   { hero, type: 'level', slot }                spend a skill point (needed when autoLevel is off)
 *   { hero, type: 'buy', item }                  buy an item at the fountain or while dead (needed when autoBuy is off)
 *   { hero, type: 'action', action: [kind, direction, distance, target] }   an action of the ML interface
 */
export interface Order {
  hero: number;
  type: string;
  x?: number;
  y?: number;
  target?: number;
  slot?: number | string;
  item?: string;
  action?: number[];
}

export interface GatewayOptions {
  /** Present Red's coordinates mirrored, so the own base is always bottom-left (default true). */
  mirror?: boolean;
}

const r1 = (v: number) => Math.round(v * 10) / 10;
const r2 = (v: number) => Math.round(v * 100) / 100;
const r3 = (v: number) => Math.round(v * 1000) / 1000;

export class TeamGateway {
  readonly game: Game;
  readonly team: number;
  readonly mirror: boolean;
  /** The five hero slots of this team in the game (0-4 or 5-9). */
  readonly slots: number[];
  readonly memory = new FogMemory();
  /** ML observation and mask of the five heroes, refreshed by `observeMl`. */
  readonly obs = new Float32Array(5 * OBS_SIZE);
  readonly mask = new Uint8Array(5 * MASK_SIZE);
  private targets = [0, 1, 2, 3, 4].map(() => new Int32Array(N_TARGETS));
  private mlFresh = -1;
  /** Real unit id -> the alias this team knows it by, and back. */
  private alias = new Map<number, number>();
  private real: number[] = [0];
  private shotAlias = new Map<number, number>();
  private nextShot = 1;
  /** Orders dropped so far, and why (the first few of each decision are passed on to the agent's log). */
  rejected = 0;

  constructor(game: Game, team: number, opts: GatewayOptions = {}) {
    this.game = game;
    this.team = team;
    this.mirror = (opts.mirror ?? true) && team === 1;
    this.slots = [0, 1, 2, 3, 4].map((k) => team * 5 + k);
    this.memory.reset(game);
    // Heroes and structures get their aliases up front, in an order that is the same for both teams' point of view.
    for (const s of this.slots) this.id(game.heroes[s]);
    for (let k = 0; k < 5; k++) this.id(game.heroes[(1 - team) * 5 + k]);
    const n = STRUCTURES.length / 2;
    for (let k = 0; k < n; k++) this.id(game.structures[team * n + k]);
    for (let k = 0; k < n; k++) this.id(game.structures[(1 - team) * n + k]);
  }

  /** Call after every simulation tick. */
  onTick() {
    this.memory.update(this.game);
  }

  private id(u: Unit): number {
    let a = this.alias.get(u.id);
    if (a === undefined) {
      a = this.real.length;
      this.alias.set(u.id, a);
      this.real.push(u.id);
    }
    return a;
  }

  /** Alias of a unit the agent may know about, -1 otherwise. */
  private idOf(realId: number): number {
    if (realId < 0) return -1;
    const u = this.game.byId.get(realId);
    if (!u || !this.game.canSee(this.team, u)) return -1;
    return this.id(u);
  }

  /** The unit an agent means by `alias`, if it exists and the team can see it. */
  private unit(alias: unknown): Unit | undefined {
    if (typeof alias !== 'number' || !Number.isInteger(alias) || alias <= 0 || alias >= this.real.length) return undefined;
    const u = this.game.byId.get(this.real[alias]);
    return u && u.alive && this.game.canSee(this.team, u) ? u : undefined;
  }

  private px = (x: number, y: number) => r1(this.mirror ? y : x);
  private py = (x: number, y: number) => r1(this.mirror ? x : y);
  private ang = (a: number) => {
    let v = this.mirror ? Math.PI / 2 - a : a;
    if (v > Math.PI) v -= 2 * Math.PI;
    else if (v <= -Math.PI) v += 2 * Math.PI;
    return r3(v);
  };

  /** Whether the team has sight of a point (heroes and structures see SIGHT_RANGE, minions 800). */
  private sees(x: number, y: number): boolean {
    for (const o of this.game.unitsIn(x, y, SIGHT_RANGE)) {
      if (o.team !== this.team) continue;
      if (o.kind !== 'minion') return true;
      const dx = o.x - x, dy = o.y - y;
      if (dx * dx + dy * dy <= 800 * 800) return true;
    }
    return false;
  }

  private buffs(u: Unit): BuffView[] {
    const out: BuffView[] = [];
    for (const b of u.buffs) out.push({ kind: b.kind, time: r2(b.t), value: r2(b.val) });
    return out;
  }

  /** The team's view of the game right now. */
  view(): TeamView {
    const g = this.game, team = this.team, px = this.px, py = this.py;
    const heroes: OwnHeroView[] = [];
    for (let k = 0; k < 5; k++) {
      const u = g.heroes[team * 5 + k], h = u.hero!;
      const o = h.order;
      heroes.push({
        id: this.id(u), index: k, hero: HEROES[h.def].id,
        x: px(u.x, u.y), y: py(u.x, u.y), vx: px(u.vx, u.vy), vy: py(u.vx, u.vy), facing: this.ang(u.facing),
        alive: u.alive, respawn: u.alive ? 0 : r2(Math.max(0, h.respawn)),
        hp: r1(u.alive ? u.hp : 0), maxHp: r1(u.maxHp), mana: r1(u.mana), maxMana: r1(u.maxMana), shield: r1(u.shield),
        level: h.level, xp: r1(h.xp), xpToNext: h.level >= MAX_LEVEL ? 0 : xpForLevel(h.level),
        gold: Math.floor(h.gold), skillPoints: h.skillPoints,
        ranks: h.ranks.slice(), cooldowns: h.cds.map(r2), ready: [0, 1, 2, 3].map((s) => g.castError(u, s) === ''),
        canLevel: [0, 1, 2, 3].map((s) => g.canLevel(u, s)),
        items: h.items.slice(), build: h.build.slice(), canShop: !u.alive || g.inFountain(u),
        ad: r1(u.ad), ap: r1(u.ap), armor: r1(u.armor), mr: r1(u.mr), attackSpeed: r3(u.as), range: r1(u.range), moveSpeed: r1(u.ms),
        stunned: u.stunned, rooted: u.rooted, stealthed: u.stealthed, buffs: this.buffs(u),
        order: { type: o.type === 'amove' ? 'attack_move' : o.type, x: px(o.x, o.y), y: py(o.x, o.y), target: o.type === 'attack' || o.type === 'cast' ? this.idOf(o.targetId) : -1, slot: o.type === 'cast' ? o.slot : -1 },
        recall: o.type === 'recall' ? r2(h.recallT) : 0,
        attackCooldown: r2(Math.max(0, u.attackCd)), attacking: u.windup > 0, target: this.idOf(u.windup > 0 ? u.windupTarget : u.targetId),
        inFountain: g.inFountain(u), kills: h.kills, deaths: h.deaths, assists: h.assists, lastHits: h.cs,
      });
    }

    const enemies: EnemyHeroView[] = [];
    const mem = this.memory;
    for (let k = 0; k < 5; k++) {
      const u = g.heroes[(1 - team) * 5 + k], h = u.hero!, s = h.slot;
      const visible = u.alive && g.canSee(team, u);
      const e: EnemyHeroView = {
        id: this.id(u), index: k, hero: HEROES[h.def].id, visible, alive: u.alive, respawn: u.alive ? 0 : r2(Math.max(0, h.respawn)),
        level: h.level, kills: h.kills, deaths: h.deaths, assists: h.assists,
        lastSeen: { x: px(mem.x[s], mem.y[s]), y: py(mem.x[s], mem.y[s]), hp: r1(mem.hp[s]), age: r2((g.tick - mem.tick[s]) / TICK_RATE) },
      };
      if (visible) {
        e.x = px(u.x, u.y); e.y = py(u.x, u.y); e.vx = px(u.vx, u.vy); e.vy = py(u.vx, u.vy); e.facing = this.ang(u.facing);
        e.hp = r1(u.hp); e.maxHp = r1(u.maxHp); e.mana = r1(u.mana); e.maxMana = r1(u.maxMana); e.shield = r1(u.shield);
        e.ad = r1(u.ad); e.ap = r1(u.ap); e.armor = r1(u.armor); e.mr = r1(u.mr); e.attackSpeed = r3(u.as); e.range = r1(u.range); e.moveSpeed = r1(u.ms);
        e.stunned = u.stunned; e.rooted = u.rooted; e.buffs = this.buffs(u); e.items = h.items.slice();
        e.target = this.idOf(u.windup > 0 ? u.windupTarget : h.order.type === 'attack' ? h.order.targetId : u.targetId);
        e.dash = u.dash ? { x: px(u.dash.x, u.dash.y), y: py(u.dash.x, u.dash.y) } : null;
        e.lastSeen = { x: e.x, y: e.y, hp: e.hp, age: 0 };
      }
      enemies.push(e);
    }

    const creeps: CreepView[] = [];
    const campSeen = new Uint8Array(CAMPS.length);
    for (const u of g.units) {
      if (!u.alive || (u.kind !== 'minion' && u.kind !== 'monster')) continue;
      if (u.team !== team && !g.canSee(team, u)) continue;
      if (u.kind === 'monster' && u.camp >= 0) campSeen[u.camp] = 1;
      creeps.push({
        id: this.id(u), kind: u.kind, type: u.subtype, own: u.team === team, lane: u.kind === 'minion' ? u.lane : this.campIndex(u.camp),
        x: px(u.x, u.y), y: py(u.x, u.y), hp: r1(u.hp), maxHp: r1(u.maxHp), ad: r1(u.ad), armor: r1(u.armor), range: r1(u.range),
        target: this.idOf(u.windup > 0 ? u.windupTarget : u.targetId),
      });
    }

    const structures: StructureView[] = [];
    const n = STRUCTURES.length / 2;
    for (const t of [team, 1 - team]) {
      for (let k = 0; k < n; k++) {
        const u = g.structures[t * n + k];
        structures.push({
          id: this.id(u), index: (this.mirror ? (t === team ? 0 : n) : t * n) + k, kind: u.kind, own: t === team, lane: u.lane, tier: u.tier, x: px(u.x, u.y), y: py(u.x, u.y),
          hp: r1(u.alive ? u.hp : 0), maxHp: r1(u.maxHp), alive: u.alive, vulnerable: u.alive && u.vulnerable, fortified: u.alive && u.fortified,
          target: u.alive && u.kind === 'tower' ? this.idOf(u.targetId) : -1,
        });
      }
    }

    const projectiles: ProjectileView[] = [];
    for (const p of g.projectiles) {
      if (p.team !== team && !this.sees(p.x, p.y)) continue;
      projectiles.push({
        id: this.shot(p.id), own: p.team === team, kind: p.kind, x: px(p.x, p.y), y: py(p.x, p.y), dx: r3(this.mirror ? p.dy : p.dx), dy: r3(this.mirror ? p.dx : p.dy),
        speed: r1(p.speed), radius: r1(p.radius), range: r1(p.dist), target: this.idOf(p.targetId), pierce: p.pierce, heroOnly: p.heroOnly,
      });
    }
    const zones: ZoneView[] = [];
    for (const z of g.zones) {
      if (z.team !== team && !this.sees(z.x, z.y)) continue;
      zones.push({ id: this.shot(-z.id), own: z.team === team, kind: z.kind, x: px(z.x, z.y), y: py(z.x, z.y), radius: r1(z.radius), delay: r2(Math.max(0, z.delay)) });
    }

    const camps: CampView[] = CAMPS.map((_, index) => {
      const real = this.campIndex(index), c = CAMPS[real];
      // Like the ML observation: the own jungle and the river are known, the enemy's jungle only where the team can see.
      const known = c.side === team || c.side === 2;
      const alive = known ? g.campAlive[real] > 0 : campSeen[real] ? true : this.sees(c.x, c.y) ? g.campAlive[real] > 0 : null;
      return { index, type: c.type, side: c.side === 2 ? 'river' : c.side === team ? 'own' : 'enemy', x: px(c.x, c.y), y: py(c.x, c.y), alive };
    });

    return {
      tick: g.tick, time: r2(g.time), team,
      kills: [g.teamKills[team], g.teamKills[1 - team]], towersLost: [g.towersLost[team], g.towersLost[1 - team]], wyrmStacks: [g.wyrmStacks[team], g.wyrmStacks[1 - team]],
      heroes, enemies, creeps, structures, projectiles, zones, camps,
    };
  }

  /**
   * Camp index between the game and the view (the mapping is its own inverse). The jungle camps come in two
   * blocks of four, Blue's then Red's; under the mirror the blocks swap. The two river camps stay where they are.
   */
  private campIndex(i: number): number {
    return this.mirror && i >= 0 && i < 8 ? (i + 4) % 8 : i;
  }

  private shot(key: number): number {
    let a = this.shotAlias.get(key);
    if (a === undefined) {
      a = this.nextShot++;
      this.shotAlias.set(key, a);
      if (this.shotAlias.size > 4096) this.shotAlias.delete(this.shotAlias.keys().next().value!);
    }
    return a;
  }

  /** Refresh `obs` and `mask` (the ML interface) for the five heroes. */
  observeMl() {
    const g = this.game;
    if (this.mlFresh === g.tick) return;
    for (let k = 0; k < 5; k++) observe(g, g.heroes[this.slots[k]], this.obs, k * OBS_SIZE, this.mask, k * MASK_SIZE, this.targets[k], this.memory);
    this.mlFresh = g.tick;
  }

  /**
   * Hand the agent's orders to the game. Returns a message for every order that was dropped (malformed, or naming
   * something the team cannot see); the rest take effect on the next tick.
   */
  apply(orders: unknown, maxOrders = 40): string[] {
    const errors: string[] = [];
    if (orders === undefined || orders === null) return errors;
    if (!Array.isArray(orders)) return ['orders must be a list'];
    const g = this.game;
    let n = 0;
    for (const raw of orders) {
      if (++n > maxOrders) { errors.push(`more than ${maxOrders} orders in one decision; the rest were dropped`); break; }
      const err = this.order(g, raw);
      if (err) errors.push(err);
    }
    this.rejected += errors.length;
    return errors;
  }

  /** One row of ML actions for all five heroes: 20 integers, [kind, direction, distance, target] per hero. */
  applyActions(actions: unknown): string[] {
    const flat = Array.isArray(actions) ? (actions as unknown[]).flat() : null;
    if (!flat || flat.length !== 20 || flat.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
      this.rejected++;
      return ['actions must be 5 x [kind, direction, distance, target] integers'];
    }
    this.observeMl();
    const g = this.game;
    for (let k = 0; k < 5; k++) {
      const a = flat as number[];
      const c = decodeAction(g, g.heroes[this.slots[k]], a[k * 4] | 0, a[k * 4 + 1] | 0, a[k * 4 + 2] | 0, a[k * 4 + 3] | 0, this.targets[k]);
      if (c) g.command(this.slots[k], c);
    }
    return [];
  }

  private order(g: Game, raw: unknown): string | null {
    if (typeof raw !== 'object' || raw === null) return 'an order must be an object';
    const o = raw as Order;
    const k = o.hero;
    if (typeof k !== 'number' || !Number.isInteger(k) || k < 0 || k > 4) return `order.hero must be 0-4, got ${JSON.stringify(k)}`;
    const slot = this.slots[k];
    const u = g.heroes[slot];
    const point = (): [number, number] | null => {
      const x = o.x, y = o.y;
      if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null;
      return this.mirror ? [y, x] : [x, y];
    };
    let cmd: Command;
    switch (o.type) {
      case 'move':
      case 'attack_move': {
        const p = point();
        if (!p) return `${o.type} needs finite x and y`;
        cmd = { t: o.type === 'move' ? 'move' : 'amove', x: p[0], y: p[1] };
        break;
      }
      case 'attack': {
        const t = this.unit(o.target);
        if (!t) return `attack: no visible unit with id ${JSON.stringify(o.target)}`;
        if (t.team === this.team) return 'attack: the target is on your team';
        cmd = { t: 'attack', id: t.id };
        break;
      }
      case 'stop': cmd = { t: 'stop' }; break;
      case 'recall': cmd = { t: 'recall' }; break;
      case 'cast': {
        const s = abilitySlot(o.slot);
        if (s < 0) return `cast: slot must be 0-3 or Q, W, E, R, got ${JSON.stringify(o.slot)}`;
        const tg = HEROES[u.hero!.def].abilities[s].targeting;
        const t = o.target === undefined || o.target === null || o.target === -1 ? undefined : this.unit(o.target);
        if (o.target !== undefined && o.target !== null && o.target !== -1 && !t) return `cast: no visible unit with id ${JSON.stringify(o.target)}`;
        if (tg === 'self') cmd = { t: 'cast', slot: s, x: u.x, y: u.y };
        else if (tg === 'ally') cmd = { t: 'cast', slot: s, x: u.x, y: u.y, id: t && t.team === this.team ? t.id : u.id };
        else if (tg === 'enemy' || tg === 'enemyHero') {
          if (!t) return `cast: ${'QWER'[s]} of ${HEROES[u.hero!.def].id} needs a target`;
          cmd = { t: 'cast', slot: s, x: u.x, y: u.y, id: t.id };
        } else {
          const p = t ? [t.x, t.y] : point();
          if (!p) return `cast: ${'QWER'[s]} of ${HEROES[u.hero!.def].id} needs x and y (or a target to aim at)`;
          cmd = { t: 'cast', slot: s, x: p[0], y: p[1] };
        }
        break;
      }
      case 'level': {
        const s = abilitySlot(o.slot);
        if (s < 0) return `level: slot must be 0-3 or Q, W, E, R, got ${JSON.stringify(o.slot)}`;
        cmd = { t: 'level', slot: s };
        break;
      }
      case 'buy': {
        if (typeof o.item !== 'string' || !(o.item in ITEM_BY_ID)) return `buy: unknown item ${JSON.stringify(o.item)}`;
        cmd = { t: 'buy', item: o.item };
        break;
      }
      case 'action': {
        const a = o.action;
        if (!Array.isArray(a) || a.length !== 4 || a.some((v) => typeof v !== 'number' || !Number.isFinite(v))) return 'action must be [kind, direction, distance, target]';
        this.observeMl();
        if ((a[0] | 0) < 0 || (a[0] | 0) >= N_KINDS) return `action kind must be 0-${N_KINDS - 1}`;
        const c = decodeAction(g, u, a[0] | 0, a[1] | 0, a[2] | 0, a[3] | 0, this.targets[k]);
        if (c) g.command(slot, c);
        return null;
      }
      default:
        return `unknown order type ${JSON.stringify(o.type)}`;
    }
    g.command(slot, cmd);
    return null;
  }
}

function abilitySlot(v: unknown): number {
  if (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= 3) return v;
  if (typeof v === 'string' && v.length === 1) return 'QWER'.indexOf(v.toUpperCase());
  return -1;
}

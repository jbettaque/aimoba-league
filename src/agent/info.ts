/**
 * Everything about the game that does not change during a match, as plain JSON: the map, the heroes and their
 * abilities, items, unit stats, the rules of a match and the layout of the fixed-size observation. An agent
 * receives it once, in the `hello` message, so it never needs the simulation's source to know the numbers.
 */
import {
  FORTIFY_REDUCTION, FORTIFY_UNTIL, HERO_VS_MINION, ITEMS, MAX_ITEMS, MAX_LEVEL, MINION_STATS, MINION_VS_MINION, MONSTER_STATS, PASSIVE_GOLD, PROXIMITY_GOLD,
  PROXIMITY_GOLD_RANGE, SIGHT_RANGE, TICK_RATE, TOWER_AD, TOWER_GOLD_GLOBAL, TOWER_GOLD_LOCAL, TOWER_GOLD_RANGE, TOWER_HP, TOWER_PLATES, TOWER_PLATE_GOLD, TOWER_RANGE,
  XP_RANGE, XP_SHARE_GROUP, XP_SHARE_TWO, xpForLevel,
} from '../sim/defs.ts';
import { HEROES } from '../sim/heroes.ts';
import { CAMPS, CELL, CORE_POS, FOUNTAIN_POS, FOUNTAIN_RADIUS, GRID_N, LANES, LANE_NAMES, MAP_SIZE, STRUCTURES, blocked } from '../sim/map.ts';
import { ACTION_HEADS, ACTION_NVEC, ACTION_TARGETS, MASK_SIZE, OBS_BLOCKS, OBS_HERO_OFFSET, OBS_SIZE } from '../sim/ml.ts';

/** Version of the agent protocol and of the view / order formats. Raised when an agent written for the old one would break. */
export const PROTOCOL_VERSION = 1;

/** Limits a match is played under. The league publishes its values; local matches use these defaults unless told otherwise. */
export interface MatchRules {
  /** Game time limit in seconds. A game that reaches it is decided on structures, then kills (see `judge` in match.ts). */
  maxTime: number;
  /** Fewest and most sim ticks between two decisions an agent may ask for (30 ticks = 1 second). */
  minDecisionTicks: number;
  maxDecisionTicks: number;
  /** Thinking time: wall-clock milliseconds per second of game time, spread over the agent's decisions. */
  msPerGameSecond: number;
  /** Extra milliseconds for the whole game, drawn on whenever a decision takes longer than its share. Empty bank = forfeit. */
  bankMs: number;
  /** Milliseconds from start-up until the agent must have answered `hello`. */
  initMs: number;
  /** Most orders an agent may send in one decision; the rest are dropped. */
  maxOrders: number;
  /** Bytes of an agent's stderr kept as its match log. */
  logBytes: number;
  /** Largest single message an agent may send, in bytes. */
  maxMessageBytes: number;
}

export const DEFAULT_RULES: MatchRules = {
  maxTime: 30 * 60,
  minDecisionTicks: 3,
  maxDecisionTicks: 30,
  msPerGameSecond: 250,
  bankMs: 10_000,
  initMs: 15_000,
  maxOrders: 40,
  logBytes: 256 * 1024,
  maxMessageBytes: 1024 * 1024,
};

/** Thinking time an agent gets for one decision when it decides every `decisionTicks` ticks. */
export const stepBudgetMs = (rules: MatchRules, decisionTicks: number) => (rules.msPerGameSecond * decisionTicks) / TICK_RATE;

let cached: GameInfo | null = null;

export interface GameInfo {
  protocol: number;
  tickRate: number;
  map: {
    size: number;
    /** Walkability grid: `grid.n` x `grid.n` cells of `grid.cell` units; `grid.blocked[y * n + x]` is '1' for a wall. */
    grid: { n: number; cell: number; blocked: string };
    fountains: [number, number][];
    fountainRadius: number;
    cores: [number, number][];
    /** Lane waypoints from Blue's base to Red's core: top, mid, bot. Minions walk these. */
    lanes: { name: string; points: [number, number][] }[];
    structures: { index: number; kind: string; team: number; lane: number; tier: number; x: number; y: number }[];
    camps: { index: number; type: string; side: number; x: number; y: number; firstSpawn: number; respawn: number }[];
  };
  heroes: {
    index: number; id: string; name: string; title: string; role: string; lane: number; passive: { name: string; desc: string };
    base: Record<string, number>; perLevel: Record<string, number>;
    abilities: { slot: number; key: string; name: string; desc: string; targeting: string; range: number; radius: number; cooldown: number[]; cost: number[] }[];
    skillOrder: number[]; build: string[]; altBuilds: string[][];
  }[];
  items: { id: string; name: string; cost: number; desc: string; stats: Record<string, number>; from: string[]; passive: [string, string] | null }[];
  units: { minions: typeof MINION_STATS; monsters: typeof MONSTER_STATS; towerHp: number[]; towerAd: number[]; towerRange: number };
  constants: Record<string, number>;
  /** Experience needed to go from level L to L+1, for L = 1 .. maxLevel-1. */
  xpToLevel: number[];
  /** The fixed-size observation and the MultiDiscrete action of the ML interface (see docs/league/ml.md). */
  ml: {
    obsSize: number; maskSize: number; actionNvec: number[]; obsBlocks: [string, number, number][]; heroOffset: number;
    actionKinds: string[]; actionHeads: number[][][]; actionTargets: number[][][];
  };
  rules: MatchRules;
}

/** The static game description. `rules` are the limits this match is played under. */
export function gameInfo(rules: MatchRules = DEFAULT_RULES): GameInfo {
  if (!cached) {
    let grid = '';
    for (let i = 0; i < blocked.length; i++) grid += blocked[i] ? '1' : '0';
    cached = {
      protocol: PROTOCOL_VERSION,
      tickRate: TICK_RATE,
      map: {
        size: MAP_SIZE,
        grid: { n: GRID_N, cell: CELL, blocked: grid },
        fountains: FOUNTAIN_POS.map((p) => [p[0], p[1]]),
        fountainRadius: FOUNTAIN_RADIUS,
        cores: CORE_POS.map((p) => [p[0], p[1]]),
        lanes: LANES.map((points, i) => ({ name: LANE_NAMES[i], points: points.map((p) => [p[0], p[1]] as [number, number]) })),
        structures: STRUCTURES.map((s, index) => ({ index, kind: s.kind, team: s.team, lane: s.lane, tier: s.tier, x: s.x, y: s.y })),
        camps: CAMPS.map((c, index) => ({ index, type: c.type, side: c.side, x: c.x, y: c.y, firstSpawn: c.firstSpawn, respawn: c.respawn })),
      },
      heroes: HEROES.map((h, index) => ({
        index, id: h.id, name: h.name, title: h.title, role: h.role, lane: h.lane, passive: h.passive,
        base: { hp: h.hp, mana: h.mana, ad: h.ad, armor: h.armor, mr: h.mr, as: h.as, range: h.range, ms: h.ms, hpRegen: h.hpRegen, manaRegen: h.manaRegen },
        perLevel: { hp: h.hpL, mana: h.manaL, ad: h.adL, armor: h.armorL, mr: h.mrL, as: h.asL },
        abilities: h.abilities.map((a, slot) => ({ slot, key: 'QWER'[slot], name: a.name, desc: a.desc, targeting: a.targeting, range: a.range, radius: a.radius, cooldown: a.cd, cost: a.cost })),
        skillOrder: h.skillOrder, build: h.build, altBuilds: h.altBuilds,
      })),
      items: ITEMS.map((it) => ({ id: it.id, name: it.name, cost: it.cost, desc: it.desc, stats: it.stats as Record<string, number>, from: it.from ?? [], passive: it.passive ?? null })),
      units: { minions: MINION_STATS, monsters: MONSTER_STATS, towerHp: TOWER_HP, towerAd: TOWER_AD, towerRange: TOWER_RANGE },
      constants: {
        maxLevel: MAX_LEVEL, maxItems: MAX_ITEMS, sightRange: SIGHT_RANGE, minionSightRange: 800, xpRange: XP_RANGE, fortifyUntil: FORTIFY_UNTIL,
        fortifyReduction: FORTIFY_REDUCTION, heroVsMinion: HERO_VS_MINION, minionVsMinion: MINION_VS_MINION, passiveGold: PASSIVE_GOLD,
        proximityGold: PROXIMITY_GOLD, proximityGoldRange: PROXIMITY_GOLD_RANGE, towerPlates: TOWER_PLATES, towerPlateGold: TOWER_PLATE_GOLD,
        towerGoldLocal: TOWER_GOLD_LOCAL, towerGoldGlobal: TOWER_GOLD_GLOBAL, towerGoldRange: TOWER_GOLD_RANGE, xpShareTwo: XP_SHARE_TWO, xpShareGroup: XP_SHARE_GROUP,
        firstWave: 12, waveInterval: 28, recallSeconds: 5, startGold: 650,
      },
      xpToLevel: Array.from({ length: MAX_LEVEL - 1 }, (_, i) => xpForLevel(i + 1)),
      ml: {
        obsSize: OBS_SIZE, maskSize: MASK_SIZE, actionNvec: ACTION_NVEC, obsBlocks: OBS_BLOCKS, heroOffset: OBS_HERO_OFFSET,
        actionKinds: ['noop', 'move', 'attack', 'Q', 'W', 'E', 'R', 'recall'], actionHeads: ACTION_HEADS, actionTargets: ACTION_TARGETS,
      },
      rules: DEFAULT_RULES,
    };
  }
  return { ...cached, rules };
}

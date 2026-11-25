/**
 * Helpers handed to JavaScript bots as `ctx.lib`: geometry, map queries, shopping maths and order builders.
 *
 * Coordinates are those of the view. With `mirror` on (the default) that is the Blue frame for both sides, and
 * because the map is symmetric the map queries here are right either way. With `mirror` off, a Red bot must count
 * lane positions from the far end itself (`laneLength[lane] - s`).
 */
import { ITEM_BY_ID, canAfford, itemPrice, missingParts, nextPurchase } from '../sim/defs.ts';
import { LANE_LENGTH, MAP_SIZE, findPath, isBlocked, laneProgress, lanePoint, lineClear, nearestWalkable } from '../sim/map.ts';
import type { Order } from './view.ts';

export interface Point { x: number; y: number }
type HeroRef = number | { index: number };
type UnitRef = number | { id: number };

const heroIndex = (h: HeroRef) => (typeof h === 'number' ? h : h.index);
const unitId = (u: UnitRef) => (typeof u === 'number' ? u : u.id);

export const lib = {
  mapSize: MAP_SIZE,

  // --- Geometry -----------------------------------------------------------------------------------------------
  dist(a: Point, b: Point): number {
    return Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  },
  /** The point `d` units from `from` in the direction of `to` (not past `to` unless `overshoot`). */
  towards(from: Point, to: Point, d: number, overshoot = false): Point {
    const dx = to.x - from.x, dy = to.y - from.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1e-6 || (!overshoot && len <= d)) return { x: to.x, y: to.y };
    return { x: from.x + (dx / len) * d, y: from.y + (dy / len) * d };
  },
  /** The point `d` units from `from`, directly away from `threat`. */
  away(from: Point, threat: Point, d: number): Point {
    const dx = from.x - threat.x, dy = from.y - threat.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1e-6) return { x: from.x - d * 0.7071, y: from.y + d * 0.7071 };
    return { x: from.x + (dx / len) * d, y: from.y + (dy / len) * d };
  },
  /** The entry of `list` closest to `p` (undefined for an empty list). */
  nearest<T extends Point>(p: Point, list: T[]): T | undefined {
    let best: T | undefined, bd = Infinity;
    for (const o of list) {
      const d = (o.x - p.x) * (o.x - p.x) + (o.y - p.y) * (o.y - p.y);
      if (d < bd) { bd = d; best = o; }
    }
    return best;
  },
  /** The entries of `list` within `r` of `p`. */
  within<T extends Point>(p: Point, list: T[], r: number): T[] {
    return list.filter((o) => (o.x - p.x) * (o.x - p.x) + (o.y - p.y) * (o.y - p.y) <= r * r);
  },
  /** Where something moving at (vx, vy) will be when a shot of `speed` fired now from `from` reaches it. */
  lead(from: Point, target: Point & { vx?: number; vy?: number }, speed: number, delay = 0): Point {
    const vx = target.vx ?? 0, vy = target.vy ?? 0;
    let x = target.x, y = target.y;
    for (let i = 0; i < 3; i++) {
      const t = delay + (speed > 0 ? Math.sqrt((x - from.x) * (x - from.x) + (y - from.y) * (y - from.y)) / speed : 0);
      x = target.x + vx * t;
      y = target.y + vy * t;
    }
    return { x, y };
  },

  // --- Map ----------------------------------------------------------------------------------------------------
  isWalkable: (x: number, y: number) => !isBlocked(x, y),
  /** True when the straight line between the two points crosses no wall. */
  lineClear,
  /** A walkable path as a list of points (empty when there is none). Heroes find their own way on a `move`; this is for planning. */
  findPath(from: Point, to: Point): Point[] {
    const flat = findPath(from.x, from.y, to.x, to.y);
    const out: Point[] = [];
    for (let i = 0; i < flat.length; i += 2) out.push({ x: flat[i], y: flat[i + 1] });
    return out;
  },
  nearestWalkable(x: number, y: number): Point {
    const p = nearestWalkable(x, y);
    return { x: p[0], y: p[1] };
  },
  /** Length of each lane (0 top, 1 mid, 2 bot) from the Blue base to the Red core. */
  laneLength: LANE_LENGTH,
  /** The point `s` units along a lane, counted from the Blue base (your own base when `mirror` is on). */
  lanePoint(lane: number, s: number): Point {
    const p = lanePoint(lane, Math.max(0, Math.min(LANE_LENGTH[lane], s)));
    return { x: p[0], y: p[1] };
  },
  /** How far along a lane the lane point nearest to (x, y) is. */
  laneProgress,

  // --- Combat and shopping ----------------------------------------------------------------------------------
  /** Damage left of `amount` after `resist` armor or magic resist. */
  afterResist: (amount: number, resist: number) => amount * (100 / (100 + Math.max(0, resist))),
  item: (id: string) => ITEM_BY_ID[id],
  /** Gold still needed for an item, counting the components already owned. */
  itemPrice,
  canAfford,
  /** Components of an item that are not owned yet. */
  missingParts,
  /** The next thing to buy toward a build with this gold (null = nothing affordable). */
  nextPurchase,

  // --- Orders -----------------------------------------------------------------------------------------------
  move: (hero: HeroRef, x: number, y: number): Order => ({ hero: heroIndex(hero), type: 'move', x, y }),
  attackMove: (hero: HeroRef, x: number, y: number): Order => ({ hero: heroIndex(hero), type: 'attack_move', x, y }),
  attack: (hero: HeroRef, target: UnitRef): Order => ({ hero: heroIndex(hero), type: 'attack', target: unitId(target) }),
  /** Cast an ability (slot 0-3 or 'Q' 'W' 'E' 'R') at a point `{x, y}`, at a unit, or with neither for a self-cast. */
  cast(hero: HeroRef, slot: number | string, at?: Point | UnitRef | null): Order {
    const o: Order = { hero: heroIndex(hero), type: 'cast', slot };
    if (typeof at === 'number') o.target = at;
    else if (at && 'id' in at && typeof at.id === 'number') o.target = at.id;
    else if (at && 'x' in at) { o.x = at.x; o.y = at.y; }
    return o;
  },
  stop: (hero: HeroRef): Order => ({ hero: heroIndex(hero), type: 'stop' }),
  recall: (hero: HeroRef): Order => ({ hero: heroIndex(hero), type: 'recall' }),
  level: (hero: HeroRef, slot: number | string): Order => ({ hero: heroIndex(hero), type: 'level', slot }),
  buy: (hero: HeroRef, item: string): Order => ({ hero: heroIndex(hero), type: 'buy', item }),
  /** An action of the ML interface for one hero: [kind, direction, distance, target]. */
  action: (hero: HeroRef, action: number[]): Order => ({ hero: heroIndex(hero), type: 'action', action }),
};

export type BotLib = typeof lib;

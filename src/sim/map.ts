/**
 * Map layout, walkability grid and pathfinding.
 *
 * The map is an 8000x8000 square. Blue's base sits bottom-left, Red's top-right.
 * Everything is defined for Blue and mirrored across the river diagonal (x,y)->(y,x),
 * which makes the map exactly fair for both teams.
 */
import { hypot } from './dmath.ts';
export const MAP_SIZE = 8000;
export const CELL = 80;
export const GRID_N = MAP_SIZE / CELL;

export type Pt = [number, number];

/** Mirror across the river: swaps the two teams. */
export const mirrorTeam = (p: Pt): Pt => [p[1], p[0]];
/** Mirror across the mid lane: swaps top and bottom side. */
const mirrorSide = (p: Pt): Pt => [MAP_SIZE - p[1], MAP_SIZE - p[0]];

export const LANE_TOP = 0;
export const LANE_MID = 1;
export const LANE_BOT = 2;
export const LANE_NAMES = ['top', 'mid', 'bot'];

const TOP: Pt[] = [[1000, 6300], [1000, 1700], [1250, 1250], [1700, 1000], [6300, 1000]];
const MID: Pt[] = [[1600, 6400], [6400, 1600]];
const BOT: Pt[] = TOP.map(mirrorSide);

export const CORE_POS: Pt[] = [[1100, 6900], [6900, 1100]];
export const FOUNTAIN_POS: Pt[] = [[480, 7520], [7520, 480]];
export const FOUNTAIN_RADIUS = 650;

/** Lane waypoints in Blue's travel direction (Blue base -> Red core). */
export const LANES: Pt[][] = [TOP, MID, BOT].map((l) => [...l, CORE_POS[1]]);

/** Waypoints a minion of `team` walks along `lane`. */
// Red walks the mirrored path; mirroring the top lane gives the top lane reversed.
const RED_LANES: Pt[][] = LANES.map((l) => l.map(mirrorTeam));
export function laneWaypoints(team: number, lane: number): Pt[] {
  return team === 0 ? LANES[lane] : RED_LANES[lane];
}

// --- Lane progress helpers (used by bots / observations) -------------------
const laneCum: number[][] = LANES.map((l) => {
  const c = [0];
  for (let i = 1; i < l.length; i++) c.push(c[i - 1] + hypot(l[i][0] - l[i - 1][0], l[i][1] - l[i - 1][1]));
  return c;
});
export const LANE_LENGTH = laneCum.map((c) => c[c.length - 1]);

/** Distance along the lane (Blue-oriented) of the lane point closest to (x,y). */
export function laneProgress(lane: number, x: number, y: number): number {
  const l = LANES[lane];
  let best = Infinity;
  let bestS = 0;
  for (let i = 0; i < l.length - 1; i++) {
    const ax = l[i][0], ay = l[i][1];
    const dx = l[i + 1][0] - ax, dy = l[i + 1][1] - ay;
    const len2 = dx * dx + dy * dy;
    let t = ((x - ax) * dx + (y - ay) * dy) / len2;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const px = ax + dx * t, py = ay + dy * t;
    const d = (px - x) * (px - x) + (py - y) * (py - y);
    if (d < best) {
      best = d;
      bestS = laneCum[lane][i] + Math.sqrt(len2) * t;
    }
  }
  return bestS;
}

/** Point on the lane at Blue-oriented distance s. */
export function lanePoint(lane: number, s: number): Pt {
  const l = LANES[lane];
  const c = laneCum[lane];
  if (s <= 0) return l[0];
  for (let i = 0; i < l.length - 1; i++) {
    if (s <= c[i + 1]) {
      const t = (s - c[i]) / (c[i + 1] - c[i]);
      return [l[i][0] + (l[i + 1][0] - l[i][0]) * t, l[i][1] + (l[i + 1][1] - l[i][1]) * t];
    }
  }
  return l[l.length - 1];
}

// --- Structures --------------------------------------------------------------
export interface StructureSpec {
  kind: 'tower' | 'core';
  team: number;
  x: number;
  y: number;
  lane: number; // -1 for guardians / core
  tier: number; // 0 outer, 1 inner, 2 guardian, 3 core
}

const BLUE_TOWERS: { p: Pt; lane: number; tier: number }[] = [
  { p: [1130, 2500], lane: LANE_TOP, tier: 0 },
  { p: [1130, 4600], lane: LANE_TOP, tier: 1 },
  { p: [3320, 4820], lane: LANE_MID, tier: 0 },
  { p: [2470, 5670], lane: LANE_MID, tier: 1 },
  { p: mirrorSide([1130, 2500]), lane: LANE_BOT, tier: 0 },
  { p: mirrorSide([1130, 4600]), lane: LANE_BOT, tier: 1 },
  { p: [1300, 6300], lane: -1, tier: 2 },
  { p: [1700, 6700], lane: -1, tier: 2 },
];

export const STRUCTURES: StructureSpec[] = [];
for (let team = 0; team < 2; team++) {
  const m = (p: Pt) => (team === 0 ? p : mirrorTeam(p));
  for (const t of BLUE_TOWERS) {
    const p = m(t.p);
    STRUCTURES.push({ kind: 'tower', team, x: p[0], y: p[1], lane: t.lane, tier: t.tier });
  }
  STRUCTURES.push({ kind: 'core', team, x: CORE_POS[team][0], y: CORE_POS[team][1], lane: -1, tier: 3 });
}

// --- Jungle camps ------------------------------------------------------------
export type CampType = 'insight' | 'might' | 'stalkers' | 'golem' | 'wyrm' | 'titan';
export interface CampSpec {
  type: CampType;
  x: number;
  y: number;
  /** Which team's jungle the camp is in (2 = river / contested). */
  side: number;
  firstSpawn: number;
  respawn: number;
}

const BLUE_CAMPS: { type: CampType; p: Pt }[] = [
  { type: 'insight', p: [2000, 3700] },
  { type: 'stalkers', p: [1900, 4700] },
  { type: 'might', p: mirrorSide([2000, 3700]) },
  { type: 'golem', p: mirrorSide([1900, 4700]) },
];

export const CAMPS: CampSpec[] = [];
for (let team = 0; team < 2; team++) {
  for (const c of BLUE_CAMPS) {
    const p = team === 0 ? c.p : mirrorTeam(c.p);
    CAMPS.push({ type: c.type, x: p[0], y: p[1], side: team, firstSpawn: 30, respawn: 70 });
  }
}
export const WYRM_POS: Pt = [5500, 5500];
export const TITAN_POS: Pt = [2500, 2500];
CAMPS.push({ type: 'wyrm', x: WYRM_POS[0], y: WYRM_POS[1], side: 2, firstSpawn: 150, respawn: 150 });
CAMPS.push({ type: 'titan', x: TITAN_POS[0], y: TITAN_POS[1], side: 2, firstSpawn: 360, respawn: 240 });

// --- Walkability grid --------------------------------------------------------
/** Carved walkable capsules: [x0, y0, x1, y1, radius]. Everything else is wall. */
export const CARVES: number[][] = [];
{
  const seg = (a: Pt, b: Pt, r: number) => CARVES.push([a[0], a[1], b[0], b[1], r]);
  const LANE_R = 340;
  for (const lane of [TOP, MID, BOT]) for (let i = 0; i < lane.length - 1; i++) seg(lane[i], lane[i + 1], LANE_R);
  seg([1000, 7000], [7000, 1000], LANE_R); // mid, base to base
  seg([1000, 1000], [7000, 7000], 250); // river
  seg(TITAN_POS, TITAN_POS, 520);
  seg(WYRM_POS, WYRM_POS, 520);
  seg([1250, 1250], [1250, 1250], 520);
  seg([6750, 6750], [6750, 6750], 520);
  seg([4000, 4000], [4000, 4000], 480);
  seg([1000, 7000], [1000, 7000], 1400); // bases
  seg([7000, 1000], [7000, 1000], 1400);
  const jungle: [Pt, Pt, number][] = [
    [[1000, 3700], [2000, 3700], 200],
    [[2000, 3700], [2850, 2950], 200],
    [[2000, 3700], [1900, 4700], 200],
    [[1900, 4700], [2700, 5300], 200],
    [[1000, 5400], [1900, 4700], 200],
    [[2000, 3700], [2000, 3700], 360],
    [[1900, 4700], [1900, 4700], 360],
  ];
  const id = (p: Pt) => p;
  const both = (p: Pt) => mirrorTeam(mirrorSide(p));
  for (const f of [id, mirrorSide, mirrorTeam, both]) for (const [a, b, r] of jungle) seg(f(a), f(b), r);
}

export const blocked = new Uint8Array(GRID_N * GRID_N);
{
  const EDGE = 160;
  for (let cy = 0; cy < GRID_N; cy++) {
    for (let cx = 0; cx < GRID_N; cx++) {
      const x = (cx + 0.5) * CELL, y = (cy + 0.5) * CELL;
      let open = false;
      if (x > EDGE && y > EDGE && x < MAP_SIZE - EDGE && y < MAP_SIZE - EDGE) {
        for (const c of CARVES) {
          const dx = c[2] - c[0], dy = c[3] - c[1];
          const len2 = dx * dx + dy * dy;
          let t = len2 > 0 ? ((x - c[0]) * dx + (y - c[1]) * dy) / len2 : 0;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const px = c[0] + dx * t - x, py = c[1] + dy * t - y;
          if (px * px + py * py <= c[4] * c[4]) {
            open = true;
            break;
          }
        }
      }
      blocked[cy * GRID_N + cx] = open ? 0 : 1;
    }
  }
}

export function isBlocked(x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= MAP_SIZE || y >= MAP_SIZE) return true;
  return blocked[((y / CELL) | 0) * GRID_N + ((x / CELL) | 0)] === 1;
}

/** 2-D prefix sums of `blocked`: blockedSum[y * (GRID_N + 1) + x] counts the blocked cells in [0, x) x [0, y). */
const blockedSum = new Int32Array((GRID_N + 1) * (GRID_N + 1));
for (let cy = 0; cy < GRID_N; cy++) {
  for (let cx = 0; cx < GRID_N; cx++) {
    const W = GRID_N + 1;
    blockedSum[(cy + 1) * W + cx + 1] = blocked[cy * GRID_N + cx] + blockedSum[cy * W + cx + 1] + blockedSum[(cy + 1) * W + cx] - blockedSum[cy * W + cx];
  }
}

/** True when no cell of the inclusive cell rectangle is blocked. */
function cellsOpen(cx0: number, cy0: number, cx1: number, cy1: number): boolean {
  const W = GRID_N + 1;
  return blockedSum[(cy1 + 1) * W + cx1 + 1] - blockedSum[cy0 * W + cx1 + 1] - blockedSum[(cy1 + 1) * W + cx0] + blockedSum[cy0 * W + cx0] === 0;
}

/**
 * Whether samples a..b of lineClear's walk are all walkable. Skips whole runs of samples whose bounding box holds no
 * wall (the samples are monotone in i, so they all lie inside the box of the first and last one); the answer is
 * exactly that of testing every sample.
 */
function samplesClear(x0: number, y0: number, dx: number, dy: number, n: number, a: number, b: number): boolean {
  const xa = x0 + dx * (a / n), ya = y0 + dy * (a / n), xb = x0 + dx * (b / n), yb = y0 + dy * (b / n);
  const lx = xa < xb ? xa : xb, hx = xa < xb ? xb : xa, ly = ya < yb ? ya : yb, hy = ya < yb ? yb : ya;
  if (lx >= 0 && ly >= 0 && hx < MAP_SIZE && hy < MAP_SIZE && cellsOpen((lx / CELL) | 0, (ly / CELL) | 0, (hx / CELL) | 0, (hy / CELL) | 0)) return true;
  if (b - a < 6) {
    for (let i = a; i <= b; i++) {
      const t = i / n;
      if (isBlocked(x0 + dx * t, y0 + dy * t)) return false;
    }
    return true;
  }
  const m = (a + b) >> 1;
  return samplesClear(x0, y0, dx, dy, n, a, m) && samplesClear(x0, y0, dx, dy, n, m + 1, b);
}

/** True when the straight segment crosses no wall (tested every 40 units along it). */
export function lineClear(x0: number, y0: number, x1: number, y1: number): boolean {
  const dx = x1 - x0, dy = y1 - y0;
  const n = Math.ceil(hypot(dx, dy) / 40);
  return n < 1 || samplesClear(x0, y0, dx, dy, n, 1, n);
}

/** lineClear with a safety margin on both sides, so paths do not graze wall corners. */
function lineClearWide(x0: number, y0: number, x1: number, y1: number): boolean {
  const dx = x1 - x0, dy = y1 - y0;
  const d = hypot(dx, dy);
  if (d < 1) return true;
  const nx = (-dy / d) * 45, ny = (dx / d) * 45;
  return (
    lineClear(x0, y0, x1, y1) &&
    lineClear(x0 + nx, y0 + ny, x1 + nx, y1 + ny) &&
    lineClear(x0 - nx, y0 - ny, x1 - nx, y1 - ny)
  );
}

/** Furthest point along the ray from (x0,y0) toward (x1,y1) that is still walkable. */
export function rayClamp(x0: number, y0: number, x1: number, y1: number): Pt {
  const dx = x1 - x0, dy = y1 - y0;
  const n = Math.ceil(hypot(dx, dy) / 20);
  let bx = x0, by = y0;
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const x = x0 + dx * t, y = y0 + dy * t;
    if (isBlocked(x, y)) break;
    bx = x;
    by = y;
  }
  return [bx, by];
}

/** Nearest walkable point to (x,y) (returns the input when already walkable). */
export function nearestWalkable(x: number, y: number): Pt {
  x = Math.min(MAP_SIZE - 1, Math.max(0, x));
  y = Math.min(MAP_SIZE - 1, Math.max(0, y));
  if (!isBlocked(x, y)) return [x, y];
  const cx = (x / CELL) | 0, cy = (y / CELL) | 0;
  for (let r = 1; r < GRID_N; r++) {
    let best = Infinity;
    let bp: Pt | null = null;
    for (let oy = -r; oy <= r; oy++) {
      for (let ox = -r; ox <= r; ox++) {
        if (Math.max(Math.abs(ox), Math.abs(oy)) !== r) continue;
        const nx = cx + ox, ny = cy + oy;
        if (nx < 0 || ny < 0 || nx >= GRID_N || ny >= GRID_N || blocked[ny * GRID_N + nx]) continue;
        const px = (nx + 0.5) * CELL, py = (ny + 0.5) * CELL;
        const d = (px - x) * (px - x) + (py - y) * (py - y);
        if (d < best) {
          best = d;
          bp = [px, py];
        }
      }
    }
    if (bp) return bp;
  }
  return [x, y];
}

// --- A* ----------------------------------------------------------------------
const NN = GRID_N * GRID_N;
const gScore = new Float32Array(NN);
const fScore = new Float32Array(NN);
const cameFrom = new Int32Array(NN);
const stamp = new Uint32Array(NN);
const closed = new Uint32Array(NN);
let curStamp = 0;
const heap: number[] = [];

function heapPush(n: number) {
  let i = heap.length;
  heap.push(n);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (fScore[heap[p]] <= fScore[n]) break;
    heap[i] = heap[p];
    i = p;
  }
  heap[i] = n;
}
function heapPop(): number {
  const top = heap[0];
  const last = heap.pop()!;
  const len = heap.length;
  if (len > 0) {
    let i = 0;
    for (;;) {
      let c = i * 2 + 1;
      if (c >= len) break;
      if (c + 1 < len && fScore[heap[c + 1]] < fScore[heap[c]]) c++;
      if (fScore[heap[c]] >= fScore[last]) break;
      heap[i] = heap[c];
      i = c;
    }
    heap[i] = last;
  }
  return top;
}

const NB = [1, 0, -1, 0, 0, 1, 0, -1, 1, 1, 1, -1, -1, 1, -1, -1];

/**
 * Find a walkable path. Returns a flat [x0,y0,x1,y1,...] list of waypoints
 * (excluding the start), smoothed by line of sight.
 */
export function findPath(sx: number, sy: number, gx: number, gy: number): number[] {
  [gx, gy] = nearestWalkable(gx, gy);
  if (lineClearWide(sx, sy, gx, gy)) return [gx, gy];
  const [wsx, wsy] = nearestWalkable(sx, sy);
  const start = ((wsy / CELL) | 0) * GRID_N + ((wsx / CELL) | 0);
  const goal = ((gy / CELL) | 0) * GRID_N + ((gx / CELL) | 0);
  if (start === goal) return [gx, gy];
  const gcx = goal % GRID_N, gcy = (goal / GRID_N) | 0;
  curStamp++;
  heap.length = 0;
  gScore[start] = 0;
  fScore[start] = 0;
  cameFrom[start] = -1;
  stamp[start] = curStamp;
  heapPush(start);
  let found = false;
  while (heap.length > 0) {
    const cur = heapPop();
    if (cur === goal) {
      found = true;
      break;
    }
    if (closed[cur] === curStamp) continue;
    closed[cur] = curStamp;
    const cx = cur % GRID_N, cy = (cur / GRID_N) | 0;
    for (let k = 0; k < 16; k += 2) {
      const ox = NB[k], oy = NB[k + 1];
      const nx = cx + ox, ny = cy + oy;
      if (nx < 0 || ny < 0 || nx >= GRID_N || ny >= GRID_N) continue;
      const ni = ny * GRID_N + nx;
      if (blocked[ni] || closed[ni] === curStamp) continue;
      if (ox !== 0 && oy !== 0 && (blocked[cy * GRID_N + nx] || blocked[ny * GRID_N + cx])) continue;
      const g = gScore[cur] + (ox !== 0 && oy !== 0 ? 1.4142 : 1);
      if (stamp[ni] !== curStamp || g < gScore[ni]) {
        stamp[ni] = curStamp;
        gScore[ni] = g;
        const hx = Math.abs(nx - gcx), hy = Math.abs(ny - gcy);
        fScore[ni] = g + (hx > hy ? hx + 0.4142 * hy : hy + 0.4142 * hx);
        cameFrom[ni] = cur;
        heapPush(ni);
      }
    }
  }
  if (!found) return [gx, gy];
  // Reconstruct (goal -> start), then reverse.
  const pts: number[] = [];
  let c = goal;
  while (c !== -1 && c !== start) {
    pts.push((c % GRID_N + 0.5) * CELL, (((c / GRID_N) | 0) + 0.5) * CELL);
    c = cameFrom[c];
  }
  const n = pts.length / 2;
  const xs: number[] = new Array(n), ys: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = pts[(n - 1 - i) * 2];
    ys[i] = pts[(n - 1 - i) * 2 + 1];
  }
  xs[n - 1] = gx;
  ys[n - 1] = gy;
  // String pulling.
  const out: number[] = [];
  let ax = sx, ay = sy;
  let i = 0;
  while (i < n) {
    let j = Math.min(n - 1, i + 24);
    while (j > i && !lineClearWide(ax, ay, xs[j], ys[j])) j--;
    out.push(xs[j], ys[j]);
    ax = xs[j];
    ay = ys[j];
    i = j + 1;
  }
  return out;
}

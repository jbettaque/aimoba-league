/**
 * Tournaments: who plays whom, and who won the day.
 *
 * Two formats, chosen by the size of the field:
 *
 *   round robin   up to `roundRobinMax` entries: every pair plays `gamesPerPair` games.
 *   Swiss         larger fields: `swissRounds` rounds; each round pairs entries with the same score that have
 *                 not met yet, so the strong meet the strong without everyone having to play everyone.
 *
 * Games always come in twos: the same seed played once from each side, so neither entry is favoured by the side
 * or by the luck of that game's random stream.
 *
 * Standings are by points (win 1, draw 0.5), ties broken by Sonneborn-Berger (the points of the opponents you
 * beat, plus half those of the ones you drew with), then wins, then the rating the entry came in with.
 *
 * Everything here is pure: given the same field and seed it produces the same fixtures, which makes a tournament
 * resumable after a restart (see `Tournament` in the platform) and testable.
 */

export interface Entrant {
  id: string;
  /** Rating before the tournament, for seeding and the last tie-break. */
  rating: number;
}

export interface Fixture {
  /** Position in the schedule; results are applied to ratings in this order. */
  index: number;
  round: number;
  blue: string;
  red: string;
  seed: number;
}

export interface GameResult {
  blue: string;
  red: string;
  /** 0 = blue won, 1 = red won, -1 = draw, null = not played (void: does not count). */
  winner: number | null;
}

export interface Standing {
  id: string;
  rank: number;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  /** Sonneborn-Berger score. */
  sb: number;
}

export interface FormatConfig {
  roundRobinMax: number;
  /** Games per pair in a round robin (rounded up to an even number). */
  gamesPerPair: number;
  swissRounds: number;
}

export const DEFAULT_FORMAT: FormatConfig = { roundRobinMax: 16, gamesPerPair: 4, swissRounds: 9 };

export type Format = { kind: 'round-robin'; rounds: number } | { kind: 'swiss'; rounds: number };

export function chooseFormat(entrants: number, cfg: FormatConfig = DEFAULT_FORMAT): Format {
  if (entrants <= cfg.roundRobinMax) return { kind: 'round-robin', rounds: Math.max(1, Math.ceil(cfg.gamesPerPair / 2)) };
  // More rounds than log2(n) are needed to sort the middle of the field; fewer than n-1 or it is a round robin.
  return { kind: 'swiss', rounds: Math.min(entrants - 1, Math.max(cfg.swissRounds, Math.ceil(Math.log2(entrants)) + 2)) };
}

/** A well-mixed 31-bit seed from a tournament seed and two numbers. */
export function mixSeed(seed: number, a: number, b: number): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (const v of [a, b]) {
    h = Math.imul(h ^ (v + 0x7f4a7c15), 0x85ebca6b) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
    h ^= h >>> 16;
  }
  return (h >>> 1) || 1;
}

/** The two games of a pairing: same seed, sides swapped. */
function twoGames(a: string, b: string, round: number, seed: number, out: Fixture[]) {
  out.push({ index: out.length, round, blue: a, red: b, seed });
  out.push({ index: out.length, round, blue: b, red: a, seed });
}

/**
 * All fixtures of a round robin. `rounds` passes over all pairs; each pass uses a fresh seed per pair.
 * Entrants are ordered by id first, so the schedule does not depend on the order they were given in.
 */
export function roundRobin(entrants: Entrant[], rounds: number, seed: number): Fixture[] {
  const ids = entrants.map((e) => e.id).sort();
  const out: Fixture[] = [];
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        // Alternate who gets Blue first, so across passes neither id always opens on the same side.
        const flip = (i + j + r) % 2 === 1;
        twoGames(flip ? ids[j] : ids[i], flip ? ids[i] : ids[j], r, mixSeed(seed, r * 100003 + i, j), out);
      }
    }
  }
  return out;
}

const pairKey = (a: string, b: string) => (a < b ? `${a}\n${b}` : `${b}\n${a}`);

/**
 * Pair one Swiss round. Entrants are sorted by points (then rating); each takes the highest-placed opponent it
 * has not met. If that leaves someone without an opponent, earlier pairings are undone and retried (backtracking),
 * and only if no perfect pairing exists are rematches allowed. With an odd field the lowest-placed entrant that
 * has not had one yet sits the round out with a bye.
 */
export function swissRound(entrants: Entrant[], results: GameResult[], byes: string[]): { pairs: [string, string][]; bye: string | null } {
  const points = new Map(entrants.map((e) => [e.id, 0]));
  const met = new Set<string>();
  for (const g of results) {
    if (g.winner === null) continue;
    met.add(pairKey(g.blue, g.red));
    if (points.has(g.blue)) points.set(g.blue, points.get(g.blue)! + (g.winner === 0 ? 1 : g.winner < 0 ? 0.5 : 0));
    if (points.has(g.red)) points.set(g.red, points.get(g.red)! + (g.winner === 1 ? 1 : g.winner < 0 ? 0.5 : 0));
  }
  for (const b of byes) if (points.has(b)) points.set(b, points.get(b)! + 1);
  const order = entrants.slice().sort((a, b) => points.get(b.id)! - points.get(a.id)! || b.rating - a.rating || (a.id < b.id ? -1 : 1)).map((e) => e.id);

  let bye: string | null = null;
  if (order.length % 2 === 1) {
    const had = new Set(byes);
    for (let i = order.length - 1; i >= 0; i--) if (!had.has(order[i])) { bye = order[i]; break; }
    bye ??= order[order.length - 1];
    order.splice(order.indexOf(bye), 1);
  }

  const solve = (allowRematch: boolean): [string, string][] | null => {
    const used = new Array(order.length).fill(false);
    const pairs: [string, string][] = [];
    let budget = 200_000; // the search is exponential in the worst case; give up and allow rematches rather than hang
    const go = (): boolean => {
      const i = used.indexOf(false);
      if (i < 0) return true;
      used[i] = true;
      for (let j = i + 1; j < order.length; j++) {
        if (used[j] || (!allowRematch && met.has(pairKey(order[i], order[j])))) continue;
        if (--budget < 0) break;
        used[j] = true;
        pairs.push([order[i], order[j]]);
        if (go()) return true;
        pairs.pop();
        used[j] = false;
      }
      used[i] = false;
      return false;
    };
    return go() ? pairs : null;
  };
  return { pairs: solve(false) ?? solve(true) ?? [], bye };
}

/** The fixtures of one Swiss round from its pairs. `offset` is the number of fixtures scheduled before it. */
export function swissFixtures(pairs: [string, string][], round: number, seed: number, offset: number): Fixture[] {
  const out: Fixture[] = [];
  pairs.forEach(([a, b], i) => {
    const flip = (round + i) % 2 === 1;
    twoGames(flip ? b : a, flip ? a : b, round, mixSeed(seed, round * 100003 + i, 7), out);
  });
  return out.map((f) => ({ ...f, index: f.index + offset }));
}

/** The table. `byes` lists who sat a Swiss round out (a bye scores one point, half the points of a round). */
export function standings(entrants: Entrant[], results: GameResult[], byes: string[] = []): Standing[] {
  const rows = new Map<string, Standing>(entrants.map((e) => [e.id, { id: e.id, rank: 0, played: 0, wins: 0, draws: 0, losses: 0, points: 0, sb: 0 }]));
  for (const g of results) {
    if (g.winner === null) continue;
    const b = rows.get(g.blue), r = rows.get(g.red);
    if (!b || !r) continue;
    b.played++; r.played++;
    if (g.winner < 0) { b.draws++; r.draws++; b.points += 0.5; r.points += 0.5; }
    else if (g.winner === 0) { b.wins++; r.losses++; b.points += 1; }
    else { r.wins++; b.losses++; r.points += 1; }
  }
  for (const id of byes) { const row = rows.get(id); if (row) row.points += 1; }
  for (const g of results) {
    if (g.winner === null) continue;
    const b = rows.get(g.blue), r = rows.get(g.red);
    if (!b || !r) continue;
    if (g.winner < 0) { b.sb += r.points / 2; r.sb += b.points / 2; }
    else if (g.winner === 0) b.sb += r.points;
    else r.sb += b.points;
  }
  const rating = new Map(entrants.map((e) => [e.id, e.rating]));
  const table = [...rows.values()].sort((a, b) => b.points - a.points || b.sb - a.sb || b.wins - a.wins || rating.get(b.id)! - rating.get(a.id)! || (a.id < b.id ? -1 : 1));
  table.forEach((row, i) => {
    const prev = table[i - 1];
    row.rank = prev && prev.points === row.points && prev.sb === row.sb && prev.wins === row.wins ? prev.rank : i + 1;
  });
  return table;
}

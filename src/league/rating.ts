/**
 * Elo ratings.
 *
 * Every bot starts at 1500. After a game the winner takes points from the loser in proportion to how unexpected
 * the result was; a draw moves both toward each other. A bot's first games move its rating twice as fast, so a
 * new entry finds its level within a tournament or two.
 */

export const START_RATING = 1500;
/** Games during which a rating counts as provisional. */
export const PROVISIONAL_GAMES = 20;
const K_PROVISIONAL = 40;
const K_SETTLED = 20;

export interface Rated {
  rating: number;
  /** Rated games played so far. */
  games: number;
}

/** The score (0..1) `a` is expected to make against `b`. */
export function expectedScore(a: number, b: number): number {
  return 1 / (1 + Math.pow(10, (b - a) / 400));
}

export const kFactor = (games: number) => (games < PROVISIONAL_GAMES ? K_PROVISIONAL : K_SETTLED);

/**
 * Rating changes for one game. `score` is a's result: 1 win, 0.5 draw, 0 loss. Each side moves by its own K, so
 * a settled bot is not thrown about by a newcomer whose rating still means little.
 */
export function eloDelta(a: Rated, b: Rated, score: number): [number, number] {
  const e = expectedScore(a.rating, b.rating);
  return [kFactor(a.games) * (score - e), kFactor(b.games) * (e - score)];
}

/** Apply a list of games in order. `results[i] = [a, b, score of a]`. Mutates the rated objects. */
export function applyGames<T extends Rated>(results: [T, T, number][]) {
  for (const [a, b, score] of results) {
    const [da, db] = eloDelta(a, b, score);
    a.rating += da;
    b.rating += db;
    a.games++;
    b.games++;
  }
}

/** 95% confidence interval of a win rate (Wilson score); draws count as half a win. */
export function wilson(score: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = score / n;
  const centre = p + (z * z) / (2 * n);
  const half = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, (centre - half) / (1 + (z * z) / n)), Math.min(1, (centre + half) / (1 + (z * z) / n))];
}

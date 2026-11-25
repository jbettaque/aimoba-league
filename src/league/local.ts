/**
 * Tournaments on your own machine: the same scheduling, scoring and ratings as the league, over a list of agents.
 * Used by `aimoba tournament` and `aimoba bench`, and by the tests.
 */
import type { AgentSpec } from '../agent/entry.ts';
import type { MatchRules } from '../agent/info.ts';
import type { MatchResult } from '../agent/match.ts';
import type { MatchJob, MatchPool } from '../agent/pool.ts';
import type { SandboxConfig } from '../agent/sandbox.ts';
import { START_RATING, applyGames, type Rated } from './rating.ts';
import { DEFAULT_FORMAT, chooseFormat, roundRobin, standings, swissFixtures, swissRound, type Fixture, type FormatConfig, type GameResult, type Standing } from './tournament.ts';

export interface LocalEntrant { id: string; spec: AgentSpec; rating?: number; games?: number }

export interface LocalTournamentOptions {
  entrants: LocalEntrant[];
  pool: MatchPool;
  seed: number;
  format?: Partial<FormatConfig>;
  rules?: Partial<MatchRules>;
  sandbox?: SandboxConfig;
  inProcess?: boolean;
  /** Where to write replays: called per fixture, return a path or undefined for none. */
  replayFile?: (f: Fixture) => string | undefined;
  onGame?: (f: Fixture, r: MatchResult | null, done: number, total: number) => void;
}

export interface LocalTournamentResult {
  format: string;
  standings: (Standing & { rating: number; ratingChange: number })[];
  games: { fixture: Fixture; result: MatchResult | null }[];
}

/** Play a list of fixtures on the pool; a match the infrastructure failed on is tried once more, then left void. */
export async function playFixtures(fixtures: Fixture[], spec: (id: string) => AgentSpec, opts: Pick<LocalTournamentOptions, 'pool' | 'rules' | 'sandbox' | 'inProcess' | 'replayFile'>, onGame?: (f: Fixture, r: MatchResult | null) => void): Promise<(MatchResult | null)[]> {
  return Promise.all(fixtures.map(async (f) => {
    const job: MatchJob = { blue: spec(f.blue), red: spec(f.red), seed: f.seed, rules: opts.rules, sandbox: opts.sandbox, inProcess: opts.inProcess, replayFile: opts.replayFile?.(f), record: opts.replayFile?.(f) !== undefined };
    let result: MatchResult | null = null;
    for (let attempt = 0; attempt < 2 && !result; attempt++) {
      try { result = await opts.pool.run(job); } catch { result = null; }
    }
    onGame?.(f, result);
    return result;
  }));
}

export async function runLocalTournament(opts: LocalTournamentOptions): Promise<LocalTournamentResult> {
  const cfg = { ...DEFAULT_FORMAT, ...opts.format };
  const entrants = opts.entrants.map((e) => ({ id: e.id, rating: e.rating ?? START_RATING }));
  const specs = new Map(opts.entrants.map((e) => [e.id, e.spec]));
  const format = chooseFormat(entrants.length, cfg);
  const games: { fixture: Fixture; result: MatchResult | null }[] = [];
  const byes: string[] = [];
  let done = 0;
  const toResult = (g: { fixture: Fixture; result: MatchResult | null }): GameResult => ({ blue: g.fixture.blue, red: g.fixture.red, winner: g.result ? g.result.winner : null });

  const play = async (fixtures: Fixture[], total: number) => {
    const results = await playFixtures(fixtures, (id) => specs.get(id)!, opts, (f, r) => opts.onGame?.(f, r, ++done, total));
    fixtures.forEach((fixture, i) => games.push({ fixture, result: results[i] }));
  };

  if (format.kind === 'round-robin') {
    const fixtures = roundRobin(entrants, format.rounds, opts.seed);
    await play(fixtures, fixtures.length);
  } else {
    const perRound = Math.floor(entrants.length / 2) * 2;
    for (let round = 0; round < format.rounds; round++) {
      const { pairs, bye } = swissRound(entrants, games.map(toResult), byes);
      if (bye) byes.push(bye);
      await play(swissFixtures(pairs, round, opts.seed, games.length), perRound * format.rounds);
    }
  }

  // Ratings move game by game, in schedule order.
  const rated = new Map<string, Rated>(opts.entrants.map((e) => [e.id, { rating: e.rating ?? START_RATING, games: e.games ?? 0 }]));
  const before = new Map([...rated].map(([id, r]) => [id, r.rating]));
  const ordered = games.slice().sort((a, b) => a.fixture.index - b.fixture.index);
  applyGames(ordered.filter((g) => g.result).map((g) => [rated.get(g.fixture.blue)!, rated.get(g.fixture.red)!, g.result!.winner === 0 ? 1 : g.result!.winner < 0 ? 0.5 : 0]));
  const table = standings(entrants, ordered.map(toResult), byes);
  return {
    format: format.kind === 'round-robin' ? `round robin, ${format.rounds * 2} games per pair` : `Swiss, ${format.rounds} rounds`,
    standings: table.map((s) => ({ ...s, rating: rated.get(s.id)!.rating, ratingChange: rated.get(s.id)!.rating - before.get(s.id)! })),
    games: ordered,
  };
}

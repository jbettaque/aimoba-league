/**
 * Plays one match between two agents and scores it.
 *
 * The game only advances when both agents have answered, so a match plays out the same on a fast and a slow
 * machine; thinking is on the clock instead (`MatchRules`): every decision has a share of time, anything beyond it
 * comes out of the agent's bank for the game, and an agent whose bank is empty forfeits. An agent that crashes or
 * breaks the protocol forfeits too. A game that reaches the time limit goes to the team that destroyed more
 * structures, then to the one with more kills; only if both are level is it a draw.
 *
 * Both agents decide on the same state: orders from one are never visible to the other before the next decision.
 */
import { TICK_RATE } from '../sim/defs.ts';
import { Game } from '../sim/game.ts';
import { HEROES } from '../sim/heroes.ts';
import { makeReplay, type Replay } from '../sim/ml.ts';
import { createAgent, type Agent, type StepInput } from './agents.ts';
import type { AgentSpec } from './entry.ts';
import { DEFAULT_RULES, gameInfo, stepBudgetMs, type MatchRules } from './info.ts';
import { AgentFailure, SandboxError, normalizeOptions, type ActMessage, type AgentOptions, type FailureKind } from './protocol.ts';
import type { SandboxConfig } from './sandbox.ts';
import { TeamGateway } from './view.ts';

export interface MatchOptions {
  blue: AgentSpec;
  red: AgentSpec;
  seed: number;
  rules?: Partial<MatchRules>;
  sandbox?: SandboxConfig;
  /** Run JavaScript entries inside this process: much faster, but unprotected and without a clock that can stop them. */
  inProcess?: boolean;
  /** Keep the replay in the result (default true). */
  record?: boolean;
  /** Extra fields for the replay's `meta` (a match id, a tournament name, ...). */
  meta?: Record<string, unknown>;
  /** Called about once per second of game time. */
  onProgress?: (game: Game) => void;
}

export interface HeroResult {
  hero: string; level: number; kills: number; deaths: number; assists: number; lastHits: number; gold: number; damage: number; items: string[];
}

export interface TeamResult {
  name: string;
  /** 'builtin', or the entry's runtime. */
  runtime: string;
  kills: number;
  /** Enemy structures destroyed. */
  structures: number;
  gold: number;
  lastHits: number;
  heroes: HeroResult[];
  /** Thinking time: decisions made, mean and longest in milliseconds, and what is left of the bank. */
  decisions: number;
  meanMs: number;
  maxMs: number;
  bankLeftMs: number;
  initMs: number;
  /** Orders that were dropped as invalid. */
  rejectedOrders: number;
  /** Set when this team lost by forfeit. */
  failure?: { kind: FailureKind; message: string };
  /** What the agent wrote to stderr, plus notes from the runner about dropped orders. */
  log: string;
}

export interface MatchResult {
  /** 0 = Blue won, 1 = Red won, -1 = draw. */
  winner: number;
  /**
   * 'core' (a core fell), 'time' (time limit, decided on structures then kills), 'draw' (time limit, level),
   * or a forfeit: 'crash', 'timeout', 'protocol', 'init'.
   */
  reason: string;
  seed: number;
  ticks: number;
  /** Game time in seconds. */
  duration: number;
  teams: [TeamResult, TeamResult];
  replay?: Replay;
  /** Wall-clock milliseconds the match took, start-up included. */
  wallMs: number;
}

interface Side {
  spec: AgentSpec;
  agent: Agent | null;
  options: AgentOptions | null;
  gateway: TeamGateway | null;
  budgetMs: number;
  bankMs: number;
  decisions: number;
  totalMs: number;
  maxMs: number;
  initMs: number;
  failure: AgentFailure | null;
  notes: string[];
  noted: number;
}

const NOTE_LIMIT = 30;

function note(side: Side, tick: number, errors: string[]) {
  for (const e of errors) {
    side.noted++;
    if (side.notes.length < NOTE_LIMIT) side.notes.push(`[runner] tick ${tick}: order dropped: ${e}`);
  }
}

/** Who wins a game that ran into the time limit: more structures destroyed, then more kills, else nobody. */
export function judge(g: Game): number {
  if (g.winner >= 0) return g.winner;
  const s = g.towersLost[1] - g.towersLost[0]; // structures Blue destroyed minus those Red destroyed
  if (s !== 0) return s > 0 ? 0 : 1;
  const k = g.teamKills[0] - g.teamKills[1];
  return k === 0 ? -1 : k > 0 ? 0 : 1;
}

function teamResult(g: Game | null, team: number, side: Side, rules: MatchRules): TeamResult {
  const heroes: HeroResult[] = [];
  let gold = 0, cs = 0;
  if (g) {
    for (let k = 0; k < 5; k++) {
      const u = g.heroes[team * 5 + k], h = u.hero!;
      heroes.push({ hero: HEROES[h.def].id, level: h.level, kills: h.kills, deaths: h.deaths, assists: h.assists, lastHits: h.cs, gold: Math.round(h.totalGold), damage: Math.round(h.dmgDealt), items: h.items.slice() });
      gold += h.totalGold;
      cs += h.cs;
    }
  }
  const log = (side.agent?.log() ?? '') + (side.notes.length ? (side.notes.join('\n') + (side.noted > side.notes.length ? `\n[runner] ... and ${side.noted - side.notes.length} more dropped orders` : '') + '\n') : '');
  const r: TeamResult = {
    name: side.options?.name ?? side.spec.name,
    runtime: side.spec.kind === 'builtin' ? 'builtin' : side.spec.manifest.runtime,
    kills: g ? g.teamKills[team] : 0, structures: g ? g.towersLost[1 - team] : 0, gold: Math.round(gold), lastHits: cs, heroes,
    decisions: side.decisions, meanMs: side.decisions ? side.totalMs / side.decisions : 0, maxMs: side.maxMs,
    bankLeftMs: Math.max(0, Math.round(side.bankMs)), initMs: Math.round(side.initMs), rejectedOrders: side.gateway?.rejected ?? 0,
    log: log.length > rules.logBytes + 4096 ? log.slice(0, rules.logBytes + 4096) : log,
  };
  if (side.failure) r.failure = { kind: side.failure.kind, message: side.failure.message };
  return r;
}

export async function runMatch(opts: MatchOptions): Promise<MatchResult> {
  const t0 = performance.now();
  const rules: MatchRules = { ...DEFAULT_RULES, ...opts.rules };
  const sandbox: SandboxConfig = opts.sandbox ?? { driver: 'process' };
  const sides: Side[] = [opts.blue, opts.red].map((spec) => ({
    spec, agent: createAgent(spec, { rules, sandbox, inProcess: opts.inProcess ?? false }), options: null, gateway: null,
    budgetMs: 0, bankMs: rules.bankMs, decisions: 0, totalMs: 0, maxMs: 0, initMs: 0, failure: null, notes: [], noted: 0,
  }));
  const info = gameInfo(rules);
  let game: Game | null = null;

  try {
    // 1. Bring both agents up (at the same time: neither waits for the other's start-up).
    await Promise.all(sides.map(async (side) => {
      if (!side.agent) return;
      const t = performance.now();
      try {
        const ready = await side.agent.init(info, rules.initMs);
        side.options = normalizeOptions(ready, rules, side.spec.name);
        side.budgetMs = stepBudgetMs(rules, side.options.decisionTicks);
      } catch (err) {
        if (err instanceof SandboxError) throw err; // the match cannot be played; nobody forfeits
        side.failure = err instanceof AgentFailure ? err : new AgentFailure('init', String((err as Error)?.message ?? err));
      }
      side.initMs = performance.now() - t;
    }));

    if (!sides.some((s) => s.failure)) {
      // 2. The game. Slots of a scripted bot are driven by the simulation; an agent's heroes take orders.
      const per = <T>(f: (side: Side, k: number) => T): T[] => Array.from({ length: 10 }, (_, s) => f(sides[s < 5 ? 0 : 1], s % 5));
      game = new Game({
        seed: opts.seed, maxTime: rules.maxTime, recordCommands: opts.record ?? true,
        bots: per((side) => side.agent === null),
        botKind: per((side) => (side.spec.kind === 'builtin' ? side.spec.bot : 'basic')),
        autoLevel: per((side, k) => (side.options ? side.options.autoLevel[k] : null)),
        autoBuy: per((side, k) => (side.options ? side.options.autoBuy[k] : null)),
        autoAttack: per((side, k) => (side.options ? side.options.autoAttack[k] : null)),
      });
      const g = game;
      sides.forEach((side, team) => {
        if (!side.agent) return;
        side.gateway = new TeamGateway(g, team, { mirror: side.options!.mirror });
        side.agent.start(team, opts.seed);
      });

      const input: StepInput[] = [{ tick: 0 }, { tick: 0 }];
      const due: number[] = [];
      const answers: (ActMessage | Promise<ActMessage> | null)[] = [null, null];
      const started = [0, 0], elapsed = [0, 0];

      const settle = (team: number, act: ActMessage) => {
        const side = sides[team];
        const ms = elapsed[team];
        side.decisions++;
        side.totalMs += ms;
        if (ms > side.maxMs) side.maxMs = ms;
        if (ms > side.budgetMs) side.bankMs -= ms - side.budgetMs;
        if (side.bankMs < 0) {
          side.failure = new AgentFailure('timeout', `out of thinking time at tick ${g.tick} (${Math.round(ms)} ms for one decision; ${Math.round(side.budgetMs)} ms each plus a bank of ${rules.bankMs} ms per game)`);
          return;
        }
        const gw = side.gateway!;
        if (act.actions !== undefined && act.actions !== null) note(side, g.tick, gw.applyActions(act.actions));
        if (act.orders !== undefined && act.orders !== null) note(side, g.tick, gw.apply(act.orders, rules.maxOrders));
      };
      const failed = (team: number, err: unknown) => {
        sides[team].failure = err instanceof AgentFailure ? err : new AgentFailure('crash', String((err as Error)?.message ?? err));
      };

      while (!g.over) {
        // 3. Ask every agent whose turn it is. Both see the state before either's orders are applied.
        due.length = 0;
        for (let team = 0; team < 2; team++) {
          const side = sides[team];
          if (!side.agent || g.tick % side.options!.decisionTicks !== 0) continue;
          const gw = side.gateway!, inp = input[team];
          inp.tick = g.tick;
          inp.view = side.options!.observations.includes('view') ? gw.view() : undefined;
          if (side.options!.observations.includes('obs')) { gw.observeMl(); inp.obs = gw.obs; inp.mask = gw.mask; }
          due.push(team);
        }
        let pending = false;
        for (const team of due) {
          const side = sides[team];
          started[team] = performance.now();
          try {
            const a = side.agent!.step(input[team], side.budgetMs + side.bankMs);
            if (a instanceof Promise) {
              pending = true;
              answers[team] = a.then((v) => { elapsed[team] = performance.now() - started[team]; return v; });
            } else {
              elapsed[team] = performance.now() - started[team];
              answers[team] = a;
            }
          } catch (err) {
            answers[team] = null;
            failed(team, err);
          }
        }
        if (pending) {
          await Promise.all(due.map(async (team) => {
            const a = answers[team];
            if (!a) return;
            try { settle(team, await a); } catch (err) { failed(team, err); }
          }));
        } else {
          for (const team of due) { const a = answers[team]; if (a) settle(team, a as ActMessage); }
        }
        if (sides[0].failure || sides[1].failure) break;

        g.step();
        sides[0].gateway?.onTick();
        sides[1].gateway?.onTick();
        if (opts.onProgress && g.tick % TICK_RATE === 0) opts.onProgress(g);
      }
    }

    // 4. The verdict.
    const f0 = sides[0].failure, f1 = sides[1].failure;
    let winner: number, reason: string;
    if (f0 || f1) {
      winner = f0 && f1 ? -1 : f0 ? 1 : 0;
      reason = (f0 ?? f1)!.kind;
    } else {
      const g = game!;
      winner = judge(g);
      reason = g.winner >= 0 ? 'core' : winner >= 0 ? 'time' : 'draw';
    }

    await Promise.all(sides.map((side, team) => side.agent?.end({
      result: winner < 0 ? 'draw' : winner === team ? 'win' : 'loss', reason,
      summary: game ? { ticks: game.tick, kills: [game.teamKills[team], game.teamKills[1 - team]], structures: [game.towersLost[1 - team], game.towersLost[team]] } : undefined,
    })));

    const teams = [teamResult(game, 0, sides[0], rules), teamResult(game, 1, sides[1], rules)] as [TeamResult, TeamResult];
    const result: MatchResult = { winner, reason, seed: opts.seed, ticks: game?.tick ?? 0, duration: game?.time ?? 0, teams, wallMs: 0 };
    if (game && game.cmdLog && (opts.record ?? true)) {
      result.replay = makeReplay(game, { ...opts.meta, teams: [teams[0].name, teams[1].name], result: winner, reason });
    }
    result.wallMs = Math.round(performance.now() - t0);
    return result;
  } finally {
    for (const side of sides) side.agent?.kill();
  }
}

/** One line about a finished match, for logs and the command line. */
export function describeResult(r: MatchResult): string {
  const [a, b] = r.teams;
  const who = r.winner < 0 ? 'draw' : `${r.teams[r.winner].name} wins`;
  const why: Record<string, string> = { core: 'core destroyed', time: 'time limit, ahead on structures or kills', draw: 'time limit, level', crash: 'opponent crashed', timeout: 'opponent ran out of thinking time', protocol: 'opponent broke the protocol', init: 'opponent failed to start' };
  const m = Math.floor(r.duration / 60), s = Math.floor(r.duration % 60);
  return `${who} (${why[r.reason] ?? r.reason}) at ${m}:${String(s).padStart(2, '0')} | kills ${a.kills}-${b.kills} | structures ${a.structures}-${b.structures} | gold ${a.gold}-${b.gold}`;
}

/**
 * The agent protocol: how a program that plays one team talks to the match runner.
 *
 * An agent is any program that reads JSON messages from stdin and writes JSON messages to stdout, one per line
 * (UTF-8, no line breaks inside a message). Whatever it writes to stderr is kept as its log. The conversation:
 *
 *   runner -> agent   {"type":"hello","protocol":1,"info":{...}}          the static game description (info.ts)
 *   agent  -> runner  {"type":"ready", ...options}                        within `info.rules.initMs`
 *   runner -> agent   {"type":"start","team":0}                           0 = Blue, 1 = Red
 *   then, every `decisionTicks` simulation ticks until the game ends:
 *   runner -> agent   {"type":"step","tick":120,"view":{...},"obs":"...","mask":"..."}
 *   agent  -> runner  {"type":"act","orders":[...]}        and / or  "actions":[[kind,dir,dist,target] x 5]
 *   runner -> agent   {"type":"end","result":"win","reason":"core","summary":{...}}
 *
 * The runner waits for every `act` (the game does not advance meanwhile), but the wait is on the clock: see
 * `MatchRules`. Options the agent can set in `ready` (all optional):
 *
 *   name            what to call it in logs
 *   observations    ["view"] (default), ["obs"] or both: which forms of the game state to send each step.
 *                   "view" is the structured view (view.ts); "obs" is the ML interface: `obs` is base64 of
 *                   5 x obsSize little-endian float32, `mask` base64 of 5 x maskSize bytes, one row per hero.
 *   decisionTicks   sim ticks between decisions (default 6 = five decisions a second; the rules give the range)
 *   mirror          true (default): Red sees the map mirrored, so its own base is always bottom-left
 *   autoLevel       true (default): heroes spend skill points by themselves. A list of five sets it per hero.
 *   autoBuy         true (default): heroes shop along their build by themselves when at the fountain
 *   autoAttack      false (default): idle heroes attack enemies in range by themselves
 */
import type { GameInfo, MatchRules } from './info.ts';
import type { TeamView } from './view.ts';

export type Observation = 'view' | 'obs';

/** An agent's options after defaults and limits have been applied. */
export interface AgentOptions {
  name: string;
  observations: Observation[];
  decisionTicks: number;
  mirror: boolean;
  autoLevel: boolean[];
  autoBuy: boolean[];
  autoAttack: boolean[];
}

export interface HelloMessage { type: 'hello'; protocol: number; info: GameInfo }
export interface StartMessage { type: 'start'; team: number }
export interface StepMessage { type: 'step'; tick: number; view?: TeamView; obs?: string; mask?: string }
export interface EndMessage { type: 'end'; result: 'win' | 'loss' | 'draw'; reason: string; summary?: unknown }
export type RunnerMessage = HelloMessage | StartMessage | StepMessage | EndMessage;

export interface ReadyMessage {
  type: 'ready';
  name?: string;
  observations?: string[];
  decisionTicks?: number;
  mirror?: boolean;
  autoLevel?: boolean | boolean[];
  autoBuy?: boolean | boolean[];
  autoAttack?: boolean | boolean[];
}
export interface ActMessage { type: 'act'; orders?: unknown; actions?: unknown }

const five = (v: unknown, dflt: boolean): boolean[] => {
  if (typeof v === 'boolean') return new Array(5).fill(v);
  if (Array.isArray(v) && v.length === 5) return v.map((x) => (typeof x === 'boolean' ? x : dflt));
  return new Array(5).fill(dflt);
};

/** Apply defaults and the rules' limits to what an agent asked for in `ready`. Never throws: bad values fall back to the default. */
export function normalizeOptions(raw: unknown, rules: MatchRules, fallbackName: string): AgentOptions {
  const r = (typeof raw === 'object' && raw !== null ? raw : {}) as ReadyMessage;
  const wanted = Array.isArray(r.observations) ? r.observations.filter((o): o is Observation => o === 'view' || o === 'obs') : [];
  const ticks = typeof r.decisionTicks === 'number' && Number.isFinite(r.decisionTicks) ? Math.round(r.decisionTicks) : 6;
  return {
    name: typeof r.name === 'string' && r.name.trim() ? r.name.trim().slice(0, 60) : fallbackName,
    observations: wanted.length ? [...new Set(wanted)] : ['view'],
    decisionTicks: Math.min(rules.maxDecisionTicks, Math.max(rules.minDecisionTicks, ticks)),
    mirror: typeof r.mirror === 'boolean' ? r.mirror : true,
    autoLevel: five(r.autoLevel, true),
    autoBuy: five(r.autoBuy, true),
    autoAttack: five(r.autoAttack, false),
  };
}

export const toBase64 = (a: Float32Array | Uint8Array) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');

/**
 * The sandbox itself failed (the container could not be created, the daemon did not answer). That is nobody's
 * forfeit: the match is abandoned and whoever asked for it decides whether to play it again.
 */
export class SandboxError extends Error {}

/** Why an agent lost a game without the game deciding it. */
export type FailureKind = 'crash' | 'timeout' | 'protocol' | 'init';

export class AgentFailure extends Error {
  kind: FailureKind;
  constructor(kind: FailureKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

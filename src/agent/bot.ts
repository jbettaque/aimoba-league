/**
 * The JavaScript bot interface.
 *
 * A bot is a module whose default export is an object (or a class, or a function returning the object):
 *
 *   export default {
 *     options: { observations: ['view'], decisionTicks: 6 },   // optional, see protocol.ts
 *     init(ctx) {},              // once, before the game: ctx.info is the static game description, ctx.lib the helpers
 *     start(ctx) {},             // the game begins; ctx.team says which side
 *     act(state, ctx) {          // every decision: return a list of orders (see view.ts), or nothing
 *       return state.heroes.map((h) => ctx.lib.attackMove(h, 6900, 1100));
 *     },
 *     end(result, ctx) {},       // { result: 'win' | 'loss' | 'draw', reason, summary }
 *   };
 *
 * `state` is the team's view. If the bot asked for the ML observation, `state.obs` (Float32Array, 5 x obsSize) and
 * `state.mask` (Uint8Array, 5 x maskSize) are there too; with observations ['obs'] alone, `state` holds just those
 * and the tick. `act` may also return `{ orders, actions }`, where `actions` is 5 x [kind, direction, distance, target].
 *
 * `BotRunner` adapts such a module to the agent protocol. The host (host.ts) uses it inside the bot's own process,
 * and the match runner uses it directly when a bot is run in-process.
 */
import { pathToFileURL } from 'node:url';
import type { GameInfo } from './info.ts';
import { lib, type BotLib } from './lib.ts';
import type { ActMessage, EndMessage, ReadyMessage, StepMessage } from './protocol.ts';
import type { Order, TeamView } from './view.ts';

export interface BotContext {
  info: GameInfo;
  lib: BotLib;
  /** 0 = Blue, 1 = Red; -1 before the game starts. */
  team: number;
  /** Write a line to the bot's log (stderr). `console.log` goes there too. */
  log(...args: unknown[]): void;
  /** Free for the bot's own use. */
  memory: Record<string, unknown>;
}

export type BotState = Partial<TeamView> & { tick: number; obs?: Float32Array; mask?: Uint8Array };
export type BotReply = Order[] | { orders?: Order[]; actions?: number[] | number[][] } | null | undefined | void;

export interface Bot {
  name?: string;
  options?: Omit<ReadyMessage, 'type'>;
  init?(ctx: BotContext): void | Promise<void>;
  start?(ctx: BotContext): void | Promise<void>;
  act(state: BotState, ctx: BotContext): BotReply | Promise<BotReply>;
  end?(result: Omit<EndMessage, 'type'>, ctx: BotContext): void | Promise<void>;
}

/** Load a bot module: the default export may be the bot, a class to instantiate, or a function that returns the bot. */
export async function loadBot(file: string): Promise<Bot> {
  const mod = await import(pathToFileURL(file).href);
  let bot: unknown = mod.default ?? mod.bot;
  if (typeof bot === 'function') {
    const fn = bot as { prototype?: { act?: unknown } } & (new () => unknown) & (() => unknown);
    bot = fn.prototype && typeof fn.prototype.act === 'function' ? new fn() : await fn();
  }
  if (typeof bot !== 'object' || bot === null || typeof (bot as Bot).act !== 'function') {
    throw new Error(`${file} must export a bot as its default export: an object with an act(state, ctx) function`);
  }
  return bot as Bot;
}

const fromBase64 = (s: string) => {
  const b = Buffer.from(s, 'base64');
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

export class BotRunner {
  readonly bot: Bot;
  readonly ctx: BotContext;

  constructor(bot: Bot, log: (...args: unknown[]) => void) {
    this.bot = bot;
    this.ctx = { info: null as unknown as GameInfo, lib, team: -1, log, memory: {} };
  }

  async init(info: GameInfo): Promise<ReadyMessage> {
    this.ctx.info = info;
    await this.bot.init?.(this.ctx);
    return { type: 'ready', name: this.bot.name, ...(this.bot.options ?? {}) };
  }

  async start(team: number) {
    this.ctx.team = team;
    await this.bot.start?.(this.ctx);
  }

  /** One decision. `obs` and `mask` may arrive as typed arrays (in-process) or base64 (over the wire). */
  act(msg: Omit<StepMessage, 'type' | 'obs' | 'mask'> & { obs?: string | Float32Array; mask?: string | Uint8Array }): ActMessage | Promise<ActMessage> {
    const state: BotState = msg.view ? (msg.view as BotState) : { tick: msg.tick };
    if (msg.obs !== undefined) state.obs = typeof msg.obs === 'string' ? new Float32Array(fromBase64(msg.obs)) : msg.obs;
    if (msg.mask !== undefined) state.mask = typeof msg.mask === 'string' ? new Uint8Array(fromBase64(msg.mask)) : msg.mask;
    const out = this.bot.act(state, this.ctx);
    return out instanceof Promise ? out.then(reply) : reply(out);
  }

  async end(result: Omit<EndMessage, 'type'>) {
    await this.bot.end?.(result, this.ctx);
  }
}

function reply(out: BotReply): ActMessage {
  if (out === null || out === undefined) return { type: 'act' };
  if (Array.isArray(out)) return { type: 'act', orders: out };
  if (typeof out === 'object') return { type: 'act', orders: out.orders, actions: out.actions };
  throw new Error(`act() must return a list of orders, { orders, actions } or nothing; got ${typeof out}`);
}

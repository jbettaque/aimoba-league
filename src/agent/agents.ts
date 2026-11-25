/**
 * The things that can play a team in a match, behind one interface:
 *
 *   PolicyAgent    a network exported by python/train/export.py, run inside the runner (no code of the entrant runs)
 *   ModuleAgent    a JavaScript bot loaded into the runner itself: fast, but with no isolation and no way to stop
 *                  a bot that never returns. For local development and tests.
 *   ProcessAgent   any entry as a separate program in a sandbox (sandbox.ts), speaking the agent protocol
 *
 * The scripted bots of the game (easy, basic, expert) need no agent: the simulation drives them itself.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { MASK_SIZE, OBS_SIZE } from '../sim/ml.ts';
import { loadPolicy, type PolicyNet } from '../sim/policy.ts';
import { Rng } from '../sim/rng.ts';
import { BotRunner, loadBot } from './bot.ts';
import type { AgentSpec, Manifest } from './entry.ts';
import type { GameInfo, MatchRules } from './info.ts';
import { AgentFailure, SandboxError, toBase64, type ActMessage, type EndMessage, type ReadyMessage } from './protocol.ts';
import { launch, type Launched, type SandboxConfig } from './sandbox.ts';
import type { TeamView } from './view.ts';

export interface StepInput {
  tick: number;
  view?: TeamView;
  /** The ML observation and mask of the five heroes. The arrays are reused by the next step. */
  obs?: Float32Array;
  mask?: Uint8Array;
}

export interface Agent {
  /** Start the agent and get the options it asks for. Rejects with AgentFailure('init') when it does not come up in time. */
  init(info: GameInfo, timeoutMs: number): Promise<ReadyMessage>;
  start(team: number, seed: number): void;
  /** One decision. An answer later than `timeoutMs` is an AgentFailure('timeout'). May answer synchronously. */
  step(input: StepInput, timeoutMs: number): ActMessage | Promise<ActMessage>;
  /** Tell the agent how the game ended and shut it down. */
  end(msg: Omit<EndMessage, 'type'>): Promise<void>;
  /** Stop it now. */
  kill(): void;
  /** What the agent wrote to its log. */
  log(): string;
  /** True when a slow answer cannot be interrupted, so the time it took is only known afterwards. */
  readonly blocking: boolean;
}

/** Keeps the first `limit` bytes of a log and notes how much more there was. */
class LogBuffer {
  private parts: string[] = [];
  private size = 0;
  private dropped = 0;
  private limit: number;
  constructor(limit: number) { this.limit = limit; }
  add(text: string) {
    if (this.size >= this.limit) { this.dropped += text.length; return; }
    const room = this.limit - this.size;
    if (text.length > room) { this.dropped += text.length - room; text = text.slice(0, room); }
    this.parts.push(text);
    this.size += text.length;
  }
  toString() {
    return this.parts.join('') + (this.dropped ? `\n[log cut: ${this.dropped} more characters]\n` : '');
  }
}

// --- Exported networks ------------------------------------------------------------------------------------------
/** Refuse policy files that would make the runner allocate or compute without bound. */
function checkPolicySize(net: PolicyNet) {
  const h = net.header;
  if (!(h.layers >= 1 && h.layers <= 12) || !(h.hidden >= 8 && h.hidden <= 4096)) throw new Error(`policy has ${h.layers} layers of ${h.hidden} units; the limit is 12 layers of 4096`);
}

export class PolicyAgent implements Agent {
  readonly blocking = true;
  private net: PolicyNet | null = null;
  private rng: Rng | null = null;
  private file: string;
  private greedy: boolean;
  private out = new Int32Array(4);
  private actions: number[] = new Array(20).fill(0);
  private name: string;

  constructor(file: string, greedy: boolean, name: string) {
    this.file = file;
    this.greedy = greedy;
    this.name = name;
  }

  async init(): Promise<ReadyMessage> {
    try {
      this.net = loadPolicy(readFileSync(this.file));
      checkPolicySize(this.net);
    } catch (err) {
      throw new AgentFailure('init', `could not load the policy: ${(err as Error).message}`);
    }
    return { type: 'ready', name: this.name, observations: ['obs'], decisionTicks: 6 };
  }

  start(team: number, seed: number) {
    // Sampled play, as in training; seeded from the game so a match can be reproduced.
    this.rng = this.greedy ? null : new Rng(seed ^ 0x51ed270b ^ (team << 16));
  }

  step(input: StepInput): ActMessage {
    const net = this.net!, a = this.actions;
    for (let k = 0; k < 5; k++) {
      net.act(input.obs!, k * OBS_SIZE, input.mask!, k * MASK_SIZE, this.rng, this.out);
      a[k * 4] = this.out[0]; a[k * 4 + 1] = this.out[1]; a[k * 4 + 2] = this.out[2]; a[k * 4 + 3] = this.out[3];
    }
    return { type: 'act', actions: a };
  }

  async end() {}
  kill() {}
  log() { return ''; }
}

// --- A bot inside the runner ------------------------------------------------------------------------------------
export class ModuleAgent implements Agent {
  readonly blocking = true;
  private runner: BotRunner | null = null;
  private file: string;
  private buf: LogBuffer;

  constructor(file: string, logBytes: number) {
    this.file = file;
    this.buf = new LogBuffer(logBytes);
  }

  async init(info: GameInfo): Promise<ReadyMessage> {
    try {
      this.runner = new BotRunner(await loadBot(this.file), (...args) => this.buf.add(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ') + '\n'));
      return await this.runner.init(info);
    } catch (err) {
      this.buf.add(String((err as Error).stack ?? err) + '\n');
      throw new AgentFailure('init', `the bot failed to start: ${(err as Error).message}`);
    }
  }

  start(team: number) {
    try {
      const p = this.runner!.start(team);
      p.catch((err) => this.buf.add(String((err as Error).stack ?? err) + '\n'));
    } catch (err) {
      this.buf.add(String((err as Error).stack ?? err) + '\n');
    }
  }

  step(input: StepInput): ActMessage | Promise<ActMessage> {
    const fail = (err: unknown): never => {
      this.buf.add(String((err as Error)?.stack ?? err) + '\n');
      throw new AgentFailure('crash', `the bot threw: ${(err as Error)?.message ?? err}`);
    };
    try {
      const out = this.runner!.act(input);
      return out instanceof Promise ? out.catch(fail) : out;
    } catch (err) {
      return fail(err);
    }
  }

  async end(msg: Omit<EndMessage, 'type'>) {
    try { await this.runner?.end(msg); } catch (err) { this.buf.add(String((err as Error).stack ?? err) + '\n'); }
  }
  kill() {}
  log() { return this.buf.toString(); }
}

// --- A separate program -----------------------------------------------------------------------------------------
export class ProcessAgent implements Agent {
  readonly blocking = false;
  private manifest: Manifest;
  private dir: string;
  private sandbox: SandboxConfig;
  private rules: MatchRules;
  private proc: Launched | null = null;
  private buf: LogBuffer;
  /** Bytes received on stdout that do not yet make a whole line. */
  private pending: Buffer[] = [];
  private pendingSize = 0;
  private waiting: { resolve: (msg: Record<string, unknown>) => void; reject: (err: Error) => void; timer: NodeJS.Timeout } | null = null;
  private queue: Record<string, unknown>[] = [];
  private failure: AgentFailure | null = null;
  private exited = false;
  /** The container runtime reported that it could not start the container at all. */
  private sandboxFailed = false;

  constructor(manifest: Manifest, dir: string, sandbox: SandboxConfig, rules: MatchRules) {
    this.manifest = manifest;
    this.dir = dir;
    this.sandbox = sandbox;
    this.rules = rules;
    this.buf = new LogBuffer(rules.logBytes);
  }

  private fail(kind: AgentFailure['kind'], message: string) {
    if (!this.failure) this.failure = new AgentFailure(kind, message);
    const w = this.waiting;
    if (w) {
      this.waiting = null;
      clearTimeout(w.timer);
      w.reject(this.failure);
    }
  }

  private onData(chunk: Buffer) {
    let from = 0;
    for (;;) {
      const nl = chunk.indexOf(10, from);
      if (nl < 0) break;
      const line = this.pendingSize ? Buffer.concat([...this.pending, chunk.subarray(from, nl)]) : chunk.subarray(from, nl);
      this.pending = [];
      this.pendingSize = 0;
      from = nl + 1;
      this.onLine(line);
    }
    if (from < chunk.length) {
      this.pending.push(chunk.subarray(from));
      this.pendingSize += chunk.length - from;
      if (this.pendingSize > this.rules.maxMessageBytes) {
        this.pending = [];
        this.pendingSize = 0;
        this.fail('protocol', `the agent sent a message longer than ${this.rules.maxMessageBytes} bytes`);
        this.kill();
      }
    }
  }

  private onLine(line: Buffer) {
    if (this.failure) return;
    const text = line.toString('utf8').trim();
    if (!text) return;
    let msg: unknown;
    try {
      msg = JSON.parse(text);
    } catch {
      this.fail('protocol', `the agent wrote something to stdout that is not a JSON message: ${text.slice(0, 120)} (print debug output to stderr instead)`);
      this.kill();
      return;
    }
    if (typeof msg !== 'object' || msg === null || Array.isArray(msg)) {
      this.fail('protocol', 'the agent sent a message that is not a JSON object');
      this.kill();
      return;
    }
    const w = this.waiting;
    if (w) {
      this.waiting = null;
      clearTimeout(w.timer);
      w.resolve(msg as Record<string, unknown>);
    } else if (this.queue.length < 4) this.queue.push(msg as Record<string, unknown>);
  }

  private send(msg: unknown) {
    const p = this.proc;
    if (!p || this.exited || !p.child.stdin || p.child.stdin.destroyed) return;
    p.child.stdin.write(JSON.stringify(msg) + '\n');
  }

  private receive(type: string, timeoutMs: number, what: string): Promise<Record<string, unknown>> {
    const check = (msg: Record<string, unknown>) => {
      if (msg.type !== type) throw (this.failure = new AgentFailure('protocol', `expected a "${type}" message, got ${JSON.stringify(msg.type)}`));
      return msg;
    };
    if (this.failure) return Promise.reject(this.failure);
    const early = this.queue.shift();
    if (early) return Promise.resolve().then(() => check(early));
    return new Promise<Record<string, unknown>>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiting = null;
        this.failure = new AgentFailure(what === 'init' ? 'init' : 'timeout', what === 'init' ? `the agent did not answer "hello" within ${Math.round(timeoutMs)} ms` : `out of thinking time (no answer within ${Math.round(timeoutMs)} ms)`);
        this.kill();
        reject(this.failure);
      }, Math.max(1, timeoutMs));
      this.waiting = { resolve, reject, timer };
    }).then(check);
  }

  /**
   * Start the program and wait for its `ready`. When the container runtime itself fails (which happens: a daemon
   * restarting, a hiccup creating the container) the start is tried again, and if it keeps failing that is a
   * SandboxError, not the entry's forfeit.
   */
  async init(info: GameInfo, timeoutMs: number): Promise<ReadyMessage> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await this.bringUp(info, timeoutMs);
      } catch (err) {
        if (!this.sandboxFailed) throw err;
        this.proc?.kill();
        const detail = this.buf.toString().trim().split('\n').slice(-2).join(' ');
        if (attempt >= 2) throw new SandboxError(`the sandbox could not start the entry: ${detail || (err as Error).message}`);
        await new Promise((r) => setTimeout(r, 700 * (attempt + 1)));
        this.buf = new LogBuffer(this.rules.logBytes);
        this.failure = null;
        this.exited = this.sandboxFailed = false;
        this.pending = [];
        this.pendingSize = 0;
        this.queue = [];
      }
    }
  }

  private async bringUp(info: GameInfo, timeoutMs: number): Promise<ReadyMessage> {
    try {
      this.proc = launch(this.manifest, this.dir, this.sandbox);
    } catch (err) {
      throw new AgentFailure('init', `could not start the agent: ${(err as Error).message}`);
    }
    const { child } = this.proc;
    let ready = false;
    child.on('error', (err) => {
      this.exited = true;
      // The container CLI itself is missing or cannot be run: the sandbox's problem, not the entry's.
      if (this.sandbox.driver === 'docker') this.sandboxFailed = true;
      this.fail('init', `could not start the agent (${this.proc!.describe}): ${err.message}`);
    });
    child.on('exit', (code, signal) => {
      this.exited = true;
      // `docker run` exits with 125 when the daemon could not create or start the container.
      if (this.sandbox.driver === 'docker' && code === 125 && !ready) this.sandboxFailed = true;
      this.fail('crash', `the agent exited ${signal ? `on signal ${signal}` : `with code ${code}`} during the match`);
    });
    child.stdout!.on('data', (c: Buffer) => this.onData(c));
    child.stderr!.on('data', (c: Buffer) => this.buf.add(c.toString('utf8')));
    child.stdin!.on('error', () => { /* the agent closed its stdin or died; the exit handler reports it */ });
    this.send({ type: 'hello', protocol: info.protocol, info });
    try {
      const msg = (await this.receive('ready', timeoutMs, 'init')) as unknown as ReadyMessage;
      ready = true;
      return msg;
    } catch (err) {
      const f = err as AgentFailure;
      throw new AgentFailure('init', f.kind === 'init' ? f.message : `the agent failed before it was ready: ${f.message}`);
    }
  }

  start(team: number) {
    this.send({ type: 'start', team });
  }

  step(input: StepInput, timeoutMs: number): Promise<ActMessage> {
    const msg: Record<string, unknown> = { type: 'step', tick: input.tick };
    if (input.view) msg.view = input.view;
    if (input.obs) { msg.obs = toBase64(input.obs); msg.mask = toBase64(input.mask!); }
    this.send(msg);
    return this.receive('act', timeoutMs, 'step') as unknown as Promise<ActMessage>;
  }

  async end(msg: Omit<EndMessage, 'type'>) {
    const p = this.proc;
    if (!p) return;
    if (!this.exited) {
      // The process is about to go away one way or another; that is not a failure any more.
      this.failure ??= new AgentFailure('crash', 'the match is over');
      this.send({ type: 'end', ...msg });
      try { p.child.stdin?.end(); } catch { /* already closed */ }
      await new Promise<void>((resolve) => {
        const t = setTimeout(resolve, 400);
        p.child.once('exit', () => { clearTimeout(t); resolve(); });
      });
    }
    p.kill();
  }

  kill() {
    this.proc?.kill();
  }

  log() { return this.buf.toString(); }
}

export interface AgentFactoryOptions {
  rules: MatchRules;
  sandbox: SandboxConfig;
  /** Run JavaScript entries inside the runner (ModuleAgent) instead of as a separate program. */
  inProcess: boolean;
}

/** The agent for a spec, or null for a scripted bot of the game (which the simulation drives itself). */
export function createAgent(spec: AgentSpec, opts: AgentFactoryOptions): Agent | null {
  if (spec.kind === 'builtin') return null;
  const m = spec.manifest;
  if (m.runtime === 'policy') return new PolicyAgent(join(spec.dir, m.entry!), m.greedy ?? false, spec.name);
  if (m.runtime === 'node' && opts.inProcess) return new ModuleAgent(join(spec.dir, m.entry!), opts.rules.logBytes);
  return new ProcessAgent(m, spec.dir, opts.sandbox, opts.rules);
}

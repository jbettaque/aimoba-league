/**
 * Plays matches in parallel, one worker process per slot.
 *
 * A match keeps a core busy (its two agents add their own processes while they think), so matches are farmed out
 * to worker processes instead of being awaited side by side in one. A worker that dies takes only its current
 * match with it; the pool starts a fresh one and reports that match as failed.
 */
import { fork, type ChildProcess } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { REPO_ROOT } from './entry.ts';
import type { MatchOptions, MatchResult } from './match.ts';

export interface MatchJob extends Omit<MatchOptions, 'onProgress'> {
  /** Write the replay to this file (gzip when it ends in .gz) instead of returning it. */
  replayFile?: string;
}

interface Slot { child: ChildProcess; busy: boolean; job: Pending | null }
interface Pending { job: MatchJob; resolve: (r: MatchResult) => void; reject: (e: Error) => void }

const WORKER = join(REPO_ROOT, 'src', 'agent', 'worker.ts');

/** A sensible number of matches to play at once: two agents and the game share the cores of one match. */
export const defaultPoolSize = () => Math.max(1, Math.floor(availableParallelism() / 2));

export class MatchPool {
  readonly size: number;
  private slots: Slot[] = [];
  private queue: Pending[] = [];
  private closed = false;

  constructor(size = defaultPoolSize()) {
    this.size = Math.max(1, size);
  }

  /** Matches waiting for a worker plus those being played. */
  get load(): number {
    return this.queue.length + this.slots.filter((s) => s.busy).length;
  }

  run(job: MatchJob): Promise<MatchResult> {
    if (this.closed) return Promise.reject(new Error('the match pool is closed'));
    return new Promise<MatchResult>((resolve, reject) => {
      this.queue.push({ job, resolve, reject });
      this.pump();
    });
  }

  private spawn(): Slot {
    const child = fork(WORKER, [], { stdio: ['ignore', 'inherit', 'inherit', 'ipc'], serialization: 'advanced' });
    const slot: Slot = { child, busy: false, job: null };
    child.on('message', (msg: { ok: boolean; result?: MatchResult; error?: string }) => {
      const p = slot.job;
      slot.job = null;
      slot.busy = false;
      if (p) {
        if (msg.ok) p.resolve(msg.result!);
        else p.reject(new Error(msg.error ?? 'the match failed'));
      }
      this.pump();
    });
    child.on('exit', (code) => {
      this.slots = this.slots.filter((s) => s !== slot);
      const p = slot.job;
      slot.job = null;
      if (p) p.reject(new Error(`the match worker exited with code ${code} in the middle of a match`));
      if (!this.closed) this.pump();
    });
    this.slots.push(slot);
    return slot;
  }

  private pump() {
    while (this.queue.length) {
      const slot = this.slots.find((s) => !s.busy) ?? (this.slots.length < this.size ? this.spawn() : null);
      if (!slot) return;
      const p = this.queue.shift()!;
      slot.busy = true;
      slot.job = p;
      slot.child.send(p.job, (err) => {
        if (!err || slot.job !== p) return;
        slot.job = null;
        slot.busy = false;
        p.reject(err);
      });
    }
  }

  /** Stop the workers. Matches still running are abandoned. */
  close() {
    this.closed = true;
    for (const p of this.queue) p.reject(new Error('the match pool was closed'));
    this.queue = [];
    for (const s of this.slots) {
      try { s.child.kill(); } catch { /* already gone */ }
    }
    this.slots = [];
  }
}

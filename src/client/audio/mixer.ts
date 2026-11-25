/**
 * The mixing rules, free of WebAudio so they can be exercised headlessly: given a request to play a
 * sound, decide whether it plays at all and at what gain, pan and pitch. Keeps the mix readable
 * when a lot happens at once (repeat gaps, a voice budget by priority, distance falloff).
 */
import { dbToGain } from './dsp.ts';
import { SPEC, type Bus } from './bank.ts';

export interface PlayOpts {
  /** World position; without one the sound is not spatial. */
  x?: number;
  y?: number;
  /** Linear volume multiplier. */
  vol?: number;
  /** Playback-rate multiplier (pitch). */
  rate?: number;
  /** A sound about the player's own hero: never fades with distance, cuts through the mix. */
  own?: boolean;
  /** Start this many seconds from now. */
  delay?: number;
}

export interface Listener {
  x: number;
  y: number;
  /** Distance at which an ordinary sound has faded out completely. */
  range: number;
}

export interface Voice {
  name: string;
  take: number;
  bus: Bus;
  gain: number;
  /** -1 (left) .. 1 (right). */
  pan: number;
  rate: number;
  /** Reverb send level. */
  send: number;
  /** Far enough away to sound muffled. */
  far: boolean;
  delay: number;
  /** dB to push the rest of the mix down by while this plays. */
  duck: number;
  dur: number;
}

/** What is known about a rendered sound: seconds per take, and the dB each take sits under the loudness target. */
export interface TakeInfo {
  dur: number;
  short: number;
}

/** Sounds quieter than this are not worth a voice. */
const FLOOR = 0.012;
/** World voices allowed before sounds of each priority (index) are dropped. */
const BUDGET = [Infinity, 30, 24, 18, 12];
/** Boost for sounds about the player's own hero, in dB. */
const OWN_DB = 3;

export class Mixer {
  listener: Listener = { x: 0, y: 0, range: 2600 };
  /** Rendered takes per sound; a sound that is not here yet cannot play. */
  takes = new Map<string, TakeInfo[]>();
  /** Why sounds were refused, for the mix report. */
  dropped: Record<string, number> = {};
  private last = new Map<string, number>();
  private lastTake = new Map<string, number>();
  /** End times of playing world voices. */
  private ends: number[] = [];
  private rand: () => number;

  constructor(rand: () => number = Math.random) {
    this.rand = rand;
  }

  reset() {
    this.last.clear();
    this.ends.length = 0;
    this.dropped = {};
  }

  /** How loud and where a sound at (x, y) is for the listener: [gain 0..1, pan, far]. */
  locate(x: number, y: number, reach = 1): [number, number, boolean] {
    const L = this.listener;
    const dx = x - L.x, dy = y - L.y;
    // Screen space is wider than it is deep, so depth counts a little more.
    const d = Math.hypot(dx, dy * 1.25) / (L.range * reach);
    const att = d < 0.25 ? 1 : d >= 1 ? 0 : ((1 - d) / 0.75) ** 1.6;
    return [att, Math.max(-1, Math.min(1, dx / (L.range * 0.6))) * 0.75, d > 0.6];
  }

  voicesNow(now: number): number {
    const e = this.ends;
    for (let i = e.length - 1; i >= 0; i--) if (e[i] <= now) { e[i] = e[e.length - 1]; e.pop(); }
    return e.length;
  }

  private drop(why: string): null {
    this.dropped[why] = (this.dropped[why] ?? 0) + 1;
    return null;
  }

  /** Decide how (and whether) to play `name` at time `now` (seconds). */
  decide(name: string, o: PlayOpts, now: number): Voice | null {
    const s = SPEC[name], takes = this.takes.get(name);
    if (!s) throw new Error(`unknown sound ${name}`);
    if (!takes) return this.drop('not rendered yet');
    let g = dbToGain(s.db) * (o.vol ?? 1), pan = 0, far = false;
    if (o.x !== undefined && o.y !== undefined) {
      const [att, p, f] = this.locate(o.x, o.y, s.range);
      if (o.own) pan = p * 0.5;
      else { g *= att; pan = p; far = f; }
    }
    if (o.own) g *= dbToGain(OWN_DB);
    if (g < FLOOR) return this.drop('too quiet');
    const at = now + (o.delay ?? 0);
    if ((at - (this.last.get(name) ?? -1e9)) * 1000 < s.gap) return this.drop('repeat');
    const world = s.bus === 'world';
    if (world && !o.own && this.voicesNow(now) >= BUDGET[s.prio]) return this.drop('busy');
    this.last.set(name, at);

    let take = 0;
    if (takes.length > 1) {
      take = (this.rand() * (takes.length - 1)) | 0;
      if (take >= (this.lastTake.get(name) ?? -1)) take++;
      this.lastTake.set(name, take);
    }
    const rate = (o.rate ?? 1) * (1 + (this.rand() * 2 - 1) * s.pitch);
    if (s.prio >= 3) g *= 0.88 + this.rand() * 0.24;
    const dur = takes[take].dur / rate;
    if (world) this.ends.push(at + dur);
    return { name, take, bus: s.bus, gain: g * dbToGain(takes[take].short), pan, rate, send: s.verb * (far ? 1.6 : 1), far, delay: o.delay ?? 0, duck: s.duck, dur };
  }
}

/**
 * Headless mix check: plays a bot match through the sound director and the mixer (no WebAudio) and
 * measures what a player would be hearing: how many sounds, how many at once, and how loud the sum
 * gets. Used by the soundcheck tool and the tests to keep the mix from drifting into mud.
 */
import { DT } from '../../sim/defs.ts';
import { Game } from '../../sim/game.ts';
import { BANK, REF_DB, SPEC, renderSound, type Take } from './bank.ts';
import { SoundDirector, type AudioOut, type SoundCtx } from './director.ts';
import { SR, dbToGain, rng } from './dsp.ts';
import { Mixer, type PlayOpts } from './mixer.ts';

export interface MatchReport {
  hero: string;
  asked: number;
  played: number;
  dropped: Record<string, number>;
  voices: { p50: number; p95: number; max: number };
  /** Level of the summed mix in dBFS over 100 ms windows (silent windows excluded). */
  level: { p50: number; p95: number; max: number };
  /** Share of total sound energy per group (first part of the sound name). */
  groups: [string, number][];
  top: [string, number][];
  counts: Record<string, number>;
}

const pct = (sorted: number[], p: number) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0);

export function simulateMatch(o: { seconds: number; seed: number; mode: 'play' | 'spectate'; hero?: number; takes?: Map<string, Take[]> }): MatchReport {
  const takes = o.takes ?? new Map(BANK.map((s) => [s.name, renderSound(s)]));
  const rand = rng(o.seed);
  const mixer = new Mixer(rand);
  for (const [name, ts] of takes) mixer.takes.set(name, ts.map((t) => ({ dur: t.chans[0].length / SR, short: t.short })));
  const slot = o.hero ?? 2;
  const game = new Game({ seed: o.seed, recordEvents: true });
  const focus = game.heroes[slot];

  const WIN = 0.1, windows = new Float64Array(Math.ceil(o.seconds / WIN) + 600);
  const voiceSamples: number[] = [];
  const counts: Record<string, number> = {}, energy: Record<string, number> = {};
  let asked = 0, played = 0, now = 0;
  const loopGain = new Map<string, [string, number]>();

  /** Add a sound's power to every window it overlaps (sources are uncorrelated, so powers add). */
  const spread = (name: string, start: number, dur: number, gain: number) => {
    const power = dbToGain(REF_DB) ** 2 * gain * gain;
    const group = name.split('.')[0];
    for (let w = Math.floor(start / WIN); w * WIN < start + dur && w < windows.length; w++) {
      const overlap = Math.min(start + dur, (w + 1) * WIN) - Math.max(start, w * WIN);
      windows[w] += (power * overlap) / WIN;
      energy[group] = (energy[group] ?? 0) + power * overlap;
    }
  };
  const out: AudioOut = {
    play(name: string, opts: PlayOpts = {}) {
      asked++;
      const v = mixer.decide(name, opts, now);
      if (!v) return;
      played++;
      counts[name] = (counts[name] ?? 0) + 1;
      // A sound's loudness figure is its loudest 200 ms; over its whole length it averages lower.
      spread(name, now + v.delay, Math.min(v.dur, 0.6), v.gain / dbToGain(mixer.takes.get(name)![v.take].short));
    },
    loop(key: string, name: string, vol: number, lo = {}) {
      let g = dbToGain(SPEC[name].db) * vol;
      if (lo.x !== undefined && lo.y !== undefined && !lo.own) g *= mixer.locate(lo.x, lo.y, SPEC[name].range)[0];
      loopGain.set(key, [name, g]);
    },
    scene() {},
  };
  const director = new SoundDirector(out, rand);
  const ctx: SoundCtx = {
    game, mode: o.mode, focus,
    shown: (x, y) => {
      if (o.mode !== 'play') return true;
      for (const u of game.units) if (u.team === focus.team && u.alive && (u.x - x) ** 2 + (u.y - y) ** 2 < 1050 * 1050) return true;
      return false;
    },
    camX: focus.x, camY: focus.y, paused: false,
  };

  const steps = Math.round(o.seconds / DT);
  for (let i = 0; i < steps && !game.over; i++) {
    game.step();
    now = game.time;
    mixer.listener.x = ctx.camX = focus.x;
    mixer.listener.y = ctx.camY = focus.y;
    for (const ev of game.events) director.event(ev, ctx);
    game.events.length = 0;
    if (i % 2 === 0) {
      loopGain.clear();
      director.frame(DT * 2, ctx);
      for (const [name, g] of loopGain.values()) spread(name, now, DT * 2, g);
    }
    if (i % 3 === 0) voiceSamples.push(mixer.voicesNow(now));
  }

  const levels = [...windows.subarray(0, Math.ceil(now / WIN))].filter((p) => p > 0).map((p) => 10 * Math.log10(p)).sort((a, b) => a - b);
  voiceSamples.sort((a, b) => a - b);
  const total = Object.values(energy).reduce((a, b) => a + b, 0) || 1;
  return {
    hero: `${focus.hero!.def}`,
    asked, played, dropped: { ...mixer.dropped },
    voices: { p50: pct(voiceSamples, 0.5), p95: pct(voiceSamples, 0.95), max: voiceSamples[voiceSamples.length - 1] ?? 0 },
    level: { p50: pct(levels, 0.5), p95: pct(levels, 0.95), max: levels[levels.length - 1] ?? -Infinity },
    groups: Object.entries(energy).map(([k, v]): [string, number] => [k, v / total]).sort((a, b) => b[1] - a[1]),
    top: Object.entries(counts).sort((a, b) => b[1] - a[1]),
    counts,
  };
}

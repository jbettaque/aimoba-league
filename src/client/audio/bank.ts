/**
 * The sound bank: every effect, ambience bed and musical phrase in the game, synthesised from
 * scratch (no audio assets). Each entry is a small recipe built from ./dsp.ts plus its place in
 * the mix. Entries are rendered once (in a worker), loudness-normalised, then played as buffers.
 *
 * Mix levels (`db`) are relative to one loudness reference, so they compare directly:
 *   -3 .. -6   match-defining moments (victory, ultimates landing, your own death)
 *   -7 .. -11  abilities, important feedback for your own hero
 *  -12 .. -17  basic attacks, monsters, towers, small feedback
 *  -19 .. -27  minions and UI ticks: texture, never in the way
 */
import {
  ad, brightness, comb, crackle, dbToGain, drive, echo, fadeEdges, filt, fm, gain, hump, line, loudness, mix, modal, noise, note, osc, peak, pluck,
  reverse, rng, samples, seamless, shape, softLimit, xline,
  type Buf, type Color, type FilterMode, type Fn, type Layer, type P, type Rand, SR,
} from './dsp.ts';

export type Bus = 'world' | 'self' | 'ui' | 'sting' | 'amb' | 'music';

export interface SoundSpec {
  name: string;
  bus: Bus;
  /** Mix level in dB relative to the loudness reference. */
  db: number;
  /** 0 = always plays ... 4 = texture, first to be dropped when the mix is busy. */
  prio: number;
  /** Minimum milliseconds between two plays of this sound. */
  gap: number;
  /** Random pitch spread per play (+- fraction). */
  pitch: number;
  /** How many different takes are rendered and rotated through. */
  variants: number;
  /** Reverb send, 0..1. */
  verb: number;
  /** Multiplier on how far the sound carries. */
  range: number;
  loop: boolean;
  /** dB by which this sound pushes the world, ambience and music down while it plays. */
  duck: number;
  make: (r: Rand, variant: number) => Buf | [Buf, Buf];
}

/** Loudness every sound is normalised to before its `db` is applied. */
export const REF_DB = -12;

const DEFAULTS: [string, Partial<SoundSpec>][] = [
  ['ui.', { bus: 'ui', prio: 0, gap: 30 }],
  ['self.', { bus: 'self', prio: 0, gap: 60, verb: 0.08 }],
  ['sting.', { bus: 'sting', prio: 0, gap: 400, verb: 0.3, duck: 5 }],
  ['aa.', { prio: 3, gap: 50, pitch: 0.05, variants: 3, verb: 0.06 }],
  ['hit.', { prio: 3, gap: 60, pitch: 0.06, variants: 2, verb: 0.08 }],
  ['min.', { prio: 4, gap: 75, pitch: 0.08, variants: 3, verb: 0.05 }],
  ['mon.', { prio: 3, gap: 90, pitch: 0.06, variants: 2, verb: 0.12 }],
  ['tower.', { prio: 2, gap: 90, pitch: 0.03, verb: 0.15 }],
  ['ab.', { prio: 2, gap: 60, pitch: 0.02, verb: 0.16 }],
  ['amb.', { bus: 'amb', prio: 1 }],
  ['mus.', { bus: 'music', prio: 0 }],
];

export const BANK: SoundSpec[] = [];
export const SPEC: Record<string, SoundSpec> = {};

function def(name: string, db: number, make: SoundSpec['make'], over: Partial<SoundSpec> = {}) {
  const base = DEFAULTS.find(([p]) => name.startsWith(p))?.[1] ?? {};
  const s: SoundSpec = { name, bus: 'world', db, prio: 2, gap: 80, pitch: 0, variants: 1, verb: 0.18, range: 1, loop: false, duck: 0, make, ...base, ...over };
  BANK.push(s);
  SPEC[name] = s;
}

// --- Building blocks ---------------------------------------------------------------------------
const TAU = Math.PI * 2;
const N = note;

/** Scale to a peak of 1, so layer gains in a recipe compare like-for-like. */
function norm(b: Buf): Buf {
  const p = peak(b);
  return p > 0 ? gain(b, 1 / p) : b;
}
/** The body of an impact: a sine dropping in pitch. */
const thump = (dur: number, f0: number, f1: number, k = 6) => osc(dur, 'sine', xline(0, f0, dur * 0.5, f1), ad(0.002, dur, k));
/** A filtered noise hit: smack, crack, puff. */
const burst = (r: Rand, dur: number, mode: FilterMode, fc: P, q = 0.8, atk = 0.001, color: Color = 'white') =>
  norm(shape(filt(noise(dur, r, color), mode, fc, q), ad(atk, dur)));
/** Air being moved: band-passed noise whose centre sweeps f0 -> f1 -> f2 under a swell. */
const whoosh = (r: Rand, dur: number, f0: number, f1: number, f2: number, q = 1.4, pk = 0.4) =>
  norm(shape(filt(noise(dur, r), 'bp', xline(0, f0, dur * pk, f1, dur, f2), q), hump(dur, pk)));
/** A sparkling high band with a fast flutter. */
const shimmer = (r: Rand, dur: number, f = 6000, pk = 0.3) =>
  norm(shape(filt(noise(dur, r), 'bp', f, 3), (t) => hump(dur, pk)(t) * (0.6 + 0.4 * Math.sin(TAU * 11 * t))));
/** A blade edge: a fast upward hiss. */
const slice = (r: Rand, dur: number, f0 = 2600, f1 = 8000) => norm(shape(filt(noise(dur, r), 'bp', xline(0, f0, dur, f1), 1.6), hump(dur, 0.25)));
/** Filtered clicks: embers, grit, chain links. */
const grit = (r: Rand, dur: number, rate: P, mode: FilterMode, fc: number, q = 1, decay = 0.002) => norm(filt(crackle(dur, r, rate, decay), mode, fc, q));
/** A slow random wobble between 0 and 1. */
function drift(r: Rand, dur: number, rate: number): Fn {
  const b = norm(filt(noise(dur, r), 'lp', rate, 0.6));
  return (t) => 0.5 + 0.5 * b[Math.min(b.length - 1, (t * SR) | 0)];
}

// Struck-object spectra: [frequency ratio, amplitude, decay multiplier].
const METAL = [[1, 1, 1], [2.32, 0.6, 0.75], [3.86, 0.45, 0.55], [5.1, 0.3, 0.4], [6.71, 0.2, 0.3], [8.24, 0.12, 0.2]];
const BELL = [[0.5, 0.25, 1.3], [1, 1, 1], [2, 0.5, 0.8], [3.01, 0.3, 0.55], [4.17, 0.2, 0.4], [5.43, 0.12, 0.3]];
const GLASS = [[1, 1, 1], [2.76, 0.45, 0.6], [5.4, 0.22, 0.35], [8.93, 0.1, 0.2]];
const COIN = [[1, 1, 1], [1.51, 0.55, 0.8], [2.77, 0.4, 0.55], [4.13, 0.2, 0.35]];
const WOOD = [[1, 1, 1], [2.4, 0.35, 0.5], [5.2, 0.12, 0.3]];
const MALLET = [[1, 1, 1], [4, 0.22, 0.3], [9.2, 0.06, 0.15]];

const coin = (f: number, dur = 0.3) => modal(dur, f, COIN, dur * 0.8);
const bell = (dur: number, f: number) => modal(dur, f, BELL, dur * 0.85);
/** A soft FM chime: bright at the strike, pure as it rings. */
const chime = (dur: number, f: number) => fm(dur, f, 3.5, (t) => 2.2 * Math.exp(-t * 14), ad(0.002, dur));

/** A brass-like note: detuned saws under a low-pass that opens with the attack. */
function brass(dur: number, f: number, bright = 1, atk = 0.03): Buf {
  const b = mix(dur, osc(dur, 'saw', f * 1.004), osc(dur, 'saw', f * 0.996, 1, 0.37), [osc(dur, 'saw', f * 0.5, 1, 0.61), 0.5]);
  filt(b, 'lp', (t) => f * (1.5 + 4.5 * bright * Math.min(1, t / (atk * 2)) * (0.55 + 0.45 * Math.exp(-t * 3))), 0.9);
  return norm(shape(b, line(0, 0, atk, 1, Math.max(atk, dur - 0.16), 0.75, dur, 0)));
}
/** Several brass notes at once. */
const chord = (dur: number, midi: number[], bright = 1, atk = 0.03): Buf => norm(mix(dur, ...midi.map((m): Layer => [brass(dur, N(m), bright, atk), 1])));

/** An explosion; `size` 0..1 runs from a pop to a blast. */
function boom(r: Rand, dur: number, size: number): Buf {
  const lo = 95 - 45 * size;
  return drive(mix(dur,
    [thump(Math.min(dur, 0.35 + 0.7 * size), lo, lo * 0.4, 5), 1],
    [norm(shape(filt(noise(dur, r, 'brown'), 'lp', xline(0, 3500 + 3000 * size, dur * 0.6, 220), 0.7), ad(0.004, dur, 5))), 0.9],
    [burst(r, 0.05, 'hp', 1800, 0.7), 0.5],
    [shape(grit(r, dur, line(0, 120, dur, 10), 'lp', 3000, 0.7, 0.003), ad(0.02, dur, 4)), 0.25, 0.03],
  ), 1.6 + size);
}

/** A creature's voice: a rough, unsteady saw pushed through a moving formant. */
function growl(r: Rand, dur: number, f0: number, f1: number, form0: number, form1: number): Buf {
  const jit = filt(noise(dur, r), 'lp', 25), fr = xline(0, f0, dur, f1);
  const src = osc(dur, 'saw', (t) => fr(t) * (1 + 3 * jit[Math.min(jit.length - 1, (t * SR) | 0)]), (t) => 1 + 0.6 * Math.sin(TAU * 34 * t));
  const body = filt(src.slice(), 'lp', 420, 0.8);
  const voice = filt(mix(dur, src, [noise(dur, r, 'pink'), 0.5]), 'bp', xline(0, form0, dur, form1), 2.5);
  return norm(shape(mix(dur, body, [voice, 1.6]), hump(dur, 0.3)));
}

/** Turn a list into layers: the notes of a phrase, the coins of a cascade. */
const seq = <T>(items: T[], f: (x: T, i: number) => Layer): Layer[] => items.map(f);

// --- Interface -----------------------------------------------------------------------------------
def('ui.click', -20, (r) => mix(0.07,
  [osc(0.05, 'sine', xline(0, 1500, 0.04, 950), ad(0.001, 0.045)), 0.8],
  [burst(r, 0.012, 'bp', 3200, 1.2, 0.0005), 0.3]));
def('ui.pick', -15, () => mix(0.6, [modal(0.5, N(72), MALLET, 0.4), 1], [modal(0.5, N(79), MALLET, 0.45), 0.7, 0.07]), { verb: 0.2 });
def('ui.start', -8, (r) => mix(1.9,
  [thump(0.7, 110, 42), 1, 0.4],
  [whoosh(r, 0.5, 300, 3200, 1200, 1.2, 0.85), 0.5],
  [bell(1.4, N(62)), 0.5, 0.42], [bell(1.4, N(69)), 0.4, 0.42], [bell(1.4, N(74)), 0.3, 0.46]), { verb: 0.35 });
def('ui.open', -22, (r) => whoosh(r, 0.16, 500, 2600, 1800, 1.5, 0.7));
def('ui.close', -23, (r) => whoosh(r, 0.14, 2400, 900, 400, 1.5, 0.25));
def('ui.buy', -12, (r) => mix(0.5,
  [thump(0.09, 210, 120), 0.6],
  [coin(2700), 0.8, 0.02], [coin(3400), 0.7, 0.07], [coin(4050), 0.6, 0.13],
  [burst(r, 0.03, 'bp', 5000, 2), 0.2]), { verb: 0.12 });
def('ui.deny', -17, () => filt(mix(0.22,
  [osc(0.08, 'square', 277, ad(0.003, 0.08, 4)), 1],
  [osc(0.1, 'square', 220, ad(0.003, 0.1, 4)), 1, 0.09]), 'lp', 1300), { gap: 120 });
def('ui.rank', -13, (r) => mix(0.7,
  [chime(0.4, 660), 0.8], [chime(0.4, 880), 0.8, 0.06], [chime(0.5, 1320), 0.8, 0.12],
  [shimmer(r, 0.5, 7000), 0.2, 0.1]), { verb: 0.2 });
def('ui.ready', -23, () => modal(0.3, 1760, GLASS, 0.25), { gap: 40 });
def('ui.readyUlt', -15, () => mix(0.8, [modal(0.6, 1320, GLASS, 0.5), 1], [modal(0.6, 1976, GLASS, 0.5), 0.7, 0.08], [modal(0.5, 2640, GLASS, 0.4), 0.4, 0.16]), { verb: 0.25 });

// --- Feedback for your own hero -----------------------------------------------------------------
def('self.gold', -12, (r, v) => {
  const f = 2900 + v * 260;
  return mix(0.36, [coin(f), 1], [coin(f * 1.335, 0.28), 0.75, 0.05], [burst(r, 0.015, 'hp', 5000), 0.15]);
}, { variants: 3, pitch: 0.03, gap: 70 });
def('self.goldBig', -9, (r) => mix(0.9,
  [thump(0.12, 190, 90), 0.7],
  ...seq([0, 2, 4, 7, 9, 12], (st, i) => [coin(2093 * 2 ** (st / 12), 0.4), 0.8 - i * 0.05, 0.03 + i * 0.06]),
  [shimmer(r, 0.6, 7500), 0.2, 0.15]), { verb: 0.2, gap: 200 });
def('self.level', -7, (r) => mix(1.6,
  [thump(0.3, 130, 55), 0.5],
  ...seq([60, 64, 67], (m, i) => [brass(0.2, N(m), 1.2, 0.012), 0.55, i * 0.085]),
  ...seq([60, 64, 67, 72], (m, i) => [chime(0.5, N(m + 12)), 0.45, i * 0.085]),
  [chord(1.1, [72, 67, 64], 1.3, 0.015), 0.7, 0.255],
  [shimmer(r, 0.9), 0.22, 0.3]), { verb: 0.3 });
def('self.hurt', -9, (r, v) => drive(mix(0.2,
  [thump(0.17, 190 - v * 14, 70), 0.8],
  [burst(r, 0.06, 'bp', 900, 0.7), 1],
  [burst(r, 0.015, 'bp', 2600, 1), 0.4]), 1.8), { variants: 3, pitch: 0.05, gap: 110 });
def('self.heartbeat', -13, (r) => {
  // A chest thump with a knock on top, so it survives speakers with no bass.
  const beat = () => drive(mix(0.16, [thump(0.16, 74, 40, 5), 0.8], [thump(0.07, 230, 110), 0.9], [burst(r, 0.035, 'bp', 600, 0.7), 0.6]), 2.2);
  return mix(0.86, [beat(), 1, 0.02], [beat(), 0.7, 0.28]);
}, { loop: true, verb: 0 });
def('self.death', -5, (r) => {
  const fall = (f: number, ph: number) => osc(1.6, 'saw', xline(0, f, 1.5, f / 2), 1, ph);
  const pad = shape(filt(mix(1.6, fall(146.8, 0), fall(110.4, 0.3), [fall(174.6, 0.6), 0.6]), 'lp', xline(0, 2200, 1.4, 180), 1), ad(0.02, 1.6, 3));
  return mix(1.9, [thump(0.8, 100, 32, 4), 0.9], [norm(pad), 0.65], [burst(r, 0.5, 'lp', xline(0, 2500, 0.4, 300)), 0.5], [modal(1.6, 294, [[1, 1, 1], [1.19, 0.6, 0.9], [2.38, 0.3, 0.6]], 1.4), 0.3]);
}, { verb: 0.35, duck: 6 });
def('self.respawn', -9, (r) => mix(1.5,
  [whoosh(r, 0.7, 300, 4000, 2500, 1.2, 0.9), 0.4],
  [osc(1.2, 'sine', N(62), hump(1.2, 0.5)), 0.4], [osc(1.2, 'sine', N(69), hump(1.2, 0.5)), 0.3],
  [chime(0.8, N(74)), 0.6, 0.55], [chime(0.8, N(81)), 0.5, 0.6], [chime(0.8, N(86)), 0.4, 0.66],
  [shimmer(r, 0.8), 0.2, 0.5]), { verb: 0.3 });
def('self.stun', -9, () => mix(0.55,
  [osc(0.5, 'sine', (t) => 740 * (1 + 0.05 * Math.sin(TAU * 9 * t)), ad(0.003, 0.5, 4)), 1],
  [osc(0.5, 'sine', (t) => 1109 * (1 + 0.05 * Math.sin(TAU * 9 * t + 1)), ad(0.003, 0.4, 4)), 0.6],
  [modal(0.2, 520, METAL, 0.15), 0.5]), { gap: 300 });
def('self.root', -11, (r) => mix(0.35,
  [filt(osc(0.3, 'saw', xline(0, 320, 0.25, 130), ad(0.005, 0.3, 4)), 'lp', 900, 2), 1],
  [grit(r, 0.3, 90, 'bp', 1800, 2), 0.6],
  [thump(0.1, 160, 80), 0.6]), { gap: 300 });
def('self.towerWarn', -9, () => {
  const ping = () => mix(0.24, [modal(0.22, 1480, GLASS, 0.18), 1], [filt(osc(0.07, 'square', 1480, ad(0.002, 0.07)), 'lp', 4000), 0.2]);
  return mix(0.42, [ping(), 1], [ping(), 1, 0.14]);
}, { gap: 1500 });

// --- Announcements (short musical phrases in D minor, the key of the score) ------------------------
def('sting.kill', -5, (r) => mix(1.3,
  [thump(0.5, 120, 45), 0.9],
  [brass(0.16, N(62), 1.2), 0.6], [brass(0.16, N(69), 1.2), 0.6, 0.1], [chord(0.8, [74, 69], 1.4), 0.8, 0.2],
  [chime(0.8, N(86)), 0.35, 0.2], [chime(0.8, N(81)), 0.3, 0.26],
  [shimmer(r, 0.7), 0.15, 0.2]));
def('sting.allyKill', -10, () => mix(0.8, [brass(0.14, N(69), 1), 0.6], [brass(0.5, N(74), 1.1), 0.7, 0.1], [chime(0.6, N(86)), 0.3, 0.1]), { duck: 2 });
def('sting.allyDeath', -11, () => mix(0.9, [thump(0.3, 90, 45), 0.5], [brass(0.2, N(65), 0.75), 0.7], [brass(0.6, N(62), 0.7), 0.8, 0.17]), { duck: 2 });
def('sting.firstBlood', -4, (r) => mix(1.7,
  [reverse(burst(r, 0.35, 'hp', 3000, 0.7, 0.001)), 0.35],
  [thump(0.7, 110, 36, 4), 1, 0.35],
  [chord(1.2, [50, 57, 62, 65], 1.5, 0.02), 0.9, 0.35],
  [burst(r, 0.6, 'lp', xline(0, 4000, 0.5, 400)), 0.35, 0.35],
  [bell(1.2, N(74)), 0.3, 0.35]), { duck: 6 });
def('sting.towerGood', -6, (r) => mix(1.3,
  ...seq([65, 69, 72], (m, i) => [brass(0.15, N(m), 1.2, 0.012), 0.6, i * 0.09]),
  [chord(0.9, [77, 72, 69], 1.3, 0.015), 0.85, 0.27],
  [chime(0.8, N(89)), 0.3, 0.27], [shimmer(r, 0.6), 0.15, 0.3]));
def('sting.towerBad', -6, () => mix(1.3,
  [thump(0.5, 95, 38), 0.8],
  [brass(0.2, N(62), 0.8), 0.7], [brass(0.2, N(58), 0.8), 0.7, 0.16], [chord(0.9, [55, 43], 0.75), 0.9, 0.32]));
def('sting.objGood', -5, (r) => mix(1.9,
  [thump(0.6, 110, 40), 0.9],
  [chord(0.34, [58, 62, 65], 1.2), 0.75], [chord(0.34, [60, 64, 67], 1.2), 0.8, 0.3], [chord(1.2, [62, 66, 69, 74], 1.4), 0.9, 0.6],
  [thump(0.5, 100, 40), 0.7, 0.6],
  ...seq([86, 90, 93], (m, i) => [chime(0.8, N(m)), 0.28, 0.6 + i * 0.1]),
  [shimmer(r, 1), 0.15, 0.6]), { duck: 6 });
def('sting.objBad', -5, () => mix(1.8,
  [thump(0.7, 95, 34, 4), 0.9],
  [chord(0.4, [62, 53, 57], 0.8, 0.05), 0.8], [chord(0.4, [60, 51, 55], 0.75, 0.05), 0.8, 0.35], [chord(1.1, [58, 49, 53, 34], 0.7, 0.05), 0.9, 0.7]), { duck: 6 });
def('sting.awaken', -6, (r) => mix(2.5,
  [thump(0.9, 80, 30, 4), 0.9],
  [norm(shape(filt(mix(2.3, osc(2.3, 'saw', 73.4), osc(2.3, 'saw', 77.8, 1, 0.4)), 'lp', xline(0, 300, 1.4, 1500, 2.3, 400), 1.2), hump(2.3, 0.55))), 0.6],
  [growl(r, 1.4, 72, 54, 320, 720), 0.7, 0.35],
  [brass(1.1, N(50), 0.9, 0.2), 0.5, 1.1]), { duck: 6, verb: 0.4 });
def('sting.horn', -9, () => mix(1.7, [brass(1.5, N(50), 0.9, 0.2), 1], [brass(1.5, N(57), 0.9, 0.25), 0.7, 0.04]), { duck: 2, verb: 0.45 });
def('sting.victory', -3, (r) => mix(3.8,
  [thump(0.6, 110, 40), 0.9], [thump(0.7, 100, 36), 0.9, 0.8],
  [chord(0.44, [58, 62, 65, 70], 1.3), 0.75], [chord(0.44, [60, 64, 67, 72], 1.3), 0.8, 0.4], [chord(2.8, [62, 66, 69, 74, 50], 1.5), 0.95, 0.8],
  ...seq([86, 90, 93, 98, 93, 98], (m, i) => [chime(1, N(m)), 0.3, 0.8 + i * 0.11]),
  [shimmer(r, 2.2, 7000, 0.2), 0.16, 0.8]), { duck: 9, verb: 0.4 });
def('sting.defeat', -3, () => mix(3.8,
  [thump(0.9, 90, 30, 4), 0.9],
  ...seq([62, 60, 58], (m, i) => [chord(0.78, [m, m - 12], 0.75, 0.1), 0.8, i * 0.7]),
  [chord(1.7, [57, 50, 53, 38], 0.7, 0.12), 0.95, 2.1]), { duck: 9, verb: 0.4 });
def('sting.neutral', -11, () => mix(0.9, [chime(0.6, N(81)), 0.7], [chime(0.7, N(86)), 0.7, 0.11], [modal(0.5, N(74), MALLET, 0.4), 0.5, 0.11]), { duck: 0 });

// --- Basic attacks --------------------------------------------------------------------------------
def('aa.korvath.hit', -13, (r, v) => drive(mix(0.34,
  [thump(0.2, 140, 55), 1],
  [burst(r, 0.05, 'lp', 1800), 0.7],
  [modal(0.3, 500 + v * 45, METAL, 0.24), 0.32]), 1.5), { verb: 0.1 });
def('aa.sable.hit', -14, (r, v) => mix(0.22,
  [slice(r, 0.07, 2400, 7000), 0.8],
  [modal(0.18, 2800 + v * 240, [[1, 1, 1], [1.83, 0.5, 0.7]], 0.12), 0.3, 0.01],
  [thump(0.06, 220, 120), 0.6, 0.015]));
def('aa.ignara.fire', -17, (r) => mix(0.16,
  [norm(shape(filt(noise(0.15, r), 'lp', xline(0, 300, 0.04, 2200, 0.15, 600), 0.9), hump(0.15, 0.25))), 1],
  [osc(0.08, 'sine', xline(0, 300, 0.08, 520), hump(0.08, 0.3)), 0.4]));
def('aa.ignara.hit', -15, (r) => mix(0.2,
  [burst(r, 0.09, 'lp', xline(0, 3000, 0.08, 500)), 0.9],
  [thump(0.09, 180, 90), 0.7],
  [grit(r, 0.18, 70, 'bp', 3000, 1.5), 0.25, 0.02]));
def('aa.vesper.fire', -17, (r, v) => filt(mix(0.2,
  [norm(shape(pluck(0.16, 200 + v * 14, r, 0.7), ad(0.001, 0.16, 4))), 0.9],
  [burst(r, 0.008, 'bp', 3000, 1), 0.2],
  [whoosh(r, 0.07, 2600, 2000, 1100, 2, 0.2), 0.2, 0.005]), 'lp', 5000));
def('aa.vesper.hit', -15, (r, v) => mix(0.12,
  [thump(0.07, 260, 140), 0.9],
  [modal(0.09, 760 + v * 60, WOOD, 0.07), 0.6],
  [burst(r, 0.01, 'bp', 3000, 1), 0.3]));
def('aa.liora.fire', -18, (r) => mix(0.18,
  [fm(0.16, xline(0, 1300, 0.16, 900), 2, (t) => 1.5 * Math.exp(-t * 20), ad(0.004, 0.16)), 0.9],
  [burst(r, 0.06, 'hp', 5000, 0.7, 0.004), 0.15]));
def('aa.liora.hit', -16, (r, v) => mix(0.28,
  [modal(0.25, 1568 * (1 + v * 0.06), GLASS, 0.22), 0.9],
  [burst(r, 0.03, 'bp', 2600, 0.8, 0.002), 0.25]));
def('hit.stone', -16, (r) => mix(0.2,
  [thump(0.12, 110, 60), 1],
  [burst(r, 0.06, 'lp', 900), 0.8],
  [grit(r, 0.16, 110, 'bp', 1500, 0.8, 0.003), 0.3, 0.01]));

// --- Minions, monsters, structures ------------------------------------------------------------------
def('min.melee', -24, (r, v) => mix(0.12,
  [modal(0.09, 2100 + v * 190, [[1, 1, 1], [2.7, 0.5, 0.6]], 0.07), 0.6],
  [thump(0.045, 300, 160), 0.8],
  [burst(r, 0.012, 'bp', 2000, 1), 0.3]), { variants: 4 });
def('min.ranged.fire', -26, () => osc(0.07, 'sine', xline(0, 1100, 0.06, 480), ad(0.002, 0.07)));
def('min.ranged.hit', -27, (r) => mix(0.05, [burst(r, 0.02, 'bp', 2400, 1.2), 0.8], [thump(0.035, 320, 180), 0.7]));
def('min.siege.fire', -20, (r) => mix(0.26, [thump(0.22, 120, 45), 1], [burst(r, 0.15, 'lp', xline(0, 1500, 0.12, 300)), 0.6]), { variants: 2 });
def('min.siege.hit', -20, (r) => boom(r, 0.32, 0.1), { variants: 2 });
def('min.super', -19, (r, v) => drive(mix(0.26, [thump(0.2, 110, 50), 1], [modal(0.2, 360 + v * 30, METAL, 0.16), 0.35], [burst(r, 0.04, 'lp', 1500), 0.6]), 1.5), { variants: 2, prio: 3 });
def('min.death', -24, (r) => mix(0.16,
  [norm(shape(filt(noise(0.14, r), 'lp', xline(0, 1800, 0.12, 300), 0.8), hump(0.14, 0.15))), 0.9],
  [osc(0.09, 'sine', xline(0, 500, 0.08, 150), ad(0.002, 0.09)), 0.5]), { gap: 110 });
def('mon.hit', -17, (r) => mix(0.2, [whoosh(r, 0.08, 900, 2500, 1500, 1.5, 0.4), 0.6], [thump(0.14, 130, 60), 1, 0.03], [burst(r, 0.04, 'lp', 1200), 0.5, 0.03]));
def('mon.big', -12, (r) => drive(mix(0.5,
  [thump(0.35, 90, 35, 5), 1], [burst(r, 0.12, 'bp', 900, 0.7), 0.8],
  [growl(r, 0.3, 95, 70, 400, 800), 0.45]), 1.6), { prio: 2, verb: 0.2 });
def('mon.death', -15, (r) => mix(0.6, [growl(r, 0.5, 130, 60, 900, 350), 0.9], [thump(0.2, 110, 50), 0.7, 0.3]), { verb: 0.2 });
def('mon.bigDeath', -7, (r) => mix(2,
  [growl(r, 1.6, 110, 42, 850, 260), 1],
  [boom(r, 1.1, 0.5), 0.8, 0.7]), { variants: 1, prio: 1, verb: 0.4, range: 2 });
def('tower.fire', -13, (r) => mix(0.3,
  [fm(0.22, xline(0, 520, 0.2, 160), 1.5, line(0, 4, 0.2, 0.5), ad(0.003, 0.22)), 0.9],
  [burst(r, 0.07, 'hp', 2500, 0.7, 0.002), 0.3],
  [thump(0.14, 100, 50), 0.7]), { variants: 2 });
def('tower.hit', -12, (r) => drive(mix(0.28,
  [thump(0.2, 160, 50), 1],
  [burst(r, 0.07, 'bp', 1800, 0.7), 0.6],
  [grit(r, 0.14, 260, 'hp', 2500, 0.7, 0.001), 0.3]), 1.5), { variants: 2 });
def('tower.death', -5, (r) => mix(2.2,
  [boom(r, 1.5, 0.75), 1],
  [shape(grit(r, 1.8, line(0, 90, 1.8, 6), 'lp', 2200, 0.7, 0.004), ad(0.05, 1.8, 3)), 0.5, 0.15],
  [norm(shape(filt(osc(0.9, 'saw', xline(0, 190, 0.9, 70)), 'lp', 700, 3), hump(0.9, 0.2))), 0.35, 0.1]), { prio: 1, range: 2.5, verb: 0.4, pitch: 0 });
def('core.death', -2, (r) => mix(3.6,
  [boom(r, 2.6, 1), 1],
  [echo(boom(r, 2.2, 0.8), 0.31, 0.45, 0.6), 0.6, 0.35],
  [osc(1.8, 'sine', xline(0, 70, 1.6, 24), ad(0.01, 1.8, 4)), 0.8],
  ...seq([0, 1, 2, 3, 4, 5, 6], (i) => [modal(0.9, 1500 + 620 * i * (0.7 + 0.6 * r()), GLASS, 0.7), 0.16, 0.05 + r() * 0.5]),
  [shimmer(r, 2.4, 6000, 0.1), 0.2, 0.2]), { prio: 0, range: 20, verb: 0.5, duck: 8 });
def('hero.death', -7, (r) => mix(1.1,
  [thump(0.6, 110, 35, 5), 1],
  [norm(shape(filt(osc(0.7, 'saw', xline(0, 300, 0.6, 62)), 'lp', xline(0, 1800, 0.6, 250), 1), ad(0.005, 0.7, 4))), 0.5],
  [burst(r, 0.25, 'lp', xline(0, 3000, 0.2, 400)), 0.5],
  [modal(0.9, 330, [[1, 1, 1], [1.19, 0.6, 0.9], [2.4, 0.3, 0.5]], 0.8), 0.3]), { prio: 1, verb: 0.3 });

// --- Korvath: stone and iron -------------------------------------------------------------------------
def('ab.korvath.q', -10, (r) => mix(0.45,
  [whoosh(r, 0.42, 150, 700, 300, 1.2, 0.5), 1],
  [shape(grit(r, 0.4, 70, 'bp', 1800, 4), hump(0.4, 0.5)), 0.35],
  [thump(0.15, 110, 55), 0.6]));
def('fx.impact', -8, (r) => drive(mix(0.7,
  [thump(0.3, 150, 45), 1],
  [modal(0.6, 410, METAL, 0.5), 0.45],
  [burst(r, 0.06, 'lp', 2200), 0.7],
  [modal(0.3, 1900, GLASS, 0.25), 0.12]), 1.7));
def('ab.korvath.w', -10, (r) => mix(1,
  [modal(0.9, 660, METAL, 0.8), 0.7], [modal(0.9, 990, METAL, 0.6), 0.35],
  [osc(0.7, 'sine', 110, hump(0.7, 0.3)), 0.6],
  [burst(r, 0.02, 'bp', 3000, 1), 0.3],
  [shimmer(r, 0.6, 5000), 0.12, 0.05]));
def('fx.quake', -8, (r) => drive(mix(0.75,
  [thump(0.5, 100, 32, 5), 1],
  [burst(r, 0.09, 'bp', 700, 0.8), 0.6],
  [norm(shape(filt(noise(0.6, r, 'brown'), 'lp', xline(0, 900, 0.5, 160), 0.8), ad(0.004, 0.6, 4))), 0.9],
  [shape(grit(r, 0.6, line(0, 140, 0.6, 15), 'lp', 2500, 0.7, 0.003), ad(0.01, 0.6, 3)), 0.35, 0.03]), 2));
def('ab.korvath.r', -8, (r) => mix(0.6,
  [whoosh(r, 0.55, 200, 1500, 900, 1.2, 0.8), 0.9],
  [norm(shape(filt(osc(0.55, 'saw', xline(0, 60, 0.55, 125)), 'lp', 450, 1.5), hump(0.55, 0.75))), 0.7],
  [modal(0.3, 700, METAL, 0.25), 0.15]), { range: 1.5 });
def('fx.cataclysm', -4, (r) => mix(1.8,
  [boom(r, 1.5, 0.9), 1],
  [modal(1, 180, METAL, 0.9), 0.3],
  [shape(grit(r, 1.2, line(0, 110, 1.2, 8), 'lp', 2600, 0.7, 0.004), ad(0.03, 1.2, 3)), 0.3, 0.1]), { prio: 1, range: 1.6, verb: 0.35 });

// --- Sable: shadow and steel ---------------------------------------------------------------------------
def('ab.sable.q', -11, (r) => mix(0.25, [whoosh(r, 0.22, 500, 2600, 900, 2, 0.35), 1], [osc(0.2, 'sine', xline(0, 95, 0.2, 68), hump(0.2, 0.3)), 0.6]));
def('fx.slash', -10, (r) => mix(0.3,
  [slice(r, 0.12, 3000, 8500), 0.9],
  [modal(0.25, 3300, [[1, 1, 1], [1.8, 0.5, 0.7]], 0.2), 0.25, 0.02],
  [thump(0.1, 190, 85), 0.7, 0.03]), { variants: 2, pitch: 0.04 });
def('ab.sable.w', -11, (r) => mix(0.6,
  [growl(r, 0.5, 82, 145, 500, 1150), 0.9],
  [drive(thump(0.14, 78, 42, 5), 2), 0.7], [drive(thump(0.14, 78, 42, 5), 2), 0.6, 0.18]));
def('ab.sable.e', -11, (r) => mix(0.55,
  [reverse(norm(shape(filt(noise(0.4, r), 'hp', 2200, 0.7), ad(0.002, 0.4, 4)))), 0.6],
  [osc(0.4, 'sine', xline(0, 620, 0.4, 150), hump(0.4, 0.7)), 0.3],
  [thump(0.16, 120, 40), 0.9, 0.38]));
def('ab.sable.r', -8, (r) => mix(0.7,
  [norm(shape(filt(mix(0.35, osc(0.35, 'saw', xline(0, 73, 0.35, 110)), osc(0.35, 'saw', xline(0, 78, 0.35, 117), 1, 0.4)), 'lp', xline(0, 300, 0.35, 2600), 1.5), hump(0.35, 0.85))), 0.9],
  [filt(modal(0.6, 2600, [[1, 1, 1], [1.5, 0.6, 0.8], [2.4, 0.3, 0.6]], 0.5), 'hp', 1500), 0.3, 0.1]), { range: 1.5 });
def('fx.execute', -5, (r) => mix(1,
  [slice(r, 0.14, 2400, 9000), 0.7], [slice(r, 0.14, 2000, 8000), 0.7, 0.07],
  [thump(0.5, 120, 35, 5), 1, 0.05],
  [burst(r, 0.2, 'lp', xline(0, 2500, 0.18, 300)), 0.5, 0.05],
  [modal(0.9, 220, [[1, 1, 1], [1.19, 0.6, 0.9], [2.38, 0.35, 0.6], [3.6, 0.2, 0.4]], 0.8), 0.35, 0.05]), { prio: 1, range: 1.4, verb: 0.3 });

// --- Ignara: fire -------------------------------------------------------------------------------------
def('ab.ignara.q', -10, (r) => mix(0.42,
  [norm(shape(filt(noise(0.4, r, 'pink'), 'lp', xline(0, 500, 0.08, 4000, 0.4, 800), 0.9), hump(0.4, 0.2))), 1],
  [shape(grit(r, 0.35, 120, 'bp', 3000, 1.5), hump(0.35, 0.3)), 0.3],
  [osc(0.12, 'sine', xline(0, 200, 0.12, 420), hump(0.12, 0.4)), 0.4]));
def('fx.burst', -10, (r) => mix(0.4,
  [burst(r, 0.22, 'lp', xline(0, 3500, 0.2, 400)), 0.9],
  [thump(0.15, 170, 70), 0.8],
  [shape(grit(r, 0.3, 90, 'bp', 3200, 1.5), ad(0.01, 0.3, 4)), 0.3, 0.03]), { variants: 2, pitch: 0.04 });
def('ab.ignara.w', -13, (r) => mix(0.2, [whoosh(r, 0.16, 700, 2800, 1500, 1.5, 0.4), 0.9], [grit(r, 0.18, 110, 'bp', 3200, 1.5), 0.35]));
def('zone.scorch', -12, (r) => mix(0.65,
  [norm(shape(filt(noise(0.65, r), 'bp', 2600, 0.8), (t) => (t / 0.65) ** 2 * (0.6 + 0.4 * Math.sin(TAU * (8 + 30 * t) * t)))), 0.8],
  [osc(0.65, 'sine', xline(0, 180, 0.65, 520), (t) => (t / 0.65) ** 2), 0.35],
  [shape(grit(r, 0.65, line(0, 30, 0.65, 220), 'bp', 2800, 1.5), (t) => t / 0.65), 0.35]), { verb: 0.1 });
def('fx.explosion', -7, (r) => boom(r, 0.95, 0.45), { variants: 2, pitch: 0.04, verb: 0.25 });
def('ab.ignara.e', -11, (r) => mix(0.34,
  [shape(grit(r, 0.32, 220, 'bp', 2200, 3), (t) => hump(0.32, 0.4)(t) * (0.55 + 0.45 * Math.sin(TAU * 26 * t))), 0.8],
  [whoosh(r, 0.3, 500, 1600, 900, 1.6, 0.4), 0.6]));
def('fx.snare', -10, (r) => mix(0.4,
  [comb(shape(grit(r, 0.22, 240, 'bp', 2500, 1.5), ad(0.002, 0.22, 4)), 1400, 0.5), 0.6],
  [modal(0.25, 300, METAL, 0.2), 0.6, 0.04],
  [burst(r, 0.2, 'hp', 3000, 0.7, 0.02), 0.2, 0.04]));
def('ab.ignara.r', -8, (r) => mix(0.7,
  [fm(0.65, xline(0, 55, 0.65, 82), 0.5, 3, hump(0.65, 0.5)), 0.9],
  [whoosh(r, 0.65, 400, 3500, 5000, 1.5, 0.9), 0.35],
  [shimmer(r, 0.5, 6500, 0.8), 0.15, 0.15]), { range: 1.5 });
def('zone.starfall', -8, (r) => mix(1,
  [osc(1, 'sine', xline(0, 2400, 1, 480), (t) => t ** 1.5 * 0.9), 0.5],
  [norm(shape(filt(noise(1, r), 'bp', xline(0, 3200, 1, 600), 1.2), (t) => t ** 2.2)), 0.9],
  [osc(1, 'sine', xline(0, 110, 1, 55), (t) => t ** 2), 0.5]), { range: 1.5, verb: 0.2 });
def('fx.starfall', -3, (r) => mix(2.4,
  [boom(r, 2, 1), 1],
  [echo(boom(r, 1.4, 0.7), 0.23, 0.4, 0.5), 0.45, 0.12],
  [shape(grit(r, 1.6, line(0, 150, 1.6, 10), 'lp', 3000, 0.7, 0.004), ad(0.05, 1.6, 3)), 0.3, 0.1],
  [norm(shape(filt(noise(1.2, r), 'hp', 2500, 0.7), ad(0.05, 1.2, 4))), 0.12, 0.05]), { prio: 1, range: 2, verb: 0.4 });

// --- Vesper: bowstring and storm -------------------------------------------------------------------------
def('ab.vesper.q', -10, (r) => filt(mix(0.3,
  [thump(0.08, 240, 120), 0.8],
  [norm(shape(pluck(0.2, 160, r, 0.6), ad(0.001, 0.2, 4))), 0.9],
  [burst(r, 0.01, 'bp', 3200, 1), 0.3],
  [whoosh(r, 0.2, 3200, 2400, 1200, 2, 0.15), 0.4, 0.01]), 'lp', 6000));
def('fx.spark', -15, (r, v) => mix(0.1, [burst(r, 0.025, 'bp', 3500, 1.2), 0.8], [modal(0.08, 2100 + v * 200, WOOD, 0.06), 0.5]), { variants: 3, pitch: 0.06, gap: 50 });
def('ab.vesper.w', -11, (r) => mix(0.5,
  ...seq([0, 0.09, 0.16, 0.215, 0.26], (t, i) => [modal(0.06, 1400 * 1.09 ** i, WOOD, 0.05), 0.7, t]),
  [fm(0.42, xline(0, 600, 0.42, 1500), 2, 2, hump(0.42, 0.75)), 0.35, 0.04],
  [shimmer(r, 0.3, 6000, 0.6), 0.15, 0.15]));
def('ab.vesper.e', -11, (r) => mix(0.34, [whoosh(r, 0.32, 700, 2600, 1200, 1.2, 0.3), 1], [whoosh(r, 0.26, 3000, 6000, 4000, 1, 0.3), 0.25]));
def('ab.vesper.r', -7, (r) => {
  const jit = filt(noise(0.6, r), 'lp', 400);
  const zip = filt(osc(0.6, 'saw', (t) => 900 * (1 + 0.8 * jit[Math.min(jit.length - 1, (t * SR) | 0)]) * Math.exp(-t * 1.5)), 'hp', 1200);
  return mix(1,
    [fm(0.13, xline(0, 300, 0.13, 1400), 1.41, 5, (t) => (t / 0.13) ** 2), 0.4],
    [thump(0.3, 150, 45), 1, 0.13],
    [drive(burst(r, 0.07, 'hp', 1500, 0.7), 3), 0.7, 0.13],
    [norm(shape(zip, ad(0.003, 0.6, 4))), 0.35, 0.13],
    [burst(r, 0.5, 'lp', xline(0, 2500, 0.4, 300)), 0.4, 0.13]);
}, { range: 2, verb: 0.3 });
def('fx.thunder', -4, (r) => {
  const wob = drift(r, 1.5, 9);
  return mix(1.7,
    [drive(burst(r, 0.05, 'hp', 800, 0.7, 0.0005), 3), 1],
    [thump(0.5, 85, 30, 5), 0.9],
    [norm(shape(filt(noise(1.5, r, 'brown'), 'lp', xline(0, 1800, 1.2, 150), 0.7), (t) => ad(0.005, 1.5, 4)(t) * (0.5 + 0.5 * wob(t)))), 0.9, 0.02],
    [shape(grit(r, 0.35, 420, 'hp', 2500, 0.7, 0.001), ad(0.002, 0.35, 4)), 0.35]);
}, { prio: 1, range: 1.6, verb: 0.4 });

// --- Liora: light ---------------------------------------------------------------------------------------------
def('ab.liora.w', -11, (r) => mix(0.34,
  [fm(0.3, xline(0, 1500, 0.3, 950), 2, (t) => 1.8 * Math.exp(-t * 10), ad(0.003, 0.3)), 0.9],
  [burst(r, 0.1, 'bp', 4500, 1, 0.004), 0.1]));
def('fx.flash', -10, (r) => mix(0.45, [modal(0.4, 1760, GLASS, 0.35), 0.9], [modal(0.4, 2637, GLASS, 0.3), 0.45], [burst(r, 0.03, 'hp', 3000), 0.3]), { verb: 0.22 });
def('fx.heal', -11, (r) => mix(0.8,
  ...seq([79, 83, 86], (m, i) => [chime(0.55, N(m)), 0.6, i * 0.07]),
  [osc(0.7, 'sine', N(67), hump(0.7, 0.35)), 0.35], [osc(0.7, 'sine', N(74), hump(0.7, 0.35)), 0.25],
  [shimmer(r, 0.5, 7000), 0.12, 0.05]), { gap: 120, verb: 0.28 });
def('ab.liora.e', -10, (r) => mix(0.85,
  [modal(0.8, 1047, GLASS, 0.7), 0.8], [modal(0.8, 1568, GLASS, 0.6), 0.5, 0.03],
  [osc(0.6, 'sine', 262, hump(0.6, 0.3)), 0.4],
  [shimmer(r, 0.6, 6500, 0.2), 0.15]), { verb: 0.25 });
def('fx.sanctuary', -5, (r) => {
  // A wordless choir: a saw chord through three vowel formants.
  const src = mix(2.3, ...seq([62, 66, 69, 74, 50], (m, i) => [mix(2.3, osc(2.3, 'saw', N(m) * 1.003, 1, i * 0.21), osc(2.3, 'saw', N(m) * 0.997, 1, i * 0.37)), i === 4 ? 0.6 : 1]));
  const vowel = mix(2.3, [filt(src.slice(), 'bp', 800, 4), 1], [filt(src.slice(), 'bp', 1150, 5), 0.6], [filt(src.slice(), 'bp', 2900, 6), 0.25], [filt(src, 'lp', 400, 0.7), 0.5]);
  return mix(2.4,
    [norm(shape(vowel, line(0, 0, 0.35, 1, 1.3, 0.8, 2.3, 0))), 0.9],
    ...seq([74, 81, 86, 90], (m, i) => [chime(1, N(m)), 0.3, 0.1 + i * 0.12]),
    [shimmer(r, 1.6, 7000, 0.25), 0.15, 0.1]);
}, { prio: 1, range: 1.8, verb: 0.45 });

// --- Shared world effects ---------------------------------------------------------------------------------------
def('fx.recall', -15, (r) => {
  // Loops exactly: every tone is a whole number of cycles in two seconds.
  const tones = mix(2, [osc(2, 'sine', 440), 0.5], [osc(2, 'sine', 660.5), 0.4], [osc(2, 'sine', 880), 0.3], [osc(2, 'sine', 1321), 0.15]);
  const air = shape(seamless(filt(noise(2.4, r), 'bp', 5000, 2), 0.4), (t) => 0.55 + 0.45 * Math.sin(TAU * 8 * t));
  return mix(2, [shape(tones, (t) => 0.75 + 0.25 * Math.sin(TAU * 4 * t)), 1], [norm(air), 0.3]);
}, { loop: true, verb: 0.2, prio: 1 });
def('fx.teleport', -10, (r) => mix(0.75,
  [whoosh(r, 0.28, 500, 5000, 3000, 1.3, 0.8), 0.7],
  [modal(0.5, 880, GLASS, 0.45), 0.6, 0.2], [modal(0.5, 1320, GLASS, 0.4), 0.45, 0.2],
  [thump(0.2, 130, 50), 0.7, 0.2]), { gap: 150, verb: 0.3 });
def('fx.respawn', -15, (r) => mix(0.8, [whoosh(r, 0.5, 400, 3000, 2000, 1.2, 0.8), 0.5], [chime(0.5, N(74)), 0.6, 0.3], [chime(0.5, N(81)), 0.5, 0.36]), { verb: 0.3 });
def('fx.levelup', -17, (r) => mix(0.5, [chime(0.35, N(79)), 0.7], [chime(0.4, N(84)), 0.7, 0.07], [shimmer(r, 0.35, 7000), 0.2, 0.05]), { prio: 3, gap: 200 });

// --- Ambience (stereo loops) ---------------------------------------------------------------------------------------
function wind(r: Rand): Buf {
  const D = 10, X = 1.5, T = D + X;
  const sweep = drift(r, T, 0.25), swell = drift(r, T, 0.4), rustle = drift(r, T, 0.7);
  const air = shape(filt(noise(T, r, 'pink'), 'bp', (t) => 360 + 440 * sweep(t), 0.7), (t) => 0.45 + 0.55 * swell(t));
  const leaves = shape(filt(noise(T, r), 'hp', 3500, 0.7), (t) => 0.15 + 0.85 * rustle(t) ** 2);
  return seamless(mix(T, [norm(air), 1], [norm(leaves), 0.1]), X);
}
function river(r: Rand): Buf {
  const D = 8, X = 1, T = D + X;
  const o = gain(norm(filt(filt(noise(T, r), 'bp', 1400, 0.5), 'hp', 500, 0.7)), 0.3);
  // The babble: hundreds of tiny rising chirps, like bubbles popping.
  for (let k = 0; k < T * 40; k++) {
    const at = (r() * (T - 0.08) * SR) | 0, len = samples(0.025 + r() * 0.05);
    const f0 = 400 + r() * 1300, rise = 1.2 + r() * 0.7, amp = 0.08 + 0.3 * r() * r();
    let ph = 0;
    for (let i = 0; i < len; i++) {
      const u = i / len;
      ph += (f0 * rise ** u) / SR;
      o[at + i] += Math.sin(TAU * ph) * amp * Math.sin(Math.PI * u) ** 2;
    }
  }
  return seamless(filt(o, 'lp', 5000, 0.7), X);
}
function hum(r: Rand): Buf {
  // Loops exactly: every frequency is a multiple of 1/8 Hz. The slight detunings beat slowly.
  const D = 8, ph = r();
  const air = shape(seamless(filt(noise(D + 0.5, r), 'bp', 2400, 4), 0.5), (t) => 0.5 + 0.5 * Math.sin(TAU * (0.5 * t + ph)));
  return mix(D, [osc(D, 'sine', 55, 1, ph), 0.5], [osc(D, 'sine', 82.5, 1, ph * 3), 0.5], [osc(D, 'sine', 110.5, 1, ph * 5), 0.6],
    [osc(D, 'sine', 165.25, 1, ph * 7), 0.5], [osc(D, 'sine', 220.25, 1, ph * 2), 0.4], [osc(D, 'sine', 330.25, 1, ph * 4), 0.25], [osc(D, 'sine', 440.5, 1, ph * 6), 0.12], [norm(air), 0.3]);
}
def('amb.wind', -26, (r) => [wind(r), wind(r)], { loop: true });
def('amb.river', -22, (r) => [river(r), river(r)], { loop: true });
def('amb.base', -24, (r) => [hum(r), hum(r)], { loop: true });
def('amb.bird', -25, (r, v) => {
  const layers: Layer[] = [], style = v % 3, f0 = 2600 + r() * 2200, count = 2 + ((r() * 5) | 0);
  let t = 0;
  for (let i = 0; i < count; i++) {
    const d = 0.04 + r() * 0.09, fa = f0 * (0.85 + r() * 0.3);
    const fb = fa * (style === 0 ? 1.35 : style === 1 ? 0.72 : 1 + (r() - 0.5) * 0.5);
    layers.push([osc(d, 'sine', xline(0, fa, d, fb), hump(d, 0.3)), 0.6 + 0.4 * r(), t]);
    t += d + (style === 2 ? 0.025 : 0.02 + r() * 0.1);
  }
  return mix(t + 0.02, ...layers);
}, { variants: 6, pitch: 0.12, verb: 0.4, gap: 400, prio: 4 });

// --- Score: slow pads on a four-chord cycle, with sparse bell notes over them --------------------------------------------
export const PAD_CHORDS = [
  [38, 45, 50, 53, 57, 64], // Dm9
  [41, 46, 50, 53, 57, 62], // Bbmaj7 (over F, to keep it out of the mud)
  [41, 48, 53, 57, 60, 67], // Fadd9
  [36, 43, 48, 52, 55, 62], // C6/9
];
/** Seconds of each pad buffer, and the spacing at which the score starts them (they overlap). */
export const PAD_LEN = 15, PAD_STEP = 9.5;
/** Semitone offsets from the bell's recorded pitch (A4) that sit well over every chord: D minor pentatonic. */
export const BELL_STEPS = [0, 3, 5, 8, 10, 12, 15, 17];

function pad(midi: number[], r: Rand): [Buf, Buf] {
  const side = (s: number) => {
    const layers: Layer[] = [];
    midi.forEach((m, i) => {
      // The two bass notes sit back so the chord's colour, not its weight, carries on small speakers.
      const f = N(m), det = (r() - 0.5) * 0.006 + s * 0.0015, a = i < 2 ? 0.4 : 1;
      layers.push([osc(PAD_LEN, 'tri', f * (1 + det), 1, r()), a], [osc(PAD_LEN, 'sine', f * (1 - det), 1, r()), a * 0.6], [osc(PAD_LEN, 'saw', f * (1 + det * 2), 1, r()), a * 0.5]);
    });
    const ph = r() * TAU;
    const b = filt(mix(PAD_LEN, ...layers), 'lp', (t) => 1500 + 500 * Math.sin((TAU * t) / PAD_LEN + ph), 0.7);
    return shape(b, line(0, 0, 4.5, 1, PAD_STEP, 1, PAD_LEN, 0));
  };
  return [side(-1), side(1)];
}
PAD_CHORDS.forEach((c, i) => def(`mus.pad${i}`, -19, (r) => pad(c, r), { verb: 0.35 }));
def('mus.bell', -24, () => fadeEdges(modal(2.6, 440, [[1, 1, 1], [2, 0.35, 0.6], [3, 0.12, 0.4], [4.07, 0.08, 0.25]], 2.2), 0.006, 0.05), { verb: 0.5 });

// --- Rendering -------------------------------------------------------------------------------------------------------
function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Most gain the limiter is allowed to absorb, in dB, before a sound is left under its target loudness instead. */
const MAX_LIMIT_DB = 3;

export interface Take {
  chans: Buf[];
  /** dB this take falls short of the reference loudness (0 unless it is so spiky the limiter would have mangled it). */
  short: number;
}

/** Render one take of a sound: synthesise, fade the edges, normalise loudness, limit peaks. */
export function renderTake(spec: SoundSpec, variant: number): Take {
  const out = spec.make(rng(hash(spec.name) + variant * 7919), variant);
  const chans = out instanceof Float32Array ? [out] : out;
  if (!spec.loop) for (const c of chans) fadeEdges(c);
  const want = REF_DB - loudness(chans, spec.loop ? 0 : 0.2);
  const pk = Math.max(...chans.map(peak));
  const most = 20 * Math.log10(0.95 / Math.max(1e-9, pk)) + MAX_LIMIT_DB;
  const g = Math.min(want, most);
  for (const c of chans) softLimit(gain(c, dbToGain(g)));
  return { chans, short: want - g };
}

export function renderSound(spec: SoundSpec): Take[] {
  const takes: Take[] = [];
  for (let v = 0; v < spec.variants; v++) takes.push(renderTake(spec, v));
  return takes;
}

/** Numbers describing a rendered take, for the mix report and the tests. */
export function describe(t: Take) {
  const loud = loudness(t.chans, 0.2);
  return {
    dur: t.chans[0].length / SR,
    peak: Math.max(...t.chans.map(peak)),
    loud,
    bright: brightness(t.chans[0]),
    short: t.short,
    /** dB of loudness left on speakers with no bass (everything under 300 Hz removed); near 0 is good. */
    small: loudness(t.chans.map((c) => filt(filt(c.slice(), 'hp', 300, 0.7), 'hp', 300, 0.7)), 0.2) - loud,
  };
}

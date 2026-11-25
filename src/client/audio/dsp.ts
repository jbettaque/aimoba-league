/**
 * Sample-level synthesis toolkit. Pure functions over Float32Array with no WebAudio dependency, so
 * the sound bank renders identically in a Web Worker and in Node (for analysis and tests).
 */
export const SR = 44100;
export type Buf = Float32Array;
export type Fn = (t: number) => number;
/** A constant, or a function of time in seconds. */
export type P = number | Fn;
export type Rand = () => number;

const TAU = Math.PI * 2;
export const samples = (dur: number) => Math.max(1, Math.round(dur * SR));
/** Equal-tempered frequency of a MIDI note (69 = A4 = 440 Hz). */
export const note = (midi: number) => 440 * 2 ** ((midi - 69) / 12);
export const dbToGain = (db: number) => 10 ** (db / 20);
export const gainToDb = (g: number) => 20 * Math.log10(Math.max(1e-9, g));

/** Small deterministic generator (mulberry32) so every render of a sound is identical. */
export function rng(seed: number): Rand {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Envelopes ---------------------------------------------------------------------------------
/** Linear attack over `a` seconds, then an exponential decay that is inaudible `d` seconds later. */
export const ad = (a: number, d: number, k = 6): Fn => (t) => (t < 0 ? 0 : t < a ? t / a : Math.exp((-k * (t - a)) / d));

/** Breakpoint envelope `line(t0, v0, t1, v1, ...)`, held flat outside the points. */
export function line(...p: number[]): Fn {
  return (t) => {
    if (t <= p[0]) return p[1];
    for (let i = 2; i < p.length; i += 2) {
      if (t < p[i]) return p[i - 1] + ((p[i + 1] - p[i - 1]) * (t - p[i - 2])) / (p[i] - p[i - 2]);
    }
    return p[p.length - 1];
  };
}

/** Like `line`, but interpolating exponentially: the natural shape for frequency sweeps. */
export function xline(...p: number[]): Fn {
  return (t) => {
    if (t <= p[0]) return p[1];
    for (let i = 2; i < p.length; i += 2) {
      if (t < p[i]) return p[i - 1] * (p[i + 1] / p[i - 1]) ** ((t - p[i - 2]) / (p[i] - p[i - 2]));
    }
    return p[p.length - 1];
  };
}

/** Smooth swell 0 -> 1 -> 0 over `dur`, peaking at fraction `peak` of the way through. */
export const hump = (dur: number, peak = 0.5, sharp = 1): Fn => (t) => {
  if (t <= 0 || t >= dur) return 0;
  const u = t / dur;
  const x = u < peak ? u / peak : (1 - u) / (1 - peak);
  return (0.5 - 0.5 * Math.cos(Math.PI * x)) ** sharp;
};

// --- Generators --------------------------------------------------------------------------------
export type Wave = 'sine' | 'tri' | 'saw' | 'square';

/** Band-limiting correction for the jump in a saw / square wave (polyBLEP). */
function blep(t: number, dt: number): number {
  if (t < dt) { const x = t / dt; return x + x - x * x - 1; }
  if (t > 1 - dt) { const x = (t - 1) / dt; return x * x + x + x + 1; }
  return 0;
}

export function osc(dur: number, wave: Wave, f: P, amp: P = 1, phase = 0): Buf {
  const n = samples(dur), o = new Float32Array(n);
  const ff = typeof f === 'function' ? f : null, af = typeof amp === 'function' ? amp : null;
  let ph = phase - Math.floor(phase);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const dp = Math.min(0.45, (ff ? ff(t) : (f as number)) / SR);
    let v: number;
    if (wave === 'sine') v = Math.sin(TAU * ph);
    else if (wave === 'tri') v = 1 - 4 * Math.abs(ph - 0.5);
    else if (wave === 'saw') v = 2 * ph - 1 - blep(ph, dp);
    else { const p2 = ph + 0.5; v = (ph < 0.5 ? 1 : -1) + blep(ph, dp) - blep(p2 - Math.floor(p2), dp); }
    o[i] = v * (af ? af(t) : (amp as number));
    ph += dp;
    if (ph >= 1) ph -= 1;
  }
  return o;
}

/** Two-operator FM: a sine carrier phase-modulated by a sine at `ratio` times its frequency. */
export function fm(dur: number, f: P, ratio: number, index: P, amp: P = 1): Buf {
  const n = samples(dur), o = new Float32Array(n);
  const ff = typeof f === 'function' ? f : null, xf = typeof index === 'function' ? index : null, af = typeof amp === 'function' ? amp : null;
  let pc = 0, pm = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const fr = ff ? ff(t) : (f as number);
    o[i] = Math.sin(TAU * pc + (xf ? xf(t) : (index as number)) * Math.sin(TAU * pm)) * (af ? af(t) : (amp as number));
    pc += fr / SR;
    pm += (fr * ratio) / SR;
    if (pc >= 1) pc -= 1;
    if (pm >= 1) pm -= Math.floor(pm);
  }
  return o;
}

export type Color = 'white' | 'pink' | 'brown';

export function noise(dur: number, r: Rand, color: Color = 'white'): Buf {
  const n = samples(dur), o = new Float32Array(n);
  if (color === 'white') {
    for (let i = 0; i < n; i++) o[i] = r() * 2 - 1;
  } else if (color === 'pink') {
    // Paul Kellet's economy filter: about -3 dB per octave.
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < n; i++) {
      const w = r() * 2 - 1;
      b0 = 0.99765 * b0 + w * 0.099046;
      b1 = 0.963 * b1 + w * 0.2965164;
      b2 = 0.57 * b2 + w * 1.0526913;
      o[i] = (b0 + b1 + b2 + w * 0.1848) * 0.22;
    }
  } else {
    let b = 0;
    for (let i = 0; i < n; i++) {
      b = (b + 0.02 * (r() * 2 - 1)) / 1.02;
      o[i] = b * 3.5;
    }
  }
  return o;
}

/**
 * A struck resonator (bells, metal, glass, wood): decaying sine partials given as
 * [frequency ratio, amplitude, decay multiplier]. `decay` is the time the fundamental takes to fade.
 */
export function modal(dur: number, f0: number, parts: number[][], decay: number): Buf {
  const n = samples(dur), o = new Float32Array(n);
  for (const [ratio, amp, dm] of parts) {
    const f = f0 * ratio;
    if (f > SR * 0.45) continue;
    const w = (TAU * f) / SR, k = Math.exp(-6.9 / (decay * dm * SR));
    let a = amp;
    for (let i = 0; i < n; i++) {
      o[i] += Math.sin(w * i) * a;
      a *= k;
    }
  }
  return o;
}

/** Plucked string (Karplus-Strong). `damp` 0..1: higher loses its brightness faster. */
export function pluck(dur: number, f: number, r: Rand, damp = 0.5): Buf {
  const n = samples(dur), o = new Float32Array(n);
  const L = Math.max(2, Math.round(SR / f));
  const d = new Float32Array(L);
  for (let i = 0; i < L; i++) d[i] = r() * 2 - 1;
  const keep = 0.999 - damp * 0.03;
  for (let i = 0, j = 0; i < n; i++) {
    const k = j + 1 === L ? 0 : j + 1;
    o[i] = d[j];
    d[j] = (d[j] + d[k]) * 0.5 * keep;
    j = k;
  }
  return o;
}

/** Sparse random clicks (fire crackle, debris, sparks, rattles): about `rate` per second. */
export function crackle(dur: number, r: Rand, rate: P, decay = 0.0015): Buf {
  const n = samples(dur), o = new Float32Array(n);
  const rf = typeof rate === 'function' ? rate : null;
  const k = Math.exp(-1 / (decay * SR));
  let v = 0;
  for (let i = 0; i < n; i++) {
    const p = (rf ? rf(i / SR) : (rate as number)) / SR;
    if (r() < p) v += (r() < 0.5 ? -1 : 1) * (0.35 + 0.65 * r());
    o[i] = v;
    v *= k;
  }
  return o;
}

// --- Processors (in place; they return the buffer so calls can be nested) --------------------
export type FilterMode = 'lp' | 'hp' | 'bp';

/** State-variable filter (stable under fast cutoff sweeps). The band-pass has unity gain at its centre. */
export function filt(b: Buf, mode: FilterMode, fc: P, q = 0.707): Buf {
  const cf = typeof fc === 'function' ? fc : null;
  const k = 1 / q;
  let g = 0, a1 = 0, a2 = 0, a3 = 0, ic1 = 0, ic2 = 0;
  const set = (f: number) => {
    g = Math.tan((Math.PI * Math.min(SR * 0.45, Math.max(20, f))) / SR);
    a1 = 1 / (1 + g * (g + k));
    a2 = g * a1;
    a3 = g * a2;
  };
  if (!cf) set(fc as number);
  for (let i = 0; i < b.length; i++) {
    if (cf && (i & 15) === 0) set(cf(i / SR));
    const v3 = b[i] - ic2;
    const v1 = a1 * ic1 + a2 * v3;
    const v2 = ic2 + a2 * ic1 + a3 * v3;
    ic1 = 2 * v1 - ic1;
    ic2 = 2 * v2 - ic2;
    b[i] = mode === 'lp' ? v2 : mode === 'bp' ? v1 * k : b[i] - k * v1 - v2;
  }
  return b;
}

/** Multiply by an amplitude envelope. */
export function shape(b: Buf, env: Fn): Buf {
  for (let i = 0; i < b.length; i++) b[i] *= env(i / SR);
  return b;
}

export function gain(b: Buf, g: number): Buf {
  for (let i = 0; i < b.length; i++) b[i] *= g;
  return b;
}

/** Soft saturation: adds grit and weight, and tames peaks. Output stays within +-1. */
export function drive(b: Buf, amount: number): Buf {
  const norm = 1 / Math.tanh(amount);
  for (let i = 0; i < b.length; i++) b[i] = Math.tanh(b[i] * amount) * norm;
  return b;
}

/** Feedback delay mixed into the signal (rumbling echoes, flams). */
export function echo(b: Buf, time: number, feedback: number, wet: number): Buf {
  const d = samples(time);
  const line_ = new Float32Array(d);
  for (let i = 0, j = 0; i < b.length; i++) {
    const y = line_[j];
    line_[j] = b[i] + y * feedback;
    b[i] += y * wet;
    if (++j === d) j = 0;
  }
  return b;
}

/** Resonant comb at `f` Hz: a cheap metallic / hollow colouring. */
export function comb(b: Buf, f: number, feedback: number): Buf {
  const d = Math.max(1, Math.round(SR / f));
  const line_ = new Float32Array(d);
  for (let i = 0, j = 0; i < b.length; i++) {
    const y = b[i] + line_[j] * feedback;
    line_[j] = y;
    b[i] = y;
    if (++j === d) j = 0;
  }
  return b;
}

export function reverse(b: Buf): Buf {
  b.reverse();
  return b;
}

/** A layer for `mix`: a buffer, or [buffer, gain, start time in seconds]. */
export type Layer = Buf | [Buf, number, number?];

/** Sum layers into a new buffer of `dur` seconds; anything past the end is cut. */
export function mix(dur: number, ...layers: Layer[]): Buf {
  const o = new Float32Array(samples(dur));
  for (const l of layers) {
    const [b, g, at] = l instanceof Float32Array ? [l, 1, 0] : [l[0], l[1], l[2] ?? 0];
    const off = Math.round(at * SR);
    const n = Math.min(b.length, o.length - off);
    for (let i = 0; i < n; i++) o[off + i] += b[i] * g;
  }
  return o;
}

/** Short linear fades so a sound never starts or stops with a click. */
export function fadeEdges(b: Buf, fadeIn = 0.002, fadeOut = 0.01): Buf {
  const a = Math.min(b.length >> 1, samples(fadeIn)), z = Math.min(b.length >> 1, samples(fadeOut));
  for (let i = 0; i < a; i++) b[i] *= i / a;
  for (let i = 0; i < z; i++) b[b.length - 1 - i] *= i / z;
  return b;
}

/** Turn a buffer into a seamless loop by cross-fading its tail into its head (shortens it by `xf`). */
export function seamless(b: Buf, xf: number): Buf {
  const x = samples(xf), n = b.length - x;
  const o = b.slice(0, n);
  for (let i = 0; i < x; i++) {
    const u = i / x;
    o[i] = b[i] * Math.sqrt(u) + b[n + i] * Math.sqrt(1 - u);
  }
  return o;
}

// --- Measurement ---------------------------------------------------------------------------------
function biquad(b: Buf, b0: number, b1: number, b2: number, a1: number, a2: number): Buf {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < b.length; i++) {
    const x = b[i];
    const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x; y2 = y1; y1 = y;
    b[i] = y;
  }
  return b;
}

/** Approximate K-weighting (ITU-R BS.1770): a +4 dB shelf above 1.5 kHz and a 38 Hz high-pass. Returns a copy. */
export function kWeight(b: Buf): Buf {
  const o = b.slice();
  {
    const A = 10 ** (4 / 40), w = (TAU * 1500) / SR, c = Math.cos(w), al = (Math.sin(w) / 2) * Math.SQRT2, s = 2 * Math.sqrt(A) * al;
    const a0 = A + 1 - (A - 1) * c + s;
    biquad(o, (A * (A + 1 + (A - 1) * c + s)) / a0, (-2 * A * (A - 1 + (A + 1) * c)) / a0, (A * (A + 1 + (A - 1) * c - s)) / a0, (2 * (A - 1 - (A + 1) * c)) / a0, (A + 1 - (A - 1) * c - s) / a0);
  }
  {
    const w = (TAU * 38) / SR, c = Math.cos(w), al = Math.sin(w) / (2 * 0.5), a0 = 1 + al;
    biquad(o, (1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, (-2 * c) / a0, (1 - al) / a0);
  }
  return o;
}

export function peak(b: Buf): number {
  let m = 0;
  for (let i = 0; i < b.length; i++) { const a = Math.abs(b[i]); if (a > m) m = a; }
  return m;
}

/**
 * Perceived loudness in dB: the loudest `window` seconds of the K-weighted signal (mean power).
 * Sounds shorter than the window read quieter, like the ear's own integration time. With
 * `window` <= 0 the whole buffer is averaged (for loops and beds).
 */
export function loudness(chans: Buf[], window = 0.2): number {
  let best = 0;
  const n = chans[0].length;
  const w = window > 0 ? samples(window) : n;
  const sq = new Float64Array(n);
  for (const c of chans) {
    const k = kWeight(c);
    for (let i = 0; i < n; i++) sq[i] += (k[i] * k[i]) / chans.length;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += sq[i];
    if (i >= w) sum -= sq[i - w];
    if (sum > best) best = sum;
  }
  return 10 * Math.log10(Math.max(1e-12, best / w));
}

/** Spectral centre of gravity in Hz, estimated from zero crossings of the differentiated signal. Cheap and good enough to compare brightness. */
export function brightness(b: Buf): number {
  let num = 0, den = 0;
  for (let i = 1; i < b.length; i++) {
    const d = b[i] - b[i - 1];
    num += d * d;
    den += b[i] * b[i];
  }
  // For a sinusoid, sqrt(num/den) = 2 sin(pi f / SR).
  return den > 0 ? (Math.asin(Math.min(1, Math.sqrt(num / den) / 2)) * SR) / Math.PI : 0;
}

/** Limit peaks to just under full scale, bending only the part of the wave above the knee. */
export function softLimit(b: Buf, knee = 0.6, ceiling = 0.95): Buf {
  const span = ceiling - knee;
  for (let i = 0; i < b.length; i++) {
    const a = Math.abs(b[i]);
    if (a > knee) b[i] = Math.sign(b[i]) * (knee + span * Math.tanh((a - knee) / span));
  }
  return b;
}

/**
 * WebAudio playback of the synthesised sound bank.
 *
 *   voices ─► world ─┐                          (far-away voices pass a low-pass first)
 *   ambience loops ──┼─► muffle ─┐
 *   reverb return ───┘           ├─► duck ─┐
 *   music ───────────────────────┘         ├─► master ─► limiter ─► out
 *   own hero / announcements / interface ──┘
 *
 * `muffle` closes when the player is dead, paused or in the menu, so the world recedes while the
 * sounds meant for the player stay clear. `duck` dips everything else under an announcement.
 */
import { BANK, BELL_STEPS, PAD_CHORDS, PAD_STEP, SPEC, renderSound, type Bus, type Take } from './bank.ts';
import type { AudioOut, LoopOpts, Scene } from './director.ts';
import { SR, dbToGain, filt, noise, rng, shape, xline } from './dsp.ts';
import { Mixer, type PlayOpts } from './mixer.ts';

export interface AudioSettings {
  master: number;
  effects: number;
  music: number;
  ambience: number;
  muted: boolean;
}

const STORE_KEY = 'aimoba.audio';
const DEFAULTS: AudioSettings = { master: 0.8, effects: 1, music: 0.7, ambience: 0.8, muted: false };
/** Which volume slider governs each bus. */
const BUS_SETTING: Record<Bus, 'effects' | 'music' | 'ambience'> = { world: 'effects', self: 'effects', ui: 'effects', sting: 'effects', amb: 'ambience', music: 'music' };
/** The limiter adds make-up gain; this brings the result back to a sensible level. */
const OUT_TRIM = 0.9;

interface LoopVoice {
  src: AudioBufferSourceNode;
  gain: GainNode;
  pan: StereoPannerNode | null;
  seen: number;
}

export class AudioEngine implements AudioOut {
  readonly mixer = new Mixer();
  settings: AudioSettings = { ...DEFAULTS };
  private ctx: AudioContext | null = null;
  private rendered = new Map<string, Take[]>();
  private buffers = new Map<string, AudioBuffer[]>();
  private bus = {} as Record<Bus, GainNode>;
  private far!: BiquadFilterNode;
  private muffle!: BiquadFilterNode;
  private worldTrim!: GainNode;
  private duck!: GainNode;
  private master!: GainNode;
  /** The last node before the speakers (a tap point for meters and recordings). */
  private out!: GainNode;
  private verbIn!: GainNode;
  private loops = new Map<string, LoopVoice>();
  private sceneKey = '';
  private menu = true;
  private duckUntil = 0;
  private padAt = 0;
  private padIdx = 0;
  private started = false;

  /** Begin rendering the bank in the background and arm the first-gesture unlock. Call once at boot. */
  init() {
    if (this.started) return;
    this.started = true;
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY) ?? '{}');
      for (const k of Object.keys(DEFAULTS) as (keyof AudioSettings)[]) if (typeof saved[k] === typeof DEFAULTS[k]) (this.settings as any)[k] = saved[k];
    } catch { /* keep defaults */ }
    this.render();
    // Browsers only let audio start from a user gesture.
    const unlock = () => this.unlock();
    window.addEventListener('pointerdown', unlock, true);
    window.addEventListener('keydown', unlock, true);
    document.addEventListener('visibilitychange', () => {
      if (!this.ctx) return;
      if (document.hidden) void this.ctx.suspend();
      else void this.ctx.resume();
    });
  }

  /** Synthesise every sound, off the main thread when possible. */
  private render() {
    const accept = (name: string, takes: Take[]) => {
      this.rendered.set(name, takes);
      this.mixer.takes.set(name, takes.map((t) => ({ dur: t.chans[0].length / SR, short: t.short })));
    };
    const inline = (from: number) => {
      // Fallback: one sound per timer tick on the main thread.
      let i = from;
      const next = () => {
        if (i >= BANK.length) return;
        const s = BANK[i++];
        if (!this.rendered.has(s.name)) accept(s.name, renderSound(s));
        setTimeout(next, 0);
      };
      next();
    };
    try {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<{ name: string; takes: Take[] }>) => accept(e.data.name, e.data.takes);
      w.onerror = () => { w.terminate(); inline(0); };
    } catch {
      inline(0);
    }
  }

  unlock() {
    if (!this.ctx) {
      try {
        this.build(new AudioContext({ latencyHint: 'interactive' }));
      } catch {
        return;
      }
    }
    if (this.ctx!.state === 'suspended' && !document.hidden) void this.ctx!.resume();
  }

  private build(ctx: AudioContext) {
    this.ctx = ctx;
    const g = (v = 1) => { const n = ctx.createGain(); n.gain.value = v; return n; };
    const lp = (f: number) => { const n = ctx.createBiquadFilter(); n.type = 'lowpass'; n.frequency.value = f; n.Q.value = 0.5; return n; };
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -16;
    limiter.knee.value = 12;
    limiter.ratio.value = 6;
    limiter.attack.value = 0.004;
    limiter.release.value = 0.18;
    this.master = g(0);
    this.out = g(OUT_TRIM);
    this.master.connect(limiter).connect(this.out).connect(ctx.destination);
    this.duck = g();
    this.duck.connect(this.master);
    this.muffle = lp(20000);
    this.worldTrim = g();
    this.muffle.connect(this.worldTrim).connect(this.duck);
    for (const b of ['world', 'self', 'ui', 'sting', 'amb', 'music'] as Bus[]) this.bus[b] = g();
    this.bus.world.connect(this.muffle);
    this.bus.amb.connect(this.muffle);
    this.bus.music.connect(this.duck);
    for (const b of ['self', 'ui', 'sting'] as Bus[]) this.bus[b].connect(this.master);
    this.far = lp(2400);
    this.far.connect(this.bus.world);

    // Reverb: a synthesised hall, two seconds of decaying noise that darkens as it fades.
    // (Built at the bank's rate but played at the context's: for decaying noise the difference is nothing.)
    const len = 2, ir = ctx.createBuffer(2, len * SR, ctx.sampleRate), r = rng(4242);
    for (let c = 0; c < 2; c++) {
      const b = shape(noise(len, r), (t) => (t < 0.012 ? 0 : Math.exp(-(t - 0.012) * 3.4)));
      filt(filt(b, 'lp', xline(0, 9000, len, 1200), 0.7), 'hp', 180, 0.7);
      ir.copyToChannel(b as Float32Array<ArrayBuffer>, c);
    }
    const verb = ctx.createConvolver();
    verb.buffer = ir;
    this.verbIn = g();
    this.verbIn.connect(verb).connect(g(0.9)).connect(this.muffle);
    this.applySettings();
  }

  // --- Settings -----------------------------------------------------------------------------------
  set(patch: Partial<AudioSettings>) {
    Object.assign(this.settings, patch);
    try { localStorage.setItem(STORE_KEY, JSON.stringify(this.settings)); } catch { /* private mode */ }
    this.applySettings();
  }

  private applySettings() {
    if (!this.ctx) return;
    const s = this.settings, t = this.ctx.currentTime;
    this.master.gain.setTargetAtTime(s.muted ? 0 : s.master, t, 0.03);
    for (const b of Object.keys(this.bus) as Bus[]) this.bus[b].gain.setTargetAtTime(s[BUS_SETTING[b]] * (b === 'music' && this.menu ? 1.5 : 1), t, 0.05);
  }

  private get live(): boolean {
    return this.ctx !== null && this.ctx.state === 'running' && !this.settings.muted;
  }

  // --- Voices -------------------------------------------------------------------------------------
  private buffer(name: string, take: number): AudioBuffer | null {
    let list = this.buffers.get(name);
    if (!list) {
      const takes = this.rendered.get(name);
      if (!takes) return null;
      list = takes.map((t) => {
        const b = this.ctx!.createBuffer(t.chans.length, t.chans[0].length, SR);
        t.chans.forEach((c, i) => b.copyToChannel(c as Float32Array<ArrayBuffer>, i));
        return b;
      });
      this.buffers.set(name, list);
      this.rendered.delete(name); // the AudioBuffers hold the samples now
    }
    return list[take];
  }

  play(name: string, o: PlayOpts = {}) {
    if (!this.live) return;
    const ctx = this.ctx!, now = ctx.currentTime;
    const v = this.mixer.decide(name, o, now);
    if (!v) return;
    const buf = this.buffer(name, v.take);
    if (!buf) return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = v.rate;
    const gain = ctx.createGain();
    gain.gain.value = v.gain;
    let tail: AudioNode = src.connect(gain);
    if (v.pan !== 0 && buf.numberOfChannels === 1) {
      const pan = ctx.createStereoPanner();
      pan.pan.value = v.pan;
      tail = tail.connect(pan);
    }
    tail.connect(v.far ? this.far : this.bus[v.bus]);
    if (v.send > 0.01) {
      const send = ctx.createGain();
      send.gain.value = v.send;
      tail.connect(send).connect(this.verbIn);
    }
    src.start(now + v.delay);
    if (v.duck > 0) {
      // Dip the rest of the mix for the length of the phrase, then let it back in gently.
      const a = now + v.delay, end = a + v.dur * 0.8, p = this.duck.gain;
      if (end > this.duckUntil) {
        this.duckUntil = end;
        p.cancelScheduledValues(a);
        p.setTargetAtTime(dbToGain(-v.duck), a, 0.04);
        p.setTargetAtTime(1, end, 0.35);
      }
    }
  }

  loop(key: string, name: string, vol: number, o: LoopOpts = {}) {
    if (!this.live) return;
    const ctx = this.ctx!, now = ctx.currentTime, s = SPEC[name];
    let g = dbToGain(s.db) * vol, pan = 0;
    if (o.x !== undefined && o.y !== undefined) {
      const [att, p] = this.mixer.locate(o.x, o.y, s.range);
      if (o.own) pan = p * 0.5;
      else { g *= att; pan = p; }
    }
    let l = this.loops.get(key);
    if (!l) {
      if (g < 0.003) return;
      const buf = this.buffer(name, 0);
      if (!buf) return;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      let tail: AudioNode = src.connect(gain);
      let panner: StereoPannerNode | null = null;
      if (buf.numberOfChannels === 1) {
        panner = ctx.createStereoPanner();
        tail = tail.connect(panner);
      }
      tail.connect(this.bus[s.bus]);
      if (s.verb > 0.01) {
        const send = ctx.createGain();
        send.gain.value = s.verb;
        tail.connect(send).connect(this.verbIn);
      }
      // Beds start somewhere inside the loop so they never open on the same note twice.
      src.start(now, s.bus === 'amb' ? Math.random() * buf.duration : 0);
      l = { src, gain, pan: panner, seen: now };
      this.loops.set(key, l);
    }
    l.seen = now;
    l.gain.gain.setTargetAtTime(g * dbToGain(this.mixer.takes.get(name)?.[0].short ?? 0), now, s.bus === 'amb' ? 0.5 : 0.07);
    l.src.playbackRate.setTargetAtTime(o.rate ?? 1, now, 0.1);
    if (l.pan) l.pan.pan.setTargetAtTime(pan, now, 0.1);
  }

  scene(s: Scene) {
    const key = `${s.menu}|${s.dead}|${s.paused}`;
    if (!this.ctx || key === this.sceneKey) return;
    this.sceneKey = key;
    const t = this.ctx.currentTime;
    // How open the world sounds, and how loud: the menu backdrop is the most distant.
    const [cutoff, trim] = s.menu ? [1500, 0.3] : s.paused ? [900, 0.35] : s.dead ? [750, 0.7] : [20000, 1];
    this.muffle.frequency.cancelScheduledValues(t);
    this.muffle.frequency.setTargetAtTime(cutoff, t, cutoff > 5000 ? 0.5 : 0.18);
    this.worldTrim.gain.setTargetAtTime(trim, t, 0.2);
    if (s.menu !== this.menu) {
      this.menu = s.menu;
      this.applySettings();
    }
  }

  /** Per-frame upkeep: where the listener is, loops that are no longer wanted, the score. */
  frame(x: number, y: number, range: number) {
    const L = this.mixer.listener;
    L.x = x;
    L.y = y;
    L.range = range;
    if (!this.ctx || this.ctx.state !== 'running') return;
    const now = this.ctx.currentTime;
    for (const [key, l] of this.loops) {
      if (now - l.seen < 0.2 && !this.settings.muted) continue;
      l.gain.gain.cancelScheduledValues(now);
      l.gain.gain.setTargetAtTime(0, now, 0.08);
      l.src.stop(now + 0.5);
      this.loops.delete(key);
    }
    this.score(now);
  }

  /** The score: overlapping pads walking a four-chord cycle, with a few bell notes scattered over each. */
  private score(now: number) {
    if (!this.live || this.settings.music <= 0 || now < this.padAt - 0.2) return;
    const name = `mus.pad${this.padIdx}`;
    if (!this.mixer.takes.has(name) || !this.mixer.takes.has('mus.bell')) return;
    const delay = Math.max(0, this.padAt - now);
    this.play(name, { delay });
    this.padAt = now + delay + PAD_STEP;
    this.padIdx = (this.padIdx + 1) % PAD_CHORDS.length;
    const bells = this.menu ? 2 + ((Math.random() * 3) | 0) : Math.random() < 0.6 ? 1 : 0;
    for (let i = 0; i < bells; i++) {
      const step = BELL_STEPS[(Math.random() * BELL_STEPS.length) | 0];
      this.play('mus.bell', { delay: delay + 2 + Math.random() * (PAD_STEP - 2), rate: 2 ** (step / 12), vol: 0.5 + Math.random() * 0.5 });
    }
  }
}

export const audio = new AudioEngine();

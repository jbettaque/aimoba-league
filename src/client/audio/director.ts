/**
 * The sound director: turns simulation events and the state of the watched hero into sounds.
 * It only talks to an `AudioOut`, so the same logic drives the real engine in the browser and a
 * recording stand-in in headless mix tests.
 *
 * Three rules shape what is heard:
 *  - Fog of war applies to sound. Nothing is heard from a place the watched team cannot see.
 *  - Anything about the player's own hero is exempt from distance and gets a boost, so their own
 *    actions always read clearly; everyone else's sound is placed in the world.
 *  - Team-wide news (kills, towers, the Wyrm and Titan) is told in short musical phrases whose
 *    mood says who it is good for.
 */
import type { GameEvent, Unit } from '../../sim/defs.ts';
import type { Game } from '../../sim/game.ts';
import { HEROES } from '../../sim/heroes.ts';
import { CORE_POS } from '../../sim/map.ts';
import { SPEC } from './bank.ts';
import type { PlayOpts } from './mixer.ts';

export interface LoopOpts {
  rate?: number;
  x?: number;
  y?: number;
  own?: boolean;
}

export interface Scene {
  /** The match is only a backdrop to the main menu. */
  menu: boolean;
  /** The player's hero is dead: the world sounds distant until they respawn. */
  dead: boolean;
  paused: boolean;
}

export interface AudioOut {
  play(name: string, o?: PlayOpts): void;
  /** Keep a named loop sounding. Call every frame; a loop that stops being asked for fades out. */
  loop(key: string, name: string, vol: number, o?: LoopOpts): void;
  scene(s: Scene): void;
}

export interface SoundCtx {
  game: Game;
  mode: 'menu' | 'play' | 'spectate' | 'replay';
  /** The player's hero in play mode, otherwise the hero the camera follows (if any). */
  focus: Unit | undefined;
  /** Whether a world point is in view of whoever is watching. */
  shown(x: number, y: number): boolean;
  camX: number;
  camY: number;
  paused: boolean;
}

/** Visual effect names from the simulation that have a sound of their own. */
export const FX_SOUND: Record<string, string> = {
  quake: 'fx.quake', cataclysm: 'fx.cataclysm', impact: 'fx.impact', slash: 'fx.slash', execute: 'fx.execute', burst: 'fx.burst',
  explosion: 'fx.explosion', starfall: 'fx.starfall', snare: 'fx.snare', spark: 'fx.spark', thunder: 'fx.thunder', flash: 'fx.flash',
  heal: 'fx.heal', sanctuary: 'fx.sanctuary', teleport: 'fx.teleport',
};
/** Delayed area spells: the sound of them building up, as long as the delay. */
export const ZONE_SOUND: Record<string, string> = { ignW: 'zone.scorch', ignR: 'zone.starfall' };
const MINION_HIT: Record<string, string> = { melee: 'min.melee', ranged: 'min.ranged.hit', siege: 'min.siege.hit', super: 'min.super' };
/** Pitch of the "ability ready" tick for Q, W and E. */
const READY_RATE = [1, 1.122, 1.26];
const HEALTH_WARN = 0.32;

/** Distance from a point to the river, which runs along the map's main diagonal. */
function riverDistance(x: number, y: number): number {
  const t = Math.max(1000, Math.min(7000, (x + y) / 2));
  return Math.hypot(x - t, y - t);
}

export class SoundDirector {
  private out: AudioOut;
  private rand: () => number;
  private heroId = -1;
  private stunned = false;
  private rooted = false;
  private aggro = false;
  private cds = [0, 0, 0, 0];
  private waves = -1;
  private firstBloodTick = -1;
  private birdIn = 4;

  constructor(out: AudioOut, rand: () => number = Math.random) {
    this.out = out;
    this.rand = rand;
  }

  /** Forget everything about the previous match. */
  reset() {
    this.heroId = -1;
    this.waves = -1;
    this.firstBloodTick = -1;
    this.birdIn = 4;
  }

  /** One-off interface sounds. */
  ui(name: string, o?: PlayOpts) {
    this.out.play('ui.' + name, o);
  }

  event(ev: GameEvent, c: SoundCtx) {
    const g = c.game, out = this.out;
    const me = c.mode === 'play' && c.focus ? c.focus : null;
    const x = ev.x ?? 0, y = ev.y ?? 0;
    switch (ev.type) {
      case 'attack': {
        const u = g.unit(ev.id!);
        if (!u || u.range < 250) return; // a melee swing lands at once: its hit is the sound
        const own = u === me;
        if (!own && !c.shown(x, y)) return;
        if (u.hero) out.play(`aa.${HEROES[u.hero.def].id}.fire`, { x, y, own });
        else if (u.kind === 'tower') out.play('tower.fire', { x, y, vol: me && ev.other === me.id ? 1.5 : 1 });
        else if (u.kind === 'minion') out.play(u.subtype === 'siege' ? 'min.siege.fire' : 'min.ranged.fire', { x, y });
        return;
      }
      case 'dmg': {
        const src = ev.other! >= 0 ? g.unit(ev.other!) : undefined;
        const hitMe = me !== null && ev.id === me.id;
        // Being hurt is felt, in proportion to how much it took off. Burn ticks stay quiet.
        if (hitMe && ev.slot !== 0 && ev.amount! > me.maxHp * 0.012) out.play('self.hurt', { vol: Math.min(1, 0.4 + (ev.amount! / me.maxHp) * 6) });
        if (ev.slot !== 1 || !src) return; // abilities sound through their own effects
        const own = hitMe || src === me;
        if (!own && !c.shown(x, y)) return;
        let name: string;
        if (src.hero) name = `aa.${HEROES[src.hero.def].id}.hit`;
        else if (src.kind === 'tower') name = 'tower.hit';
        else if (src.kind === 'minion') name = MINION_HIT[src.subtype] ?? 'min.melee';
        else name = src.subtype === 'wyrm' || src.subtype === 'titan' ? 'mon.big' : 'mon.hit';
        out.play(name, { x, y, own });
        const t = g.unit(ev.id!);
        if (src.hero && t && (t.kind === 'tower' || t.kind === 'core')) out.play('hit.stone', { x, y, own });
        return;
      }
      case 'gold':
        if (!me || ev.id !== me.id) return;
        if (ev.amount! >= 60) out.play('self.goldBig');
        else out.play('self.gold', { rate: 1 + Math.min(0.2, ev.amount! / 250) });
        return;
      case 'cast': {
        const u = g.unit(ev.id!);
        if (!u || !u.hero) return;
        const own = u === me;
        if (own || c.shown(x, y)) {
          const name = `ab.${HEROES[u.hero.def].id}.${'qwer'[ev.slot!]}`;
          if (SPEC[name]) out.play(name, { x, y, own });
        }
        for (const z of g.zones) {
          const zs = ZONE_SOUND[z.kind];
          if (zs && z.srcId === u.id && z.total - z.delay < 0.05 && (own || c.shown(z.x, z.y))) out.play(zs, { x: z.x, y: z.y });
        }
        return;
      }
      case 'fx': {
        const name = FX_SOUND[ev.name!];
        if (!name) return;
        const own = me !== null && ev.id === me.id;
        if (own || c.shown(x, y)) out.play(name, { x, y, own });
        return;
      }
      case 'levelup':
        if (me && ev.id === me.id) out.play('self.level');
        else if (c.shown(x, y)) out.play('fx.levelup', { x, y });
        return;
      case 'respawn':
        if (me && ev.id === me.id) out.play('self.respawn');
        else if (c.shown(x, y)) out.play('fx.respawn', { x, y });
        return;
      case 'death':
        if (ev.name === 'tower') out.play('tower.death', { x, y });
        else if (ev.name === 'core') out.play('core.death', { x, y });
        else if (!c.shown(x, y)) return;
        else if (ev.name === 'minion') out.play('min.death', { x, y });
        else if (ev.name === 'monster') out.play(ev.sub === 'wyrm' || ev.sub === 'titan' ? 'mon.bigDeath' : 'mon.death', { x, y });
        else if (!me || ev.id !== me.id) out.play('hero.death', { x, y });
        return;
    }
    if (c.mode === 'menu') return; // the backdrop match makes no announcements
    switch (ev.type) {
      case 'kill': {
        if (me && ev.id === me.id) { out.play('self.death'); return; }
        if (g.tick === this.firstBloodTick) return; // First Blood already said it
        if (!me) out.play('sting.neutral');
        else if (ev.other === me.id) out.play('sting.kill');
        else out.play(ev.team === me.team ? 'sting.allyDeath' : 'sting.allyKill');
        return;
      }
      case 'announce': {
        const ours = !me || ev.team === me.team;
        if (ev.name === 'firstblood') { this.firstBloodTick = g.tick; out.play('sting.firstBlood'); }
        else if (ev.name === 'tower') out.play(!me ? 'sting.neutral' : ours ? 'sting.towerGood' : 'sting.towerBad');
        else if (ev.name === 'objective') out.play(ours ? 'sting.objGood' : 'sting.objBad');
        else if (ev.name === 'awaken') out.play('sting.awaken');
        return;
      }
      case 'gameover':
        out.play(!me || ev.team === me.team ? 'sting.victory' : ev.team! < 0 ? 'sting.neutral' : 'sting.defeat');
        return;
    }
  }

  /** Per-frame work: state the events do not report, loops, ambience. `dt` is real seconds. */
  frame(dt: number, c: SoundCtx) {
    const g = c.game, out = this.out;
    const menu = c.mode === 'menu';
    const me = c.mode === 'play' && c.focus ? c.focus : null;
    out.scene({ menu, dead: me !== null && !me.alive, paused: c.paused });

    if (me) this.watchHero(me, g);
    else this.heroId = -1;

    if (!menu) {
      // The first wave leaving the bases is the real start of the match.
      if (this.waves === 0 && g.waveCount > 0) out.play('sting.horn');
      this.waves = g.waveCount;
    }

    // Recalls hum and rise while they are channelled.
    for (const h of g.heroes) {
      if (!h.alive || h.hero!.order.type !== 'recall') continue;
      const own = h === me;
      if (!own && !c.shown(h.x, h.y)) continue;
      const p = Math.max(0, Math.min(1, 1 - h.hero!.recallT / 5));
      out.loop('recall' + h.id, 'fx.recall', 0.35 + 0.65 * p, { rate: 0.8 + 0.7 * p, x: h.x, y: h.y, own });
    }

    // Ambience follows the camera: open air everywhere, water by the river, a hum near either Core.
    out.loop('wind', 'amb.wind', menu ? 0.6 : 1);
    if (menu) return;
    const river = Math.max(0, Math.min(1, 1 - (riverDistance(c.camX, c.camY) - 250) / 1200));
    if (river > 0) out.loop('river', 'amb.river', river ** 1.5);
    const core = Math.min(...CORE_POS.map(([x, y]) => Math.hypot(c.camX - x, c.camY - y)));
    if (core < 2000) out.loop('base', 'amb.base', (1 - core / 2000) ** 1.3);
    if (!c.paused && (this.birdIn -= dt) <= 0) {
      this.birdIn = 5 + this.rand() * 10;
      out.play('amb.bird', { x: c.camX + (this.rand() - 0.5) * 3000, y: c.camY + (this.rand() - 0.5) * 2000, vol: 0.6 + this.rand() * 0.4 });
    }
  }

  /** Things the player needs to know about their own hero that no event announces. */
  private watchHero(me: Unit, g: Game) {
    const out = this.out, h = me.hero!;
    const fresh = this.heroId !== me.id;
    this.heroId = me.id;

    const frac = me.hp / me.maxHp;
    if (me.alive && frac < HEALTH_WARN) {
      const danger = Math.min(1, (HEALTH_WARN - frac) / 0.2);
      out.loop('heart', 'self.heartbeat', 0.35 + 0.65 * danger, { rate: 1 + 0.3 * danger });
    }

    if (me.stunned && !this.stunned && !fresh) out.play('self.stun');
    else if (me.rooted && !this.rooted && !fresh) out.play('self.root');
    this.stunned = me.stunned;
    this.rooted = me.rooted;

    let aggro = false;
    if (me.alive) for (const s of g.structures) if (s.alive && s.kind === 'tower' && s.team !== me.team && s.targetId === me.id) { aggro = true; break; }
    if (aggro && !this.aggro && !fresh) out.play('self.towerWarn');
    this.aggro = aggro;

    for (let s = 0; s < 4; s++) {
      const cd = h.cds[s];
      if (!fresh && me.alive && this.cds[s] > 0 && cd <= 0 && h.ranks[s] > 0) {
        if (s === 3) out.play('ui.readyUlt');
        else out.play('ui.ready', { rate: READY_RATE[s] });
      }
      this.cds[s] = cd;
    }
  }
}

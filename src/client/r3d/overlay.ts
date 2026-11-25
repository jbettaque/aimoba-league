/**
 * Screen-space overlay on its own transparent 2D canvas, stacked above the WebGL canvas:
 * health / mana bars, level badges, names and floating combat text.
 */
import { TEAM_NEUTRAL } from '../../sim/defs.ts';
import { attackDamage } from '../../sim/game.ts';
import { HEROES } from '../../sim/heroes.ts';
import type { Fx } from '../fx.ts';
import type { Shown } from './ground2d.ts';
import { LOOK } from './look.ts';

export class Overlay {
  canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;

  constructor(below: HTMLCanvasElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'overlay2d';
    this.canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;display:block;pointer-events:none';
    below.insertAdjacentElement('afterend', this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
  }

  resize(w: number, h: number, dpr: number) {
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
  }

  /**
   * `project(simX, simY, worldHeight)` maps to CSS pixels. Bars sit above each unit's head.
   */
  draw(shown: Shown[], fx: Fx, focusId: number, w: number, h: number, project: (x: number, y: number, hgt: number) => [number, number]) {
    const c = this.ctx, TEAM = LOOK.teamColors;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    // The hero whose attacks decide which creeps are marked as ready for the last hit.
    const focus = shown.find((s) => s.u.id === focusId)?.u;
    const striker = focus && focus.hero && focus.alive ? focus : null;
    for (const s of shown) {
      const u = s.u;
      if (!u.alive) continue;
      const [sx, top] = project(s.x, s.y, s.wy + s.height);
      if (sx < -80 || sx > w + 80 || top < -40 || top > h + 80) continue;
      const frac = Math.max(0, u.hp / u.maxHp);
      if (u.kind === 'hero') {
        const hs = u.hero!;
        const bw = 92, bh = 10, bx = Math.round(sx - bw / 2) + 8, by = Math.round(top - 20);
        const own = u.id === focusId;
        const col = own ? '#63e06c' : TEAM[u.team];
        c.globalAlpha = u.stealthed ? 0.5 : 1;
        c.fillStyle = 'rgba(6,10,16,0.88)';
        this.rr(bx - 2, by - 2, bw + 4, bh + 9, 3); c.fill();
        c.fillStyle = '#1d2530';
        c.fillRect(bx, by, bw, bh);
        const total = Math.max(u.maxHp, u.hp + u.shield);
        c.fillStyle = col;
        c.fillRect(bx, by, bw * (u.hp / total), bh);
        if (u.shield > 0) { c.fillStyle = '#e8f3ff'; c.fillRect(bx + bw * (u.hp / total), by, bw * (u.shield / total), bh); }
        c.fillStyle = 'rgba(255,255,255,0.25)';
        c.fillRect(bx, by, bw * (u.hp / total), 3);
        c.fillStyle = 'rgba(0,0,0,0.5)';
        for (let v = 250; v < total; v += 250) c.fillRect(bx + (bw * v) / total, by, 1, v % 1000 === 0 ? bh : bh * 0.55);
        c.fillStyle = '#3b8cff';
        c.fillRect(bx, by + bh + 1, bw * (u.maxMana > 0 ? u.mana / u.maxMana : 0), 4);
        c.fillStyle = 'rgba(6,10,16,0.92)';
        c.beginPath(); c.arc(bx - 11, by + 6, 11, 0, 6.283); c.fill();
        c.strokeStyle = col; c.lineWidth = 1.5; c.stroke();
        c.fillStyle = '#fff';
        c.font = "700 12px 'Alegreya Sans', system-ui, sans-serif";
        c.fillText(String(hs.level), bx - 11, by + 6.5);
        c.font = "700 13px 'Alegreya Sans', system-ui, sans-serif";
        c.lineWidth = 3;
        c.strokeStyle = 'rgba(0,0,0,0.8)';
        const name = HEROES[hs.def].name + (hs.isBot ? '' : own ? '' : ' ★');
        c.strokeText(name, sx + 2, by - 10);
        c.fillStyle = own ? '#c9ffd0' : u.team === 0 ? '#cfe6ff' : '#ffd3d3';
        c.fillText(name, sx + 2, by - 10);
        c.globalAlpha = 1;
      } else if (u.kind === 'tower' || u.kind === 'core') {
        const bw = u.kind === 'core' ? 130 : 90, bh = 8, bx = sx - bw / 2, by = top - 16;
        c.fillStyle = 'rgba(6,10,16,0.88)';
        this.rr(bx - 2, by - 2, bw + 4, bh + 4, 3); c.fill();
        c.fillStyle = TEAM[u.team];
        c.fillRect(bx, by, bw * frac, bh);
        c.fillStyle = 'rgba(255,255,255,0.25)';
        c.fillRect(bx, by, bw * frac, 2.5);
        if (u.kind === 'tower') {
          // Five plates: each one broken pays the attackers who are there.
          c.fillStyle = 'rgba(6,10,16,0.75)';
          for (let k = 1; k < 5; k++) c.fillRect(bx + (bw * k) / 5 - 0.75, by, 1.5, bh);
        }
        if (u.fortified) {
          // Fortified in the opening minutes: a pale frame and a small shield beside the bar.
          c.strokeStyle = '#e8f3ff';
          c.lineWidth = 2;
          c.strokeRect(bx - 2, by - 2, bw + 4, bh + 4);
          const gx = bx - 13, gy = by + bh / 2;
          c.fillStyle = '#e8f3ff';
          c.beginPath();
          c.moveTo(gx - 6, gy - 7); c.lineTo(gx + 6, gy - 7); c.lineTo(gx + 6, gy); c.quadraticCurveTo(gx + 5, gy + 6, gx, gy + 9); c.quadraticCurveTo(gx - 5, gy + 6, gx - 6, gy);
          c.closePath(); c.fill();
        }
      } else {
        const big = u.kind === 'monster' && u.radius > 50;
        const bw = big ? 96 : u.kind === 'monster' ? 44 : 34, bh = big ? 7 : 4, bx = sx - bw / 2, by = top - (big ? 14 : 8);
        c.fillStyle = 'rgba(6,10,16,0.85)';
        c.fillRect(bx - 1, by - 1, bw + 2, bh + 2);
        // A creep the followed hero's next attack would kill is marked: bright bar, gold frame.
        const ready = striker !== null && u.team !== striker.team && attackDamage(striker, u) >= u.hp;
        c.fillStyle = ready ? '#fff6c9' : u.team === TEAM_NEUTRAL ? '#e0b84c' : TEAM[u.team];
        c.fillRect(bx, by, bw * frac, bh);
        if (ready) {
          c.strokeStyle = '#ffd75e';
          c.lineWidth = 2;
          c.strokeRect(bx - 2.5, by - 2.5, bw + 5, bh + 5);
        }
      }
    }

    // Floating combat text.
    for (const t of fx.texts) {
      const [sx, sy] = project(t.x, t.y, 0.9);
      if (sx < -50 || sy < -50 || sx > w + 50 || sy > h + 50) continue;
      const f = t.life / t.max;
      c.globalAlpha = Math.min(1, f * 2.2);
      const pop = 1 + Math.max(0, f - 0.8) * 2;
      c.font = `800 ${Math.round(t.size * pop)}px 'Alegreya Sans', system-ui, sans-serif`;
      c.lineWidth = 3.5;
      c.strokeStyle = 'rgba(0,0,0,0.85)';
      c.strokeText(t.text, sx, sy);
      c.fillStyle = t.color;
      c.fillText(t.text, sx, sy);
    }
    c.globalAlpha = 1;
  }

  private rr(x: number, y: number, w: number, h: number, r: number) {
    const c = this.ctx;
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }
}

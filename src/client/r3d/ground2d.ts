/**
 * The flat, top-down layer of the game, drawn with Canvas 2D in sim coordinates and draped
 * over the terrain by DecalLayer: fountain pulses, ability telegraphs, ground particles, aim
 * indicators, tower range rings, team / hover / buff rings and status rings.
 */
import { TOWER_RANGE, type Unit } from '../../sim/defs.ts';
import type { Game } from '../../sim/game.ts';
import { HEROES } from '../../sim/heroes.ts';
import { FOUNTAIN_POS } from '../../sim/map.ts';
import type { Fx } from '../fx.ts';
import { LOOK } from './look.ts';

export interface GroundView {
  team: number;
  focusId: number;
  hoverId: number;
  aimSlot: number;
  amove: boolean;
  mouseX: number;
  mouseY: number;
}
export interface Shown { u: Unit; x: number; y: number; wy: number; height: number }

export function drawGroundLayer(
  c: CanvasRenderingContext2D, g: Game, view: GroundView, shown: Shown[], fx: Fx, time: number, alpha: number,
  x0: number, y0: number, x1: number, y1: number, sees: (u: Unit) => boolean,
) {
  const TEAM = LOOK.teamColors;
  const spectator = view.team < 0;
  c.lineCap = 'butt';
  c.setLineDash([]);

  // Fountains.
  for (let team = 0; team < 2; team++) {
    const [fx0, fy0] = FOUNTAIN_POS[team];
    if (fx0 < x0 - 700 || fx0 > x1 + 700 || fy0 < y0 - 700 || fy0 > y1 + 700) continue;
    c.strokeStyle = team === 0 ? 'rgba(120,200,255,0.45)' : 'rgba(255,140,120,0.45)';
    c.lineWidth = 8;
    for (let k = 0; k < 3; k++) {
      const rr = (time * 60 + k * 215) % 650;
      c.globalAlpha = 1 - rr / 650;
      c.beginPath(); c.arc(fx0, fy0, rr, 0, 6.283); c.stroke();
    }
    c.globalAlpha = 1;
  }

  // Ability telegraphs.
  for (const z of g.zones) {
    if (z.x < x0 - 500 || z.x > x1 + 500 || z.y < y0 - 500 || z.y > y1 + 500) continue;
    const k = 1 - z.delay / z.total;
    const enemy = !spectator && z.team !== view.team;
    const col = enemy ? '255,80,60' : '255,170,60';
    c.fillStyle = `rgba(${col},0.16)`;
    c.beginPath(); c.arc(z.x, z.y, z.radius, 0, 6.283); c.fill();
    c.fillStyle = `rgba(${col},0.32)`;
    c.beginPath(); c.arc(z.x, z.y, z.radius * k, 0, 6.283); c.fill();
    c.strokeStyle = `rgba(${col},0.95)`;
    c.lineWidth = 7;
    c.beginPath(); c.arc(z.x, z.y, z.radius, 0, 6.283); c.stroke();
  }
  fx.draw(c, true, x0, y0, x1, y1);
  c.globalAlpha = 1;
  c.globalCompositeOperation = 'source-over';

  const focus = view.focusId >= 0 ? g.unit(view.focusId) : undefined;
  // Aim indicators for the player's pending cast.
  if (focus && focus.alive && focus.hero && !spectator) {
    const fxp = focus.px + (focus.x - focus.px) * alpha, fyp = focus.py + (focus.y - focus.py) * alpha;
    if (view.aimSlot >= 0) {
      const ab = HEROES[focus.hero.def].abilities[view.aimSlot];
      const dx = view.mouseX - fxp, dy = view.mouseY - fyp;
      const d = Math.hypot(dx, dy) || 1;
      c.strokeStyle = 'rgba(130,210,255,0.65)';
      c.fillStyle = 'rgba(130,210,255,0.16)';
      c.lineWidth = 5;
      if (ab.range > 0 && ab.range < 3000) { c.beginPath(); c.arc(fxp, fyp, ab.range, 0, 6.283); c.stroke(); }
      if (ab.targeting === 'dir') {
        const len = Math.min(ab.range, 1400), wdt = Math.max(30, ab.radius);
        c.save();
        c.translate(fxp, fyp);
        c.rotate(Math.atan2(dy, dx));
        const gr = c.createLinearGradient(0, 0, len, 0);
        gr.addColorStop(0, 'rgba(130,210,255,0.38)');
        gr.addColorStop(1, 'rgba(130,210,255,0.1)');
        c.fillStyle = gr;
        c.beginPath(); c.moveTo(0, -wdt); c.lineTo(len - wdt, -wdt); c.lineTo(len, 0); c.lineTo(len - wdt, wdt); c.lineTo(0, wdt); c.closePath(); c.fill(); c.stroke();
        c.restore();
      } else if (ab.targeting === 'point') {
        const k = Math.min(1, ab.range / d);
        c.beginPath(); c.arc(fxp + dx * k, fyp + dy * k, ab.radius, 0, 6.283); c.fill(); c.stroke();
      } else if (ab.targeting === 'self' && ab.radius > 0) {
        c.beginPath(); c.arc(fxp, fyp, ab.radius, 0, 6.283); c.fill(); c.stroke();
      }
    } else if (view.amove) {
      c.strokeStyle = 'rgba(255,120,100,0.6)';
      c.lineWidth = 5;
      c.beginPath(); c.arc(fxp, fyp, focus.range + 45, 0, 6.283); c.stroke();
    }
  }

  // Tower range rings near the focused hero, and what each tower is shooting at.
  for (const s of g.structures) {
    if (!s.alive || s.kind !== 'tower') continue;
    if (s.x < x0 - 900 || s.x > x1 + 900 || s.y < y0 - 900 || s.y > y1 + 900) continue;
    const threat = focus && focus.alive && s.team !== focus.team && Math.hypot(s.x - focus.x, s.y - focus.y) < TOWER_RANGE + 450;
    const hot = view.focusId >= 0 && s.targetId === view.focusId;
    if (threat || hot) {
      c.strokeStyle = hot ? 'rgba(255,70,50,0.9)' : 'rgba(255,170,90,0.45)';
      c.fillStyle = hot ? 'rgba(255,60,40,0.08)' : 'rgba(255,170,90,0.035)';
      c.lineWidth = hot ? 8 : 5;
      c.beginPath(); c.arc(s.x, s.y, TOWER_RANGE, 0, 6.283); c.fill(); c.stroke();
    }
    const tgt = s.targetId >= 0 ? g.unit(s.targetId) : undefined;
    if (tgt && tgt.alive && sees(tgt)) {
      c.strokeStyle = s.team === 0 ? 'rgba(120,190,255,0.45)' : 'rgba(255,120,110,0.45)';
      c.lineWidth = 4;
      c.setLineDash([18, 14]);
      c.beginPath(); c.moveTo(s.x, s.y); c.lineTo(tgt.x, tgt.y); c.stroke();
      c.setLineDash([]);
    }
  }

  // Per-unit rings.
  for (const { u, x: ux, y: uy } of shown) {
    if (u.kind === 'tower' || u.kind === 'core') {
      if (u.id === view.hoverId) {
        c.strokeStyle = 'rgba(255,90,70,0.95)';
        c.lineWidth = 7;
        c.beginPath(); c.arc(ux, uy, u.radius + 14, 0, 6.283); c.stroke();
      }
      if (u.alive && !u.vulnerable) {
        c.strokeStyle = 'rgba(190,225,255,0.55)';
        c.lineWidth = 5;
        c.setLineDash([22, 16]);
        c.beginPath(); c.arc(ux, uy, u.radius * 1.3, time * 0.5, time * 0.5 + 6.283); c.stroke();
        c.setLineDash([]);
      }
      continue;
    }
    const fade = u.stealthed ? 0.45 : 1;
    if (u.kind === 'hero') {
      const isFocus = u.id === view.focusId;
      c.globalAlpha = 0.95 * fade;
      c.strokeStyle = isFocus ? '#8dffa0' : TEAM[u.team];
      c.lineWidth = isFocus ? 7 : 6;
      c.beginPath(); c.arc(ux, uy, u.radius + 14, 0, 6.283); c.stroke();
      c.fillStyle = u.team === 0 ? 'rgba(74,168,255,0.2)' : 'rgba(255,93,93,0.2)';
      c.fill();
      c.globalAlpha = fade;
      for (const b of u.buffs) {
        const col = b.kind === 'might' ? '255,90,60' : b.kind === 'insight' ? '90,160,255' : b.kind === 'titan' ? '190,110,255' : '';
        if (!col) continue;
        c.strokeStyle = `rgba(${col},0.8)`;
        c.lineWidth = 6;
        c.setLineDash([20, 18]);
        const rr = u.radius + (b.kind === 'titan' ? 34 : 24);
        c.beginPath(); c.arc(ux, uy, rr, time * 1.5, time * 1.5 + 6.283); c.stroke();
        c.setLineDash([]);
      }
    } else {
      // A faint team disc keeps small units readable against the ground.
      c.globalAlpha = 0.5 * fade;
      c.strokeStyle = TEAM[u.team] ?? TEAM[2];
      c.lineWidth = 3;
      c.beginPath(); c.arc(ux, uy, u.radius * 0.9, 0, 6.283); c.stroke();
    }
    c.globalAlpha = 1;
    if (u.id === view.hoverId) {
      c.strokeStyle = u.team === (focus ? focus.team : 0) ? 'rgba(255,255,255,0.9)' : 'rgba(255,90,70,0.95)';
      c.lineWidth = 6;
      c.beginPath(); c.arc(ux, uy, u.radius + 8, 0, 6.283); c.stroke();
    }
    if (u.shield > 0) {
      c.strokeStyle = 'rgba(210,235,255,0.9)';
      c.fillStyle = 'rgba(180,220,255,0.16)';
      c.lineWidth = 5;
      c.beginPath(); c.arc(ux, uy, u.radius + 20, 0, 6.283); c.fill(); c.stroke();
    }
    if (u.rooted) {
      c.strokeStyle = 'rgba(255,170,70,0.95)';
      c.lineWidth = 7;
      c.setLineDash([14, 10]);
      c.beginPath(); c.arc(ux, uy, u.radius + 6, -time * 2, -time * 2 + 6.283); c.stroke();
      c.setLineDash([]);
    }
    if (u.hero) {
      if (u.hero.order.type === 'recall') {
        const k = 1 - u.hero.recallT / 5;
        c.strokeStyle = 'rgba(140,220,255,0.95)';
        c.lineWidth = 9;
        c.beginPath(); c.arc(ux, uy, u.radius + 32, -1.57, -1.57 + k * 6.283); c.stroke();
      }
      for (const b of u.buffs) {
        if (b.kind === 'slow') { c.strokeStyle = 'rgba(120,190,255,0.8)'; c.lineWidth = 5; c.beginPath(); c.arc(ux, uy + 6, u.radius + 2, 0.4, 2.74); c.stroke(); }
      }
    }
  }
}

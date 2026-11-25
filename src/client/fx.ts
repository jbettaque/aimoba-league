/** Lightweight particle / floating-text system driven by simulation events. */

interface Part {
  kind: 'dot' | 'ring' | 'disc' | 'line' | 'shard';
  x: number; y: number; vx: number; vy: number;
  life: number; max: number;
  size: number; size2: number;
  color: string;
  x2: number; y2: number;
  ground: boolean;
  drag: number;
  rot: number;
  /** Height above the ground (sim units) and vertical speed, for the 3D renderer. */
  z: number;
  vz: number;
  grav: number;
}

interface FloatText {
  x: number; y: number; vy: number; vx: number;
  text: string; color: string; size: number;
  life: number; max: number;
}

export class Fx {
  parts: Part[] = [];
  texts: FloatText[] = [];
  shake = 0;

  clear() {
    this.parts.length = 0;
    this.texts.length = 0;
    this.shake = 0;
  }

  private add(p: Partial<Part>) {
    if (this.parts.length > 2500) return;
    this.parts.push({ kind: 'dot', x: 0, y: 0, vx: 0, vy: 0, life: 0.5, max: 0.5, size: 6, size2: 0, color: '#fff', x2: 0, y2: 0, ground: false, drag: 2, rot: 0, z: 55, vz: 0, grav: 0, ...p });
    const n = this.parts[this.parts.length - 1];
    n.max = n.life;
  }

  burst(x: number, y: number, color: string, n: number, speed: number, size: number, life: number) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.35 + Math.random() * 0.65);
      this.add({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, color, size: size * (0.5 + Math.random() * 0.8), life: life * (0.6 + Math.random() * 0.6), drag: 3, z: 35 + Math.random() * 40, vz: s * (0.15 + Math.random() * 0.75), grav: 900 });
    }
  }

  shards(x: number, y: number, color: string, n: number, speed: number, size: number) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.4 + Math.random() * 0.6);
      this.add({ kind: 'shard', x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, color, size: size * (0.6 + Math.random() * 0.8), life: 0.7 + Math.random() * 0.5, drag: 2.5, rot: Math.random() * 6, ground: true });
    }
  }

  ring(x: number, y: number, r0: number, r1: number, color: string, life: number, width = 8, ground = true) {
    this.add({ kind: 'ring', x, y, size: r0, size2: r1, color, life, x2: width, ground });
  }

  disc(x: number, y: number, r: number, color: string, life: number, ground = true) {
    this.add({ kind: 'disc', x, y, size: r, color, life, ground });
  }

  line(x: number, y: number, x2: number, y2: number, color: string, life: number, width = 6, z = 60) {
    this.add({ kind: 'line', x, y, x2, y2, color, life, size: width, z });
  }

  trail(x: number, y: number, color: string, size: number, life = 0.25, z = 55) {
    this.add({ x: x + (Math.random() - 0.5) * size, y: y + (Math.random() - 0.5) * size, color, size, life, vx: (Math.random() - 0.5) * 40, vy: (Math.random() - 0.5) * 40, z: z + (Math.random() - 0.5) * size, vz: 20 + Math.random() * 40 });
  }

  text(x: number, y: number, text: string, color: string, size = 16, life = 0.9) {
    if (this.texts.length > 200) this.texts.shift();
    this.texts.push({ x: x + (Math.random() - 0.5) * 50, y: y - 40, vx: (Math.random() - 0.5) * 30, vy: -95, text, color, size, life, max: life });
  }

  update(dt: number) {
    const ps = this.parts;
    for (let i = ps.length - 1; i >= 0; i--) {
      const p = ps[i];
      p.life -= dt;
      if (p.life <= 0) { ps[i] = ps[ps.length - 1]; ps.pop(); continue; }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      p.vz -= p.grav * dt;
      if (p.z < 6) { p.z = 6; p.vz = 0; }
      const k = Math.max(0, 1 - p.drag * dt);
      p.vx *= k;
      p.vy *= k;
    }
    const ts = this.texts;
    for (let i = ts.length - 1; i >= 0; i--) {
      const t = ts[i];
      t.life -= dt;
      if (t.life <= 0) { ts.splice(i, 1); continue; }
      t.x += t.vx * dt;
      t.y += t.vy * dt;
      t.vy *= 1 - 1.6 * dt;
    }
    this.shake = Math.max(0, this.shake - dt * 28);
  }

  /**
   * Draw one layer in world space on a 2D canvas. The 3D renderer uses this for the ground
   * layer only (it is projected onto the terrain); airborne parts become billboards.
   */
  draw(c: CanvasRenderingContext2D, ground: boolean, x0: number, y0: number, x1: number, y1: number) {
    c.save();
    if (!ground) c.globalCompositeOperation = 'lighter';
    for (const p of this.parts) {
      if (p.ground !== ground) continue;
      if (p.x < x0 - 500 || p.x > x1 + 500 || p.y < y0 - 500 || p.y > y1 + 500) continue;
      const f = p.life / p.max;
      switch (p.kind) {
        case 'dot':
          c.globalAlpha = Math.min(1, f * 1.6);
          c.fillStyle = p.color;
          c.beginPath();
          c.arc(p.x, p.y, p.size * (0.4 + 0.6 * f), 0, 6.283);
          c.fill();
          break;
        case 'shard':
          c.globalAlpha = Math.min(1, f * 2);
          c.fillStyle = p.color;
          c.save();
          c.translate(p.x, p.y);
          c.rotate(p.rot + p.life * 3);
          c.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.7);
          c.restore();
          break;
        case 'ring': {
          const r = p.size + (p.size2 - p.size) * (1 - f * f);
          c.globalAlpha = f;
          c.strokeStyle = p.color;
          c.lineWidth = p.x2 * (0.4 + 0.6 * f);
          c.beginPath();
          c.arc(p.x, p.y, Math.max(1, r), 0, 6.283);
          c.stroke();
          break;
        }
        case 'disc':
          c.globalAlpha = f * 0.55;
          c.fillStyle = p.color;
          c.beginPath();
          c.arc(p.x, p.y, p.size, 0, 6.283);
          c.fill();
          break;
        case 'line':
          c.globalAlpha = f;
          c.strokeStyle = p.color;
          c.lineWidth = p.size * f;
          c.lineCap = 'round';
          c.beginPath();
          c.moveTo(p.x, p.y);
          c.lineTo(p.x2, p.y2);
          c.stroke();
          break;
      }
    }
    c.restore();
  }
}

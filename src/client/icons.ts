/** Procedurally drawn icons for abilities and items (no image assets needed). */

type G = CanvasRenderingContext2D;

function star(c: G, x: number, y: number, r1: number, r2: number, n: number, rot = -Math.PI / 2) {
  c.beginPath();
  for (let i = 0; i < n * 2; i++) {
    const r = i % 2 ? r2 : r1;
    const a = rot + (i * Math.PI) / n;
    c.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
  }
  c.closePath();
}

function poly(c: G, pts: number[]) {
  c.beginPath();
  for (let i = 0; i < pts.length; i += 2) c.lineTo(pts[i], pts[i + 1]);
  c.closePath();
}

/** Glyphs are drawn in a 64x64 box using the current fill/stroke style. */
const GLYPHS: Record<string, (c: G) => void> = {
  rush(c) { poly(c, [10, 22, 36, 22, 36, 12, 56, 32, 36, 52, 36, 42, 10, 42]); c.fill(); },
  shield(c) { c.beginPath(); c.moveTo(32, 8); c.lineTo(54, 16); c.quadraticCurveTo(54, 44, 32, 58); c.quadraticCurveTo(10, 44, 10, 16); c.closePath(); c.fill(); },
  quake(c) { c.lineWidth = 5; for (const r of [10, 19, 28]) { c.beginPath(); c.ellipse(32, 40, r, r * 0.55, 0, 0, 7); c.stroke(); } poly(c, [28, 6, 38, 6, 35, 30, 30, 30]); c.fill(); },
  meteor(c) { c.beginPath(); c.arc(40, 42, 13, 0, 7); c.fill(); poly(c, [8, 6, 34, 30, 28, 44]); c.fill(); poly(c, [22, 4, 46, 28, 36, 32]); c.fill(); },
  dagger(c) { poly(c, [50, 8, 56, 14, 28, 44, 20, 36]); c.fill(); poly(c, [14, 34, 30, 50, 26, 54, 10, 38]); c.fill(); poly(c, [16, 44, 20, 48, 10, 58, 6, 54]); c.fill(); },
  claw(c) { c.lineWidth = 6; c.lineCap = 'round'; for (const o of [-14, 0, 14]) { c.beginPath(); c.moveTo(20 + o, 10); c.quadraticCurveTo(36 + o, 30, 26 + o, 54); c.stroke(); } },
  veil(c) { c.beginPath(); c.arc(32, 26, 18, Math.PI, 0); c.lineTo(50, 54); c.lineTo(42, 46); c.lineTo(32, 56); c.lineTo(22, 46); c.lineTo(14, 54); c.closePath(); c.fill(); c.globalCompositeOperation = 'destination-out'; c.beginPath(); c.arc(25, 28, 4, 0, 7); c.arc(39, 28, 4, 0, 7); c.fill(); c.globalCompositeOperation = 'source-over'; },
  skull(c) { c.beginPath(); c.arc(32, 26, 19, 0, 7); c.fill(); c.fillRect(22, 38, 20, 16); c.globalCompositeOperation = 'destination-out'; c.beginPath(); c.arc(24, 27, 5.5, 0, 7); c.arc(40, 27, 5.5, 0, 7); c.fill(); c.fillRect(30, 44, 4, 12); c.globalCompositeOperation = 'source-over'; },
  bolt(c) { poly(c, [8, 28, 40, 20, 58, 32, 40, 44, 8, 36, 20, 32]); c.fill(); },
  flame(c) { c.beginPath(); c.moveTo(32, 4); c.bezierCurveTo(52, 24, 54, 40, 44, 52); c.bezierCurveTo(40, 58, 24, 58, 20, 52); c.bezierCurveTo(8, 40, 22, 30, 22, 18); c.bezierCurveTo(28, 22, 30, 16, 32, 4); c.fill(); },
  snare(c) { c.lineWidth = 6; c.beginPath(); c.arc(32, 32, 20, 0, 7); c.stroke(); c.beginPath(); c.arc(32, 32, 8, 0, 7); c.fill(); for (let i = 0; i < 4; i++) { const a = i * 1.57 + 0.78; c.beginPath(); c.moveTo(32 + Math.cos(a) * 20, 32 + Math.sin(a) * 20); c.lineTo(32 + Math.cos(a) * 30, 32 + Math.sin(a) * 30); c.stroke(); } },
  arrow(c) { c.lineWidth = 5; c.beginPath(); c.moveTo(8, 56); c.lineTo(46, 18); c.stroke(); poly(c, [58, 6, 52, 30, 34, 12]); c.fill(); poly(c, [8, 56, 6, 42, 14, 50, 22, 58]); c.fill(); },
  volley(c) { for (const o of [-14, 0, 14]) { poly(c, [32 + o, 6, 40 + o, 22, 35 + o, 22, 35 + o, 56, 29 + o, 56, 29 + o, 22, 24 + o, 22]); c.fill(); } },
  wind(c) { c.lineWidth = 5; c.lineCap = 'round'; for (const [y, w] of [[18, 34], [32, 44], [46, 28]]) { c.beginPath(); c.moveTo(8, y); c.lineTo(8 + w, y); c.arc(8 + w, y - 6, 6, 1.57, -1.2, true); c.stroke(); } },
  thunder(c) { poly(c, [36, 4, 14, 36, 28, 36, 22, 60, 50, 24, 34, 24]); c.fill(); },
  heal(c) { c.fillRect(25, 10, 14, 44); c.fillRect(10, 25, 44, 14); },
  chain(c) { c.lineWidth = 6; for (const [x, y] of [[20, 20], [32, 32], [44, 44]]) { c.beginPath(); c.ellipse(x, y, 11, 7, 0.785, 0, 7); c.stroke(); } },
  aegis(c) { c.lineWidth = 5; c.beginPath(); c.arc(32, 32, 22, 0, 7); c.stroke(); star(c, 32, 32, 14, 6, 4); c.fill(); },
  sun(c) { c.beginPath(); c.arc(32, 32, 12, 0, 7); c.fill(); c.lineWidth = 5; c.lineCap = 'round'; for (let i = 0; i < 8; i++) { const a = i * 0.785; c.beginPath(); c.moveTo(32 + Math.cos(a) * 18, 32 + Math.sin(a) * 18); c.lineTo(32 + Math.cos(a) * 27, 32 + Math.sin(a) * 27); c.stroke(); } },
  // Items.
  boot(c) { poly(c, [20, 8, 36, 8, 36, 34, 54, 42, 54, 54, 14, 54, 14, 40, 20, 34]); c.fill(); },
  sword(c) { poly(c, [48, 6, 58, 6, 58, 16, 28, 46, 18, 36]); c.fill(); poly(c, [12, 34, 30, 52, 26, 56, 8, 38]); c.fill(); poly(c, [14, 46, 18, 50, 8, 60, 4, 56]); c.fill(); },
  book(c) { c.fillRect(10, 12, 20, 40); c.fillRect(34, 12, 20, 40); c.globalAlpha *= 0.6; c.fillRect(30, 12, 4, 40); },
  heart(c) { c.beginPath(); c.moveTo(32, 54); c.bezierCurveTo(4, 34, 10, 10, 32, 22); c.bezierCurveTo(54, 10, 60, 34, 32, 54); c.fill(); },
  bow(c) { c.lineWidth = 5; c.beginPath(); c.arc(14, 32, 30, -1.1, 1.1); c.stroke(); c.lineWidth = 2; c.beginPath(); c.moveTo(27.6, 5.3); c.lineTo(27.6, 58.7); c.stroke(); c.lineWidth = 3; c.beginPath(); c.moveTo(10, 32); c.lineTo(56, 32); c.stroke(); poly(c, [60, 32, 50, 26, 50, 38]); c.fill(); },
  axe(c) { c.fillRect(29, 8, 6, 50); c.beginPath(); c.moveTo(35, 12); c.quadraticCurveTo(60, 14, 54, 38); c.quadraticCurveTo(46, 28, 35, 30); c.closePath(); c.fill(); c.beginPath(); c.moveTo(29, 12); c.quadraticCurveTo(4, 14, 10, 38); c.quadraticCurveTo(18, 28, 29, 30); c.closePath(); c.fill(); },
  fang(c) { c.beginPath(); c.moveTo(14, 8); c.quadraticCurveTo(40, 14, 30, 58); c.quadraticCurveTo(26, 30, 14, 8); c.fill(); c.beginPath(); c.moveTo(34, 8); c.quadraticCurveTo(58, 14, 48, 50); c.quadraticCurveTo(46, 26, 34, 8); c.fill(); },
  staff(c) { c.lineWidth = 5; c.beginPath(); c.moveTo(14, 58); c.lineTo(40, 22); c.stroke(); star(c, 44, 16, 14, 6, 4); c.fill(); },
  crystal(c) { poly(c, [32, 4, 48, 24, 32, 60, 16, 24]); c.fill(); c.globalAlpha *= 0.5; poly(c, [8, 30, 18, 20, 20, 44]); c.fill(); poly(c, [56, 30, 46, 20, 44, 44]); c.fill(); },
  orb(c) { c.beginPath(); c.arc(32, 30, 20, 0, 7); c.fill(); c.fillRect(20, 52, 24, 6); c.globalCompositeOperation = 'destination-out'; c.beginPath(); c.arc(25, 23, 6, 0, 7); c.fill(); c.globalCompositeOperation = 'source-over'; },
  prism(c) { poly(c, [32, 6, 58, 52, 6, 52]); c.fill(); c.globalCompositeOperation = 'destination-out'; poly(c, [32, 24, 44, 46, 20, 46]); c.fill(); c.globalCompositeOperation = 'source-over'; },
  cloak(c) { c.beginPath(); c.moveTo(22, 8); c.lineTo(42, 8); c.quadraticCurveTo(58, 34, 54, 58); c.lineTo(10, 58); c.quadraticCurveTo(6, 34, 22, 8); c.fill(); },
  glove(c) { c.fillRect(18, 40, 26, 16); c.beginPath(); c.roundRect(16, 18, 30, 26, 6); c.fill(); for (const x of [18, 25, 32, 39]) { c.beginPath(); c.roundRect(x, 6, 6, 18, 3); c.fill(); } c.beginPath(); c.roundRect(44, 22, 12, 7, 3); c.fill(); },
  wand(c) { c.lineWidth = 6; c.lineCap = 'round'; c.beginPath(); c.moveTo(12, 54); c.lineTo(40, 26); c.stroke(); star(c, 46, 18, 13, 5, 5); c.fill(); },
  lens(c) { c.lineWidth = 6; c.beginPath(); c.arc(28, 28, 17, 0, 7); c.stroke(); c.lineCap = 'round'; c.lineWidth = 8; c.beginPath(); c.moveTo(41, 41); c.lineTo(54, 54); c.stroke(); c.globalAlpha *= 0.45; c.beginPath(); c.arc(28, 28, 12, 0, 7); c.fill(); },
  gem(c) { poly(c, [18, 12, 46, 12, 56, 26, 32, 56, 8, 26]); c.fill(); c.globalAlpha *= 0.5; poly(c, [8, 26, 56, 26, 32, 56]); c.fill(); },
  idol(c) { c.beginPath(); c.arc(32, 18, 11, 0, 7); c.fill(); poly(c, [20, 30, 44, 30, 48, 54, 16, 54]); c.fill(); c.fillRect(12, 52, 40, 6); c.globalCompositeOperation = 'destination-out'; c.beginPath(); c.arc(32, 18, 4, 0, 7); c.fill(); c.globalCompositeOperation = 'source-over'; },
  belt(c) { c.fillRect(6, 24, 52, 16); c.globalCompositeOperation = 'destination-out'; c.fillRect(26, 27, 12, 10); c.globalCompositeOperation = 'source-over'; c.lineWidth = 4; c.strokeRect(22, 20, 20, 24); },
  bead(c) { c.beginPath(); c.arc(32, 34, 18, 0, 7); c.fill(); c.lineWidth = 4; c.beginPath(); c.moveTo(32, 16); c.quadraticCurveTo(32, 6, 42, 6); c.stroke(); c.globalCompositeOperation = 'destination-out'; c.beginPath(); c.arc(25, 28, 5, 0, 7); c.fill(); c.globalCompositeOperation = 'source-over'; },
  vest(c) { poly(c, [20, 6, 28, 6, 32, 16, 36, 6, 44, 6, 56, 16, 50, 26, 48, 58, 16, 58, 14, 26, 8, 16]); c.fill(); c.globalCompositeOperation = 'destination-out'; c.fillRect(30, 20, 4, 36); c.globalCompositeOperation = 'source-over'; },
  cup(c) { c.beginPath(); c.moveTo(12, 8); c.lineTo(52, 8); c.quadraticCurveTo(50, 36, 35, 38); c.lineTo(35, 50); c.lineTo(46, 56); c.lineTo(18, 56); c.lineTo(29, 50); c.lineTo(29, 38); c.quadraticCurveTo(14, 36, 12, 8); c.fill(); },
};

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${f(n >> 16)},${f((n >> 8) & 255)},${f(n & 255)})`;
}

const cache = new Map<string, string>();

/** Returns a data URL for a square icon with the given glyph on a tinted background. */
export function iconUrl(glyph: string, color: string, size = 96): string {
  const key = `${glyph}|${color}|${size}`;
  let url = cache.get(key);
  if (url) return url;
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const c = cv.getContext('2d')!;
  const bg = c.createRadialGradient(size * 0.35, size * 0.3, size * 0.05, size / 2, size / 2, size * 0.75);
  bg.addColorStop(0, shade(color, 0.75));
  bg.addColorStop(1, shade(color, 0.18));
  c.fillStyle = bg;
  c.fillRect(0, 0, size, size);
  // Glyph on its own layer so cut-outs (destination-out) do not punch through the background.
  const layer = document.createElement('canvas');
  layer.width = layer.height = size;
  const l = layer.getContext('2d')!;
  l.scale((size / 64) * 0.78, (size / 64) * 0.78);
  l.translate(9, 9);
  l.fillStyle = l.strokeStyle = '#fff';
  (GLYPHS[glyph] ?? GLYPHS.sun)(l);
  l.setTransform(1, 0, 0, 1, 0, 0);
  l.globalCompositeOperation = 'source-in';
  const fg = l.createLinearGradient(0, 0, 0, size);
  fg.addColorStop(0, '#ffffff');
  fg.addColorStop(1, shade(color, 1.25));
  l.fillStyle = fg;
  l.fillRect(0, 0, size, size);
  c.shadowColor = 'rgba(0,0,0,0.7)';
  c.shadowBlur = size * 0.06;
  c.shadowOffsetY = size * 0.03;
  c.drawImage(layer, 0, 0);
  url = cv.toDataURL();
  cache.set(key, url);
  return url;
}

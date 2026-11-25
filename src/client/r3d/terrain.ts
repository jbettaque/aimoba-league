/**
 * Terrain: a height field derived from the walkability carves (playable ground is flat at
 * y = 0, forest is raised into plateaus with a clear bank, the river bed is sunk), a painted
 * albedo texture, and an animated stylised water surface.
 */
import * as THREE from 'three';
import { CAMPS, CARVES, CELL, CORE_POS, FOUNTAIN_POS, FOUNTAIN_RADIUS, GRID_N, LANES, MAP_SIZE, blocked } from '../../sim/map.ts';
import type { Fow } from './fow.ts';
import { LOOK, WORLD } from './look.ts';

const T = LOOK.terrain;
/** Walkable cells poke at most ~57 units past their carve; start the bank a little outside it. */
const BANK_OFFSET = 36;
const EDGE = 160;

function hash2(x: number, y: number): number {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x: number, y: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  let fx = x - xi, fy = y - yi;
  fx = fx * fx * (3 - 2 * fx);
  fy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
const fbm = (x: number, y: number) => vnoise(x, y) * 0.55 + vnoise(x * 2.1 + 7.3, y * 2.1 - 3.1) * 0.3 + vnoise(x * 4.3 - 1.7, y * 4.3 + 9.2) * 0.15;
const sstep = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
function segDist(x: number, y: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(ax + dx * t - x, ay + dy * t - y);
}

/** Signed distance (sim units) from the walkable region: <= 0 inside, > 0 inside the forest. */
export function walkSdf(x: number, y: number): number {
  let best = Infinity;
  for (const c of CARVES) {
    const d = segDist(x, y, c[0], c[1], c[2], c[3]) - c[4];
    if (d < best) best = d;
  }
  return Math.max(best, EDGE - Math.min(x, y, MAP_SIZE - x, MAP_SIZE - y));
}

const RIVER = [1150, 1150, 6850, 6850];
const LANE_LINES: number[][][] = [LANES[0].slice(0, -1), [[1000, 7000], [7000, 1000]], LANES[2].slice(0, -1)];
function laneDist(x: number, y: number): number {
  let best = Infinity;
  for (const l of LANE_LINES) for (let i = 0; i < l.length - 1; i++) best = Math.min(best, segDist(x, y, l[i][0], l[i][1], l[i + 1][0], l[i + 1][1]));
  return best;
}

/** Analytic terrain height (world units) at a sim position. */
function heightFn(x: number, y: number): number {
  const d = walkSdf(x, y) - BANK_OFFSET;
  if (d > 0) {
    return T.bankHeight * sstep(10, 95, d) + T.plateauRise * sstep(70, 420, d) + T.noiseAmp * sstep(40, 230, d) * (fbm(x / 520, y / 520) - 0.32) * 1.6;
  }
  let depth = 0;
  const rd = segDist(x, y, RIVER[0], RIVER[1], RIVER[2], RIVER[3]);
  if (rd < 250) {
    const ford = sstep(340, 170, laneDist(x, y));
    depth = T.riverDepth * sstep(245, 110, rd) * (1 - 0.82 * ford);
  }
  for (const c of CAMPS) {
    if (c.side !== 2) continue;
    const pd = Math.hypot(x - c.x, y - c.y);
    if (pd < 470) depth = Math.max(depth, T.poolDepth * sstep(465, 330, pd));
  }
  // Never dig under the banks.
  return -depth * sstep(0, -60, d);
}

export class Terrain {
  mesh: THREE.Mesh;
  water: THREE.Mesh;
  /** Painted albedo. */
  albedo: HTMLCanvasElement;
  /** Small top-down picture for the minimap. */
  minimap: HTMLCanvasElement;
  /** Heights as a texture (R = world height), for draping decals over the ground. */
  heightTex: THREE.DataTexture;
  /** xy = sim origin of the height grid, z = 1 / sim extent. */
  heightRect: THREE.Vector3;
  private n: number;
  private step: number;
  private origin: number;
  private heights: Float32Array;
  private waterMat: THREE.ShaderMaterial;

  constructor(fow: Fow) {
    const step = CELL / T.res;
    const origin = -T.margin;
    const n = Math.round((MAP_SIZE + 2 * T.margin) / step) + 1;
    this.n = n;
    this.step = step;
    this.origin = origin;
    const heights = new Float32Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) heights[j * n + i] = heightFn(origin + i * step, origin + j * step);
    this.heights = heights;

    // --- Ground mesh -----------------------------------------------------------
    const pos = new Float32Array(n * n * 3), uv = new Float32Array(n * n * 2);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const k = j * n + i, x = origin + i * step, y = origin + j * step;
        pos[k * 3] = x * WORLD;
        pos[k * 3 + 1] = heights[k];
        pos[k * 3 + 2] = y * WORLD;
        uv[k * 2] = x / MAP_SIZE;
        uv[k * 2 + 1] = y / MAP_SIZE;
      }
    }
    const idx = new Uint32Array((n - 1) * (n - 1) * 6);
    let q = 0;
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
        // Alternate the diagonal so banks do not show a directional grain.
        if ((i + j) & 1) { idx[q++] = a; idx[q++] = c; idx[q++] = b; idx[q++] = b; idx[q++] = c; idx[q++] = d; }
        else { idx[q++] = a; idx[q++] = c; idx[q++] = d; idx[q++] = a; idx[q++] = d; idx[q++] = b; }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    // Vertex tint: rocky on steep banks, a little variation on the plateaus.
    const nrm = geo.getAttribute('normal') as THREE.BufferAttribute;
    const col = new Float32Array(n * n * 3);
    const cliff = new THREE.Color(T.cliffTint), forest = new THREE.Color(T.forestTint), tmp = new THREE.Color();
    for (let k = 0; k < n * n; k++) {
      const ny = nrm.getY(k), h = heights[k];
      const steep = sstep(0.93, 0.6, ny);
      const top = sstep(0.5, 0.9, h) * (1 - steep);
      const v = 0.9 + 0.2 * hash2(k % n, (k / n) | 0);
      tmp.setRGB(1, 1, 1).lerp(forest, top * 0.6).lerp(cliff, steep * 0.75).multiplyScalar(top > 0 ? v : 1);
      col[k * 3] = tmp.r; col[k * 3 + 1] = tmp.g; col[k * 3 + 2] = tmp.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

    const painted = paintGround(T.textureSize);
    this.albedo = painted;
    this.minimap = paintMinimap(painted);
    const map = new THREE.CanvasTexture(painted);
    map.colorSpace = THREE.SRGBColorSpace;
    map.flipY = false;
    map.anisotropy = 8;
    map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping;
    const mat = fow.patch(new THREE.MeshStandardMaterial({ map, vertexColors: true, roughness: T.roughness, metalness: 0 }));
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = true;
    this.mesh.name = 'terrain';

    // --- Height texture ---------------------------------------------------------
    const half = new Uint16Array(n * n);
    for (let k = 0; k < n * n; k++) half[k] = THREE.DataUtils.toHalfFloat(heights[k]);
    this.heightTex = new THREE.DataTexture(half, n, n, THREE.RedFormat, THREE.HalfFloatType);
    this.heightTex.minFilter = this.heightTex.magFilter = THREE.LinearFilter;
    this.heightTex.needsUpdate = true;
    this.heightRect = new THREE.Vector3(origin - step / 2, origin - step / 2, 1 / (n * step));

    // --- Water ---------------------------------------------------------------------
    const level = T.waterLevel;
    const remap = new Int32Array(n * n).fill(-1);
    const wpos: number[] = [], wdepth: number[] = [], widx: number[] = [];
    const vert = (k: number) => {
      if (remap[k] < 0) {
        remap[k] = wpos.length / 3;
        wpos.push(pos[k * 3], level, pos[k * 3 + 2]);
        wdepth.push(-heights[k]);
      }
      return remap[k];
    };
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
        if (Math.min(heights[a], heights[b], heights[c], heights[d]) > level) continue;
        widx.push(vert(a), vert(c), vert(d), vert(a), vert(d), vert(b));
      }
    }
    const wgeo = new THREE.BufferGeometry();
    wgeo.setAttribute('position', new THREE.Float32BufferAttribute(wpos, 3));
    wgeo.setAttribute('aDepth', new THREE.Float32BufferAttribute(wdepth, 1));
    wgeo.setIndex(widx);
    const W = LOOK.water;
    this.waterMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      fog: true,
      uniforms: {
        ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
        uTime: { value: 0 },
        uLevel: { value: level },
        uShallow: { value: new THREE.Color(W.shallow) },
        uDeep: { value: new THREE.Color(W.deep) },
        uFoam: { value: new THREE.Color(W.foam) },
        uOpacity: { value: W.opacity },
        uGlow: { value: W.glow },
        uFowMap: fow.uniforms.uFowMap,
        uFowParams: fow.uniforms.uFowParams,
        uFowTint: fow.uniforms.uFowTint,
      },
      vertexShader: /* glsl */ `
        attribute float aDepth;
        varying float vDepth;
        varying vec2 vXZ;
        #include <fog_pars_vertex>
        void main() {
          vDepth = aDepth;
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vXZ = wp.xz;
          vec4 mvPosition = viewMatrix * wp;
          gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: /* glsl */ `
        uniform float uTime, uLevel, uOpacity, uGlow;
        uniform vec3 uShallow, uDeep, uFoam;
        uniform sampler2D uFowMap;
        uniform vec4 uFowParams;
        uniform vec3 uFowTint;
        varying float vDepth;
        varying vec2 vXZ;
        #include <common>
        #include <fog_pars_fragment>
        float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
        float noise(vec2 p) {
          vec2 i = floor(p), f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
        }
        void main() {
          float s = vDepth + uLevel;
          if (s <= 0.0) discard;
          vec2 flow = vec2(0.7071, 0.7071);
          float along = dot(vXZ, flow), across = dot(vXZ, vec2(-flow.y, flow.x));
          vec2 q = vec2(along * 0.9 - uTime * 0.35, across * 2.2);
          float n1 = noise(q * 1.3), n2 = noise(q * 2.9 + vec2(uTime * 0.2, 3.7));
          float ripple = smoothstep(0.6, 0.7, n1 * 0.6 + n2 * 0.4);
          float streak = smoothstep(0.8, 0.87, noise(vec2(along * 0.55 - uTime * 0.6, across * 5.0)));
          float deep = smoothstep(0.0, 0.15, s);
          vec3 col = mix(uShallow, uDeep, deep);
          col += (ripple * 0.09 + streak * 0.2) * uFoam;
          float shore = 1.0 - smoothstep(0.0, 0.04, s);
          float foamN = noise(vXZ * 6.0 + vec2(uTime * 0.3, -uTime * 0.22));
          float foam = shore * smoothstep(0.42, 0.72, foamN + shore * 0.3);
          col = mix(col, uFoam, foam * 0.6);
          col *= 1.0 + uGlow;
          float alpha = smoothstep(0.0, 0.02, s) * mix(0.5, uOpacity, deep);
          alpha = max(alpha, foam * 0.6);
          float vis = smoothstep(0.05, 0.85, texture2D(uFowMap, clamp(vXZ * uFowParams.z, 0.0, 1.0)).r);
          float l = dot(col, vec3(0.299, 0.587, 0.114));
          col = mix(mix(col, vec3(l), uFowParams.y) * uFowParams.x * uFowTint, col, vis);
          gl_FragColor = vec4(col, alpha);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
          #include <fog_fragment>
        }`,
    });
    this.water = new THREE.Mesh(wgeo, this.waterMat);
    this.water.name = 'water';
    this.water.renderOrder = 1;
  }

  update(time: number) {
    this.waterMat.uniforms.uTime.value = time;
  }

  /** Raw terrain height (world units) at a sim position, bilinear. */
  heightAt(x: number, y: number): number {
    const n = this.n;
    let fx = (x - this.origin) / this.step, fy = (y - this.origin) / this.step;
    fx = fx < 0 ? 0 : fx > n - 1.001 ? n - 1.001 : fx;
    fy = fy < 0 ? 0 : fy > n - 1.001 ? n - 1.001 : fy;
    const i = fx | 0, j = fy | 0, tx = fx - i, ty = fy - j, h = this.heights, k = j * n + i;
    return (h[k] * (1 - tx) + h[k + 1] * tx) * (1 - ty) + (h[k + n] * (1 - tx) + h[k + n + 1] * tx) * ty;
  }

  /** Height a unit stands at: the ground, wading down into the river bed. */
  unitY(x: number, y: number): number {
    return Math.min(0, this.heightAt(x, y));
  }
}

// --- Painting ------------------------------------------------------------------------

/** Paint the top-down ground albedo (forest floor, banks, grass, lanes, river bed, bases, camps). */
function paintGround(S: number): HTMLCanvasElement {
  const P = T.palette;
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const mk = () => {
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const g = cv.getContext('2d')!;
    g.scale(S / MAP_SIZE, S / MAP_SIZE);
    g.lineCap = 'round';
    g.lineJoin = 'round';
    return [cv, g] as const;
  };
  // Carves batched by radius so each band is a handful of (blurred) strokes.
  const byRadius = new Map<number, Path2D>();
  for (const c of CARVES) {
    let p = byRadius.get(c[4]);
    if (!p) byRadius.set(c[4], (p = new Path2D()));
    p.moveTo(c[0], c[1]);
    p.lineTo(c[2] + 0.1, c[3] + 0.1);
  }
  const strokeCarves = (g: CanvasRenderingContext2D, color: string, grow: number) => {
    g.strokeStyle = color;
    for (const [r, p] of byRadius) {
      g.lineWidth = Math.max(10, (r + grow) * 2);
      g.stroke(p);
    }
  };
  const clipEdge = (g: CanvasRenderingContext2D, inset: number) => {
    g.beginPath();
    g.rect(inset, inset, MAP_SIZE - inset * 2, MAP_SIZE - inset * 2);
    g.clip();
  };

  // --- Base canvas: forest floor and the bank strata ------------------------------
  const [cv, g] = mk();
  g.fillStyle = P.forest;
  g.fillRect(0, 0, MAP_SIZE, MAP_SIZE);
  for (let i = 0; i < 5200; i++) {
    const x = rnd() * MAP_SIZE, y = rnd() * MAP_SIZE, r = 40 + rnd() * 150;
    g.fillStyle = rnd() < 0.5 ? 'rgba(96,128,62,0.07)' : 'rgba(14,26,16,0.09)';
    g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
  }
  for (let i = 0; i < 16000; i++) { // leaf litter
    const x = rnd() * MAP_SIZE, y = rnd() * MAP_SIZE;
    g.fillStyle = rnd() < 0.4 ? 'rgba(150,120,60,0.20)' : rnd() < 0.5 ? 'rgba(120,160,70,0.16)' : 'rgba(10,20,10,0.2)';
    g.beginPath(); g.ellipse(x, y, 5 + rnd() * 9, 3 + rnd() * 5, rnd() * 3, 0, 7); g.fill();
  }
  g.save();
  clipEdge(g, EDGE - BANK_OFFSET - 140);
  g.filter = `blur(${(S / 4096) * 2.5}px)`;
  const bands: [string, number][] = [['#4f6a35', 158], ['#5d4d3a', 134], ['#7c684f', 112], ['#96805f', 86], ['#7a6a4c', 64], ['#4c5230', 52]];
  for (const [colr, grow] of bands) strokeCarves(g, colr, BANK_OFFSET + grow - 36);
  g.restore();

  // --- Walkable layer: everything painted "source-atop" the grass is clipped to it ---
  const [lv, l] = mk();
  l.save();
  clipEdge(l, EDGE - BANK_OFFSET - 8);
  l.filter = `blur(${(S / 4096) * 3}px)`;
  strokeCarves(l, P.grassDark, BANK_OFFSET + 10);
  strokeCarves(l, P.grass, BANK_OFFSET - 22);
  l.restore();
  l.filter = 'none';
  l.globalCompositeOperation = 'source-atop';
  for (let i = 0; i < 4200; i++) { // grass mottling
    const x = rnd() * MAP_SIZE, y = rnd() * MAP_SIZE, r = 50 + rnd() * 170;
    l.fillStyle = rnd() < 0.5 ? 'rgba(150,190,80,0.07)' : 'rgba(30,60,30,0.08)';
    l.beginPath(); l.arc(x, y, r, 0, 7); l.fill();
  }
  l.lineWidth = 4;
  for (let i = 0; i < 26000; i++) { // grass tufts
    const x = rnd() * MAP_SIZE, y = rnd() * MAP_SIZE, a = -1.57 + (rnd() - 0.5) * 1.2, len = 12 + rnd() * 22;
    l.strokeStyle = rnd() < 0.55 ? 'rgba(170,205,95,0.22)' : 'rgba(34,66,30,0.24)';
    l.beginPath(); l.moveTo(x, y); l.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len); l.stroke();
  }
  for (let i = 0; i < 2600; i++) { // flowers
    const x = rnd() * MAP_SIZE, y = rnd() * MAP_SIZE, t = rnd();
    l.fillStyle = t < 0.4 ? 'rgba(255,244,200,0.75)' : t < 0.7 ? 'rgba(255,214,110,0.75)' : t < 0.87 ? 'rgba(255,170,190,0.7)' : 'rgba(170,200,255,0.7)';
    l.beginPath(); l.arc(x, y, 4 + rnd() * 4, 0, 7); l.fill();
  }

  // River bed and pools.
  const river = (w: number, color: string) => {
    l.strokeStyle = color;
    l.lineWidth = w;
    l.beginPath(); l.moveTo(RIVER[0], RIVER[1]); l.lineTo(RIVER[2], RIVER[3]); l.stroke();
  };
  l.filter = `blur(${(S / 4096) * 4}px)`;
  river(560, '#8c8a62');
  river(450, P.riverBed);
  river(300, P.riverBedDeep);
  for (const c of CAMPS) {
    if (c.side !== 2) continue;
    l.fillStyle = '#8c8a62'; l.beginPath(); l.arc(c.x, c.y, 500, 0, 7); l.fill();
    l.fillStyle = P.riverBed; l.beginPath(); l.arc(c.x, c.y, 450, 0, 7); l.fill();
    l.fillStyle = P.riverBedDeep; l.beginPath(); l.arc(c.x, c.y, 360, 0, 7); l.fill();
  }
  l.filter = 'none';
  for (let i = 0; i < 2600; i++) { // pebbles
    const t = rnd(), o = (rnd() - 0.5) * 520;
    const s = RIVER[0] + (RIVER[2] - RIVER[0]) * t;
    const x = s + o * 0.707, y = s - o * 0.707;
    l.fillStyle = rnd() < 0.5 ? 'rgba(220,225,215,0.22)' : 'rgba(20,40,50,0.22)';
    l.beginPath(); l.ellipse(x, y, 8 + rnd() * 14, 6 + rnd() * 9, rnd() * 3, 0, 7); l.fill();
  }

  // Lanes (drawn over the river as fords).
  const lanePath = (pts: number[][]) => {
    l.beginPath();
    pts.forEach((p, i) => (i ? l.lineTo(p[0], p[1]) : l.moveTo(p[0], p[1])));
    l.stroke();
  };
  l.filter = `blur(${(S / 4096) * 4}px)`;
  for (const [w, colr] of [[540, 'rgba(120,104,66,0.55)'], [410, P.pathDark], [330, P.path], [150, 'rgba(214,191,142,0.45)']] as [number, string][]) {
    l.lineWidth = w;
    l.strokeStyle = colr;
    for (const ln of LANE_LINES) lanePath(ln);
  }
  l.filter = 'none';
  for (const ln of LANE_LINES) { // cobbles and wheel ruts
    for (let i = 0; i < ln.length - 1; i++) {
      const [ax, ay] = ln[i], [bx, by] = ln[i + 1];
      const len = Math.hypot(bx - ax, by - ay);
      for (let k = 0; k < len / 5; k++) {
        const t = rnd(), o = (rnd() - 0.5) * 400;
        const x = ax + (bx - ax) * t + ((by - ay) / len) * o, y = ay + (by - ay) * t - ((bx - ax) / len) * o;
        l.fillStyle = rnd() < 0.5 ? 'rgba(236,220,180,0.20)' : 'rgba(60,46,28,0.20)';
        l.beginPath(); l.ellipse(x, y, 9 + rnd() * 15, 6 + rnd() * 9, rnd() * 3, 0, 7); l.fill();
      }
    }
  }

  // Bases: tinted plazas, tiled stone around the cores, glowing fountains.
  for (let team = 0; team < 2; team++) {
    const [bx, by] = team === 0 ? [1000, 7000] : [7000, 1000];
    const grad = l.createRadialGradient(bx, by, 100, bx, by, 1450);
    grad.addColorStop(0, team === 0 ? 'rgba(128,150,186,0.85)' : 'rgba(190,134,124,0.85)');
    grad.addColorStop(0.7, team === 0 ? 'rgba(112,130,156,0.6)' : 'rgba(160,118,108,0.6)');
    grad.addColorStop(1, 'rgba(120,120,120,0)');
    l.fillStyle = grad;
    l.beginPath(); l.arc(bx, by, 1450, 0, 7); l.fill();
    const [cx, cy] = CORE_POS[team];
    l.fillStyle = team === 0 ? 'rgba(150,168,196,0.75)' : 'rgba(200,152,140,0.75)';
    l.beginPath(); l.arc(cx, cy, 620, 0, 7); l.fill();
    l.strokeStyle = 'rgba(30,34,44,0.42)';
    l.lineWidth = 9;
    for (const r of [150, 260, 420, 620]) { l.beginPath(); l.arc(cx, cy, r, 0, 7); l.stroke(); }
    for (const [r0, r1, nseg, off] of [[150, 260, 8, 0], [260, 420, 12, 0.26], [420, 620, 18, 0.1]]) {
      for (let a = 0; a < nseg; a++) {
        const ang = (a / nseg) * 6.2832 + off;
        l.beginPath();
        l.moveTo(cx + Math.cos(ang) * r0, cy + Math.sin(ang) * r0);
        l.lineTo(cx + Math.cos(ang) * r1, cy + Math.sin(ang) * r1);
        l.stroke();
      }
    }
    l.strokeStyle = team === 0 ? 'rgba(90,170,255,0.55)' : 'rgba(255,110,96,0.55)';
    l.lineWidth = 14;
    l.beginPath(); l.arc(cx, cy, 640, 0, 7); l.stroke();
    const [fx, fy] = FOUNTAIN_POS[team];
    const fg = l.createRadialGradient(fx, fy, 50, fx, fy, FOUNTAIN_RADIUS);
    fg.addColorStop(0, team === 0 ? 'rgba(150,215,255,0.8)' : 'rgba(255,170,150,0.8)');
    fg.addColorStop(0.6, team === 0 ? 'rgba(120,180,235,0.35)' : 'rgba(235,140,125,0.35)');
    fg.addColorStop(1, 'rgba(0,0,0,0)');
    l.fillStyle = fg;
    l.beginPath(); l.arc(fx, fy, FOUNTAIN_RADIUS, 0, 7); l.fill();
    l.strokeStyle = 'rgba(30,34,44,0.35)';
    l.lineWidth = 8;
    for (const r of [120, 250, 400]) { l.beginPath(); l.arc(fx, fy, r, 0, 7); l.stroke(); }
  }

  // Camp clearings.
  for (const c of CAMPS) {
    if (c.side === 2) continue;
    const cg = l.createRadialGradient(c.x, c.y, 60, c.x, c.y, 300);
    cg.addColorStop(0, 'rgba(107,90,64,0.9)');
    cg.addColorStop(0.75, 'rgba(96,84,58,0.6)');
    cg.addColorStop(1, 'rgba(96,84,58,0)');
    l.fillStyle = cg;
    l.beginPath(); l.arc(c.x, c.y, 300, 0, 7); l.fill();
    l.strokeStyle = c.type === 'might' ? 'rgba(255,120,96,0.55)' : c.type === 'insight' ? 'rgba(120,180,255,0.55)' : 'rgba(232,220,170,0.35)';
    l.lineWidth = 12;
    l.setLineDash([60, 40]);
    l.beginPath(); l.arc(c.x, c.y, 215, 0, 7); l.stroke();
    l.setLineDash([]);
  }

  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(lv, 0, 0);
  return cv;
}

function paintMinimap(albedo: HTMLCanvasElement): HTMLCanvasElement {
  const S = 520;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d')!;
  g.imageSmoothingQuality = 'high';
  g.drawImage(albedo, 0, 0, S, S);
  g.fillStyle = 'rgba(4,14,12,0.6)';
  const k = S / GRID_N;
  for (let cy = 0; cy < GRID_N; cy++) for (let cx = 0; cx < GRID_N; cx++) if (blocked[cy * GRID_N + cx]) g.fillRect(cx * k, cy * k, k + 0.5, k + 0.5);
  // Water tint so the river reads on the map.
  g.strokeStyle = 'rgba(70,170,200,0.55)';
  g.lineWidth = (300 / MAP_SIZE) * S;
  g.lineCap = 'round';
  g.beginPath(); g.moveTo((RIVER[0] / MAP_SIZE) * S, (RIVER[1] / MAP_SIZE) * S); g.lineTo((RIVER[2] / MAP_SIZE) * S, (RIVER[3] / MAP_SIZE) * S); g.stroke();
  return cv;
}

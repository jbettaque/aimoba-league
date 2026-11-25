/**
 * Environment scatter: places the props of env.glb (trees, rocks, grass, ...) over the map
 * with deterministic rules, and renders them as InstancedMeshes grouped in spatial chunks so
 * frustum culling works.
 *
 * To art-direct the forest, edit SCATTER below: each rule names a prop prefix, a density
 * (expected instances per 80x80 cell, as a function of where the cell is) and a few
 * placement options. Rules whose prefix matches no prop are skipped.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { CAMPS, CELL, GRID_N, MAP_SIZE } from '../../sim/map.ts';
import type { Fow } from './fow.ts';
import { LOOK, WORLD } from './look.ts';
import { type Terrain, walkSdf } from './terrain.ts';

export interface ScatterCtx {
  /** Cell centre, sim units. */
  x: number;
  y: number;
  /** Signed distance from the walkable region: > 0 inside the forest, < 0 on playable ground. */
  sdf: number;
  /** True when the cell is outside the map square (background forest). */
  outside: boolean;
  /** Playable ground lies just north of this cell, so tall props here would hide units. */
  occludes: boolean;
  /** Terrain height (world units); negative in the river. */
  height: number;
  /** Distance to the nearest lane centre line / jungle camp / base centre (sim units). */
  lane: number;
  camp: number;
  base: number;
  /** Deterministic random in [0,1) for this cell; `i` selects an independent stream. */
  rnd: (i: number) => number;
}

export interface ScatterRule {
  /** Props whose name starts with this are picked at random. */
  prefix: string;
  /** Expected number of instances in the cell (may exceed 1). */
  density: (c: ScatterCtx) => number;
  /** Uniform scale range. */
  scale?: [number, number] | ((c: ScatterCtx) => [number, number]);
  /** Random lean, radians. */
  tilt?: number;
  /** Cast shadows (default true). */
  shadow?: boolean;
  /** Sit on the water surface instead of the ground. */
  onWater?: boolean;
  /** Sink into the ground by this many world units (hides flat bottoms on slopes). */
  sink?: number;
  /**
   * Shrink instances (using the prop's real bounding box) so that they neither overhang the
   * playable ground by more than `overhang` world units nor hide more than `hide` world units
   * of the ground north of them from the tilted camera. Keeps lanes readable next to tall trees.
   */
  fit?: { overhang: number; hide: number; min: number };
}

const forest = (c: ScatterCtx) => c.sdf > 46;
const meadow = (c: ScatterCtx) => c.sdf < -70 && c.height > -0.01 && c.lane > 330 && c.base > 1500 && c.camp > 300;

export const SCATTER: ScatterRule[] = [
  {
    prefix: 'tree_',
    density: (c) => (!forest(c) ? 0 : c.outside ? 0.42 : c.occludes ? 0.3 : c.sdf > 230 ? 0.5 : 1),
    scale: (c) => (c.sdf < 130 ? [0.8, 1.05] : [0.95, 1.3]),
    tilt: 0.06,
    fit: { overhang: 0.25, hide: 0.1, min: 0.3 },
  },
  { prefix: 'rock_', density: (c) => (c.sdf > 20 && c.sdf < 150 && !c.outside ? 0.16 : c.sdf > 150 && !c.outside ? 0.03 : 0), scale: [0.7, 1.5], sink: 0.06 },
  { prefix: 'rock_', density: (c) => (c.height < -0.1 ? 0.05 : 0), scale: [0.4, 0.7], sink: 0.05 },
  { prefix: 'stump', density: (c) => (forest(c) && c.sdf < 200 && !c.outside ? 0.035 : 0), scale: [0.8, 1.2] },
  { prefix: 'log', density: (c) => (forest(c) && c.sdf < 220 && !c.outside ? 0.025 : 0), scale: [0.8, 1.2] },
  { prefix: 'fern', density: (c) => (c.sdf > 30 && c.sdf < 260 && !c.outside ? 0.55 : 0), scale: [0.7, 1.3], shadow: false },
  { prefix: 'mushroom', density: (c) => (c.sdf > 40 && c.sdf < 220 && !c.outside ? 0.1 : 0), scale: [0.7, 1.4], shadow: false },
  { prefix: 'grass', density: (c) => (meadow(c) ? 0.55 : c.sdf > -70 && c.sdf < 40 && c.height > -0.01 ? 0.9 : 0), scale: [0.7, 1.4], shadow: false },
  { prefix: 'flower_', density: (c) => (meadow(c) ? 0.12 : 0), scale: [0.8, 1.3], shadow: false },
  { prefix: 'reed', density: (c) => (c.height < -0.02 && c.height > -0.12 && c.lane > 380 ? 0.5 : 0), scale: [0.8, 1.3], shadow: false },
  { prefix: 'lilypad', density: (c) => (c.height < -0.13 && c.lane > 420 ? 0.12 : 0), scale: [0.7, 1.3], shadow: false, onWater: true },
  { prefix: 'crystal', density: (c) => (c.camp > 260 && c.camp < 420 && c.sdf > 10 && c.sdf < 120 ? 0.3 : 0), scale: [0.7, 1.3], sink: 0.04 },
  { prefix: 'lantern', density: (c) => (c.sdf > 46 && c.sdf < 110 && !c.occludes && c.lane < 520 && c.rnd(40) < 0.5 ? 0.12 : 0), scale: [1, 1] },
  { prefix: 'brazier', density: (c) => (c.sdf > 46 && c.sdf < 100 && !c.occludes && c.base < 1900 && c.rnd(41) < 0.5 ? 0.2 : 0), scale: [1, 1] },
  { prefix: 'totem', density: (c) => (c.camp > 280 && c.camp < 430 && c.sdf > 46 && c.sdf < 130 && !c.occludes ? 0.12 : 0), scale: [0.9, 1.15] },
  { prefix: 'cobble', density: (c) => (c.sdf < -60 && c.lane < 300 && c.height > -0.01 && c.base > 1500 ? 0.07 : 0), scale: [0.8, 1.3], shadow: false },
  { prefix: 'ruin_', density: (c) => (c.sdf > 90 && c.sdf < 240 && !c.outside && !c.occludes ? 0.012 : 0), scale: [0.9, 1.2], sink: 0.05 },
];

function hash3(x: number, y: number, i: number): number {
  let h = (x * 374761393 + y * 668265263 + i * 1442695041) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function segDist(x: number, y: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(ax + dx * t - x, ay + dy * t - y);
}
const LANE_LINES: number[][] = [
  [1000, 6300, 1000, 1700], [1000, 1700, 1250, 1250], [1250, 1250, 1700, 1000], [1700, 1000, 6300, 1000],
  [1000, 7000, 7000, 1000],
  [7000, 1700, 7000, 6300], [7000, 6300, 6750, 6750], [6750, 6750, 6300, 7000], [6300, 7000, 1700, 7000],
];

/** Tangent of the camera pitch: how far a prop of height h hides the ground behind it (h / tan). */
const HIDE = 1 / Math.tan((LOOK.camera.pitch * Math.PI) / 180);

interface Part { geo: THREE.BufferGeometry; mat: THREE.Material | THREE.Material[]; local: THREE.Matrix4 }

export class Props {
  group = new THREE.Group();
  private fow: Fow;
  private terrain: Terrain;
  private matCache = new Map<string, THREE.Material>();

  constructor(fow: Fow, terrain: Terrain) {
    this.fow = fow;
    this.terrain = terrain;
    this.group.name = 'props';
  }

  /** (Re)build the whole scatter from a list of prop templates. */
  build(env: THREE.Object3D[]) {
    for (const c of this.group.children.slice()) { this.group.remove(c); (c as THREE.InstancedMesh).dispose?.(); }
    const t0 = performance.now();
    // Split every prop into instancable parts.
    const parts = new Map<THREE.Object3D, Part[]>();
    const dims = new Map<THREE.Object3D, { r: number; h: number }>();
    const inv = new THREE.Matrix4(), rootRS = new THREE.Matrix4(), zero = new THREE.Vector3();
    for (const p of env) {
      p.updateWorldMatrix(true, true);
      inv.copy(p.matrixWorld).invert();
      rootRS.compose(zero, p.quaternion, p.scale);
      const list: Part[] = [];
      p.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        const sway = Number(o.userData.sway ?? p.userData.sway) || 0;
        const mat = Array.isArray(m.material) ? m.material.map((x) => this.material(x, sway)) : this.material(m.material, sway);
        list.push({ geo: m.geometry, mat, local: new THREE.Matrix4().multiplyMatrices(rootRS, new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld)) });
      });
      parts.set(p, mergeParts(list));
      // Footprint radius and height of the prop, for `fit` rules.
      const bb = new THREE.Box3(), tb = new THREE.Box3();
      for (const part of parts.get(p)!) {
        part.geo.computeBoundingBox();
        bb.union(tb.copy(part.geo.boundingBox!).applyMatrix4(part.local));
      }
      dims.set(p, { r: Math.max(-bb.min.x, bb.max.x, -bb.min.z, bb.max.z, 0.01), h: Math.max(bb.max.y, 0.01) });
    }
    const rules = SCATTER.map((r) => ({ r, props: env.filter((p) => p.name.startsWith(r.prefix)) })).filter((x) => x.props.length);

    // chunk key -> prop -> instance matrices (+ shadow flag)
    const CH = LOOK.props.chunk;
    const chunks = new Map<string, { prop: THREE.Object3D; shadow: boolean; mats: THREE.Matrix4[] }>();
    const M = Math.round(LOOK.terrain.margin / CELL) - 1;
    const q = new THREE.Quaternion(), e = new THREE.Euler(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
    let total = 0;
    for (let cy = -M; cy < GRID_N + M; cy++) {
      for (let cx = -M; cx < GRID_N + M; cx++) {
        const x = (cx + 0.5) * CELL, y = (cy + 0.5) * CELL;
        const sdf = walkSdf(x, y);
        let lane = Infinity, camp = Infinity;
        for (const l of LANE_LINES) lane = Math.min(lane, segDist(x, y, l[0], l[1], l[2], l[3]));
        for (const c of CAMPS) camp = Math.min(camp, Math.hypot(x - c.x, y - c.y));
        const ctx: ScatterCtx = {
          x, y, sdf,
          outside: x < 0 || y < 0 || x > MAP_SIZE || y > MAP_SIZE,
          occludes: sdf > 0 && (walkSdf(x, y - 170) < 20 || walkSdf(x, y - 320) < -40),
          height: this.terrain.heightAt(x, y),
          lane, camp,
          base: Math.min(Math.hypot(x - 1000, y - 7000), Math.hypot(x - 7000, y - 1000)),
          rnd: (i) => hash3(cx, cy, i),
        };
        for (let ri = 0; ri < rules.length; ri++) {
          const { r, props } = rules[ri];
          const d = r.density(ctx);
          if (d <= 0) continue;
          const count = Math.floor(d) + (hash3(cx, cy, ri * 16 + 1) < d - Math.floor(d) ? 1 : 0);
          for (let k = 0; k < count; k++) {
            const h = (i: number) => hash3(cx, cy, ri * 16 + 2 + k * 5 + i);
            const prop = props[Math.floor(h(0) * props.length) % props.length];
            const px = x + (h(1) - 0.5) * CELL * 0.9, py = y + (h(2) - 0.5) * CELL * 0.9;
            const sr = typeof r.scale === 'function' ? r.scale(ctx) : r.scale ?? [1, 1];
            let s = sr[0] + (sr[1] - sr[0]) * h(3);
            if (r.fit) {
              const dm = dims.get(prop)!;
              const here = walkSdf(px, py);
              let lim = (Math.max(0, here) * WORLD + r.fit.overhang) / dm.r;
              // Distance (sim units) to playable ground due north, where the camera looks.
              let north = Infinity;
              for (let dn = 40; dn <= 480; dn += 40) if (walkSdf(px, py - dn) < 0) { north = dn; break; }
              // The canopy's near edge sits `r` closer to that ground and hides `h / tan(pitch)` beyond it,
              // so trees step down in size towards the south edge of lanes and clearings.
              if (north < Infinity) lim = Math.min(lim, (north * WORLD + r.fit.hide) / (0.85 * dm.h * HIDE + dm.r));
              if (lim < r.fit.min) continue;
              s = Math.min(s, lim);
            }
            const gy = r.onWater ? LOOK.terrain.waterLevel + 0.005 : this.terrain.heightAt(px, py) - (r.sink ?? 0);
            const tilt = r.tilt ?? 0;
            e.set((h(4) - 0.5) * 2 * tilt, h(0) * 53.1, (h(1) * 7.7 % 1 - 0.5) * 2 * tilt, 'YXZ');
            q.setFromEuler(e);
            pos.set(px * WORLD, gy, py * WORLD);
            scl.setScalar(s);
            const key = `${Math.floor(pos.x / CH)},${Math.floor(pos.z / CH)},${env.indexOf(prop)},${r.shadow === false ? 0 : 1}`;
            let entry = chunks.get(key);
            if (!entry) chunks.set(key, (entry = { prop, shadow: r.shadow !== false, mats: [] }));
            entry.mats.push(new THREE.Matrix4().compose(pos, q, scl));
            total++;
          }
        }
      }
    }
    const tmp = new THREE.Matrix4();
    let meshes = 0;
    for (const entry of chunks.values()) {
      for (const part of parts.get(entry.prop)!) {
        const im = new THREE.InstancedMesh(part.geo, part.mat, entry.mats.length);
        for (let i = 0; i < entry.mats.length; i++) im.setMatrixAt(i, tmp.multiplyMatrices(entry.mats[i], part.local));
        im.instanceMatrix.needsUpdate = true;
        im.castShadow = entry.shadow;
        im.receiveShadow = true;
        im.computeBoundingSphere();
        im.computeBoundingBox();
        this.group.add(im);
        meshes++;
      }
    }
    console.info(`[r3d] scatter: ${total} props in ${meshes} instanced meshes, ${(performance.now() - t0).toFixed(0)} ms`);
  }

  private material(src: THREE.Material, sway: number): THREE.Material {
    const key = `${src.uuid}|${sway}`;
    let m = this.matCache.get(key);
    if (!m) {
      m = this.fow.patch(src.clone(), sway);
      m.name = src.name;
      this.matCache.set(key, m);
    }
    return m;
  }
}

/** Fold parts that share a material into one geometry, so a prop costs one draw call per material. */
function mergeParts(list: Part[]): Part[] {
  const out: Part[] = [], groups = new Map<THREE.Material, Part[]>();
  for (const p of list) {
    if (Array.isArray(p.mat)) { out.push(p); continue; }
    let g = groups.get(p.mat);
    if (!g) groups.set(p.mat, (g = []));
    g.push(p);
  }
  const identity = new THREE.Matrix4();
  for (const [mat, g] of groups) {
    if (g.length === 1) { out.push(g[0]); continue; }
    const merged = mergeGeometries(g.map((p) => p.geo.clone().applyMatrix4(p.local)), false);
    if (merged) out.push({ geo: merged, mat, local: identity });
    else out.push(...g);
  }
  return out;
}

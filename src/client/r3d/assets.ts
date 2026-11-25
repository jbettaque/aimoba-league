/**
 * Model assets: GLB loading (`./models/<name>.glb`), primitive placeholders used until a
 * GLB arrives (or if it is missing), and per-team material variants.
 *
 * Asset contract: 1 glTF unit = 100 sim units, +Y up, forward = +Z, origin on the ground.
 * Material `team` gets the team colour, `team_glow` the team emissive; `glow*` materials bloom.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Fow } from './fow.ts';
import { LOOK } from './look.ts';

export interface ModelAsset {
  name: string;
  template: THREE.Object3D;
  /** Bumped when the GLB replaces the placeholder, so live views can swap. */
  version: number;
  /** Top of the model in world units (for overhead bars). */
  height: number;
  placeholder: boolean;
}

export const HERO_IDS = ['korvath', 'sable', 'ignara', 'vesper', 'liora'];
export const MODEL_NAMES = [
  ...HERO_IDS.map((h) => 'hero_' + h),
  'minion_melee', 'minion_ranged', 'minion_siege', 'minion_super',
  'monster_stalker', 'monster_golem', 'monster_insight', 'monster_might', 'monster_wyrm', 'monster_titan',
  'tower', 'core', 'fountain', 'plinth',
];

const std = (name: string, color: number | string, opts: THREE.MeshStandardMaterialParameters = {}) => {
  const m = new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.05, ...opts });
  m.name = name;
  return m;
};

export class Assets {
  private loader = new GLTFLoader();
  private models = new Map<string, ModelAsset>();
  private matCache = new Map<string, THREE.Material>();
  /** Materials whose emissive intensity pulses (`userData.pulse` on a node). */
  private pulsing: { mat: THREE.MeshStandardMaterial; base: number; amt: number; speed: number }[] = [];
  /** Environment props (top-level children of env.glb), or placeholders. */
  env: THREE.Object3D[] = [];
  envVersion = 0;
  private fow: Fow;
  /** Placeholder materials. */
  ph = {
    team: std('team', '#ffffff'),
    glow: std('team_glow', '#111111', { emissive: '#ffffff', emissiveIntensity: 2.4 }),
    skin: std('skin', '#e8c9a8'),
    dark: std('dark', '#3a3f4a'),
    metal: std('metal', '#b9c0cc', { metalness: 0.6, roughness: 0.4 }),
    stone: std('stone', '#7d838c', { roughness: 0.95 }),
    stoneDark: std('stone_dark', '#4a4f57', { roughness: 0.95 }),
    wood: std('wood', '#6e4c2e'),
    hide: std('hide', '#7d705f'),
    moss: std('moss', '#5c8a4a'),
    leafA: std('leaf_a', '#3f7a3a', { roughness: 0.9 }),
    leafB: std('leaf_b', '#5d8f3a', { roughness: 0.9 }),
    leafC: std('leaf_c', '#2f6248', { roughness: 0.9 }),
    eye: std('glow_eye', '#221a08', { emissive: '#ffd34d', emissiveIntensity: 3 }),
    shield: std('glow_shield', '#9fd0ff', { emissive: '#7fc0ff', emissiveIntensity: 0.9, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }),
  };

  constructor(fow: Fow) {
    this.fow = fow;
    for (const n of MODEL_NAMES) this.models.set(n, this.finish(n, buildPlaceholder(n, this.ph), true, 0));
    this.env = buildEnvPlaceholders(this.ph);
  }

  /** Kick off loading every GLB; placeholders stay in place for anything that fails. */
  loadAll() {
    for (const n of MODEL_NAMES) {
      this.loader.loadAsync(`./models/${n}.glb`).then((gltf) => {
        const prev = this.models.get(n)!;
        const scene = gltf.scene;
        tuneGlow(scene);
        // Scene-level extras are the rig description; fall back to the first node that has them.
        if (!scene.userData.anim) scene.traverse((o) => { if (!scene.userData.anim && o.userData.anim) { scene.userData.anim = o.userData.anim; scene.userData.attack ??= o.userData.attack; scene.userData.height ??= o.userData.height; } });
        this.models.set(n, this.finish(n, scene, false, prev.version + 1));
      }).catch(() => { /* missing or unparsable: keep the placeholder */ });
    }
    this.loader.loadAsync('./models/env.glb').then((gltf) => {
      gltf.scene.updateMatrixWorld(true);
      tuneGlow(gltf.scene);
      // Props are the top-level nodes; unwrap a single wrapper node (e.g. `env`) if present.
      let props = gltf.scene.children.slice();
      while (props.length === 1 && !(props[0] as THREE.Mesh).isMesh && (props[0].name === 'env' || props[0].userData.asset === 'env' || props[0].children.some((c) => c.children.length > 0))) props = props[0].children.slice();
      if (!props.length) return;
      this.env = props;
      this.envVersion++;
    }).catch(() => {});
  }

  private finish(name: string, template: THREE.Object3D, placeholder: boolean, version: number): ModelAsset {
    template.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const mat = m.material as THREE.MeshStandardMaterial;
      const ghost = mat.transparent && mat.opacity < 0.6;
      m.castShadow = !ghost;
      m.receiveShadow = !ghost;
    });
    template.updateMatrixWorld(true);
    let height = Number(template.userData.height) || 0;
    if (!height) {
      const box = new THREE.Box3();
      // Ignore helper nodes that are not part of the standing silhouette.
      template.traverse((o) => { if ((o as THREE.Mesh).isMesh && !hasAncestor(o, 'ruin') && !hasAncestor(o, 'shield')) box.expandByObject(o); });
      height = box.isEmpty() ? 1 : box.max.y;
    }
    return { name, template, version, height, placeholder };
  }

  get(name: string): ModelAsset {
    return this.models.get(name) ?? this.models.get('minion_melee')!;
  }

  /**
   * Clone a model for a team. Geometry and materials are shared; team-tinted, fog-of-war and
   * stealth variants of each material are cached, never created per unit.
   */
  instantiate(name: string, team: number, opts: { fow?: boolean } = {}): THREE.Object3D {
    const asset = this.get(name);
    const obj = asset.template.clone(true);
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.material = Array.isArray(m.material) ? m.material.map((x) => this.variant(x, team, !!opts.fow, false)) : this.variant(m.material, team, !!opts.fow, false);
      const src = o.userData.pulse ? o : o.parent?.userData.pulse ? o.parent : null;
      if (src) for (const mat of Array.isArray(m.material) ? m.material : [m.material]) this.registerPulse(mat, Number(src.userData.pulse), Number(src.userData.pulseSpeed) || 2.4);
    });
    return obj;
  }

  /** Swap every mesh of `obj` between its normal and its stealth (ghosted) materials. */
  setStealth(obj: THREE.Object3D, team: number, on: boolean) {
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const cur = m.material as THREE.Material;
      const base = (cur.userData.base as THREE.Material) ?? cur;
      m.material = on ? this.variant(base, team, false, true, true) : base;
      m.castShadow = !on && !(base.transparent && base.opacity < 0.6);
    });
  }

  private variant(src: THREE.Material, team: number, fow: boolean, stealth: boolean, resolved = false): THREE.Material {
    const isTeam = src.name === 'team', isGlow = src.name === 'team_glow';
    if (!isTeam && !isGlow && !fow && !stealth) return src;
    const key = `${src.uuid}|${resolved ? 'r' : team}|${fow ? 1 : 0}|${stealth ? 1 : 0}`;
    let m = this.matCache.get(key);
    if (m) return m;
    m = src.clone();
    m.name = src.name;
    const s = m as THREE.MeshStandardMaterial;
    if (!resolved) {
      if (isTeam && s.color) s.color.set(LOOK.teamModel[team] ?? LOOK.teamModel[2]);
      if (isGlow && s.emissive) s.emissive.set(LOOK.teamGlow[team] ?? LOOK.teamGlow[2]);
    }
    if (stealth) {
      m.transparent = true;
      m.opacity = Math.min(m.opacity, 0.38);
      m.depthWrite = false;
      m.userData = { ...m.userData, base: src };
    }
    if (fow) this.fow.patch(m);
    this.matCache.set(key, m);
    return m;
  }

  private registerPulse(mat: THREE.Material, amt: number, speed: number) {
    const s = mat as THREE.MeshStandardMaterial;
    if (!s.emissive || this.pulsing.some((p) => p.mat === s)) return;
    this.pulsing.push({ mat: s, base: s.emissiveIntensity, amt, speed });
  }

  update(time: number) {
    for (const p of this.pulsing) p.mat.emissiveIntensity = p.base * (1 + p.amt * Math.sin(time * p.speed));
  }
}

function hasAncestor(o: THREE.Object3D, name: string): boolean {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p.name === name) return true;
  return false;
}

// --- Placeholders ---------------------------------------------------------------------
type Mats = Assets['ph'];

function mesh(geo: THREE.BufferGeometry, mat: THREE.Material, x = 0, y = 0, z = 0): THREE.Mesh {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  return m;
}
function node(name: string, x = 0, y = 0, z = 0, ...children: THREE.Object3D[]): THREE.Group {
  const g = new THREE.Group();
  g.name = name;
  g.position.set(x, y, z);
  for (const c of children) g.add(c);
  return g;
}
const box = (w: number, h: number, d: number) => new THREE.BoxGeometry(w, h, d);
const cyl = (r0: number, r1: number, h: number, seg = 10) => new THREE.CylinderGeometry(r0, r1, h, seg);
const ball = (r: number, detail = 1) => new THREE.IcosahedronGeometry(r, detail);

/** A simple articulated figure with the contract's node names; `s` is its overall scale. */
function biped(M: Mats, s: number, attack: string, opts: { bulk?: number; weapon?: 'sword' | 'staff' | 'bow' | 'blades' | 'shield' | 'fists'; skin?: THREE.Material } = {}): THREE.Group {
  const bulk = opts.bulk ?? 1, skin = opts.skin ?? M.skin;
  const hip = 0.62 * s, sh = 1.12 * s, w = 0.26 * s * bulk;
  const limb = (len: number, t: number, mat: THREE.Material) => mesh(box(t, len, t), mat, 0, -len / 2, 0);
  const armR = node('arm_r', -w - 0.07 * s, sh, 0, limb(0.5 * s, 0.13 * s * bulk, M.dark));
  const armL = node('arm_l', w + 0.07 * s, sh, 0, limb(0.5 * s, 0.13 * s * bulk, M.dark));
  const hand = -0.5 * s;
  switch (opts.weapon) {
    case 'sword': armR.add(mesh(box(0.05 * s, 0.08 * s, 0.75 * s), M.metal, 0, hand, 0.34 * s)); break;
    case 'blades':
      armR.add(mesh(box(0.04 * s, 0.07 * s, 0.5 * s), M.metal, 0, hand, 0.24 * s));
      armL.add(mesh(box(0.04 * s, 0.07 * s, 0.5 * s), M.metal, 0, hand, 0.24 * s));
      break;
    case 'staff': {
      armR.add(mesh(cyl(0.025 * s, 0.025 * s, 1.5 * s, 6), M.wood, 0, hand + 0.3 * s, 0.08 * s));
      const orb = mesh(ball(0.11 * s), M.glow, 0, hand + 1.1 * s, 0.08 * s);
      orb.userData.pulse = 0.3;
      armR.add(orb);
      break;
    }
    case 'bow': {
      const bow = mesh(new THREE.TorusGeometry(0.42 * s, 0.025 * s, 5, 12, Math.PI), M.wood, 0, hand, 0.1 * s);
      bow.rotation.set(0, Math.PI / 2, Math.PI / 2);
      armL.add(bow);
      break;
    }
    case 'shield':
      armL.add(mesh(box(0.5 * s, 0.75 * s, 0.07 * s), M.metal, 0, hand + 0.15 * s, 0.14 * s));
      armR.add(mesh(box(0.14 * s, 0.14 * s, 0.4 * s), M.metal, 0, hand, 0.16 * s));
      break;
    default: break;
  }
  const torso = node('torso', 0, hip, 0,
    mesh(box(w * 2, (sh - hip) * 1.05, 0.32 * s * bulk), M.team, 0, (sh - hip) / 2, 0),
    mesh(box(w * 2.5, 0.12 * s, 0.38 * s * bulk), M.dark, 0, sh - hip, 0));
  const head = node('head', 0, sh + 0.08 * s, 0, mesh(ball(0.19 * s, 2), skin, 0, 0.16 * s, 0), mesh(box(0.2 * s, 0.05 * s, 0.06 * s), M.dark, 0, 0.19 * s, 0.17 * s));
  const legR = node('leg_r', -0.12 * s * bulk, hip, 0, limb(hip, 0.15 * s * bulk, M.dark));
  const legL = node('leg_l', 0.12 * s * bulk, hip, 0, limb(hip, 0.15 * s * bulk, M.dark));
  const body = node('body', 0, 0, 0, torso, head, armL, armR, legL, legR);
  const root = node('root', 0, 0, 0, body);
  root.userData = { anim: 'biped', attack, height: sh + 0.5 * s };
  return root;
}

function buildPlaceholder(name: string, M: Mats): THREE.Object3D {
  switch (name) {
    case 'hero_korvath': return biped(M, 1.25, 'bash', { bulk: 1.5, weapon: 'shield' });
    case 'hero_sable': return biped(M, 1.08, 'dual', { bulk: 0.85, weapon: 'blades' });
    case 'hero_ignara': return biped(M, 1.1, 'cast', { bulk: 0.9, weapon: 'staff' });
    case 'hero_vesper': return biped(M, 1.1, 'shoot', { bulk: 0.9, weapon: 'bow' });
    case 'hero_liora': return biped(M, 1.1, 'cast', { bulk: 0.9, weapon: 'staff' });
    case 'minion_melee': return biped(M, 0.62, 'slash', { weapon: 'sword', skin: M.metal });
    case 'minion_ranged': return biped(M, 0.58, 'cast', { bulk: 0.85, weapon: 'staff', skin: M.metal });
    case 'minion_super': return biped(M, 1.0, 'slam', { bulk: 1.7, skin: M.metal });
    case 'minion_siege': {
      const wheels = [[-0.32, 0.26], [0.32, 0.26], [-0.32, -0.26], [0.32, -0.26]].map(([x, z], i) => {
        const w = node('wheel_' + i, x, 0.16, z, mesh(cyl(0.16, 0.16, 0.08, 10), M.wood));
        (w.children[0] as THREE.Mesh).rotation.z = Math.PI / 2;
        return w;
      });
      const barrel = node('barrel', 0, 0.5, 0, mesh(cyl(0.09, 0.12, 0.7, 8), M.metal, 0, 0, 0.25));
      (barrel.children[0] as THREE.Mesh).rotation.x = Math.PI / 2;
      const body = node('body', 0, 0, 0, mesh(box(0.56, 0.22, 0.8), M.team, 0, 0.3, 0), barrel, ...wheels);
      const root = node('root', 0, 0, 0, body);
      root.userData = { anim: 'cart', attack: 'shoot', height: 0.8 };
      return root;
    }
    case 'monster_stalker': {
      const leg = (n: string, x: number, z: number) => node(n, x, 0.34, z, mesh(box(0.09, 0.34, 0.09), M.hide, 0, -0.17, 0));
      const jaw = node('jaw', 0, -0.04, 0.1, mesh(box(0.16, 0.04, 0.2), M.dark, 0, 0, 0.08));
      const head = node('head', 0, 0.46, 0.34, mesh(box(0.22, 0.18, 0.3), M.hide, 0, 0.02, 0.1), mesh(box(0.24, 0.03, 0.04), M.eye, 0, 0.07, 0.24), jaw);
      const tail = node('tail', 0, 0.44, -0.34, mesh(box(0.06, 0.06, 0.34), M.hide, 0, 0, -0.17));
      const body = node('body', 0, 0, 0, mesh(box(0.28, 0.24, 0.7), M.hide, 0, 0.42, 0), head, tail, leg('leg_fl', 0.13, 0.24), leg('leg_fr', -0.13, 0.24), leg('leg_bl', 0.13, -0.26), leg('leg_br', -0.13, -0.26));
      const root = node('root', 0, 0, 0, body);
      root.userData = { anim: 'quad', attack: 'bite', height: 0.75 };
      return root;
    }
    case 'monster_golem': {
      const g = biped(M, 1.15, 'slam', { bulk: 1.8, skin: M.stone });
      g.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) m.material = m.material === M.team ? M.stone : m.material === M.dark ? M.stoneDark : m.material; });
      g.getObjectByName('torso')!.add(mesh(ball(0.12), M.eye, 0, 0.45, 0.24), mesh(ball(0.16), M.moss, 0.2, 0.62, -0.05));
      return g;
    }
    case 'monster_insight':
    case 'monster_might': {
      const glow = std('glow_' + name, '#111111', { emissive: name === 'monster_might' ? '#ff5a3c' : '#4f9bff', emissiveIntensity: 2.6 });
      const shell = std('shell_' + name, name === 'monster_might' ? '#7a2018' : '#1c4a8a', { roughness: 0.5, metalness: 0.2 });
      const ring = node('ring', 0, 0.75, 0);
      for (let i = 0; i < 6; i++) { const a = (i / 6) * 6.283; ring.add(mesh(new THREE.ConeGeometry(0.12, 0.36, 4), shell, Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52)); }
      ring.userData = { spin: 0.9 };
      const core = mesh(new THREE.OctahedronGeometry(0.34), glow, 0, 0.75, 0);
      core.userData = { spin: -1.4, bob: 0.05, pulse: 0.25 };
      const body = node('body', 0, 0, 0, ring, core);
      const root = node('root', 0, 0, 0, body);
      root.userData = { anim: 'float', attack: name === 'monster_might' ? 'bite' : 'cast', height: 1.35 };
      return root;
    }
    case 'monster_wyrm': {
      const scaleA = std('wyrm_a', '#1f8a7a'), scaleB = std('wyrm_b', '#27a892');
      const root = node('root');
      const jaw = node('jaw', 0, -0.08, 0.2, mesh(box(0.34, 0.07, 0.42), scaleA, 0, 0, 0.18));
      root.add(node('head', 0, 0.55, 0.5, mesh(box(0.5, 0.36, 0.7), scaleB, 0, 0.05, 0.3), mesh(box(0.52, 0.05, 0.06), M.eye, 0, 0.16, 0.56), jaw));
      for (let i = 1; i <= 7; i++) root.add(node('seg_' + i, 0, 0.45 - i * 0.03, 0.5 - i * 0.42, mesh(ball(0.36 - i * 0.03, 1), i % 2 ? scaleA : scaleB)));
      root.userData = { anim: 'serpent', attack: 'bite', height: 1.1 };
      return root;
    }
    case 'monster_titan': {
      const hideT = std('titan', '#5b3f9a', { roughness: 0.6 });
      const g = biped(M, 1.9, 'slam', { bulk: 1.9, skin: hideT });
      g.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) m.material = m.material === M.team ? hideT : m.material; });
      const eye = std('glow_titan', '#110011', { emissive: '#ff7aff', emissiveIntensity: 3.2 });
      const orb = mesh(ball(0.2), eye, 0, 0.7, 0.34);
      orb.userData = { pulse: 0.35 };
      g.getObjectByName('torso')!.add(orb);
      return g;
    }
    case 'tower':
    case 'core': {
      const core = name === 'core', k = core ? 1.7 : 1;
      const root = node('root');
      root.add(node('base', 0, 0, 0, mesh(cyl(0.62 * k, 0.75 * k, 0.3, core ? 12 : 8), M.stoneDark, 0, 0.15, 0)));
      root.add(node('shaft', 0, 0, 0, mesh(cyl(0.34 * k, 0.5 * k, core ? 0.9 : 1.6, core ? 12 : 8), M.stone, 0, core ? 0.75 : 1.1, 0), mesh(cyl(0.52 * k, 0.4 * k, 0.16, core ? 12 : 8), M.team, 0, core ? 1.25 : 1.95, 0)));
      const crystal = mesh(new THREE.OctahedronGeometry(core ? 0.6 : 0.34), M.glow, 0, core ? 2.2 : 2.55, 0);
      crystal.scale.y = 1.5;
      crystal.name = 'crystal';
      crystal.userData = { spin: 0.8, bob: 0.07, pulse: 0.2 };
      root.add(crystal);
      const shield = mesh(new THREE.SphereGeometry(core ? 2.0 : 1.2, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.55), M.shield, 0, 0.05, 0);
      shield.name = 'shield';
      root.add(shield);
      const ruin = node('ruin', 0, 0, 0, mesh(cyl(0.62 * k, 0.75 * k, 0.3, core ? 12 : 8), M.stoneDark, 0, 0.15, 0));
      for (let i = 0; i < 7; i++) {
        const a = i * 1.1, d = (0.15 + 0.4 * ((i * 37) % 10) / 10) * k;
        const r = mesh(box(0.3 * k, 0.24 * k, 0.26 * k), i % 2 ? M.stone : M.stoneDark, Math.cos(a) * d, 0.36 + (i % 3) * 0.08, Math.sin(a) * d);
        r.rotation.set(i, i * 2.1, i * 0.7);
        ruin.add(r);
      }
      root.add(ruin);
      root.userData = { anim: 'static', height: core ? 3.1 : 3.0 };
      return root;
    }
    case 'fountain': case 'plinth': { const r = node('root'); r.userData = { anim: 'static', height: 0 }; return r; }
    default: return biped(M, 0.6, 'slash');
  }
}

/** Stand-in environment props so the forest reads before env.glb exists. */
function buildEnvPlaceholders(M: Mats): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  const pine = (name: string, leaf: THREE.Material, h: number, r: number) => {
    const g = node(name);
    g.add(mesh(cyl(0.07, 0.11, h * 0.35, 6), M.wood, 0, h * 0.17, 0));
    for (let i = 0; i < 3; i++) g.add(mesh(new THREE.ConeGeometry(r * (1 - i * 0.24), h * 0.42, 7), leaf, 0, h * (0.42 + i * 0.22), 0));
    g.userData = { sway: 0.012 };
    return g;
  };
  const round = (name: string, leaf: THREE.Material, h: number, r: number) => {
    const g = node(name);
    g.add(mesh(cyl(0.08, 0.13, h * 0.5, 6), M.wood, 0, h * 0.25, 0));
    g.add(mesh(ball(r, 1), leaf, 0, h * 0.62, 0), mesh(ball(r * 0.7, 1), leaf, r * 0.5, h * 0.5, r * 0.2), mesh(ball(r * 0.62, 1), leaf, -r * 0.45, h * 0.52, -r * 0.3));
    g.userData = { sway: 0.01 };
    return g;
  };
  out.push(pine('tree_pine_a', M.leafC, 2.3, 0.62), pine('tree_pine_b', M.leafA, 1.9, 0.55), round('tree_oak_a', M.leafB, 1.9, 0.62), round('tree_oak_b', M.leafA, 1.6, 0.55));
  const rock = (name: string, sx: number, sy: number, sz: number) => {
    const m = mesh(new THREE.DodecahedronGeometry(0.4, 0), M.stone, 0, 0.14 * sy, 0);
    m.scale.set(sx, sy, sz);
    m.rotation.set(0.3, 0.7, 0.2);
    return node(name, 0, 0, 0, m);
  };
  out.push(rock('rock_a', 1.2, 0.8, 1), rock('rock_b', 0.8, 1.2, 0.9));
  const tuft = node('grass');
  for (let i = 0; i < 5; i++) {
    const b = mesh(new THREE.ConeGeometry(0.03, 0.22 + (i % 3) * 0.05, 3), M.leafB, Math.cos(i * 1.3) * 0.06, 0.11, Math.sin(i * 1.3) * 0.06);
    b.rotation.set(Math.sin(i) * 0.3, 0, Math.cos(i) * 0.3);
    tuft.add(b);
  }
  tuft.userData = { sway: 0.2 };
  out.push(tuft);
  return out;
}

/**
 * Apply the global glow strength to a freshly loaded model's emissive (`glow*`, `team_glow`) materials.
 * A model can scale its own glow with a `glow` extra on its root (big emissive surfaces bloom harder).
 */
function tuneGlow(root: THREE.Object3D) {
  let own = 1;
  root.traverse((o) => { if (o.userData.glow && own === 1) own = Number(o.userData.glow) || 1; });
  const seen = new Set<THREE.Material>();
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) {
      if (seen.has(mat) || !(mat.name.startsWith('glow') || mat.name === 'team_glow')) continue;
      seen.add(mat);
      (mat as THREE.MeshStandardMaterial).emissiveIntensity *= LOOK.glow.emissive * own;
    }
  });
}

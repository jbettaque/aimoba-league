/**
 * 3D world renderer (three.js): terrain, props, units, projectiles, fog of war, ground decals
 * and the overhead-bar overlay. The pieces live in ./r3d; all art-direction parameters are in
 * ./r3d/look.ts. The simulation stays 2D: sim (x, y) maps to world (x, 0, y) * 0.01.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { SIGHT_RANGE, TOWER_RANGE, type Projectile, type Unit } from '../sim/defs.ts';
import type { Game } from '../sim/game.ts';
import { HEROES } from '../sim/heroes.ts';
import { FOUNTAIN_POS, MAP_SIZE } from '../sim/map.ts';
import type { Fx } from './fx.ts';
import { animate, buildRig, type Rig } from './r3d/anim.ts';
import { Assets } from './r3d/assets.ts';
import { DecalLayer } from './r3d/decals.ts';
import { Fow } from './r3d/fow.ts';
import { GlowBatch, rgb } from './r3d/glow.ts';
import { drawGroundLayer, type Shown } from './r3d/ground2d.ts';
import { LOOK, WORLD } from './r3d/look.ts';
import { Overlay } from './r3d/overlay.ts';
import { Props } from './r3d/props.ts';
import { Terrain } from './r3d/terrain.ts';
import { Units } from './r3d/units.ts';

export const TEAM_COLORS = LOOK.teamColors;

export interface View {
  /** Whose vision we render (-1 = spectator, sees everything). */
  team: number;
  /** The player's (or followed) hero unit id, -1 if none. */
  focusId: number;
  hoverId: number;
  aimSlot: number;
  amove: boolean;
  mouseX: number;
  mouseY: number;
}

const WHITE: [number, number, number] = [1, 1, 1];
const rad = (d: number) => (d * Math.PI) / 180;

export class Renderer {
  canvas: HTMLCanvasElement;
  w = 0;
  h = 0;
  dpr = 1;
  /** Sim-space focus point; `zoom` = screen pixels per sim unit at that point. */
  cam = { x: 1000, y: 7000, zoom: 0.62 };
  zoomFactor = 1;
  /** Top-down picture of the map, for the minimap. */
  ground: HTMLCanvasElement;
  time = 0;

  private gl: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private sun: THREE.DirectionalLight;
  private fogFx: THREE.Fog;
  private fow = new Fow();
  private terrain: Terrain;
  /** Loaded models; shared with the hero-select stage. */
  readonly assets: Assets;
  private props: Props;
  private units: Units;
  private decals: DecalLayer;
  private glow = new GlowBatch();
  private overlay: Overlay;
  private envVersion = -1;
  /** Static scenery models (the two fountains). */
  private statics = new THREE.Group();
  private staticRigs: Rig[] = [];
  private staticVersion = -1;
  private shadowExtent = 0;
  private camDist = 30;
  private v3 = new THREE.Vector3();
  private v3b = new THREE.Vector3();
  /** Rolling frame-time stats (ms), exposed for profiling. */
  stats = { frame: 0, cpu: 0 };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = (this.gl = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false }));
    gl.shadowMap.enabled = true;
    gl.shadowMap.type = THREE.PCFShadowMap;
    // Khronos Neutral keeps hue and saturation of bright colours (ACES washes glowing blues/reds out to white).
    gl.toneMapping = THREE.NeutralToneMapping;
    gl.toneMappingExposure = LOOK.exposure;
    gl.outputColorSpace = THREE.SRGBColorSpace;
    gl.info.autoReset = false;

    const L = LOOK;
    this.scene.background = new THREE.Color(L.background);
    this.fogFx = new THREE.Fog(L.fog.color, 40, 140);
    this.scene.fog = this.fogFx;
    this.camera = new THREE.PerspectiveCamera(L.camera.fov, 1, L.camera.near, L.camera.far);

    // Lights.
    this.scene.add(new THREE.HemisphereLight(L.hemi.sky, L.hemi.ground, L.hemi.intensity));
    this.scene.add(new THREE.AmbientLight(L.ambient.color, L.ambient.intensity));
    const rim = new THREE.DirectionalLight(L.rim.color, L.rim.intensity);
    rim.position.set(...L.rim.dir).multiplyScalar(50);
    this.scene.add(rim);
    const sun = (this.sun = new THREE.DirectionalLight(L.sun.color, L.sun.intensity));
    sun.castShadow = true;
    sun.shadow.mapSize.set(L.sun.shadowMapSize, L.sun.shadowMapSize);
    sun.shadow.bias = L.sun.shadowBias;
    sun.shadow.normalBias = L.sun.shadowNormalBias;
    sun.shadow.radius = L.sun.shadowRadius;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 220;
    this.scene.add(sun, sun.target);

    // World.
    this.terrain = new Terrain(this.fow);
    this.ground = this.terrain.minimap;
    this.assets = new Assets(this.fow);
    this.props = new Props(this.fow, this.terrain);
    this.units = new Units(this.assets, this.terrain);
    this.decals = new DecalLayer(this.terrain);
    this.overlay = new Overlay(canvas);
    this.scene.add(this.terrain.mesh, this.terrain.water, this.props.group, this.units.group, this.decals.mesh, this.glow.group, this.statics);
    this.assets.loadAll();

    // Post: HDR scene -> bloom -> tone mapping.
    const rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(gl, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(4, 4), L.bloom.strength, L.bloom.radius, L.bloom.threshold);
    // Threshold on the brightest channel instead of luminance, so saturated emissives (team blue / red,
    // ember orange) bloom in their own hue instead of having to burn out to white first.
    const hp = this.bloom.materialHighPassFilter;
    hp.fragmentShader = hp.fragmentShader.replace('float v = luminance( texel.xyz );', 'float v = max( texel.r, max( texel.g, texel.b ) );');
    hp.uniforms.smoothWidth.value = L.bloom.knee;
    hp.needsUpdate = true;
    this.composer.addPass(this.bloom);
    // Grade (scene-linear, before tone mapping): saturation, and contrast as a power curve around mid grey.
    this.composer.addPass(new ShaderPass({
      uniforms: { tDiffuse: { value: null }, saturation: { value: L.grade.saturation }, contrast: { value: L.grade.contrast } },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `
        uniform sampler2D tDiffuse; uniform float saturation; uniform float contrast; varying vec2 vUv;
        void main() {
          vec4 c = texture2D(tDiffuse, vUv);
          float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
          c.rgb = max(mix(vec3(l), c.rgb, saturation), 0.0);
          c.rgb = 0.18 * pow(c.rgb / 0.18, vec3(contrast));
          gl_FragColor = c;
        }`,
    }));
    this.composer.addPass(new OutputPass());

    // Missing portrait renders fall back to a simple generated badge.
    window.addEventListener('error', (e) => {
      const img = e.target;
      if (!(img instanceof HTMLImageElement)) return;
      const m = /portraits\/(\w+)\.png/.exec(img.src);
      const def = m ? HEROES.findIndex((hh) => hh.id === m[1]) : -1;
      if (def >= 0) img.src = this.fallbackPortrait(def);
    }, true);

    this.resize();
  }

  // --- Camera -------------------------------------------------------------------------
  resize() {
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.gl.setPixelRatio(this.dpr);
    this.gl.setSize(this.w, this.h, false);
    this.composer.setPixelRatio(this.dpr);
    this.composer.setSize(this.w, this.h);
    this.camera.aspect = this.w / this.h;
    this.camera.updateProjectionMatrix();
    this.overlay.resize(this.w, this.h, this.dpr);
    this.decals.resize(this.w * this.dpr, this.h * this.dpr);
    this.cam.zoom = LOOK.camera.baseZoom * (this.h / 1080) * this.zoomFactor;
    this.placeCamera();
  }

  setZoom(f: number) {
    this.zoomFactor = Math.max(0.55, Math.min(1.5, f));
    this.cam.zoom = LOOK.camera.baseZoom * (this.h / 1080) * this.zoomFactor;
  }

  /** Put the camera south of the focus point, looking north and down at the configured pitch. */
  private placeCamera(shx = 0, shy = 0) {
    const C = LOOK.camera, pitch = rad(C.pitch);
    const tx = (this.cam.x + shx) * WORLD, tz = (this.cam.y + shy) * WORLD;
    const d = (this.camDist = this.h / (2 * Math.tan(rad(C.fov) / 2) * this.cam.zoom / WORLD));
    this.camera.position.set(tx, d * Math.sin(pitch), tz + d * Math.cos(pitch));
    this.camera.lookAt(tx, 0, tz);
    this.camera.updateMatrixWorld();
  }

  /** Screen (CSS px) -> sim position on the ground plane. */
  toWorld(sx: number, sy: number): [number, number] {
    this.placeCamera();
    const o = this.camera.position;
    const d = this.v3.set((sx / this.w) * 2 - 1, -(sy / this.h) * 2 + 1, 0.5).unproject(this.camera).sub(o);
    if (d.y > -1e-6) return [this.cam.x, this.cam.y];
    const t = -o.y / d.y;
    return [(o.x + d.x * t) / WORLD, (o.z + d.z * t) / WORLD];
  }

  /** Sim position (optionally lifted `height` world units) -> screen CSS px. */
  toScreen(x: number, y: number, height = 0): [number, number] {
    const p = this.v3.set(x * WORLD, height, y * WORLD).project(this.camera);
    return [(p.x * 0.5 + 0.5) * this.w, (-p.y * 0.5 + 0.5) * this.h];
  }

  /** Pan the focus point by a screen-space drag (pixels). */
  panPixels(dx: number, dy: number) {
    this.cam.x -= dx / this.cam.zoom;
    this.cam.y -= dy / this.cam.zoom / Math.sin(rad(LOOK.camera.pitch));
  }

  clampCam() {
    const hw = this.w / 2 / this.cam.zoom, hh = this.h / 2 / this.cam.zoom;
    this.cam.x = Math.max(Math.min(hw, MAP_SIZE / 2) - 200, Math.min(MAP_SIZE - Math.min(hw, MAP_SIZE / 2) + 200, this.cam.x));
    this.cam.y = Math.max(Math.min(hh, MAP_SIZE / 2) - 200, Math.min(MAP_SIZE - Math.min(hh, MAP_SIZE / 2) + 400, this.cam.y));
  }

  /** The four ground points under the screen corners (sim units): TL, TR, BR, BL. */
  viewCorners(): [number, number][] {
    return [this.toWorld(0, 0), this.toWorld(this.w, 0), this.toWorld(this.w, this.h), this.toWorld(0, this.h)];
  }

  /**
   * How far (sim units) the cursor ray through ground point (x, y) passes from a unit standing
   * at (ux, uy): the unit is treated as an upright segment as tall as its model, so pointing at
   * a body or head picks it, not only pointing at its feet.
   */
  pickDistance(u: Unit, ux: number, uy: number, x: number, y: number): number {
    const flat = Math.hypot(ux - x, uy - y);
    const o = this.camera.position;
    const hgt = this.units.heightOf(u) * 0.85;
    // Ray P(t) = o + t * d ; segment Q(s) = a + s * (0, hgt, 0), s in [0, 1].
    const d = this.v3.set(x * WORLD - o.x, -o.y, y * WORLD - o.z).normalize();
    const ax = ux * WORLD - o.x, ay = -o.y, az = uy * WORLD - o.z;
    // Minimise |a + s*e - t*d| with e = (0, hgt, 0).
    const de = d.y * hgt, ee = hgt * hgt, ad = ax * d.x + ay * d.y + az * d.z, ae = ay * hgt;
    const den = ee - de * de;
    let s = den > 1e-9 ? (de * ad - ae) / den : 0;
    s = s < 0 ? 0 : s > 1 ? 1 : s;
    const t = ad + s * de;
    const dist = Math.hypot(ax - t * d.x, ay + s * hgt - t * d.y, az - t * d.z) / WORLD;
    return Math.min(flat, dist);
  }

  // --- Portraits ----------------------------------------------------------------------
  /** URL of a hero portrait (rendered offline into public/portraits). */
  portrait(def: number, _size = 128): string {
    return `./portraits/${HEROES[def].id}.png`;
  }

  private fallbackPortrait(def: number): string {
    const size = 128, hd = HEROES[def];
    const cv = document.createElement('canvas');
    cv.width = cv.height = size;
    const c = cv.getContext('2d')!;
    const bg = c.createRadialGradient(size * 0.5, size * 0.4, 4, size / 2, size / 2, size * 0.8);
    bg.addColorStop(0, hd.color);
    bg.addColorStop(1, '#0b0f17');
    c.fillStyle = bg;
    c.fillRect(0, 0, size, size);
    c.fillStyle = hd.accent;
    c.font = `800 ${size * 0.56}px Georgia, serif`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(hd.name[0], size / 2, size * 0.54);
    return cv.toDataURL();
  }

  /** Whether a world point is inside `team`'s current vision. */
  inSight(g: Game, team: number, x: number, y: number): boolean {
    for (const u of g.units) {
      if (u.team !== team || !u.alive) continue;
      const r = u.kind === 'minion' ? 800 : SIGHT_RANGE;
      if ((u.x - x) ** 2 + (u.y - y) ** 2 < r * r) return true;
    }
    return false;
  }

  // --- Projectiles / airborne FX ------------------------------------------------------
  private bolt(x: number, h: number, y: number, dx: number, dy: number, len: number, w: number, core: string, glow: string) {
    const gb = this.glow, c = rgb(glow), k = WORLD;
    gb.ribbon((x - dx * len) * k, h, (y - dy * len) * k, (x + dx * w) * k, h, (y + dy * w) * k, w * 2.6 * k, c, 0, 0.9);
    gb.sprite(x * k, h, y * k, w * 4.2 * k, c, 0.9);
    gb.sprite(x * k, h, y * k, w * 2 * k, rgb(core), 1);
  }

  private drawProjectile(p: Projectile, x: number, y: number, fx: Fx, g: Game) {
    const TEAM = LOOK.teamColors;
    let h = 0.75;
    switch (p.kind) {
      case 'aa': {
        const src = g.unit(p.srcId);
        const hero = src && src.hero ? HEROES[src.hero.def] : null;
        if (hero) this.bolt(x, h, y, p.dx, p.dy, 60, 11, '#ffffff', hero.color);
        else this.bolt(x, 0.5, y, p.dx, p.dy, 26, 7, '#ffffff', TEAM[p.team]);
        break;
      }
      case 'tower': {
        // Tower shots arc down from the crystal to the target.
        const t = p.targetId >= 0 ? g.unit(p.targetId) : undefined;
        const far = t ? Math.min(1, Math.hypot(t.x - x, t.y - y) / TOWER_RANGE) : 0;
        h = 0.7 + far * 1.9;
        this.bolt(x, h, y, p.dx, p.dy, 90, 22, '#ffffff', TEAM[p.team]);
        fx.trail(x, y, TEAM[p.team], 14, 0.25, h * 100);
        break;
      }
      case 'ignQ': this.bolt(x, h, y, p.dx, p.dy, 150, 30, '#fff6c8', '#ff7a2e'); fx.trail(x, y, '#ff8a3d', 20, 0.35, 75); break;
      case 'ignE': {
        const c = rgb('#ffb347'), k = WORLD;
        for (let i = 0; i < 5; i++) {
          const a = this.time * 12 + i * 0.9;
          this.glow.sprite((x + Math.cos(a) * 34) * k, h, (y + Math.sin(a) * 34) * k, 0.22, c, 0.9 - i * 0.14);
        }
        this.glow.sprite(x * k, h, y * k, 0.3, rgb('#fff1c2'), 1);
        fx.trail(x, y, '#ffb347', 12, 0.25, 75);
        break;
      }
      case 'vesQ': this.bolt(x, h, y, p.dx, p.dy, 190, 16, '#eafff4', '#3ecf8e'); fx.trail(x, y, '#6ee7ae', 10, 0.2, 75); break;
      case 'vesR':
        this.bolt(x, h, y, p.dx, p.dy, 320, 46, '#ffffff', '#7df9ff');
        fx.trail(x, y, '#aef6ff', 30, 0.4, 75);
        fx.trail(x, y, '#3ecf8e', 22, 0.4, 75);
        break;
      case 'lioW': this.bolt(x, h, y, p.dx, p.dy, 120, 26, '#ffffff', '#ffd166'); fx.trail(x, y, '#ffe29a', 14, 0.3, 75); break;
      default: this.bolt(x, h, y, p.dx, p.dy, 40, 10, '#ffffff', '#cccccc');
    }
  }

  // --- Frame ----------------------------------------------------------------------------
  draw(g: Game, view: View, alpha: number, dt: number, fx: Fx) {
    const t0 = performance.now();
    this.gl.info.reset();
    dt = Math.max(0, dt);
    this.time += dt;
    const time = this.time, k = WORLD;
    if (this.assets.envVersion !== this.envVersion) {
      this.envVersion = this.assets.envVersion;
      this.props.build(this.assets.env);
    }

    const fountain = this.assets.get('fountain');
    if (fountain.version !== this.staticVersion) {
      this.staticVersion = fountain.version;
      this.statics.clear();
      this.staticRigs = [];
      for (let team = 0; team < 2; team++) {
        const m = this.assets.instantiate('fountain', team, { fow: true });
        m.position.set(FOUNTAIN_POS[team][0] * k, 0, FOUNTAIN_POS[team][1] * k);
        m.rotation.y = team === 0 ? Math.PI * 0.75 : -Math.PI * 0.25;
        this.statics.add(m);
        this.staticRigs.push(buildRig(m));
      }
    }
    for (const rig of this.staticRigs) animate(rig, { time, dt, speed: 0, phase: 0, dist: 0, atk: 0, seed: 0 });

    const shx = fx.shake > 0 ? (Math.random() - 0.5) * fx.shake : 0, shy = fx.shake > 0 ? (Math.random() - 0.5) * fx.shake : 0;
    this.placeCamera(shx, shy);
    // Visible ground rectangle (sim units).
    const cs = this.viewCorners();
    this.placeCamera(shx, shy);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [cx, cy] of cs) { x0 = Math.min(x0, cx); y0 = Math.min(y0, cy); x1 = Math.max(x1, cx); y1 = Math.max(y1, cy); }

    const spectator = view.team < 0;
    const sees = (u: Unit) => spectator || g.canSee(view.team, u);
    this.fow.update(g, view.team);
    this.fow.uniforms.uWindTime.value = time;

    const shown = this.units.sync(g, alpha, dt, time, sees, x0, y0, x1, y1);

    // Ground decals.
    const c = this.decals.begin(x0 - 60, y0 - 60, x1 + 60, y1 + 60);
    drawGroundLayer(c, g, view, shown, fx, time, alpha, x0, y0, x1, y1, sees);
    this.decals.end();

    // Glow batch: projectiles, airborne particles, status effects.
    const gb = this.glow;
    gb.begin(this.camera, (this.h * this.dpr) / (2 * Math.tan(rad(LOOK.camera.fov) / 2)));
    for (const z of g.zones) {
      if (z.kind !== 'ignR') continue;
      // Falling star.
      const kk = 1 - z.delay / z.total, hgt = (1 - kk) * 15;
      const sx = z.x * k + hgt * 0.4, sz = z.y * k - hgt * 0.15;
      gb.sprite(sx, hgt + 0.4, sz, 2.6, rgb('#ff8c32'), 0.9);
      gb.sprite(sx, hgt + 0.4, sz, 1.1, rgb('#fffad2'), 1);
      gb.ribbon(sx + 1.0, hgt + 3.0, sz - 0.4, sx, hgt + 0.4, sz, 0.9, rgb('#ff7a2e'), 0, 0.8);
      fx.trail(sx / k, sz / k, '#ff8a3d', 40, 0.4, (hgt + 0.4) * 100);
    }
    for (const s of shown) {
      const u = s.u, ux = s.x, uy = s.y;
      if (u.kind === 'tower' || u.kind === 'core') continue;
      const wx = ux * k, wz = uy * k;
      if (u.stunned) {
        const top = s.wy + s.height + 0.25, col = rgb('#ffe66d');
        for (let i = 0; i < 3; i++) {
          const a = time * 5 + i * 2.094;
          gb.sprite(wx + Math.cos(a) * 0.3, top + Math.sin(a * 0.5) * 0.04, wz + Math.sin(a) * 0.3, 0.2, col, 1);
        }
      }
      if (u.hero) {
        const hs = u.hero;
        if (hs.order.type === 'recall') {
          const kk = 1 - hs.recallT / 5;
          gb.ribbon(wx, s.wy, wz, wx, s.wy + 4.4, wz, 0.75, rgb('#78c8ff'), 0.25 + 0.5 * kk, 0);
          if (Math.random() < 0.5) fx.trail(ux + (Math.random() - 0.5) * 80, uy + (Math.random() - 0.5) * 80, '#9be0ff', 8, 0.5, 20 + Math.random() * 160);
        }
        if (u.dash) fx.trail(ux, uy, HEROES[hs.def].color, 26, 0.3, 60);
        for (const b of u.buffs) {
          if (b.kind === 'dot' && Math.random() < 0.25) fx.trail(ux, uy, '#ff7a2e', 12, 0.35, 70);
          else if (b.kind === 'as' && Math.random() < 0.2) fx.trail(ux, uy, HEROES[hs.def].accent, 9, 0.3, 70);
        }
      } else if (u.buffs.length && Math.random() < 0.25) {
        for (const b of u.buffs) if (b.kind === 'dot') fx.trail(ux, uy, '#ff7a2e', 10, 0.3, 45);
      }
    }
    for (const p of g.projectiles) {
      const px = p.px + (p.x - p.px) * alpha, py = p.py + (p.y - p.py) * alpha;
      if (px < x0 - 300 || px > x1 + 300 || py < y0 - 300 || py > y1 + 300) continue;
      if (!spectator && p.team !== view.team && !this.inSight(g, view.team, px, py)) continue;
      this.drawProjectile(p, px, py, fx, g);
    }
    for (const p of fx.parts) {
      if (p.ground) continue;
      if (p.x < x0 - 500 || p.x > x1 + 500 || p.y < y0 - 500 || p.y > y1 + 500) continue;
      const f = p.life / p.max;
      if (p.kind === 'dot') gb.sprite(p.x * k, p.z * k, p.y * k, p.size * (0.4 + 0.6 * f) * 3.2 * k, rgb(p.color), Math.min(1, f * 1.6));
      else if (p.kind === 'line') gb.ribbon(p.x * k, p.z * k, p.y * k, p.x2 * k, p.z * k, p.y2 * k, p.size * f * 3 * k, rgb(p.color), f, f);
      else if (p.kind === 'ring') {
        // Airborne ring: a circle of sprites.
        const r = p.size + (p.size2 - p.size) * (1 - f * f), col = rgb(p.color);
        for (let i = 0; i < 18; i++) { const a = (i / 18) * 6.283; gb.sprite((p.x + Math.cos(a) * r) * k, p.z * k, (p.y + Math.sin(a) * r) * k, p.x2 * 3 * k, col, f); }
      }
    }
    gb.end();

    // Sun and its shadow box follow the camera, snapped to shadow texels to avoid shimmer.
    const S = LOOK.sun;
    const ext = Math.max(12, Math.min(52, S.shadowExtent / this.zoomFactor));
    if (ext !== this.shadowExtent) {
      this.shadowExtent = ext;
      const sc = this.sun.shadow.camera;
      sc.left = -ext; sc.right = ext; sc.top = ext; sc.bottom = -ext;
      sc.updateProjectionMatrix();
    }
    const dir = this.v3.set(...S.dir).normalize();
    const right = this.v3b.set(0, 1, 0).cross(dir).normalize();
    const upx = dir.y * right.z - dir.z * right.y, upy = dir.z * right.x - dir.x * right.z, upz = dir.x * right.y - dir.y * right.x;
    const cx = this.cam.x * k, cz = this.cam.y * k - ext * 0.18;
    const texel = (2 * ext) / S.shadowMapSize;
    const a = Math.round((cx * right.x + cz * right.z) / texel) * texel, b = Math.round((cx * upx + cz * upz) / texel) * texel, cc = cx * dir.x + cz * dir.z;
    const tx = right.x * a + upx * b + dir.x * cc, ty = right.y * a + upy * b + dir.y * cc, tz = right.z * a + upz * b + dir.z * cc;
    this.sun.target.position.set(tx, ty, tz);
    this.sun.position.set(tx + dir.x * 90, ty + dir.y * 90, tz + dir.z * 90);
    this.sun.target.updateMatrixWorld();

    this.fogFx.near = this.camDist * LOOK.fog.near;
    this.fogFx.far = this.camDist * LOOK.fog.far;
    this.terrain.update(time);
    this.assets.update(time);

    const t1 = performance.now();
    this.composer.render();

    // Screen-space overlay.
    this.overlay.draw(shown, fx, view.focusId, this.w, this.h, (x, y, hgt) => this.toScreen(x, y, hgt));
    const t2 = performance.now();
    this.stats.cpu += (t1 - t0 - this.stats.cpu) * 0.05;
    this.stats.frame += (t2 - t0 - this.stats.frame) * 0.05;
  }

  /** Units drawn last frame (debug / tooling). */
  debugInfo() {
    const i = this.gl.info;
    return { calls: i.render.calls, triangles: i.render.triangles, geometries: i.memory.geometries, textures: i.memory.textures, ...this.stats };
  }
}

export type { Shown };

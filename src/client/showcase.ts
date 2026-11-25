/**
 * Hero-select stage: the chosen hero's real in-game model, standing on a display plinth and
 * turning slowly, drawn into its own small transparent WebGL canvas inside the menu.
 * Drag to turn it by hand. Changing the team recolours its cloth, exactly as in a match.
 */
import * as THREE from 'three';
import { animate, buildRig, type Rig } from './r3d/anim.ts';
import type { Assets } from './r3d/assets.ts';

const easeOut = (t: number) => 1 - (1 - t) ** 3;

export class Showcase {
  readonly canvas = document.createElement('canvas');
  private gl: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(25, 1, 0.5, 60);
  private turn = new THREE.Group();
  private heroSlot = new THREE.Group();
  private plinthSlot = new THREE.Group();
  private assets: Assets;
  private hero = '';
  private team = 0;
  private built = '';
  private rig: Rig | null = null;
  private plinthRig: Rig | null = null;
  private time = 0;
  /** Seconds since the hero was (re)selected; drives the entrance flourish. */
  private since = 9;
  private yaw = 0.42;
  private drag: { x: number; yaw: number } | null = null;
  private w = 0;
  private h = 0;

  constructor(assets: Assets) {
    this.assets = assets;
    this.canvas.className = 'm-figure';
    const gl = (this.gl = new THREE.WebGLRenderer({ canvas: this.canvas, alpha: true, antialias: true, powerPreference: 'high-performance' }));
    gl.setClearColor(0x000000, 0);
    gl.toneMapping = THREE.NeutralToneMapping;
    gl.toneMappingExposure = 1.08;
    gl.shadowMap.enabled = true;
    gl.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene.add(new THREE.HemisphereLight('#c6e2ff', '#4a4630', 1.25));
    const key = new THREE.DirectionalLight('#ffe0b0', 3.3);
    key.position.set(-3.2, 5.5, 4.2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = key.shadow.camera.bottom = -3.2;
    key.shadow.camera.right = key.shadow.camera.top = 3.2;
    key.shadow.camera.far = 20;
    key.shadow.bias = -0.0008;
    key.shadow.normalBias = 0.03;
    const rim = new THREE.DirectionalLight('#86b8ff', 1.6);
    rim.position.set(3.5, 3, -5);
    this.scene.add(key, rim);
    this.turn.add(this.heroSlot, this.plinthSlot);
    this.scene.add(this.turn);
    this.camera.position.set(0, 2.45, 7.2);
    this.camera.lookAt(0, 0.88, 0);

    this.canvas.addEventListener('pointerdown', (e) => {
      this.drag = { x: e.clientX, yaw: this.yaw };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', (e) => { if (this.drag) this.yaw = this.drag.yaw + (e.clientX - this.drag.x) * 0.012; });
    const end = () => { this.drag = null; };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
  }

  /** Show a hero (by id) in a team's colours. */
  set(heroId: string, team: number) {
    if (heroId !== this.hero) this.since = 0;
    this.hero = heroId;
    this.team = team;
  }

  private rebuild() {
    const name = 'hero_' + this.hero;
    const sig = `${name}|${this.team}|${this.assets.get(name).version}|${this.assets.get('plinth').version}`;
    if (sig === this.built) return;
    this.built = sig;
    this.heroSlot.clear();
    this.plinthSlot.clear();
    const hero = this.assets.instantiate(name, this.team);
    const plinth = this.assets.instantiate('plinth', this.team);
    this.heroSlot.add(hero);
    plinth.scale.setScalar(0.74);
    this.plinthSlot.add(plinth);
    this.rig = buildRig(hero);
    this.plinthRig = buildRig(plinth);
  }

  frame(dt: number) {
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if (!this.hero || !cw || !ch) return;
    if (cw !== this.w || ch !== this.h) {
      this.w = cw;
      this.h = ch;
      this.gl.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      this.gl.setSize(cw, ch, false);
      this.camera.aspect = cw / ch;
      this.camera.updateProjectionMatrix();
    }
    this.rebuild();
    this.time += dt;
    this.since += dt;
    // Entrance: the figure spins into place, settles with a small pop and strikes its attack pose once.
    const enter = easeOut(Math.min(1, this.since / 0.55));
    const atk = this.since > 0.25 && this.since < 1.0 ? Math.sin(((this.since - 0.25) / 0.75) * Math.PI) : 0;
    this.heroSlot.scale.setScalar(0.82 + 0.18 * enter);
    this.heroSlot.rotation.y = (1 - enter) * -1.6;
    this.turn.rotation.y = this.yaw + (this.drag ? 0 : Math.sin(this.time * 0.45) * 0.28);
    if (this.rig) animate(this.rig, { time: this.time, dt, speed: 0, phase: 0, dist: 0, atk, seed: 0 });
    if (this.plinthRig) animate(this.plinthRig, { time: this.time, dt, speed: 0, phase: 0, dist: 0, atk: 0, seed: 0 });
    this.gl.render(this.scene, this.camera);
  }
}

/**
 * Unit views: one model instance per visible unit, synced to the simulation every frame
 * (interpolated position, smoothed facing, procedural animation, stealth, death fade,
 * structure ruins). Models are pooled per (model, team).
 */
import * as THREE from 'three';
import type { Unit } from '../../sim/defs.ts';
import type { Game } from '../../sim/game.ts';
import { HEROES } from '../../sim/heroes.ts';
import { animate, buildRig, type Rig } from './anim.ts';
import type { Assets } from './assets.ts';
import type { Shown } from './ground2d.ts';
import { ANIM, WORLD } from './look.ts';
import type { Terrain } from './terrain.ts';

interface UnitView {
  id: number;
  key: string;
  team: number;
  version: number;
  root: THREE.Group;
  model: THREE.Object3D;
  rig: Rig;
  height: number;
  structure: boolean;
  ruin: THREE.Object3D | undefined;
  shield: THREE.Object3D | undefined;
  intact: THREE.Object3D[];
  lx: number;
  lz: number;
  speed: number;
  phase: number;
  dist: number;
  facing: number;
  seed: number;
  stealth: boolean;
  /** 1 = alive, counts down to 0 while the death fade plays. */
  fade: number;
  seen: boolean;
}

export function modelKey(u: Unit): string {
  switch (u.kind) {
    case 'hero': return 'hero_' + HEROES[u.hero!.def].id;
    case 'minion': return 'minion_' + (u.subtype || 'melee');
    case 'monster': return 'monster_' + u.subtype;
    default: return u.kind; // 'tower' | 'core'
  }
}

export class Units {
  group = new THREE.Group();
  private views = new Map<number, UnitView>();
  private pool = new Map<string, { model: THREE.Object3D; rig: Rig; version: number }[]>();
  private assets: Assets;
  private terrain: Terrain;
  private game: Game | null = null;
  private shown: Shown[] = [];

  constructor(assets: Assets, terrain: Terrain) {
    this.assets = assets;
    this.terrain = terrain;
    this.group.name = 'units';
  }

  /** Height (world units) of a unit's model, for picking and overhead bars. */
  heightOf(u: Unit): number {
    return this.views.get(u.id)?.height ?? this.assets.get(modelKey(u)).height;
  }

  private make(u: Unit): UnitView {
    const key = modelKey(u), structure = u.kind === 'tower' || u.kind === 'core';
    const asset = this.assets.get(key);
    const pk = key + '|' + u.team;
    const pooled = this.pool.get(pk);
    let inst = pooled?.pop();
    while (inst && inst.version !== asset.version) inst = pooled?.pop();
    if (!inst) {
      const model = this.assets.instantiate(key, u.team, { fow: structure });
      // The rig description lives on the template root (scene extras).
      model.userData.anim ??= asset.template.userData.anim;
      model.userData.attack ??= asset.template.userData.attack;
      inst = { model, rig: buildRig(model), version: asset.version };
    }
    const root = new THREE.Group();
    root.add(inst.model);
    this.group.add(root);
    const ruin = structure ? inst.model.getObjectByName('ruin') : undefined;
    const shield = structure ? inst.model.getObjectByName('shield') : undefined;
    const intact = ruin && ruin.parent ? ruin.parent.children.filter((c) => c !== ruin && c !== shield) : [];
    const v: UnitView = {
      id: u.id, key, team: u.team, version: asset.version, root, model: inst.model, rig: inst.rig, height: asset.height, structure, ruin, shield, intact,
      lx: NaN, lz: NaN, speed: 0, phase: (u.id * 2.399) % 6.283, dist: 0, facing: u.facing, seed: ((u.id * 0.618034) % 1), stealth: false, fade: 1, seen: true,
    };
    this.views.set(u.id, v);
    return v;
  }

  private release(v: UnitView) {
    this.group.remove(v.root);
    v.root.remove(v.model);
    if (v.stealth) this.assets.setStealth(v.model, v.team, false);
    v.model.scale.setScalar(1);
    const pk = v.key + '|' + v.team;
    let list = this.pool.get(pk);
    if (!list) this.pool.set(pk, (list = []));
    if (list.length < 24) list.push({ model: v.model, rig: v.rig, version: v.version });
    this.views.delete(v.id);
  }

  clear() {
    for (const v of [...this.views.values()]) this.release(v);
  }

  /**
   * Bring the views in line with the simulation. Returns the units drawn this frame with their
   * interpolated sim positions (used by the decal layer and the overlay).
   */
  sync(g: Game, alpha: number, dt: number, time: number, sees: (u: Unit) => boolean, x0: number, y0: number, x1: number, y1: number): Shown[] {
    if (g !== this.game) { this.clear(); this.game = g; }
    const shown = this.shown;
    shown.length = 0;
    for (const v of this.views.values()) v.seen = false;

    for (const u of g.units) {
      const structure = u.kind === 'tower' || u.kind === 'core';
      if (!structure && (!u.alive || !sees(u))) continue;
      let v = this.views.get(u.id);
      if (v && (v.version !== this.assets.get(v.key).version || v.team !== u.team || v.key !== modelKey(u))) { this.release(v); v = undefined; }
      if (!v) v = this.make(u);
      v.seen = true;
      v.fade = 1;

      let ux = u.x, uy = u.y;
      if (Math.abs(u.x - u.px) + Math.abs(u.y - u.py) < 300) { ux = u.px + (u.x - u.px) * alpha; uy = u.py + (u.y - u.py) * alpha; }
      const wy = structure ? 0 : this.terrain.unitY(ux, uy);
      const wx = ux * WORLD, wz = uy * WORLD;
      v.root.position.set(wx, wy, wz);
      v.root.scale.setScalar(1);
      shown.push({ u, x: ux, y: uy, wy, height: v.height });

      if (structure) {
        // Structures face the enemy base along the map diagonal.
        v.root.rotation.y = u.team === 0 ? Math.PI * 0.75 : -Math.PI * 0.25;
        if (v.ruin) {
          v.ruin.visible = !u.alive;
          for (const o of v.intact) o.visible = u.alive;
        }
        if (v.shield) v.shield.visible = u.alive && !u.vulnerable;
        if (u.alive) animate(v.rig, { time, dt, speed: 0, phase: 0, dist: 0, atk: 0, seed: v.seed });
        continue;
      }

      // Facing, smoothed like the 2D renderer.
      let d = u.facing - v.facing;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      v.facing += d * Math.min(1, dt * ANIM.facingRate);
      v.root.rotation.y = Math.PI / 2 - v.facing;

      // Locomotion from the distance actually covered on screen.
      let step = Number.isNaN(v.lx) ? 0 : Math.hypot(wx - v.lx, wz - v.lz);
      if (step > 3) step = 0; // teleport / respawn
      v.lx = wx;
      v.lz = wz;
      const inst = dt > 1e-4 ? step / dt : 0;
      v.speed += (inst - v.speed) * Math.min(1, dt * 12);
      const stride = (ANIM.stride as Record<string, number>)[v.rig.kind] ?? 0.7;
      v.phase += step * stride * Math.PI * 2;
      v.dist += step;

      if (u.stealthed !== v.stealth) {
        v.stealth = u.stealthed;
        this.assets.setStealth(v.model, v.team, v.stealth);
      }
      // Skip the pose work for units well outside the view.
      if (ux > x0 - 300 && ux < x1 + 300 && uy > y0 - 300 && uy < y1 + 300) {
        const atk = u.attackAnim > 0 ? Math.sin(Math.min(1, u.attackAnim / 0.4) * Math.PI) : 0;
        animate(v.rig, { time, dt, speed: v.speed, phase: v.phase, dist: v.dist, atk, seed: v.seed });
      }
    }

    // Anything not seen this frame either died (quick sink / shrink) or slipped into the fog.
    for (const v of [...this.views.values()]) {
      if (v.seen) continue;
      const u = g.unit(v.id);
      const died = !v.structure && (!u || !u.alive);
      if (!died || v.fade <= 0) { this.release(v); continue; }
      v.fade -= dt / ANIM.deathTime;
      if (v.fade <= 0) { this.release(v); continue; }
      const k = v.fade * v.fade * (3 - 2 * v.fade);
      v.root.scale.set(0.6 + 0.4 * k, k, 0.6 + 0.4 * k);
      v.root.rotation.z = (1 - k) * 0.5 * (v.seed > 0.5 ? 1 : -1);
    }
    return shown;
  }
}

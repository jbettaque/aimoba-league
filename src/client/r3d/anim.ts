/**
 * Data-driven procedural animation. A model is described by node names and glTF extras
 * (`userData`), not baked clips:
 *
 *   root.userData.anim    'biped' | 'quad' | 'serpent' | 'cart' | 'float' | 'static'
 *   root.userData.attack  'slash' | 'dual' | 'bash' | 'cast' | 'shoot' | 'bite' | 'slam'
 *   any node              userData.spin (rad/s about local Y), userData.bob (+ bobSpeed, phase),
 *                         userData.rock (radians, + rockAxis 'x'|'y'|'z', rockSpeed): gentle to-and-fro, e.g. wings
 *
 * Every frame the named nodes are reset to their rest pose and offset from it, so the
 * tuning constants in look.ts (ANIM) and the tables below can be changed freely.
 *
 * Axes: forward = +Z, up = +Y, so +X is the model's LEFT. A limb hanging down swings
 * forward with a NEGATIVE rotation about X; positive torso yaw brings the right shoulder forward.
 */
import * as THREE from 'three';
import { ANIM } from './look.ts';

interface Rest { px: number; py: number; pz: number; rx: number; ry: number; rz: number }
interface Generic { node: THREE.Object3D; spin: number; bob: number; bobSpeed: number; phase: number; baseY: number; rock: number; rockAxis: THREE.Vector3; rockSpeed: number; restQ: THREE.Quaternion }

const AXES: Record<string, THREE.Vector3> = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };
const tmpQ = new THREE.Quaternion();

export interface Rig {
  kind: string;
  attack: string;
  n: Record<string, THREE.Object3D | undefined>;
  rest: Map<THREE.Object3D, Rest>;
  segs: THREE.Object3D[];
  wheels: { node: THREE.Object3D; radius: number }[];
  generic: Generic[];
}

/** What the animator needs to know about a unit this frame. */
export interface AnimState {
  time: number;
  dt: number;
  /** Ground speed in world units / second (already smoothed). */
  speed: number;
  /** Accumulated walk phase (radians). */
  phase: number;
  /** Total distance travelled (world units), for wheels. */
  dist: number;
  /** Attack pulse 0..1 (sin over the swing), exactly like the old 2D renderer. */
  atk: number;
  /** Per-unit random offset so crowds do not move in lockstep. */
  seed: number;
}

const NAMED = ['body', 'torso', 'head', 'arm_l', 'arm_r', 'leg_l', 'leg_r', 'jaw', 'tail', 'leg_fl', 'leg_fr', 'leg_bl', 'leg_br', 'barrel'];

export function buildRig(model: THREE.Object3D): Rig {
  const rig: Rig = { kind: String(model.userData.anim ?? 'static'), attack: String(model.userData.attack ?? ''), n: {}, rest: new Map(), segs: [], wheels: [], generic: [] };
  const remember = (o: THREE.Object3D) => rig.rest.set(o, { px: o.position.x, py: o.position.y, pz: o.position.z, rx: o.rotation.x, ry: o.rotation.y, rz: o.rotation.z });
  const segs: [number, THREE.Object3D][] = [];
  model.traverse((o) => {
    if (NAMED.includes(o.name) && !rig.n[o.name]) { rig.n[o.name] = o; remember(o); }
    const seg = /^seg_(\d+)$/.exec(o.name);
    if (seg) { segs.push([Number(seg[1]), o]); remember(o); }
    if (o.name.startsWith('wheel')) {
      // Wheel radius: explicit, or the height of its hub above the ground.
      rig.wheels.push({ node: o, radius: Number(o.userData.radius) || Math.max(0.05, o.position.y) });
      remember(o);
    }
    const ud = o.userData;
    if (ud.spin || ud.bob || ud.rock) rig.generic.push({ node: o, spin: Number(ud.spin) || 0, bob: Number(ud.bob) || 0, bobSpeed: Number(ud.bobSpeed) || 1.6, phase: Number(ud.phase) || 0, baseY: o.position.y, rock: Number(ud.rock) || 0, rockAxis: AXES[String(ud.rockAxis)] ?? AXES.z, rockSpeed: Number(ud.rockSpeed) || 1.6, restQ: o.quaternion.clone() });
  });
  rig.segs = segs.sort((a, b) => a[0] - b[0]).map((s) => s[1]);
  return rig;
}

/** Per-attack-style offsets at full swing (radians / world units). Missing nodes are skipped. */
type Pose = Partial<Record<string, Partial<Rest>>>;
const ATTACKS: Record<string, Pose> = {
  slash: { arm_r: { rx: -1.9, rz: -0.25 }, torso: { ry: 0.45, rx: 0.12 }, body: { pz: 0.06 } },
  dual: { arm_r: { rx: -1.6, rz: -0.5 }, arm_l: { rx: -1.6, rz: 0.5 }, torso: { rx: 0.2 }, body: { pz: 0.1 } },
  bash: { arm_l: { rx: -1.35, rz: 0.2 }, torso: { ry: -0.35, rx: 0.1 }, body: { pz: 0.16 } },
  cast: { arm_r: { rx: -2.3, rz: -0.15 }, arm_l: { rx: -0.6 }, torso: { rx: -0.1 } },
  shoot: { arm_l: { rx: -1.5 }, arm_r: { rx: -1.15, pz: -0.14 }, torso: { ry: -0.3 } },
  slam: { arm_l: { rx: -2.7 }, arm_r: { rx: -2.7 }, torso: { rx: 0.22 }, body: { py: 0.08, pz: 0.06 } },
  bite: { head: { pz: 0.22, rx: -0.15 }, jaw: { rx: 0.55 }, body: { pz: 0.12 } },
};

export function animate(rig: Rig, s: AnimState) {
  // Rest pose first, then offsets.
  for (const [o, r] of rig.rest) { o.position.set(r.px, r.py, r.pz); o.rotation.set(r.rx, r.ry, r.rz); }
  const n = rig.n, t = s.time + s.seed * 10;
  const move = Math.min(1, s.speed / ANIM.walkSpeed);
  const sw = Math.sin(s.phase), cw = Math.cos(s.phase);
  const breathe = Math.sin(t * ANIM.breatheSpeed) * ANIM.breathe;
  const add = (o: THREE.Object3D | undefined, d: Partial<Rest>, k = 1) => {
    if (!o) return;
    if (d.px) o.position.x += d.px * k;
    if (d.py) o.position.y += d.py * k;
    if (d.pz) o.position.z += d.pz * k;
    if (d.rx) o.rotation.x += d.rx * k;
    if (d.ry) o.rotation.y += d.ry * k;
    if (d.rz) o.rotation.z += d.rz * k;
  };

  switch (rig.kind) {
    case 'biped': {
      const still = 1 - move;
      add(n.leg_l, { rx: -sw * ANIM.legSwing * move });
      add(n.leg_r, { rx: sw * ANIM.legSwing * move });
      add(n.arm_l, { rx: sw * ANIM.armSwing * move + Math.sin(t * 1.3) * 0.04 * still, rz: breathe * 2 });
      add(n.arm_r, { rx: -sw * ANIM.armSwing * move + Math.sin(t * 1.3 + 1) * 0.04 * still, rz: -breathe * 2 });
      add(n.body, { py: Math.abs(cw) * ANIM.bob * move + breathe * still });
      add(n.torso, { ry: sw * ANIM.torsoTwist * move, rx: ANIM.lean * move });
      add(n.head, { rx: -ANIM.lean * 0.5 * move + breathe * still, ry: Math.sin(t * 0.7) * 0.08 * still });
      break;
    }
    case 'quad': {
      const sw2 = Math.sin(s.phase + Math.PI);
      const k = ANIM.legSwing * 0.9 * move;
      add(n.leg_fl, { rx: -sw * k });
      add(n.leg_br, { rx: -sw * k });
      add(n.leg_fr, { rx: -sw2 * k });
      add(n.leg_bl, { rx: -sw2 * k });
      add(n.body, { py: Math.abs(cw) * ANIM.bob * 0.7 * move + breathe * (1 - move), rx: Math.sin(s.phase * 2) * 0.03 * move });
      add(n.head, { rx: Math.sin(s.phase * 2 + 0.6) * 0.06 * move + breathe * 2, ry: Math.sin(t * 0.8) * 0.12 * (1 - move) });
      add(n.tail, { ry: Math.sin(t * 2.2 + s.phase) * 0.35, rx: -0.1 * move });
      break;
    }
    case 'serpent': {
      // Lateral wave travelling down the chain; grows towards the tail.
      const amp = 0.14 + 0.1 * move;
      const w = t * 2.6 + s.phase;
      add(n.head, { px: Math.sin(w) * amp * 0.35, ry: Math.cos(w) * 0.18, py: Math.sin(t * 1.7) * 0.03 });
      for (let i = 0; i < rig.segs.length; i++) {
        const o = rig.segs[i], ph = w - (i + 1) * 0.85, k = Math.min(1, 0.45 + i * 0.15);
        o.position.x += Math.sin(ph) * amp * k;
        o.rotation.y += Math.cos(ph) * 0.35 * k;
        o.position.y += Math.sin(ph * 0.5 + 1) * 0.02;
      }
      break;
    }
    case 'cart': {
      for (const wh of rig.wheels) wh.node.rotation.x += s.dist / wh.radius;
      add(n.body, { py: Math.abs(Math.sin(s.phase * 3)) * 0.012 * move, rz: Math.sin(s.phase * 1.5) * 0.015 * move });
      break;
    }
    case 'float': {
      add(n.body, { py: Math.sin(t * 1.6) * 0.07, rx: 0.14 * move, rz: Math.sin(t * 0.9) * 0.05 });
      break;
    }
    default: break;
  }

  // Attack, layered on top of locomotion.
  if (s.atk > 0.001) {
    if (rig.kind === 'cart') add(n.barrel, { pz: -0.16 * s.atk, rx: -0.06 * s.atk });
    else if (rig.kind === 'float') add(n.body, { pz: 0.22 * s.atk, rx: 0.25 * s.atk });
    else {
      const pose = ATTACKS[rig.attack] ?? ATTACKS.bite;
      for (const key in pose) add(n[key], pose[key]!, s.atk);
      if (rig.kind === 'serpent') add(n.head, { pz: 0.25, py: 0.1 }, s.atk);
    }
  }

  for (const gnr of rig.generic) {
    // Spin / rock about the node's own axes, on top of whatever tilt it was authored with.
    if (gnr.spin) gnr.node.quaternion.copy(gnr.restQ).multiply(tmpQ.setFromAxisAngle(AXES.y, gnr.spin * t + gnr.phase));
    else if (gnr.rock) gnr.node.quaternion.copy(gnr.restQ).multiply(tmpQ.setFromAxisAngle(gnr.rockAxis, Math.sin(t * gnr.rockSpeed + gnr.phase) * gnr.rock));
    if (gnr.bob) gnr.node.position.y = gnr.baseY + Math.sin(t * gnr.bobSpeed + gnr.phase) * gnr.bob;
  }
}

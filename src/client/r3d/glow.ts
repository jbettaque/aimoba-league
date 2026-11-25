/**
 * Additive, HDR billboards for everything that glows above the ground: airborne particles,
 * projectile bolts and their trails, beams, stun stars. Two dynamic batches, rebuilt every
 * frame: soft round sprites (GL points) and camera-facing ribbons.
 */
import * as THREE from 'three';
import { LOOK } from './look.ts';

const MAX_SPRITES = 4096;
const MAX_RIBBONS = 512;
const colorCache = new Map<string, [number, number, number]>();
const tmpColor = new THREE.Color();

/** CSS colour -> linear RGB (cached). */
export function rgb(css: string): [number, number, number] {
  let c = colorCache.get(css);
  if (!c) {
    tmpColor.set(css);
    c = [tmpColor.r, tmpColor.g, tmpColor.b];
    colorCache.set(css, c);
  }
  return c;
}

export class GlowBatch {
  group = new THREE.Group();
  private pPos = new Float32Array(MAX_SPRITES * 3);
  private pCol = new Float32Array(MAX_SPRITES * 4);
  private pSize = new Float32Array(MAX_SPRITES);
  private pCount = 0;
  private rPos = new Float32Array(MAX_RIBBONS * 12);
  private rCol = new Float32Array(MAX_RIBBONS * 16);
  private rUv = new Float32Array(MAX_RIBBONS * 8);
  private rCount = 0;
  private points: THREE.Points;
  private ribbons: THREE.Mesh;
  private pointMat: THREE.ShaderMaterial;
  private camPos = new THREE.Vector3();

  constructor() {
    const additive = { transparent: true, depthWrite: false, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation } as const;
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.pPos, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('aColor', new THREE.BufferAttribute(this.pCol, 4).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('aSize', new THREE.BufferAttribute(this.pSize, 1).setUsage(THREE.DynamicDrawUsage));
    this.pointMat = new THREE.ShaderMaterial({
      ...additive,
      uniforms: { uScale: { value: 1000 }, uGlow: { value: LOOK.fx.glow } },
      vertexShader: /* glsl */ `
        attribute vec4 aColor;
        attribute float aSize;
        uniform float uScale;
        varying vec4 vColor;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = aSize * uScale / -mv.z;
          vColor = aColor;
        }`,
      fragmentShader: /* glsl */ `
        uniform float uGlow;
        varying vec4 vColor;
        void main() {
          float d = length(gl_PointCoord - 0.5) * 2.0;
          float a = smoothstep(1.0, 0.0, d);
          a *= a;
          vec3 c = vColor.rgb + vec3(0.3) * smoothstep(0.4, 0.0, d);
          gl_FragColor = vec4(c * (a * vColor.a * uGlow), 1.0);
        }`,
    });
    this.points = new THREE.Points(pg, this.pointMat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 5;

    const rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.BufferAttribute(this.rPos, 3).setUsage(THREE.DynamicDrawUsage));
    rg.setAttribute('aColor', new THREE.BufferAttribute(this.rCol, 4).setUsage(THREE.DynamicDrawUsage));
    rg.setAttribute('uv', new THREE.BufferAttribute(this.rUv, 2));
    const idx = new Uint16Array(MAX_RIBBONS * 6);
    for (let i = 0; i < MAX_RIBBONS; i++) {
      idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 2, i * 4 + 1, i * 4 + 3], i * 6);
      this.rUv.set([0, 0, 0, 1, 1, 0, 1, 1], i * 8);
    }
    rg.setIndex(new THREE.BufferAttribute(idx, 1));
    const ribbonMat = new THREE.ShaderMaterial({
      ...additive,
      side: THREE.DoubleSide,
      uniforms: { uGlow: { value: LOOK.fx.glow } },
      vertexShader: /* glsl */ `
        attribute vec4 aColor;
        varying vec4 vColor;
        varying vec2 vUv;
        void main() {
          vColor = aColor;
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uGlow;
        varying vec4 vColor;
        varying vec2 vUv;
        void main() {
          float e = 1.0 - abs(vUv.y * 2.0 - 1.0);
          float a = e * e;
          vec3 c = vColor.rgb + vec3(0.45) * smoothstep(0.6, 1.0, e);
          gl_FragColor = vec4(c * (a * vColor.a * uGlow), 1.0);
        }`,
    });
    this.ribbons = new THREE.Mesh(rg, ribbonMat);
    this.ribbons.frustumCulled = false;
    this.ribbons.renderOrder = 5;
    this.group.add(this.points, this.ribbons);
  }

  /** Start a frame. `pxScale` = device pixels per world unit at distance 1. */
  begin(camera: THREE.Camera, pxScale: number) {
    this.pCount = 0;
    this.rCount = 0;
    this.camPos.copy(camera.position);
    this.pointMat.uniforms.uScale.value = pxScale;
  }

  /** A soft round glow of `size` world units across at a world position. */
  sprite(x: number, y: number, z: number, size: number, c: [number, number, number], alpha = 1) {
    const i = this.pCount;
    if (i >= MAX_SPRITES || alpha <= 0) return;
    this.pPos[i * 3] = x; this.pPos[i * 3 + 1] = y; this.pPos[i * 3 + 2] = z;
    this.pCol[i * 4] = c[0]; this.pCol[i * 4 + 1] = c[1]; this.pCol[i * 4 + 2] = c[2]; this.pCol[i * 4 + 3] = alpha;
    this.pSize[i] = size;
    this.pCount++;
  }

  /** A camera-facing streak from A to B, `width` world units, fading from alphaA to alphaB. */
  ribbon(ax: number, ay: number, az: number, bx: number, by: number, bz: number, width: number, c: [number, number, number], alphaA: number, alphaB: number) {
    const i = this.rCount;
    if (i >= MAX_RIBBONS) return;
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const vx = this.camPos.x - (ax + bx) / 2, vy = this.camPos.y - (ay + by) / 2, vz = this.camPos.z - (az + bz) / 2;
    let sx = dy * vz - dz * vy, sy = dz * vx - dx * vz, sz = dx * vy - dy * vx;
    const sl = Math.hypot(sx, sy, sz) || 1;
    const k = width / 2 / sl;
    sx *= k; sy *= k; sz *= k;
    const p = this.rPos, o = i * 12;
    p[o] = ax - sx; p[o + 1] = ay - sy; p[o + 2] = az - sz;
    p[o + 3] = ax + sx; p[o + 4] = ay + sy; p[o + 5] = az + sz;
    p[o + 6] = bx - sx; p[o + 7] = by - sy; p[o + 8] = bz - sz;
    p[o + 9] = bx + sx; p[o + 10] = by + sy; p[o + 11] = bz + sz;
    const col = this.rCol, q = i * 16;
    for (let v = 0; v < 4; v++) {
      col[q + v * 4] = c[0]; col[q + v * 4 + 1] = c[1]; col[q + v * 4 + 2] = c[2];
      col[q + v * 4 + 3] = v < 2 ? alphaA : alphaB;
    }
    this.rCount++;
  }

  end() {
    const pg = this.points.geometry, rg = this.ribbons.geometry;
    pg.setDrawRange(0, this.pCount);
    (pg.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (pg.getAttribute('aColor') as THREE.BufferAttribute).needsUpdate = true;
    (pg.getAttribute('aSize') as THREE.BufferAttribute).needsUpdate = true;
    rg.setDrawRange(0, this.rCount * 6);
    (rg.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    (rg.getAttribute('aColor') as THREE.BufferAttribute).needsUpdate = true;
    this.points.visible = this.pCount > 0;
    this.ribbons.visible = this.rCount > 0;
  }
}

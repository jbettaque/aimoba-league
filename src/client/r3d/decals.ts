/**
 * Ground decal layer. The flat, top-down graphics of the game (ability telegraphs, range and
 * team rings, aim indicators, ground particles) are still drawn with Canvas 2D in sim
 * coordinates, into an offscreen canvas that covers the visible ground rectangle. That canvas
 * is then draped over the terrain as one transparent, depth-tested quad.
 */
import * as THREE from 'three';
import { LOOK, WORLD } from './look.ts';
import type { Terrain } from './terrain.ts';

export class DecalLayer {
  mesh: THREE.Mesh;
  private canvas = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  private tex: THREE.CanvasTexture;
  private mat: THREE.ShaderMaterial;

  constructor(terrain: Terrain) {
    this.ctx = this.canvas.getContext('2d')!;
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.colorSpace = THREE.NoColorSpace;
    this.tex.flipY = false;
    this.tex.premultiplyAlpha = true;
    this.tex.generateMipmaps = false;
    this.tex.minFilter = this.tex.magFilter = THREE.LinearFilter;
    const hr = terrain.heightRect;
    this.mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
      uniforms: {
        uMap: { value: this.tex },
        uRect: { value: new THREE.Vector4(0, 0, 1, 1) },
        uHeight: { value: terrain.heightTex },
        uHRect: { value: new THREE.Vector3(hr.x * WORLD, hr.y * WORLD, hr.z / WORLD) },
        uLevel: { value: LOOK.terrain.waterLevel },
        uBoost: { value: LOOK.fx.decal },
      },
      vertexShader: /* glsl */ `
        uniform vec4 uRect;
        uniform sampler2D uHeight;
        uniform vec3 uHRect;
        uniform float uLevel;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          vec2 xz = mix(uRect.xy, uRect.zw, uv);
          float h = texture2D(uHeight, (xz - uHRect.xy) * uHRect.z).r;
          gl_Position = projectionMatrix * viewMatrix * vec4(xz.x, max(h, uLevel) + 0.035, xz.y, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap;
        uniform float uBoost;
        varying vec2 vUv;
        void main() {
          vec4 t = texture2D(uMap, vUv);
          if (t.a < 0.004) discard;
          vec3 c = pow(t.rgb / t.a, vec3(2.2));
          gl_FragColor = vec4(c * t.a * uBoost, t.a);
        }`,
    });
    const geo = new THREE.PlaneGeometry(1, 1, 96, 72);
    // PlaneGeometry puts v = 1 at the top; we want uv (0,0) at the rectangle's (x0, y0).
    const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
    this.mesh = new THREE.Mesh(geo, this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    this.mesh.name = 'decals';
  }

  resize(w: number, h: number) {
    const k = Math.min(1, 2048 / Math.max(w, h));
    this.canvas.width = Math.max(64, Math.round(w * k));
    this.canvas.height = Math.max(64, Math.round(h * k));
    this.tex.dispose();
  }

  /** Clear and return a context whose coordinate system is sim units over the given rectangle. */
  begin(x0: number, y0: number, x1: number, y1: number): CanvasRenderingContext2D {
    const c = this.ctx, w = this.canvas.width, h = this.canvas.height;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    c.globalCompositeOperation = 'source-over';
    c.clearRect(0, 0, w, h);
    c.setTransform(w / (x1 - x0), 0, 0, h / (y1 - y0), (-x0 * w) / (x1 - x0), (-y0 * h) / (y1 - y0));
    (this.mat.uniforms.uRect.value as THREE.Vector4).set(x0 * WORLD, y0 * WORLD, x1 * WORLD, y1 * WORLD);
    return c;
  }

  end() {
    this.tex.needsUpdate = true;
  }
}

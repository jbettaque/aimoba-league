/**
 * Fog of war as a world-space visibility texture, plus the shader patch that makes
 * "world" materials (terrain, props, structures) darken where the viewing team cannot see.
 * The same patch optionally adds wind sway for instanced foliage.
 */
import * as THREE from 'three';
import { SIGHT_RANGE } from '../../sim/defs.ts';
import type { Game } from '../../sim/game.ts';
import { MAP_SIZE } from '../../sim/map.ts';
import { LOOK, WORLD } from './look.ts';

const SIZE = 256;

export class Fow {
  private canvas = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  tex: THREE.CanvasTexture;
  /** Shared by every patched material. */
  uniforms = {
    uFowMap: { value: null as THREE.Texture | null },
    uFowParams: { value: new THREE.Vector4(LOOK.fow.dark, LOOK.fow.desat, 1 / (MAP_SIZE * WORLD), 0) },
    uFowTint: { value: new THREE.Vector3(...LOOK.fow.tint) },
    uWindTime: { value: 0 },
  };
  private wasAll = false;

  constructor() {
    this.canvas.width = this.canvas.height = SIZE;
    this.ctx = this.canvas.getContext('2d')!;
    this.ctx.fillStyle = '#fff';
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.colorSpace = THREE.NoColorSpace;
    this.tex.flipY = false;
    this.tex.generateMipmaps = false;
    this.tex.minFilter = this.tex.magFilter = THREE.LinearFilter;
    this.uniforms.uFowMap.value = this.tex;
  }

  /** Repaint visibility for `team` (-1 = spectator, everything visible). */
  update(g: Game, team: number) {
    const c = this.ctx;
    if (team < 0) {
      if (this.wasAll) return;
      c.globalCompositeOperation = 'source-over';
      c.fillStyle = '#fff';
      c.fillRect(0, 0, SIZE, SIZE);
      this.tex.needsUpdate = true;
      this.wasAll = true;
      return;
    }
    this.wasAll = false;
    const k = SIZE / MAP_SIZE;
    c.globalCompositeOperation = 'source-over';
    c.fillStyle = '#000';
    c.fillRect(0, 0, SIZE, SIZE);
    c.globalCompositeOperation = 'lighter';
    for (const u of g.units) {
      if (u.team !== team || !u.alive) continue;
      const r = (u.kind === 'minion' ? 800 : SIGHT_RANGE) * k;
      const x = u.x * k, y = u.y * k;
      const gr = c.createRadialGradient(x, y, r * 0.55, x, y, r);
      gr.addColorStop(0, 'rgba(255,255,255,1)');
      gr.addColorStop(1, 'rgba(255,255,255,0)');
      c.fillStyle = gr;
      c.beginPath();
      c.arc(x, y, r, 0, 6.283);
      c.fill();
    }
    this.tex.needsUpdate = true;
  }

  /**
   * Patch a built-in material so it samples the visibility texture by world XZ.
   * `sway` > 0 additionally bends vertices in the wind (amount in radians at 1 unit height).
   */
  patch<T extends THREE.Material>(mat: T, sway = 0): T {
    const u = this.uniforms;
    const swayAmt = sway;
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uFowMap = u.uFowMap;
      shader.uniforms.uFowParams = u.uFowParams;
      shader.uniforms.uFowTint = u.uFowTint;
      shader.uniforms.uWindTime = u.uWindTime;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vFowXZ;\nuniform float uWindTime;')
        .replace(
          '#include <project_vertex>',
          `${swayAmt > 0 ? `
          {
            vec3 swOrigin = vec3(0.0);
            #ifdef USE_INSTANCING
              swOrigin = instanceMatrix[3].xyz;
            #endif
            float swPhase = swOrigin.x * 1.7 + swOrigin.z * 2.3;
            float swH = max(transformed.y, 0.0);
            float swA = ${swayAmt.toFixed(4)} * swH;
            transformed.x += swA * (sin(uWindTime * 1.3 + swPhase) + 0.35 * sin(uWindTime * 3.1 + swPhase * 1.7 + transformed.y * 2.0));
            transformed.z += swA * 0.6 * sin(uWindTime * 1.1 + swPhase * 1.3 + 1.7);
          }` : ''}
          #include <project_vertex>
          {
            vec4 fowWp = vec4(transformed, 1.0);
            #ifdef USE_INSTANCING
              fowWp = instanceMatrix * fowWp;
            #endif
            fowWp = modelMatrix * fowWp;
            vFowXZ = fowWp.xz;
          }`,
        );
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying vec2 vFowXZ;\nuniform sampler2D uFowMap;\nuniform vec4 uFowParams;\nuniform vec3 uFowTint;')
        .replace(
          '#include <opaque_fragment>',
          `#include <opaque_fragment>
          {
            float fowVis = texture2D(uFowMap, clamp(vFowXZ * uFowParams.z, 0.0, 1.0)).r;
            fowVis = smoothstep(0.05, 0.85, fowVis);
            float fowL = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));
            vec3 fowC = mix(gl_FragColor.rgb, vec3(fowL), uFowParams.y) * uFowParams.x * uFowTint;
            gl_FragColor.rgb = mix(fowC, gl_FragColor.rgb, fowVis);
          }`,
        );
    };
    mat.customProgramCacheKey = () => `fow|${swayAmt.toFixed(4)}`;
    mat.needsUpdate = true;
    return mat;
  }
}

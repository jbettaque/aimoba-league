/**
 * Every art-direction knob of the 3D renderer lives here: camera, lights, tone mapping,
 * bloom, fog, fog of war, terrain palette, water and FX intensity.
 *
 * World scale: 1 three.js unit = 100 simulation units. X = sim x, Z = sim y, Y is up.
 */
export const WORLD = 0.01;

export const LOOK = {
  camera: {
    fov: 32,
    /** Degrees above the horizon the camera looks down from. */
    pitch: 55,
    /** Screen pixels per sim unit at the focus point, at zoomFactor 1 on a 1080p screen. */
    baseZoom: 0.62,
    near: 2,
    far: 420,
  },

  /** Flat UI colours (bars, rings, minimap). */
  teamColors: ['#4aa8ff', '#ff5d5d', '#c9b37a'],
  /** Colours pushed into materials named `team` / `team_glow` (lit, so a little richer). */
  teamModel: ['#2f86f0', '#ee4a42', '#c9a45e'],
  /** Emissive colour of `team_glow` parts. Deeper than the UI colours: tone mapping lifts bright emissives toward white. */
  teamGlow: ['#2b8cff', '#ff3d2e', '#ffb84a'],

  background: '#141c24',
  exposure: 1.06,
  /** Final colour grade, applied in scene-linear before tone mapping. 1 = unchanged. */
  grade: { saturation: 1.05, contrast: 1.1 },

  /** Warm low key light. `dir` points from the scene towards the sun. */
  sun: {
    color: '#ffd9a6',
    intensity: 3.1,
    dir: [-0.62, 0.7, 0.16] as [number, number, number],
    shadowMapSize: 4096,
    /** Half-size (world units) of the shadow box at zoomFactor 1; scales with 1/zoom. */
    shadowExtent: 24,
    shadowBias: -0.0006,
    shadowNormalBias: 0.035,
    shadowRadius: 2.2,
  },
  /** Cool teal fill from the sky, mossy bounce from the ground. */
  hemi: { sky: '#8fc4d6', ground: '#3d4a2e', intensity: 1.15 },
  ambient: { color: '#5b6f88', intensity: 0.25 },
  /** Faint cool rim from behind-right, so silhouettes separate from the ground. */
  rim: { color: '#7fb6ff', intensity: 0.55, dir: [0.55, 0.45, -0.7] as [number, number, number] },

  /** Distance fog, expressed as multiples of the camera distance. */
  fog: { color: '#1b2a33', near: 1.25, far: 4.2 },

  /**
   * Only HDR values whose brightest channel exceeds `threshold` bloom (soft knee of width `knee`):
   * emissive (`glow*`, `team_glow`) materials and FX. Sunlit surfaces stay below it.
   */
  bloom: { strength: 0.6, radius: 0.55, threshold: 1.12, knee: 0.4 },
  /** Multiplier on the emissive strength authored in the models (art/blender/kit.py GLOW_SCALE). */
  glow: { emissive: 1.2 },

  /** Fog of war: unseen ground is darkened, desaturated and tinted. */
  fow: { dark: 0.36, desat: 0.6, tint: [0.72, 0.86, 1.08] as [number, number, number] },

  terrain: {
    /** Height-field vertices per walkability cell. */
    res: 3,
    /** How far (sim units) the terrain continues beyond the map as raised forest. */
    margin: 1600,
    textureSize: 4096,
    bankHeight: 0.62,
    plateauRise: 0.5,
    noiseAmp: 0.28,
    riverDepth: 0.24,
    poolDepth: 0.17,
    waterLevel: -0.06,
    roughness: 0.96,
    cliffTint: '#8a7358',
    forestTint: '#b9c8a0',
    palette: {
      forest: '#33472b',
      grassDark: '#48682f',
      grass: '#5f8a3c',
      grassLight: '#86ab4d',
      path: '#b89b68',
      pathDark: '#8f7650',
      pathLight: '#d6bf8e',
      riverBed: '#6f8a80',
      riverBedDeep: '#3f6a72',
      camp: '#6b5a40',
      stone: '#9aa0a6',
    },
  },

  water: {
    shallow: '#58c8c4',
    deep: '#1c6f8e',
    foam: '#e9fbf6',
    opacity: 0.86,
    /** Emissive-ish lift so the river reads at dusk. */
    glow: 0.35,
  },

  fx: {
    /** HDR multiplier for additive particles / bolts (drives how much they bloom). */
    glow: 1.45,
    /** Brightness of the 2D ground decal layer (telegraphs, rings). */
    decal: 1.25,
  },

  /** Edge (world units) of the square chunks props are instanced in (smaller = better culling, more draw calls). */
  props: { chunk: 14 },
};

/** Animation constants for the procedural animator (see anim.ts). */
export const ANIM = {
  /** Walk cycles per world unit travelled, by rig kind. */
  stride: { biped: 0.72, quad: 0.85, serpent: 0.5, cart: 1, float: 0.4 },
  legSwing: 0.62,
  armSwing: 0.5,
  bob: 0.045,
  torsoTwist: 0.09,
  lean: 0.1,
  breathe: 0.012,
  breatheSpeed: 2.1,
  /** Speed (world units / s) at which the walk blend is fully on. */
  walkSpeed: 1.2,
  deathTime: 0.4,
  facingRate: 14,
};

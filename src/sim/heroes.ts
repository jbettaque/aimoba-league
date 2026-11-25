/** Hero roster: stats, abilities and all ability effect handlers. */
import type { Game } from './game.ts';
import { MAGIC, PHYS, type Dash, type Projectile, type Unit, type Zone } from './defs.ts';
import { nearestWalkable, rayClamp } from './map.ts';
import { atan2, cos, hypot, sin, sq } from './dmath.ts';

export type Targeting = 'dir' | 'point' | 'enemy' | 'enemyHero' | 'ally' | 'self';

export interface AbilityDef {
  name: string;
  desc: string;
  targeting: Targeting;
  range: number;
  /** Area radius (point / self AoE) or projectile half-width (dir) for indicators. */
  radius: number;
  cd: number[];
  cost: number[];
  icon: string;
  cast(g: Game, u: Unit, rank: number, tx: number, ty: number, target: Unit | null): void;
}

export interface HeroDef {
  id: string;
  name: string;
  title: string;
  role: string;
  lane: number; // 0 top, 1 mid, 2 bot, -1 jungle
  color: string;
  accent: string;
  blurb: string;
  passive: { name: string; desc: string };
  hp: number; hpL: number;
  mana: number; manaL: number;
  ad: number; adL: number;
  armor: number; armorL: number;
  mr: number; mrL: number;
  as: number; asL: number;
  range: number;
  ms: number;
  hpRegen: number;
  manaRegen: number;
  abilities: AbilityDef[];
  /** Order in which bots / auto-level max basic abilities (ultimate always first). */
  skillOrder: number[];
  /** Recommended purchase order. A component entry is an early buy, satisfied once owned or built into something owned. */
  build: string[];
  /** Other builds for this hero. Each game every hero follows one of them (or `build`), so games differ. */
  altBuilds: string[][];
}

const R = (a: number[], rank: number) => a[Math.min(a.length, Math.max(1, rank)) - 1];

function aim(u: Unit, tx: number, ty: number): [number, number] {
  let dx = tx - u.x, dy = ty - u.y;
  const d = hypot(dx, dy);
  if (d < 1) {
    dx = cos(u.facing);
    dy = sin(u.facing);
  } else {
    dx /= d;
    dy /= d;
  }
  u.facing = atan2(dy, dx);
  return [dx, dy];
}

function clampPoint(u: Unit, tx: number, ty: number, range: number): [number, number] {
  const dx = tx - u.x, dy = ty - u.y;
  const d = hypot(dx, dy);
  if (d > range) return [u.x + (dx / d) * range, u.y + (dy / d) * range];
  return [tx, ty];
}

function skillshot(g: Game, u: Unit, kind: string, rank: number, tx: number, ty: number, speed: number, range: number, radius: number, dmg: number, opts: Partial<Projectile> = {}) {
  const [dx, dy] = aim(u, tx, ty);
  g.spawnProjectile({ kind, srcId: u.id, team: u.team, x: u.x + dx * 30, y: u.y + dy * 30, dx, dy, speed, dist: range, radius, dmg, rank, ...opts });
}

// ---------------------------------------------------------------------------
// Korvath, the Bastion — tank / initiator
// ---------------------------------------------------------------------------
const KORVATH: HeroDef = {
  id: 'korvath', name: 'Korvath', title: 'the Bastion', role: 'Vanguard', lane: 0,
  color: '#d9a441', accent: '#fff1c2',
  blurb: 'An unbreakable frontliner who starts fights and soaks damage for the team.',
  passive: { name: 'Stoneheart', desc: 'Basic attacks deal bonus magic damage equal to 1.5% of his maximum health.' },
  hp: 690, hpL: 98, mana: 280, manaL: 35, ad: 62, adL: 3.5, armor: 38, armorL: 4.2, mr: 32, mrL: 2.2,
  as: 0.66, asL: 0.02, range: 150, ms: 340, hpRegen: 2.2, manaRegen: 1.7,
  skillOrder: [2, 0, 1], build: ['vest', 'boots', 'plate', 'mantle', 'cleaver', 'colossus', 'chalice'],
  altBuilds: [['blade', 'boots', 'cleaver', 'plate', 'mantle', 'colossus', 'fang'], ['vest', 'boots', 'colossus', 'plate', 'mantle', 'chalice', 'cleaver']],
  abilities: [
    {
      name: 'Shield Rush', icon: 'rush', targeting: 'dir', range: 520, radius: 110,
      desc: 'Charges forward, stopping at the first enemy hit. Deals 60/95/130/165 (+50% AD) physical damage and stuns for 1s.',
      cd: [11, 10, 9, 8], cost: [45, 45, 45, 45],
      cast(g, u, rank, tx, ty) {
        const [dx, dy] = aim(u, tx, ty);
        const [ex, ey] = rayClamp(u.x, u.y, u.x + dx * 520, u.y + dy * 520);
        g.startDash(u, ex, ey, 1300, 'korvQ', rank, -1);
      },
    },
    {
      name: 'Iron Resolve', icon: 'shield', targeting: 'self', range: 0, radius: 0,
      desc: 'Gains a shield for 70/105/140/175 (+10% max health) and 15/20/25/30 armor and magic resist for 4s.',
      cd: [16, 15, 14, 13], cost: [50, 50, 50, 50],
      cast(g, u, rank) {
        g.addShield(u, u, R([70, 105, 140, 175], rank) + 0.1 * u.maxHp, 4);
        g.addBuff(u, 'resist', 4, R([15, 20, 25, 30], rank), u.id);
        g.fx('shield', u.x, u.y, 90, u.team, u.id);
      },
    },
    {
      name: 'Tremor', icon: 'quake', targeting: 'self', range: 0, radius: 330,
      desc: 'Slams the ground, dealing 70/110/150/190 (+30% AP, +3% max health) magic damage to nearby enemies and slowing them by 40% for 2s.',
      cd: [9, 8.5, 8, 7.5], cost: [50, 50, 50, 50],
      cast(g, u, rank) {
        const dmg = R([70, 110, 150, 190], rank) + 0.3 * u.ap + 0.03 * u.maxHp;
        for (const e of g.enemiesIn(u.team, u.x, u.y, 330)) {
          g.damage(u, e, dmg, MAGIC, 2);
          g.addBuff(e, 'slow', 2, 0.4, u.id);
        }
        g.fx('quake', u.x, u.y, 330, u.team);
      },
    },
    {
      name: 'Cataclysm', icon: 'meteor', targeting: 'point', range: 750, radius: 400,
      desc: 'Leaps to the target location. On landing deals 200/325/450 (+50% AP, +6% max health) magic damage and stuns enemies for 1.25s.',
      cd: [110, 95, 80], cost: [100, 100, 100],
      cast(g, u, rank, tx, ty) {
        const [cx, cy] = clampPoint(u, tx, ty, 750);
        const [ex, ey] = nearestWalkable(cx, cy);
        aim(u, ex, ey);
        g.fx('telegraph', ex, ey, 400, u.team);
        g.startDash(u, ex, ey, 1500, 'korvR', rank, -1);
      },
    },
  ],
};

// ---------------------------------------------------------------------------
// Sable, the Nightblade — assassin / jungler
// ---------------------------------------------------------------------------
const SABLE: HeroDef = {
  id: 'sable', name: 'Sable', title: 'the Nightblade', role: 'Assassin', lane: -1,
  color: '#9b6dff', accent: '#e6d9ff',
  blurb: 'A jungle predator who stalks from stealth and executes wounded targets.',
  passive: { name: 'Predator', desc: 'Deals 12% more damage to targets below 40% health. Hero takedowns refund 60% of the cooldowns of Shadow Strike and Final Hour.' },
  hp: 610, hpL: 90, mana: 260, manaL: 35, ad: 64, adL: 3.8, armor: 31, armorL: 3.7, mr: 30, mrL: 1.7,
  as: 0.7, asL: 0.03, range: 150, ms: 350, hpRegen: 2.0, manaRegen: 1.7,
  skillOrder: [0, 1, 2], build: ['blade', 'boots', 'edge', 'fang', 'cleaver', 'mantle', 'plate'],
  altBuilds: [['blade', 'boots', 'fang', 'edge', 'stormbow', 'mantle', 'plate'], ['blade', 'boots', 'cleaver', 'edge', 'plate', 'mantle', 'fang']],
  abilities: [
    {
      name: 'Shadow Strike', icon: 'dagger', targeting: 'enemy', range: 500, radius: 0,
      desc: 'Dashes to an enemy, dealing 65/100/135/170 (+70% AD) physical damage. Killing the target halves the cooldown.',
      cd: [8, 7.5, 7, 6.5], cost: [40, 40, 40, 40],
      cast(g, u, rank, _tx, _ty, target) {
        if (target) g.startDash(u, target.x, target.y, 1600, 'sableQ', rank, target.id);
      },
    },
    {
      name: 'Bloodlust', icon: 'claw', targeting: 'self', range: 0, radius: 0,
      desc: 'For 5s gains 35/50/65/80% attack speed and 10/13/16/19% lifesteal.',
      cd: [15, 14, 13, 12], cost: [40, 40, 40, 40],
      cast(g, u, rank) {
        g.addBuff(u, 'as', 5, R([0.35, 0.5, 0.65, 0.8], rank), u.id);
        g.addBuff(u, 'lifesteal', 5, R([0.1, 0.13, 0.16, 0.19], rank), u.id);
        g.fx('buff', u.x, u.y, 80, u.team, u.id);
      },
    },
    {
      name: 'Shadow Veil', icon: 'veil', targeting: 'self', range: 0, radius: 0,
      desc: 'Vanishes for 3s, gaining 30% move speed. Her next attack within 5s deals 40/70/100/130 (+40% AD) bonus damage. Attacking or casting ends the stealth.',
      cd: [18, 17, 16, 15], cost: [50, 50, 50, 50],
      cast(g, u, rank) {
        g.addBuff(u, 'stealth', 3, 1, u.id);
        g.addBuff(u, 'haste', 3, 0.3, u.id);
        g.addBuff(u, 'empower', 5, R([40, 70, 100, 130], rank) + 0.4 * u.ad, u.id);
        g.fx('smoke', u.x, u.y, 120, u.team);
      },
    },
    {
      name: 'Final Hour', icon: 'skull', targeting: 'enemyHero', range: 600, radius: 0,
      desc: 'Lunges at an enemy hero for 150/250/350 (+80% AD) physical damage, increased by up to 100% against low-health targets.',
      cd: [90, 75, 60], cost: [100, 100, 100],
      cast(g, u, rank, _tx, _ty, target) {
        if (target) g.startDash(u, target.x, target.y, 1800, 'sableR', rank, target.id);
      },
    },
  ],
};

// ---------------------------------------------------------------------------
// Ignara, the Emberweaver — burst mage
// ---------------------------------------------------------------------------
const IGNARA: HeroDef = {
  id: 'ignara', name: 'Ignara', title: 'the Emberweaver', role: 'Mage', lane: 1,
  color: '#ff6a3d', accent: '#ffe0b3',
  blurb: 'A long-range spellcaster who controls the mid lane with explosive area damage.',
  passive: { name: 'Kindle', desc: 'Ability damage ignites enemies, burning them for 3s for magic damage each second.' },
  hp: 525, hpL: 80, mana: 400, manaL: 45, ad: 52, adL: 2.8, armor: 21, armorL: 3.3, mr: 28, mrL: 1.1,
  as: 0.63, asL: 0.015, range: 550, ms: 330, hpRegen: 1.5, manaRegen: 2.4,
  skillOrder: [0, 1, 2], build: ['tome', 'boots', 'orb', 'staff', 'prism', 'scepter', 'mantle'],
  altBuilds: [['tome', 'boots', 'staff', 'prism', 'orb', 'scepter', 'mantle'], ['tome', 'boots', 'scepter', 'orb', 'staff', 'prism', 'plate']],
  abilities: [
    {
      name: 'Fire Lance', icon: 'bolt', targeting: 'dir', range: 950, radius: 70,
      desc: 'Hurls a lance of flame that deals 80/125/170/215 (+75% AP) magic damage to the first enemy hit.',
      cd: [6, 5.5, 5, 4.5], cost: [50, 55, 60, 65],
      cast(g, u, rank, tx, ty) {
        skillshot(g, u, 'ignQ', rank, tx, ty, 1500, 950, 70, R([80, 125, 170, 215], rank) + 0.75 * u.ap);
      },
    },
    {
      name: 'Scorch Field', icon: 'flame', targeting: 'point', range: 850, radius: 230,
      desc: 'After a short delay the ground erupts, dealing 90/140/190/240 (+70% AP) magic damage in an area.',
      cd: [10, 9.5, 9, 8.5], cost: [70, 70, 70, 70],
      cast(g, u, rank, tx, ty) {
        const [x, y] = clampPoint(u, tx, ty, 850);
        aim(u, x, y);
        g.spawnZone({ kind: 'ignW', srcId: u.id, team: u.team, x, y, radius: 230, delay: 0.65, rank });
      },
    },
    {
      name: 'Cinder Snare', icon: 'snare', targeting: 'dir', range: 850, radius: 80,
      desc: 'Launches a snare that roots the first enemy hit for 1.2/1.4/1.6/1.8s and deals 50/75/100/125 (+40% AP) magic damage.',
      cd: [14, 13, 12, 11], cost: [60, 60, 60, 60],
      cast(g, u, rank, tx, ty) {
        skillshot(g, u, 'ignE', rank, tx, ty, 1150, 850, 80, R([50, 75, 100, 125], rank) + 0.4 * u.ap);
      },
    },
    {
      name: 'Starfall', icon: 'meteor', targeting: 'point', range: 950, radius: 400,
      desc: 'Calls down a falling star. After 1s it deals 300/450/600 (+90% AP) magic damage in a large area.',
      cd: [100, 85, 70], cost: [100, 100, 100],
      cast(g, u, rank, tx, ty) {
        const [x, y] = clampPoint(u, tx, ty, 950);
        aim(u, x, y);
        g.spawnZone({ kind: 'ignR', srcId: u.id, team: u.team, x, y, radius: 400, delay: 1.0, rank });
      },
    },
  ],
};

// ---------------------------------------------------------------------------
// Vesper, the Stormshot — marksman
// ---------------------------------------------------------------------------
const VESPER: HeroDef = {
  id: 'vesper', name: 'Vesper', title: 'the Stormshot', role: 'Marksman', lane: 2,
  color: '#3ecf8e', accent: '#d3ffe9',
  blurb: 'A fragile ranged carry whose sustained attacks melt enemies late in the game.',
  passive: { name: 'Focus', desc: 'Consecutive attacks against the same target grant 6% attack speed, stacking up to 5 times.' },
  hp: 545, hpL: 84, mana: 300, manaL: 35, ad: 60, adL: 3.5, armor: 24, armorL: 3.4, mr: 28, mrL: 1.1,
  as: 0.68, asL: 0.032, range: 575, ms: 330, hpRegen: 1.5, manaRegen: 1.8,
  skillOrder: [0, 1, 2], build: ['blade', 'boots', 'stormbow', 'edge', 'fang', 'mantle', 'plate'],
  altBuilds: [['blade', 'boots', 'edge', 'stormbow', 'cleaver', 'mantle', 'plate'], ['blade', 'boots', 'stormbow', 'fang', 'edge', 'plate', 'mantle']],
  abilities: [
    {
      name: 'Piercing Bolt', icon: 'arrow', targeting: 'dir', range: 1000, radius: 65,
      desc: 'Fires a bolt that pierces enemies for 50/85/120/155 (+100% AD) physical damage, 15% less per enemy hit.',
      cd: [8, 7.5, 7, 6.5], cost: [50, 50, 50, 50],
      cast(g, u, rank, tx, ty) {
        skillshot(g, u, 'vesQ', rank, tx, ty, 1700, 1000, 65, R([50, 85, 120, 155], rank) + u.ad, { pierce: true });
      },
    },
    {
      name: 'Rapid Fire', icon: 'volley', targeting: 'self', range: 0, radius: 0,
      desc: 'For 5s gains 40/55/70/85% attack speed and 75 attack range.',
      cd: [18, 17, 16, 15], cost: [50, 50, 50, 50],
      cast(g, u, rank) {
        g.addBuff(u, 'as', 5, R([0.4, 0.55, 0.7, 0.85], rank), u.id);
        g.addBuff(u, 'range', 5, 75, u.id);
        g.fx('buff', u.x, u.y, 80, u.team, u.id);
      },
    },
    {
      name: 'Gale Step', icon: 'wind', targeting: 'dir', range: 400, radius: 0,
      desc: 'Dashes a short distance. Her next attack within 4s deals 30/50/70/90 (+30% AD) bonus damage.',
      cd: [14, 12.5, 11, 9.5], cost: [40, 40, 40, 40],
      cast(g, u, rank, tx, ty) {
        const [dx, dy] = aim(u, tx, ty);
        const [ex, ey] = rayClamp(u.x, u.y, u.x + dx * 400, u.y + dy * 400);
        g.addBuff(u, 'empower', 4, R([30, 50, 70, 90], rank) + 0.3 * u.ad, u.id);
        g.startDash(u, ex, ey, 1400, 'vesE', rank, -1);
      },
    },
    {
      name: 'Thunder Arrow', icon: 'thunder', targeting: 'dir', range: 5000, radius: 110,
      desc: 'Looses a huge arrow across the map. The first enemy hero hit takes 250/375/500 (+100% AD) magic damage and is stunned for 1s; nearby enemies take half.',
      cd: [90, 80, 70], cost: [100, 100, 100],
      cast(g, u, rank, tx, ty) {
        skillshot(g, u, 'vesR', rank, tx, ty, 2200, 5000, 110, R([250, 375, 500], rank) + u.ad, { heroOnly: true });
      },
    },
  ],
};

// ---------------------------------------------------------------------------
// Liora, the Dawnkeeper — support / enchanter
// ---------------------------------------------------------------------------
const LIORA: HeroDef = {
  id: 'liora', name: 'Liora', title: 'the Dawnkeeper', role: 'Support', lane: 2,
  color: '#ffd166', accent: '#fff8dc',
  blurb: 'A guardian who keeps allies alive with heals and shields and locks down threats.',
  passive: { name: 'Radiance', desc: 'Her heals and shields also grant the target 20% move speed for 2s.' },
  hp: 535, hpL: 82, mana: 380, manaL: 45, ad: 50, adL: 2.6, armor: 25, armorL: 3.4, mr: 28, mrL: 1.1,
  as: 0.63, asL: 0.015, range: 525, ms: 335, hpRegen: 1.6, manaRegen: 2.6,
  skillOrder: [0, 2, 1], build: ['tome', 'boots', 'chalice', 'orb', 'mantle', 'scepter', 'plate'],
  altBuilds: [['tome', 'boots', 'chalice', 'scepter', 'orb', 'plate', 'mantle'], ['tome', 'boots', 'orb', 'chalice', 'staff', 'mantle', 'plate']],
  abilities: [
    {
      name: 'Mending Light', icon: 'heal', targeting: 'ally', range: 750, radius: 0,
      desc: 'Heals an allied hero (or herself) for 70/110/150/190 (+45% AP).',
      cd: [8, 7.5, 7, 6.5], cost: [60, 70, 80, 90],
      cast(g, u, rank, _tx, _ty, target) {
        const t = target ?? u;
        g.heal(u, t, R([70, 110, 150, 190], rank) + 0.45 * u.ap);
        g.addBuff(t, 'haste', 2, 0.2, u.id);
        g.fx('heal', t.x, t.y, 90, u.team, t.id);
        if (t !== u) g.fx('beam', u.x, u.y, 0, u.team, t.id, t.x, t.y);
      },
    },
    {
      name: 'Luminous Chain', icon: 'chain', targeting: 'dir', range: 900, radius: 80,
      desc: 'Fires a bolt of light. The first enemy hit takes 60/100/140/180 (+50% AP) magic damage and is slowed by 50% for 2s.',
      cd: [10, 9.5, 9, 8.5], cost: [60, 60, 60, 60],
      cast(g, u, rank, tx, ty) {
        skillshot(g, u, 'lioW', rank, tx, ty, 1300, 900, 80, R([60, 100, 140, 180], rank) + 0.5 * u.ap);
      },
    },
    {
      name: 'Aegis', icon: 'aegis', targeting: 'ally', range: 750, radius: 0,
      desc: 'Shields an allied hero (or herself) for 70/110/150/190 (+40% AP) for 3.5s.',
      cd: [12, 11, 10, 9], cost: [60, 60, 60, 60],
      cast(g, u, rank, _tx, _ty, target) {
        const t = target ?? u;
        g.addShield(u, t, R([70, 110, 150, 190], rank) + 0.4 * u.ap, 3.5);
        g.addBuff(t, 'haste', 2, 0.2, u.id);
        g.fx('shield', t.x, t.y, 90, u.team, t.id);
      },
    },
    {
      name: 'Sanctuary', icon: 'sun', targeting: 'self', range: 0, radius: 700,
      desc: 'Heals all nearby allied heroes for 150/250/350 (+60% AP), removes stuns, roots and slows, and grants 20/30/40 armor and magic resist for 4s.',
      cd: [120, 105, 90], cost: [100, 100, 100],
      cast(g, u, rank) {
        for (const a of g.heroes) {
          if (a.team !== u.team || !a.alive) continue;
          if (sq(a.x - u.x) + sq(a.y - u.y) > 700 * 700) continue;
          g.heal(u, a, R([150, 250, 350], rank) + 0.6 * u.ap);
          a.buffs = a.buffs.filter((b) => b.kind !== 'stun' && b.kind !== 'root' && b.kind !== 'slow');
          g.addBuff(a, 'resist', 4, R([20, 30, 40], rank), u.id);
          g.fx('heal', a.x, a.y, 90, u.team, a.id);
        }
        g.fx('sanctuary', u.x, u.y, 700, u.team);
      },
    },
  ],
};

export const HEROES: HeroDef[] = [KORVATH, SABLE, IGNARA, VESPER, LIORA];
export const HERO_KORVATH = 0, HERO_SABLE = 1, HERO_IGNARA = 2, HERO_VESPER = 3, HERO_LIORA = 4;

// ---------------------------------------------------------------------------
// Effect handlers (kept out of projectile / zone state so state stays plain data)
// ---------------------------------------------------------------------------
export const PROJ_HIT: Record<string, (g: Game, p: Projectile, src: Unit | undefined, t: Unit) => void> = {
  ignQ(g, p, src, t) {
    g.damage(src, t, p.dmg, MAGIC, 2);
    g.fx('burst', t.x, t.y, 90, p.team);
  },
  ignE(g, p, src, t) {
    g.damage(src, t, p.dmg, MAGIC, 2);
    g.addBuff(t, 'root', [1.2, 1.4, 1.6, 1.8][p.rank - 1] ?? 1.2, 1, p.srcId);
    g.fx('snare', t.x, t.y, 80, p.team, t.id);
  },
  vesQ(g, p, src, t) {
    const falloff = Math.max(0.55, 1 - 0.15 * p.hits.length);
    g.damage(src, t, p.dmg * falloff, PHYS, 2);
    g.fx('spark', t.x, t.y, 50, p.team);
  },
  vesR(g, p, src, t) {
    g.damage(src, t, p.dmg, MAGIC, 2);
    g.addBuff(t, 'stun', 1, 1, p.srcId);
    for (const e of g.enemiesIn(p.team, t.x, t.y, 280)) if (e !== t) g.damage(src, e, p.dmg * 0.5, MAGIC, 2);
    g.fx('thunder', t.x, t.y, 280, p.team);
  },
  lioW(g, p, src, t) {
    g.damage(src, t, p.dmg, MAGIC, 2);
    g.addBuff(t, 'slow', 2, 0.5, p.srcId);
    g.addBuff(t, 'reveal', 2, 1, p.srcId);
    g.fx('flash', t.x, t.y, 80, p.team);
  },
};

export const ZONE_HIT: Record<string, (g: Game, z: Zone, src: Unit | undefined) => void> = {
  ignW(g, z, src) {
    const dmg = ([90, 140, 190, 240][z.rank - 1] ?? 90) + 0.7 * (src ? src.ap : 0);
    for (const e of g.enemiesIn(z.team, z.x, z.y, z.radius)) g.damage(src, e, dmg, MAGIC, 2);
    g.fx('explosion', z.x, z.y, z.radius, z.team);
  },
  ignR(g, z, src) {
    const dmg = ([300, 450, 600][z.rank - 1] ?? 300) + 0.9 * (src ? src.ap : 0);
    for (const e of g.enemiesIn(z.team, z.x, z.y, z.radius)) g.damage(src, e, dmg, MAGIC, 2);
    g.fx('starfall', z.x, z.y, z.radius, z.team);
  },
};

/** Called every tick while dashing; return true to end the dash early. */
export const DASH_TICK: Record<string, (g: Game, u: Unit, d: Dash) => boolean> = {
  korvQ(g, u, d) {
    for (const e of g.enemiesIn(u.team, u.x, u.y, 110)) {
      g.damage(u, e, ([60, 95, 130, 165][d.rank - 1] ?? 60) + 0.5 * u.ad, PHYS, 2);
      g.addBuff(e, 'stun', 1, 1, u.id);
      g.fx('impact', e.x, e.y, 100, u.team);
      return true;
    }
    return false;
  },
};

export const DASH_END: Record<string, (g: Game, u: Unit, d: Dash, target: Unit | undefined) => void> = {
  korvR(g, u, d) {
    const dmg = ([200, 325, 450][d.rank - 1] ?? 200) + 0.5 * u.ap + 0.06 * u.maxHp;
    for (const e of g.enemiesIn(u.team, u.x, u.y, 400)) {
      g.damage(u, e, dmg, MAGIC, 2);
      g.addBuff(e, 'stun', 1.25, 1, u.id);
    }
    g.fx('cataclysm', u.x, u.y, 400, u.team);
  },
  sableQ(g, u, d, t) {
    if (!t || !t.alive) return;
    const h = u.hero!;
    h.refund = [0, g.tick + 2, t.id];
    g.damage(u, t, ([65, 100, 135, 170][d.rank - 1] ?? 65) + 0.7 * u.ad, PHYS, 2);
    g.fx('slash', t.x, t.y, 90, u.team);
    if (t.alive) g.orderAttack(u, t);
  },
  sableR(g, u, d, t) {
    if (!t || !t.alive) return;
    const missing = 1 - t.hp / t.maxHp;
    g.damage(u, t, (([150, 250, 350][d.rank - 1] ?? 150) + 0.8 * u.ad) * (1 + missing), PHYS, 2);
    g.fx('execute', t.x, t.y, 130, u.team);
    if (t.alive) g.orderAttack(u, t);
  },
};

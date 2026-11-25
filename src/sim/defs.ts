/** Core data types and balance tables shared by the whole simulation. */
export const TICK_RATE = 30;
export const DT = 1 / TICK_RATE;

export const TEAM_BLUE = 0;
export const TEAM_RED = 1;
export const TEAM_NEUTRAL = 2;

export const PHYS = 0;
export const MAGIC = 1;
export const TRUE_DMG = 2;

export const MAX_LEVEL = 15;
export const MAX_ITEMS = 6;
export const SIGHT_RANGE = 1050;
export const XP_RANGE = 1300;

export type UnitKind = 'hero' | 'minion' | 'monster' | 'tower' | 'core';

export type BuffKind =
  | 'stun' | 'root' | 'slow' | 'haste' | 'shield' | 'as' | 'ad' | 'ap' | 'resist' | 'lifesteal'
  | 'dot' | 'hot' | 'stealth' | 'empower' | 'range' | 'might' | 'insight' | 'titan' | 'reveal';

export interface Buff {
  kind: BuffKind;
  /** Remaining duration in seconds. */
  t: number;
  val: number;
  /** Unit id of whoever applied it (-1 for none). */
  src: number;
}

export interface Order {
  type: 'idle' | 'move' | 'amove' | 'attack' | 'cast' | 'recall';
  x: number;
  y: number;
  targetId: number;
  slot: number;
}

/** Commands are the only way controllers (humans, bots, ML agents) influence a hero. */
export type Command =
  | { t: 'move'; x: number; y: number }
  | { t: 'amove'; x: number; y: number }
  | { t: 'attack'; id: number }
  | { t: 'stop' }
  | { t: 'cast'; slot: number; x: number; y: number; id?: number }
  | { t: 'recall' }
  | { t: 'level'; slot: number }
  | { t: 'buy'; item: string };

export interface Dash {
  x: number;
  y: number;
  speed: number;
  kind: string;
  rank: number;
  targetId: number;
}

export class HeroState {
  def: number; // index into HEROES
  slot: number; // 0-4 blue, 5-9 red
  level = 1;
  xp = 0;
  gold = 650;
  totalGold = 650;
  /** Gold earned through play (kills, last hits, objectives) - excludes passive income. */
  earnedGold = 0;
  kills = 0;
  deaths = 0;
  assists = 0;
  cs = 0;
  ranks = [0, 0, 0, 0];
  cds = [0, 0, 0, 0];
  skillPoints = 1;
  items: string[] = [];
  /** The item build this hero follows when shopping automatically (one of its hero's builds, drawn per game). */
  build: string[] = [];
  /** Which of the hero's builds that is (0 = the default). */
  buildIdx = 0;
  respawn = 0;
  order: Order = { type: 'idle', x: 0, y: 0, targetId: -1, slot: 0 };
  recallT = 0;
  castLock = 0;
  isBot = true;
  /** Which scripted AI drives this hero when isBot: 'easy', 'basic' or 'expert'. */
  botKind = 'basic';
  /** Private scratch memory of the expert bot. */
  botMem: unknown = null;
  autoLevel = true;
  autoBuy = true;
  /** Auto-attack enemies in range while idle (on for human players). */
  autoAttack = false;
  /** Tick at which each hero slot last damaged us (assist tracking). */
  dmgBy: number[] = new Array(10).fill(-9999);
  /** Tick we last damaged an enemy hero (tower aggro). */
  aggroTick = -9999;
  focusId = -1;
  focusStacks = 0;
  /** Pending cooldown-refund window: [slot, untilTick, targetId]. */
  refund: number[] = [-1, 0, -1];
  // Item effects.
  /** Attacks landed, for effects that trigger every few attacks. */
  hitCount = 0;
  /** Until this tick the next attack carries the spellblade bonus (-1 = not primed). */
  spellbladeUntil = -1;
  /** Tick from which the spell ward can trigger again. */
  wardAt = 0;
  // Lifetime stats (also used for ML rewards).
  dmgDealt = 0;
  dmgTaken = 0;
  healDone = 0;
  towerDmg = 0;
  // Aggregated item stats.
  it: ItemStats = emptyItemStats();
  /** Total gold value of the inventory (components count at their price). */
  itemGold = 0;
  // Bot scratch state.
  botState = '';
  botNextThink = 0;
  botCamp = -1;
  /** Per-game personality: >1 takes riskier fights. */
  botAggro = 1;
  constructor(def: number, slot: number) {
    this.def = def;
    this.slot = slot;
  }
}

export class Unit {
  id = 0;
  kind: UnitKind = 'minion';
  team = 0;
  x = 0;
  y = 0;
  px = 0; // previous-tick position (render interpolation)
  py = 0;
  vx = 0;
  vy = 0;
  radius = 40;
  facing = 0;
  alive = true;
  hp = 1;
  maxHp = 1;
  mana = 0;
  maxMana = 0;
  // Derived stats.
  ad = 0;
  ap = 0;
  armor = 0;
  mr = 0;
  as = 0.7;
  range = 120;
  ms = 300;
  hpRegen = 0;
  manaRegen = 0;
  lifesteal = 0;
  haste = 0;
  armorPen = 0;
  magicPen = 0;
  healPower = 0;
  dmgAmp = 1;
  // Base values for non-hero units (heroes derive from their def + level).
  baseAd = 0;
  baseArmor = 0;
  baseMr = 0;
  baseMs = 0;
  baseAs = 0.7;
  baseRange = 120;
  buffs: Buff[] = [];
  statsDirty = true;
  stunned = false;
  rooted = false;
  stealthed = false;
  shield = 0;
  // Combat.
  targetId = -1;
  attackCd = 0;
  windup = 0;
  windupTarget = -1;
  attackAnim = 0;
  lastDamageTick = -9999;
  lastHeroDamageTick = -9999;
  // Movement.
  path: number[] | null = null;
  pathIdx = 0;
  pathGoalX = 0;
  pathGoalY = 0;
  repathAt = 0;
  dash: Dash | null = null;
  // Vision: visible to blue / red.
  vis0 = false;
  vis1 = false;
  // Minions.
  lane = -1;
  wp = 0;
  subtype = '';
  thinkOffset = 0;
  // Monsters.
  camp = -1;
  homeX = 0;
  homeY = 0;
  // Structures.
  tier = 0;
  vulnerable = true;
  /** Outer tower in the opening minutes: takes much less damage (see FORTIFY_UNTIL). */
  fortified = false;
  towerHits = 0;
  // Rewards granted on death.
  goldValue = 0;
  xpValue = 0;
  hero: HeroState | null = null;
}

export interface Projectile {
  id: number;
  kind: string;
  srcId: number;
  team: number;
  x: number;
  y: number;
  px: number;
  py: number;
  dx: number;
  dy: number;
  speed: number;
  dist: number;
  radius: number;
  targetId: number;
  pierce: boolean;
  heroOnly: boolean;
  dmg: number;
  dmg2: number;
  rank: number;
  hits: number[];
  /** Tick the projectile was spawned (bots notice it only after their reaction time). */
  born: number;
}

export interface Zone {
  id: number;
  kind: string;
  srcId: number;
  team: number;
  x: number;
  y: number;
  radius: number;
  delay: number;
  total: number;
  rank: number;
  /** Tick the zone was spawned (bots notice it only after their reaction time). */
  born: number;
}

export interface GameEvent {
  type: string;
  id?: number;
  x?: number;
  y?: number;
  x2?: number;
  y2?: number;
  r?: number;
  amount?: number;
  dtype?: number;
  team?: number;
  name?: string;
  /** Unit subtype, on death events (which minion or monster it was). */
  sub?: string;
  text?: string;
  other?: number;
  slot?: number;
}

// --- Non-hero unit stats -----------------------------------------------------
export interface UnitStats {
  hp: number;
  hpPerMin: number;
  ad: number;
  adPerMin: number;
  armor: number;
  mr: number;
  as: number;
  range: number;
  ms: number;
  radius: number;
  gold: number;
  xp: number;
}

export const MINION_STATS: Record<string, UnitStats> = {
  melee: { hp: 460, hpPerMin: 22, ad: 13, adPerMin: 0.9, armor: 0, mr: 0, as: 0.8, range: 110, ms: 300, radius: 34, gold: 30, xp: 58 },
  ranged: { hp: 290, hpPerMin: 14, ad: 23, adPerMin: 1.4, armor: 0, mr: 0, as: 0.65, range: 480, ms: 300, radius: 30, gold: 24, xp: 32 },
  siege: { hp: 850, hpPerMin: 34, ad: 40, adPerMin: 2, armor: 15, mr: 0, as: 0.5, range: 620, ms: 300, radius: 44, gold: 66, xp: 90 },
  super: { hp: 1700, hpPerMin: 60, ad: 85, adPerMin: 4, armor: 40, mr: 20, as: 0.75, range: 130, ms: 300, radius: 50, gold: 88, xp: 110 },
};

export const MONSTER_STATS: Record<string, UnitStats> = {
  stalker: { hp: 300, hpPerMin: 18, ad: 12, adPerMin: 0.8, armor: 0, mr: 0, as: 0.6, range: 110, ms: 320, radius: 34, gold: 30, xp: 48 },
  golem: { hp: 950, hpPerMin: 55, ad: 30, adPerMin: 2, armor: 10, mr: 0, as: 0.55, range: 130, ms: 300, radius: 55, gold: 94, xp: 160 },
  insight: { hp: 1050, hpPerMin: 60, ad: 32, adPerMin: 2, armor: 10, mr: 0, as: 0.55, range: 300, ms: 300, radius: 52, gold: 75, xp: 150 },
  might: { hp: 1050, hpPerMin: 60, ad: 34, adPerMin: 2, armor: 10, mr: 0, as: 0.55, range: 130, ms: 300, radius: 52, gold: 75, xp: 150 },
  wyrm: { hp: 3000, hpPerMin: 160, ad: 70, adPerMin: 4, armor: 25, mr: 25, as: 0.6, range: 420, ms: 300, radius: 80, gold: 110, xp: 300 },
  titan: { hp: 7000, hpPerMin: 220, ad: 130, adPerMin: 6, armor: 50, mr: 50, as: 0.6, range: 480, ms: 300, radius: 110, gold: 200, xp: 600 },
};

export const TOWER_HP = [3000, 3400, 3800, 5500];
export const TOWER_AD = [150, 170, 190, 0];
export const TOWER_RANGE = 760;
/** Outer towers take this much less damage until FORTIFY_UNTIL seconds, so a lane cannot be rushed before it has been played. */
export const FORTIFY_UNTIL = 300;
export const FORTIFY_REDUCTION = 0.85;
/**
 * Tower gold goes to the heroes who are there, shared between them, so a hero pushing alone earns what five together
 * would split. A tower's health is five plates; each plate broken pays TOWER_PLATE_GOLD, the tower's fall pays
 * TOWER_GOLD_LOCAL (both split among the attackers' heroes within TOWER_GOLD_RANGE), and every team-mate gets TOWER_GOLD_GLOBAL.
 */
export const TOWER_PLATES = 5;
export const TOWER_PLATE_GOLD = 50;
export const TOWER_GOLD_LOCAL = 200;
export const TOWER_GOLD_GLOBAL = 20;
export const TOWER_GOLD_RANGE = 1500;
/**
 * Experience from a kill is shared by the heroes nearby without growing: two get XP_SHARE_TWO each, three or more
 * split XP_SHARE_GROUP between them. Heroes who move as a pack therefore level slowly.
 */
export const XP_SHARE_TWO = 0.55;
export const XP_SHARE_GROUP = 1;

// --- Laning ------------------------------------------------------------------
/** Minions wear each other down slowly and heroes hit them hard, which leaves a fair window for the last hit. */
export const MINION_VS_MINION = 0.5;
export const HERO_VS_MINION = 1.25;
/** When a minion dies without a hero's last hit, enemy heroes this close split this share of its gold. */
export const PROXIMITY_GOLD = 0.4;
export const PROXIMITY_GOLD_RANGE = 900;
/** Gold per second every hero receives regardless of what they do. */
export const PASSIVE_GOLD = 1;

export const BUFF_DURATION = 100;
export const TITAN_BUFF_DURATION = 150;

// --- Items -------------------------------------------------------------------
export interface ItemStats {
  ad: number;
  ap: number;
  hp: number;
  mana: number;
  armor: number;
  mr: number;
  as: number;
  ms: number;
  haste: number;
  lifesteal: number;
  hpRegen: number;
  manaRegen: number;
  armorPen: number;
  magicPen: number;
  healPower: number;
  onHit: number;
  spellSlow: number;
  // Signature effects of the finished items (0 = not owned).
  /** Chance for an attack to strike for CRIT_DAMAGE times its damage. */
  crit: number;
  /** Share of an attack's damage that also hits the enemies around the target. */
  cleave: number;
  /** Magic damage of the lightning every third attack arcs to two more enemies. */
  chain: number;
  /** Extra damage dealt to enemies below EXECUTE_BELOW of their health. */
  execute: number;
  /** Multiplier on total ability power. */
  apAmp: number;
  /** Magic damage (plus SPELLBLADE_AP of ability power) on the first attack after casting an ability. */
  spellblade: number;
  /** Ability damage to a hero also deals this share of the hero's current health. */
  burn: number;
  /** Magic damage (plus THORNS_ARMOR of the wearer's armor) returned to whoever attacks the wearer. */
  thorns: number;
  /** Ability damage from a hero grants a shield of this share of maximum health (WARD_COOLDOWN). */
  ward: number;
  /** Share of maximum health regenerated per second after VIGOR_DELAY seconds without taking damage. */
  vigor: number;
}
export function emptyItemStats(): ItemStats {
  return {
    ad: 0, ap: 0, hp: 0, mana: 0, armor: 0, mr: 0, as: 0, ms: 0, haste: 0, lifesteal: 0, hpRegen: 0, manaRegen: 0, armorPen: 0, magicPen: 0, healPower: 0, onHit: 0, spellSlow: 0,
    crit: 0, cleave: 0, chain: 0, execute: 0, apAmp: 0, spellblade: 0, burn: 0, thorns: 0, ward: 0, vigor: 0,
  };
}
export const CRIT_DAMAGE = 1.75;
export const CLEAVE_RADIUS = 260;
export const CHAIN_RANGE = 450;
export const EXECUTE_BELOW = 0.4;
export const SPELLBLADE_AP = 0.3;
export const SPELLBLADE_WINDOW = 4;
export const THORNS_ARMOR = 0.15;
export const WARD_DURATION = 3;
export const WARD_COOLDOWN = 20;
export const VIGOR_DELAY = 5;

export interface ItemDef {
  id: string;
  name: string;
  /** Total price. For a recipe item, owned components count toward it and are consumed. */
  cost: number;
  desc: string;
  color: string;
  glyph: string;
  stats: Partial<ItemStats>;
  /** A finished item's signature effect: its name and what it does. */
  passive?: [string, string];
  /** Components this item is built from (duplicates allowed). Absent for components and standalone items. */
  from?: string[];
}

export const ITEMS: ItemDef[] = [
  // Components: cheap stat sticks that build into the finished items below.
  { id: 'blade', name: 'Iron Blade', cost: 375, color: '#ff9d6b', glyph: 'sword', desc: '+15 attack damage', stats: { ad: 15 } },
  { id: 'dagger', name: 'Leech Dagger', cost: 675, color: '#ff6b6b', glyph: 'dagger', desc: '+15 attack damage, +8% lifesteal', stats: { ad: 15, lifesteal: 0.08 } },
  { id: 'knife', name: 'Serrated Knife', cost: 675, color: '#b197fc', glyph: 'claw', desc: '+15 attack damage, +10 armor penetration', stats: { ad: 15, armorPen: 10 } },
  { id: 'gloves', name: 'Quickdraw Gloves', cost: 500, color: '#ffe066', glyph: 'glove', desc: '+20% attack speed', stats: { as: 0.2 } },
  { id: 'tome', name: 'Apprentice Tome', cost: 375, color: '#c08bff', glyph: 'book', desc: '+25 ability power', stats: { ap: 25 } },
  { id: 'rod', name: 'Blasting Rod', cost: 775, color: '#9775fa', glyph: 'wand', desc: '+45 ability power', stats: { ap: 45 } },
  { id: 'lens', name: 'Void Lens', cost: 600, color: '#da77f2', glyph: 'lens', desc: '+15% magic penetration', stats: { magicPen: 0.15 } },
  { id: 'crystal', name: 'Mana Crystal', cost: 300, color: '#66d9e8', glyph: 'gem', desc: '+250 mana', stats: { mana: 250 } },
  { id: 'idol', name: 'Focus Idol', cost: 425, color: '#ffd8a8', glyph: 'idol', desc: '+10 ability haste', stats: { haste: 10 } },
  { id: 'charm', name: 'Vital Charm', cost: 350, color: '#74e08a', glyph: 'heart', desc: '+200 health', stats: { hp: 200 } },
  { id: 'girdle', name: 'Stout Girdle', cost: 775, color: '#f783ac', glyph: 'belt', desc: '+400 health', stats: { hp: 400 } },
  { id: 'bead', name: 'Mending Bead', cost: 250, color: '#8ce99a', glyph: 'bead', desc: '+6 health regen', stats: { hpRegen: 6 } },
  { id: 'vest', name: 'Chain Vest', cost: 425, color: '#ced4da', glyph: 'vest', desc: '+25 armor', stats: { armor: 25 } },
  { id: 'ward', name: 'Warding Cloak', cost: 425, color: '#96f2d7', glyph: 'cloak', desc: '+25 magic resist', stats: { mr: 25 } },
  // Standalone.
  { id: 'boots', name: 'Swift Boots', cost: 600, color: '#7fd4ff', glyph: 'boot', desc: '+55 move speed', stats: { ms: 55 } },
  // Finished items.
  { id: 'edge', name: "Executioner's Edge", cost: 2200, color: '#ff6b6b', glyph: 'sword', desc: '+60 attack damage, +12% lifesteal', stats: { ad: 60, lifesteal: 0.12, crit: 0.25 }, passive: ['Deathblow', 'Attacks have a 25% chance to strike for 175% damage.'], from: ['dagger', 'blade', 'blade'] },
  { id: 'stormbow', name: 'Stormbow', cost: 2125, color: '#ffe066', glyph: 'bow', desc: '+30 attack damage, +45% attack speed, attacks deal +35 magic damage', stats: { ad: 30, as: 0.45, onHit: 35, chain: 70 }, passive: ['Stormcall', 'Every third attack arcs lightning to two more enemies for 70 magic damage.'], from: ['gloves', 'gloves', 'blade'] },
  { id: 'cleaver', name: 'Titan Cleaver', cost: 2300, color: '#ffa94d', glyph: 'axe', desc: '+42 attack damage, +400 health, +15 ability haste', stats: { ad: 42, hp: 400, haste: 15, cleave: 0.35 }, passive: ['Cleave', 'Attacks also hit the enemies around the target for 35% of their damage.'], from: ['girdle', 'blade', 'idol'] },
  { id: 'fang', name: 'Shadow Fang', cost: 2200, color: '#b197fc', glyph: 'fang', desc: '+55 attack damage, +18 armor penetration, +10 ability haste', stats: { ad: 55, armorPen: 18, haste: 10, execute: 0.15 }, passive: ['Predator', 'Deal 15% more damage to enemies below 40% health.'], from: ['knife', 'blade', 'idol'] },
  { id: 'staff', name: 'Archmage Staff', cost: 2475, color: '#9775fa', glyph: 'staff', desc: '+115 ability power', stats: { ap: 115, apAmp: 0.25 }, passive: ['Overflow', 'Increases your total ability power by 25%.'], from: ['rod', 'rod'] },
  { id: 'scepter', name: 'Frost Scepter', cost: 2200, color: '#74c0fc', glyph: 'crystal', desc: '+70 ability power, +400 health', stats: { ap: 70, hp: 400, spellSlow: 0.2 }, passive: ['Frostbite', 'Ability damage slows by 20% for a second.'], from: ['rod', 'girdle'] },
  { id: 'orb', name: "Sage's Orb", cost: 1875, color: '#66d9e8', glyph: 'orb', desc: '+55 ability power, +400 mana, +20 ability haste, +3 mana regen', stats: { ap: 55, mana: 400, haste: 20, manaRegen: 3, spellblade: 40 }, passive: ['Spellweave', 'After casting an ability, your next attack within 4 seconds deals +40 (+30% ability power) magic damage.'], from: ['tome', 'crystal', 'idol'] },
  { id: 'prism', name: 'Void Prism', cost: 2200, color: '#da77f2', glyph: 'prism', desc: '+70 ability power, +30% magic penetration', stats: { ap: 70, magicPen: 0.3, burn: 0.04 }, passive: ['Unmaking', 'Ability damage to a hero also deals 4% of their current health.'], from: ['rod', 'lens', 'tome'] },
  { id: 'plate', name: 'Bulwark Plate', cost: 2050, color: '#ced4da', glyph: 'shield', desc: '+450 health, +55 armor', stats: { hp: 450, armor: 55, thorns: 15 }, passive: ['Thorns', 'Whoever attacks you takes 15 (+15% of your armor) magic damage.'], from: ['girdle', 'vest', 'vest'] },
  { id: 'mantle', name: 'Spirit Mantle', cost: 2050, color: '#96f2d7', glyph: 'cloak', desc: '+400 health, +55 magic resist, +6 health regen', stats: { hp: 400, mr: 55, hpRegen: 6, ward: 0.12 }, passive: ['Spell ward', 'Ability damage from a hero shields you for 12% of your health for 3 seconds (20 second cooldown).'], from: ['girdle', 'ward', 'ward'] },
  { id: 'colossus', name: 'Heart of the Colossus', cost: 2375, color: '#f783ac', glyph: 'heart', desc: '+800 health, +14 health regen', stats: { hp: 800, hpRegen: 14, vigor: 0.025 }, passive: ['Unyielding', 'After 5 seconds without taking damage, regenerate 2.5% of your health per second.'], from: ['girdle', 'girdle', 'bead'] },
  { id: 'chalice', name: 'Chalice of Dawn', cost: 1700, color: '#ffd43b', glyph: 'cup', desc: '+40 ability power, +200 health, +20 ability haste', passive: ['Dawnlight', 'Your heals and shields are 20% stronger.'], stats: { ap: 40, hp: 200, haste: 20, healPower: 0.2 }, from: ['tome', 'charm', 'idol'] },
];
export const ITEM_BY_ID: Record<string, ItemDef> = Object.fromEntries(ITEMS.map((i) => [i.id, i]));
/** For each component, the finished items it builds into. */
export const BUILDS_INTO: Record<string, string[]> = {};
for (const it of ITEMS) for (const c of new Set(it.from ?? [])) (BUILDS_INTO[c] ??= []).push(it.id);
export const isComponent = (id: string) => id in BUILDS_INTO;

// --- Purchasing (pure helpers shared by the sim, the bots and the HUD) ---------------
/** Inventory indexes of the owned components that buying `item` would consume. */
export function recipeUse(items: readonly string[], item: string): number[] {
  const used: number[] = [];
  for (const c of ITEM_BY_ID[item]?.from ?? []) {
    const i = items.findIndex((x, j) => x === c && !used.includes(j));
    if (i >= 0) used.push(i);
  }
  return used;
}

/** Gold still needed for `item` once owned components are counted. */
export function itemPrice(items: readonly string[], item: string): number {
  let p = ITEM_BY_ID[item].cost;
  for (const i of recipeUse(items, item)) p -= ITEM_BY_ID[items[i]].cost;
  return p;
}

/** Recipe components of `item` not yet owned (with multiplicity). */
export function missingParts(items: readonly string[], item: string): string[] {
  const used = recipeUse(items, item).map((i) => items[i]);
  const out: string[] = [];
  for (const c of ITEM_BY_ID[item]?.from ?? []) {
    const k = used.indexOf(c);
    if (k >= 0) used.splice(k, 1); else out.push(c);
  }
  return out;
}

/** Whether `item` can be bought with this inventory and gold (ignores where the hero stands). */
export function canAfford(items: readonly string[], gold: number, item: string): boolean {
  const def = ITEM_BY_ID[item];
  if (!def) return false;
  // Finished and standalone items are unique; components may be stacked.
  if (!isComponent(item) && items.includes(item)) return false;
  const used = recipeUse(items, item);
  if (items.length - used.length + 1 > MAX_ITEMS) return false;
  return gold >= itemPrice(items, item);
}

/** Inventory after buying `item`: consumed components are removed and the item takes the first freed slot. */
export function applyPurchase(items: readonly string[], item: string): string[] {
  const used = recipeUse(items, item);
  if (!used.length) return [...items, item];
  const first = Math.min(...used);
  return items.flatMap((x, i) => (i === first ? [item] : used.includes(i) ? [] : [x]));
}

/** Whether a build entry is done: owned, or (for a component) already built into something owned. */
export function buildDone(items: readonly string[], id: string): boolean {
  return items.includes(id) || (BUILDS_INTO[id]?.some((f) => items.includes(f)) ?? false);
}

/** The first build entry not yet done. */
export function buildTarget(items: readonly string[], build: readonly string[]): string | undefined {
  return build.find((id) => !buildDone(items, id));
}

/**
 * Next step toward a build: the next build item itself if affordable, otherwise its most
 * expensive affordable missing component. Null when nothing useful can be bought.
 */
export function nextPurchase(items: readonly string[], gold: number, build: readonly string[]): string | null {
  const target = buildTarget(items, build);
  if (!target) return null;
  if (canAfford(items, gold, target)) return target;
  let best: string | null = null;
  for (const c of missingParts(items, target)) {
    if (canAfford(items, gold, c) && (!best || ITEM_BY_ID[c].cost > ITEM_BY_ID[best].cost)) best = c;
  }
  return best;
}

/** Simulates a shopping trip: gold spent and whether a build item gets completed. */
export function planShopping(items: readonly string[], gold: number, build: readonly string[]): { spent: number; completes: boolean } {
  let inv = [...items], spent = 0, completes = false;
  for (let id = nextPurchase(inv, gold - spent, build); id; id = nextPurchase(inv, gold - spent, build)) {
    spent += itemPrice(inv, id);
    if (build.includes(id)) completes = true;
    inv = applyPurchase(inv, id);
  }
  return { spent, completes };
}

export function xpForLevel(level: number): number {
  return 200 + 85 * (level - 1);
}

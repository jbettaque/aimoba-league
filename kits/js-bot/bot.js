/**
 * A rule-based starter bot. It plays all five heroes of one team with the same few rules:
 *
 *   1. hurt, or rich -> go home (heroes shop by themselves at the fountain), recalling when nobody is near
 *   2. under tower -> step out of an enemy tower's range unless our minions take its shots
 *   3. enemy near  -> fight if we are the stronger group, otherwise fall back along the lane
 *   4. creeps      -> last-hit, or hit the nearest
 *   5. structure   -> attack it while our minions stand in front
 *   6. otherwise   -> follow the lane up to our front minion
 *
 * It is a starting point, not a contender: it still loses most of its games to the built-in `easy` bots.
 * Every rule is an obvious place to do better.
 *
 *   ./aimoba play <your-bot> easy --watch      play a game and watch it
 *   ./aimoba bench <your-bot> basic -n 20      win rate over 20 games
 *
 * `state` is everything your team can see (docs/league/api.md lists every field); `ctx.lib` has geometry, map and
 * order helpers; `ctx.info` is the static game data (heroes, abilities, items, map). Coordinates are mirrored for
 * Red, so your base is always bottom-left and the same code plays both sides.
 */

const RETREAT_BELOW = 0.33;  // go home below this share of health
const HEALED_ABOVE = 0.85;   // leave the fountain above this
const FIGHT_RANGE = 1100;    // enemy heroes this close get attention
const FARM_RANGE = 800;
const SHOP_AT = 900;         // go home to shop with this much gold in the pocket
const EDGE = 1.0;            // fight when our group's strength is at least this many times theirs
const GROUP_AT = 9 * 60;     // seconds of game time after which the team gathers in the middle lane

export default {
  options: {
    observations: ['view'],  // add 'obs' to also get the fixed-size ML observation
    decisionTicks: 6,        // decide five times a second (30 ticks = 1 second)
  },

  init(ctx) {
    // Which lane each hero plays: 0 top, 1 mid, 2 bot. The jungler (lane -1) starts in its jungle.
    this.lanes = ctx.info.heroes.map((h) => h.lane);
    this.abilities = ctx.info.heroes.map((h) => h.abilities);
    this.home = { x: ctx.info.map.fountains[0][0], y: ctx.info.map.fountains[0][1] };
    this.towerRange = ctx.info.units.towerRange;
    this.healing = [false, false, false, false, false];
  },

  act(state, ctx) {
    const orders = [];
    for (const hero of state.heroes) {
      if (!hero.alive) continue;
      const order = this.decide(hero, state, ctx.lib);
      if (order) orders.push(...[order].flat());
    }
    return orders;
  },

  decide(hero, state, lib) {
    const hp = hero.hp / hero.maxHp;
    const enemies = state.enemies.filter((e) => e.visible && e.alive);
    const near = lib.within(hero, enemies, FIGHT_RANGE);

    // 1. Hurt, or carrying enough gold for the next item: go home and stay there until healed.
    //    Gold only turns into items at the fountain, where heroes follow their build by themselves.
    const canBuy = hero.gold >= SHOP_AT && lib.nextPurchase(hero.items, hero.gold, hero.build) !== null;
    if (hp < RETREAT_BELOW || (canBuy && !near.length)) this.healing[hero.index] = true;
    else if (hp > HEALED_ABOVE && !canBuy) this.healing[hero.index] = false;
    if (this.healing[hero.index]) {
      if (hero.inFountain) return null;
      if (hero.recall > 0 && !near.length) return null;            // already recalling
      return near.length ? lib.move(hero, this.home.x, this.home.y) : lib.recall(hero);
    }

    // 2. Standing in an enemy tower's range with no minions to take its shots: get out.
    if (this.threat(hero, state, lib)) return this.fallBack(hero, lib);

    // 3. Enemy heroes nearby: everyone picks the weakest one. Fight if our group is the stronger, else fall back.
    if (near.length) {
      const strength = (list) => list.reduce((sum, u) => sum + u.hp * (1 + 0.12 * u.level), 0);
      const ours = strength(lib.within(hero, state.heroes.filter((h) => h.alive), FIGHT_RANGE));
      const target = near.reduce((a, b) => (a.hp < b.hp ? a : b));
      if (ours >= strength(near) * EDGE && !this.threat(target, state, lib)) return this.fight(hero, target, state, lib);
      // Weaker: keep farming, but not within their reach.
      const closest = lib.nearest(hero, near);
      if (lib.dist(hero, closest) < closest.range + 250) return this.fallBack(hero, lib);
    }

    // 4. Creeps: take the last hit when one is ready, otherwise hit the nearest enemy creep.
    const jungler = this.lanes[hero.index] < 0;
    const creeps = lib.within(hero, state.creeps.filter((c) => !c.own && (c.kind === 'minion' || jungler)), FARM_RANGE).filter((c) => !this.threat(c, state, lib));
    if (creeps.length) {
      const killable = creeps.filter((c) => c.hp <= lib.afterResist(hero.ad, c.armor) * 1.2);
      const pool = killable.length ? killable : creeps;
      // Stay on the creep we are already hitting: switching target throws away the swing in progress.
      return lib.attack(hero, pool.find((c) => c.id === hero.target) ?? lib.nearest(hero, pool));
    }

    // 5. An enemy structure we may attack, with our minions in front of us.
    const structure = lib.nearest(hero, state.structures.filter((s) => !s.own && s.vulnerable && lib.dist(hero, s) < 1100));
    if (structure && !this.threat(structure, state, lib)) return lib.attack(hero, structure);

    // 6. Nothing to do here: go where the lane is being fought.
    return this.advance(hero, state, lib);
  },

  /** Cast whatever is ready at the target, then attack it. */
  fight(hero, target, state, lib) {
    const orders = [];
    const d = lib.dist(hero, target);
    hero.ready.forEach((ready, slot) => {
      if (!ready) return;
      const ability = this.abilities[hero.index][slot];
      if (ability.targeting === 'self') {
        if (d < 450) orders.push(lib.cast(hero, slot));
      } else if (ability.targeting === 'ally') {
        const hurt = lib.within(hero, state.heroes.filter((h) => h.alive && h.hp < h.maxHp * 0.6), ability.range);
        if (hurt.length) orders.push(lib.cast(hero, slot, hurt.reduce((a, b) => (a.hp < b.hp ? a : b))));
      } else if (d <= ability.range) {
        // Unit-targeted spells take the target; aimed ones a point, a little ahead of where it is walking.
        const aimed = ability.targeting === 'dir' || ability.targeting === 'point';
        orders.push(lib.cast(hero, slot, aimed ? lib.lead(hero, target, 1400) : target));
      }
    });
    orders.push(lib.attack(hero, target));
    return orders;
  },

  /** Walk back toward our base along the nearest lane. */
  fallBack(hero, lib) {
    const lanes = [0, 1, 2];
    const lane = lanes.reduce((a, b) => (lib.dist(hero, lib.lanePoint(a, lib.laneProgress(a, hero.x, hero.y))) < lib.dist(hero, lib.lanePoint(b, lib.laneProgress(b, hero.x, hero.y))) ? a : b));
    const back = lib.lanePoint(lane, lib.laneProgress(lane, hero.x, hero.y) - 700);
    return lib.move(hero, back.x, back.y);
  },

  /** True when an enemy tower would shoot us at `spot` because none of our minions stands in its range. */
  threat(spot, state, lib) {
    for (const s of state.structures) {
      if (s.own || !s.alive || s.kind !== 'tower' || lib.dist(spot, s) > this.towerRange + 150) continue;
      const shield = state.creeps.filter((c) => c.own && c.kind === 'minion' && lib.dist(c, s) < this.towerRange - 50);
      if (shield.length < 2) return true;
    }
    return false;
  },

  /** Walk along the lane to just behind our foremost minion (or to the lane's first tower before minions arrive). */
  advance(hero, state, lib) {
    let lane = this.lanes[hero.index];
    if (state.time > GROUP_AT) lane = 1;                             // late game: everyone pushes the middle together
    else if (lane < 0) {
      // The jungler: clear a camp in our own jungle if one is up, otherwise help in the middle.
      const camp = lib.nearest(hero, state.camps.filter((c) => c.side === 'own' && c.alive));
      if (camp && hero.level < 9) return lib.attackMove(hero, camp.x, camp.y);
      lane = 1;
    }
    const minions = state.creeps.filter((c) => c.own && c.kind === 'minion' && c.lane === lane);
    const front = minions.length ? Math.max(...minions.map((m) => lib.laneProgress(lane, m.x, m.y))) : lib.laneLength[lane] * 0.3;
    const spot = lib.lanePoint(lane, front - 150);
    if (this.threat(spot, state, lib)) return lib.move(hero, ...xy(lib.lanePoint(lane, front - 700)));
    return lib.attackMove(hero, spot.x, spot.y);
  },
};

const xy = (p) => [p.x, p.y];

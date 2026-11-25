# What a bot sees and can do

A bot plays one team: five heroes, always the same five in the same order.

| Index | Hero | Role | Lane |
| --- | --- | --- | --- |
| 0 | korvath | Vanguard (tank) | top |
| 1 | sable | Assassin | jungle |
| 2 | ignara | Mage | mid |
| 3 | vesper | Marksman | bot |
| 4 | liora | Support | bot |

Each decision the bot receives the state of the game as its team sees it and answers with orders.
`./aimoba info` prints the heroes with their abilities; `./aimoba info --json` prints everything a bot is told
before the game.

## The bot

JavaScript (`runtime: "node"`): the default export of the entry file.

```js
export default {
  options: { observations: ['view'], decisionTicks: 6 },
  init(ctx) {},              // once, before the game
  start(ctx) {},             // the game begins; ctx.team is 0 (Blue) or 1 (Red)
  act(state, ctx) { return [ /* orders */ ]; },
  end(result, ctx) {},       // result.result is 'win', 'loss' or 'draw'
};
```

`ctx.info` is the static game data, `ctx.lib` the helpers listed [below](#helpers), `ctx.log(...)` writes to
the log. `act` may be `async`. The default export may also be a class, or a function that returns the object.
TypeScript files work as they are (`bot.ts`).

Python (`runtime: "python"`): a subclass of `Bot` from `aimoba_agent.py`, which `init` copies into your bot.

```python
from aimoba_agent import Bot, run, attack_move

class MyBot(Bot):
    observations = ("view",)
    decision_ticks = 6

    def init(self, info): ...
    def act(self, state):
        return [attack_move(h, 6900, 1100) for h in state["heroes"] if h["alive"]]

run(MyBot())
```

The view is the same in both languages; in Python it is a dict (`state["heroes"][0]["hp"]`).

## Options

| Option (JS / Python) | Default | Meaning |
| --- | --- | --- |
| `observations` | `['view']` | What to receive each decision: `'view'` (described here), `'obs'` (the [ML observation](ml.md)), or both. |
| `decisionTicks` / `decision_ticks` | 6 | Game ticks between decisions. 30 ticks are one second, so 6 is five decisions a second. Allowed: 3 to 30. Thinking time is per second of game, so deciding more often gives less time per decision. |
| `mirror` | true | Mirror the map for Red, so your base is always bottom-left. Turn it off to get world coordinates. |
| `autoLevel` / `auto_level` | true | Heroes spend skill points by themselves (ultimate first, then the hero's usual order). A list of five sets it per hero. |
| `autoBuy` / `auto_buy` | true | Heroes shop along their build whenever they are at the fountain or dead. |
| `autoAttack` / `auto_attack` | false | Idle heroes attack enemies that come into range. |

With `autoLevel` or `autoBuy` off, your bot sends the `level` and `buy` orders itself.

## Coordinates

The map is a square of 8000 by 8000 units. With `mirror` on, your fountain is at (480, 7520), bottom-left, and
the enemy core at (6900, 1100), top-right, whichever side you play. The mirror swaps x and y, so lanes keep
their names. `info.map` describes the map in the same frame: lanes as lists of points from your base to the
enemy core, structures and camps with yours first, and a walkability grid.

Distances: a hero walks about 340 units a second, sees 1050 units, and a tower shoots 760 units.

## The view

All of this is what your team can see. Enemy units outside your team's sight are not in it.

| Field | Meaning |
| --- | --- |
| `tick`, `time` | Game tick and time in seconds. |
| `team` | 0 Blue, 1 Red. |
| `kills`, `towersLost`, `wyrmStacks` | `[yours, theirs]`. |
| `heroes` | Your five heroes, by index. |
| `enemies` | The five enemy heroes, by index. |
| `creeps` | Your minions, and the enemy minions and jungle monsters you can see. |
| `structures` | All 18 towers, guardians and cores. |
| `projectiles` | Shots in flight that you can see. |
| `zones` | Ground effects about to go off (delayed area spells). |
| `camps` | The ten monster camps and whether they are up. |

**Your heroes** (`heroes[i]`)

| Field | Meaning |
| --- | --- |
| `id`, `index`, `hero` | Unit id (to target it with heals and shields), index in the team, hero name. |
| `x`, `y`, `vx`, `vy`, `facing` | Position, velocity in units per second, facing in radians. |
| `alive`, `respawn` | `respawn`: seconds until it is back. |
| `hp`, `maxHp`, `mana`, `maxMana`, `shield` | |
| `level`, `xp`, `xpToNext`, `gold`, `skillPoints` | |
| `ranks`, `cooldowns`, `ready`, `canLevel` | Per ability (Q, W, E, R): rank, seconds of cooldown left, castable right now, whether a skill point can go there. |
| `items`, `build`, `canShop` | Inventory, the build the automatic shopper follows, and whether `buy` works now. |
| `ad`, `ap`, `armor`, `mr`, `attackSpeed`, `range`, `moveSpeed` | Current stats. |
| `stunned`, `rooted`, `stealthed`, `buffs` | `buffs`: `{kind, time, value}`. |
| `order` | What the hero is doing: `{type, x, y, target, slot}`. |
| `recall`, `attackCooldown`, `attacking`, `target` | Seconds left on a recall; attack timing; the unit being attacked (-1 none). |
| `inFountain`, `kills`, `deaths`, `assists`, `lastHits` | |

**Enemy heroes** (`enemies[i]`): `id`, `index`, `hero`, `visible`, `alive`, `respawn`, `level`, `kills`,
`deaths`, `assists` are always there. `lastSeen` is `{x, y, hp, age}`: where and how your team last saw it, and
how many seconds ago. While `visible` is true there is also `x`, `y`, `vx`, `vy`, `facing`, `hp`, `maxHp`,
`mana`, `maxMana`, `shield`, the stats, `stunned`, `rooted`, `buffs`, `items`, `target` and `dash`
(`{x, y}` where a charge or leap in progress will end).

**Creeps**: `id`, `kind` (`minion` or `monster`), `type` (melee, ranged, siege, super; or the camp type), `own`,
`lane` (lane index for minions, camp index for monsters), `x`, `y`, `hp`, `maxHp`, `ad`, `armor`, `range`,
`target`.

**Structures**: `id`, `index`, `kind` (`tower` or `core`), `own`, `lane`, `tier` (0 outer tower, 1 inner tower,
2 guardian, 3 core), `x`, `y`, `hp`, `maxHp`, `alive`, `vulnerable` (can be damaged now: what stands in front
of it is down), `fortified` (outer towers take 85% less damage for the first five minutes), `target`.

**Projectiles**: `id`, `own`, `kind`, `x`, `y`, `dx`, `dy` (unit direction), `speed`, `radius`, `range`
(distance left), `target` (the unit it homes in on; -1 for a skillshot, which flies straight and can be
dodged), `pierce`, `heroOnly`.

**Zones**: `id`, `own`, `kind`, `x`, `y`, `radius`, `delay` (seconds until it goes off).

**Camps**: `index`, `type`, `side` (`own`, `enemy`, `river`), `x`, `y`, `alive` (`null` for the enemy's jungle
unless you can see into it).

Unit ids are your team's own: they are handed out when your team first sees a unit and stay the same for the
game. The other team has different ids for the same units.

## Orders

`act` returns a list of orders. Each names one of your heroes by index.

| Order | Meaning |
| --- | --- |
| `{hero, type: 'move', x, y}` | Walk there. The game finds the path around walls. |
| `{hero, type: 'attack_move', x, y}` | Walk there, attacking whatever comes into range on the way. |
| `{hero, type: 'attack', target}` | Attack a unit until it dies or is lost from sight. |
| `{hero, type: 'cast', slot, x, y, target}` | Cast an ability. `slot` is 0 to 3 or `'Q'`, `'W'`, `'E'`, `'R'`. Aimed abilities (`dir`, `point`) take `x`, `y`, or a `target` to aim at where it stands now; `enemy` and `enemyHero` abilities need a `target`; `ally` abilities take an allied hero (yourself if none); `self` abilities take nothing. |
| `{hero, type: 'stop'}` | Stand still. |
| `{hero, type: 'recall'}` | Teleport home after five seconds of standing still. Taking damage, a stun or a new order cancels it. |
| `{hero, type: 'level', slot}` | Spend a skill point. |
| `{hero, type: 'buy', item}` | Buy an item (at the fountain or while dead). Owned components are used up and count toward the price. |
| `{hero, type: 'action', action}` | An action of the [ML interface](ml.md): `[kind, direction, distance, target]`. |

An order stays in force until another replaces it: you do not have to repeat it. Several orders for one hero
in the same decision are applied in the order given (a cast, then an attack, for example).

An order that makes no sense is dropped and noted in your bot's log: an unknown type, a target your team
cannot see, coordinates that are not numbers. An order the game cannot carry out right now (an ability on
cooldown, a purchase without the gold) is ignored silently; check `ready`, `canLevel`, `canShop` and `gold`
first. At most 40 orders count per decision.

## Helpers

JavaScript bots get these as `ctx.lib`. Python has the same order builders and geometry as functions of
`aimoba_agent`, and the map queries on `Map(info)`.

| Helper | Does |
| --- | --- |
| `move`, `attackMove`, `attack`, `cast`, `stop`, `recall`, `level`, `buy`, `action` | Build orders. They take a hero from the view or its index: `lib.cast(hero, 'Q', enemy)`. |
| `dist(a, b)`, `nearest(p, list)`, `within(p, list, r)` | Distances and filters over anything with `x` and `y`. |
| `towards(from, to, d)`, `away(from, threat, d)` | A point at distance `d` toward or away from something. |
| `lead(from, target, speed, delay)` | Where a moving target will be when a shot of that speed arrives. |
| `isWalkable(x, y)`, `lineClear(x0, y0, x1, y1)`, `findPath(from, to)`, `nearestWalkable(x, y)` | Map queries. Heroes find their own way on a `move`; these are for planning. |
| `lanePoint(lane, s)`, `laneProgress(lane, x, y)`, `laneLength` | Positions along a lane, measured from your base. |
| `afterResist(damage, resist)` | Damage left after armor or magic resist. |
| `item(id)`, `itemPrice(items, id)`, `canAfford(items, gold, id)`, `missingParts(items, id)`, `nextPurchase(items, gold, build)` | Shopping. |

## Game data

`ctx.info` (Python: the argument of `init`) holds what never changes:

- `info.heroes[i]`: base stats and growth per level, `abilities` (name, description, `targeting`, `range`,
  `radius`, `cooldown` and `cost` per rank), `skillOrder`, `build` and `altBuilds`
- `info.items`: id, name, cost, stats, the components it is built `from`, its passive effect
- `info.units`: minion and monster stats, tower health and damage, tower range
- `info.map`: size, `fountains`, `cores`, `lanes`, `structures`, `camps`, and `grid` (walkability)
- `info.constants`: sight range, experience range, gold rules, wave timing
- `info.rules`: the limits the match is played under
- `info.ml`: the layout of the ML observation and action

## Things that catch people out

- **Fog of war.** You only see enemies within sight of your heroes, minions and structures. `lastSeen` is all
  you have otherwise.
- **Towers.** An enemy tower shoots your minions first, and you as soon as none is left in its range or you
  attack an enemy hero under it. A structure can only be damaged when `vulnerable` is true.
- **Last hits.** Only the killing blow on a minion pays in full. A hero's attack on a minion deals 25% more.
- **Moving as a pack costs.** Experience and tower gold are split between the heroes present.
- **Printing.** `console.log` and `print` go to your log. Do not write to stdout yourself: it carries the
  protocol.
- **Randomness.** Seed your own random numbers if you want to reproduce a game: `./aimoba play a b --seed 7`
  replays the same game only if both bots make the same decisions.

"""A rule-based starter bot. It plays all five heroes of one team with the same few rules:

  1. hurt, or rich -> go home (heroes shop by themselves at the fountain), recalling when nobody is near
  2. under tower -> step out of an enemy tower's range unless our minions take its shots
  3. enemy near  -> fight if we are the stronger group, otherwise fall back along the lane
  4. creeps      -> last-hit, or hit the nearest
  5. structure   -> attack it while our minions stand in front
  6. otherwise   -> follow the lane up to our front minion

It is a starting point, not a contender: it still loses most of its games to the built-in ``easy`` bots.
Every rule is an obvious place to do better.

    ./aimoba play <your-bot> easy --watch      play a game and watch it
    ./aimoba bench <your-bot> basic -n 20      win rate over 20 games

``state`` is everything your team can see (docs/league/api.md lists every field). Coordinates are mirrored for
Red, so your base is always bottom-left and the same code plays both sides.
"""
from aimoba_agent import Bot, Map, after_resist, attack, attack_move, cast, dist, lead, move, nearest, recall, run, within

RETREAT_BELOW = 0.33  # go home below this share of health
HEALED_ABOVE = 0.85   # leave the fountain above this
FIGHT_RANGE = 1100    # enemy heroes this close get attention
FARM_RANGE = 800
SHOP_AT = 900        # go home to shop with this much gold in the pocket
EDGE = 1.0            # fight when our group's strength is at least this many times theirs
GROUP_AT = 9 * 60     # seconds of game time after which the team gathers in the middle lane


class StarterBot(Bot):
    observations = ("view",)  # add "obs" to also get the fixed-size ML observation
    decision_ticks = 6        # decide five times a second (30 ticks = 1 second)

    def init(self, info):
        self.map = Map(info)
        # Which lane each hero plays: 0 top, 1 mid, 2 bot. The jungler (lane -1) starts in its jungle.
        self.lanes = [h["lane"] for h in info["heroes"]]
        self.abilities = [h["abilities"] for h in info["heroes"]]
        self.items = {item["id"]: item for item in info["items"]}
        self.home = tuple(info["map"]["fountains"][0])
        self.tower_range = info["units"]["towerRange"]
        self.healing = [False] * 5

    def act(self, state):
        orders = []
        for hero in state["heroes"]:
            if hero["alive"]:
                orders.extend(self.decide(hero, state))
        return orders

    def decide(self, hero, state):
        hp = hero["hp"] / hero["maxHp"]
        enemies = [e for e in state["enemies"] if e["visible"] and e["alive"]]
        near = within(hero, enemies, FIGHT_RANGE)
        i = hero["index"]

        # 1. Hurt, or carrying enough gold for the next item: go home and stay there until healed.
        #    Gold only turns into items at the fountain, where heroes follow their build by themselves.
        can_buy = hero["gold"] >= SHOP_AT and self.affordable(hero)
        if hp < RETREAT_BELOW or (can_buy and not near):
            self.healing[i] = True
        elif hp > HEALED_ABOVE and not can_buy:
            self.healing[i] = False
        if self.healing[i]:
            if hero["inFountain"] or (hero["recall"] > 0 and not near):
                return []
            return [move(hero, *self.home)] if near else [recall(hero)]

        # 2. Standing in an enemy tower's range with no minions to take its shots: get out.
        if self.threat(hero, state):
            return [self.fall_back(hero)]

        # 3. Enemy heroes nearby: everyone picks the weakest one. Fight if our group is the stronger, else fall back.
        if near:
            strength = lambda units: sum(u["hp"] * (1 + 0.12 * u["level"]) for u in units)  # noqa: E731
            ours = strength(within(hero, [h for h in state["heroes"] if h["alive"]], FIGHT_RANGE))
            target = min(near, key=lambda e: e["hp"])
            if ours >= strength(near) * EDGE and not self.threat(target, state):
                return self.fight(hero, target, state)
            # Weaker: keep farming, but not within their reach.
            closest = nearest(hero, near)
            if dist(hero, closest) < closest["range"] + 250:
                return [self.fall_back(hero)]

        # 4. Creeps: take the last hit when one is ready, otherwise hit the nearest enemy creep.
        jungler = self.lanes[i] < 0
        creeps = [c for c in within(hero, state["creeps"], FARM_RANGE)
                  if not c["own"] and (c["kind"] == "minion" or jungler) and not self.threat(c, state)]
        if creeps:
            killable = [c for c in creeps if c["hp"] <= after_resist(hero["ad"], c["armor"]) * 1.2]
            pool = killable or creeps
            # Stay on the creep we are already hitting: switching target throws away the swing in progress.
            current = next((c for c in pool if c["id"] == hero["target"]), None)
            return [attack(hero, current or nearest(hero, pool))]

        # 5. An enemy structure we may attack, with our minions in front of us.
        structure = nearest(hero, [s for s in state["structures"] if not s["own"] and s["vulnerable"] and dist(hero, s) < 1100])
        if structure and not self.threat(structure, state):
            return [attack(hero, structure)]

        # 6. Nothing to do here: go where the lane is being fought.
        return [self.advance(hero, state)]

    def affordable(self, hero):
        """Whether the next item of the hero's build, or one of its components, can be bought with the gold it carries."""
        owned = list(hero["items"])
        for wanted in hero["build"]:
            item = self.items[wanted]
            if wanted in owned or any(wanted in self.items[o]["from"] for o in owned):
                continue  # already owned, or built into something owned
            parts = list(item["from"])
            price = item["cost"]
            for o in owned:
                if o in parts:
                    parts.remove(o)
                    price -= self.items[o]["cost"]
            return hero["gold"] >= price or any(hero["gold"] >= self.items[p]["cost"] for p in parts)
        return False

    def fight(self, hero, target, state):
        """Cast whatever is ready at the target, then attack it."""
        orders = []
        d = dist(hero, target)
        for slot, ready in enumerate(hero["ready"]):
            if not ready:
                continue
            ability = self.abilities[hero["index"]][slot]
            targeting = ability["targeting"]
            if targeting == "self":
                if d < 450:
                    orders.append(cast(hero, slot))
            elif targeting == "ally":
                hurt = within(hero, [h for h in state["heroes"] if h["alive"] and h["hp"] < h["maxHp"] * 0.6], ability["range"])
                if hurt:
                    orders.append(cast(hero, slot, min(hurt, key=lambda h: h["hp"])))
            elif d <= ability["range"]:
                # Unit-targeted spells take the target; aimed ones a point, a little ahead of where it is walking.
                aimed = targeting in ("dir", "point")
                orders.append(cast(hero, slot, lead(hero, target, 1400) if aimed else target))
        orders.append(attack(hero, target))
        return orders

    def fall_back(self, hero):
        """Walk back toward our base along the nearest lane."""
        lane = self.map.nearest_lane(hero["x"], hero["y"])
        return move(hero, *self.map.lane_point(lane, self.map.lane_progress(lane, hero["x"], hero["y"]) - 700))

    def threat(self, spot, state):
        """True when an enemy tower would shoot us at ``spot`` because none of our minions stands in its range."""
        for s in state["structures"]:
            if s["own"] or not s["alive"] or s["kind"] != "tower" or dist(spot, s) > self.tower_range + 150:
                continue
            shield = [c for c in state["creeps"] if c["own"] and c["kind"] == "minion" and dist(c, s) < self.tower_range - 50]
            if len(shield) < 2:
                return True
        return False

    def advance(self, hero, state):
        """Walk along the lane to just behind our foremost minion (or to the lane's first tower before minions arrive)."""
        lane = self.lanes[hero["index"]]
        if state["time"] > GROUP_AT:
            lane = 1  # late game: everyone pushes the middle together
        elif lane < 0:
            # The jungler: clear a camp in our own jungle if one is up, otherwise help in the middle.
            camp = nearest(hero, [c for c in state["camps"] if c["side"] == "own" and c["alive"]])
            if camp and hero["level"] < 9:
                return attack_move(hero, camp["x"], camp["y"])
            lane = 1
        minions = [c for c in state["creeps"] if c["own"] and c["kind"] == "minion" and c["lane"] == lane]
        if minions:
            front = max(self.map.lane_progress(lane, m["x"], m["y"]) for m in minions)
        else:
            front = self.map.lane_length[lane] * 0.3
        spot = self.map.lane_point(lane, front - 150)
        if self.threat({"x": spot[0], "y": spot[1]}, state):
            return move(hero, *self.map.lane_point(lane, front - 700))
        return attack_move(hero, *spot)


if __name__ == "__main__":
    run(StarterBot())

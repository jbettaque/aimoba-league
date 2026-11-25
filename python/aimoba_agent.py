"""AIMOBA agent SDK for Python: one file, standard library only (numpy is used when it is installed).

A bot is a class with an ``act`` method; ``run`` connects it to the match runner::

    from aimoba_agent import Bot, run, attack_move

    class MyBot(Bot):
        def act(self, state):
            return [attack_move(hero, 6900, 1100) for hero in state["heroes"] if hero["alive"]]

    if __name__ == "__main__":
        run(MyBot())

The runner starts this script, sends it the game state as JSON lines on stdin and reads orders from stdout
(the protocol is described in docs/league/protocol.md). ``print`` is redirected to stderr, which the runner
keeps as your bot's log, so printing never breaks the protocol.

``state`` is your team's view of the game as a dict (docs/league/api.md lists every field). Coordinates are
mirrored for Red, so your base is always bottom-left and the same code plays both sides. If the bot asks for
the ML observation (``observations = ("obs",)`` or both), ``state["obs"]`` is a float32 array of shape
(5, obs_size) and ``state["mask"]`` a uint8 array of shape (5, mask_size), one row per hero.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import traceback
from typing import Any, Iterable, Sequence

try:  # numpy is optional: without it obs and mask arrive as lists of rows
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

__all__ = [
    "Bot", "run", "Map", "PROTOCOL",
    "move", "attack_move", "attack", "cast", "stop", "recall", "level", "buy", "action",
    "dist", "towards", "away", "nearest", "within", "lead", "after_resist",
]

PROTOCOL = 1


# --- Orders -----------------------------------------------------------------------------------------------------
def _hero(h: Any) -> int:
    return h if isinstance(h, int) else h["index"]


def _unit(u: Any) -> int:
    return u if isinstance(u, int) else u["id"]


def move(hero: Any, x: float, y: float) -> dict:
    """Walk to a point (the game finds the path)."""
    return {"hero": _hero(hero), "type": "move", "x": x, "y": y}


def attack_move(hero: Any, x: float, y: float) -> dict:
    """Walk to a point, attacking what comes into range on the way."""
    return {"hero": _hero(hero), "type": "attack_move", "x": x, "y": y}


def attack(hero: Any, target: Any) -> dict:
    """Attack a unit (a dict from the view, or its id) until it dies or is lost from sight."""
    return {"hero": _hero(hero), "type": "attack", "target": _unit(target)}


def cast(hero: Any, slot: int | str, at: Any = None) -> dict:
    """Cast an ability (slot 0-3 or "Q" "W" "E" "R") at a point ``(x, y)``, at a unit, or with neither for a self-cast."""
    order: dict = {"hero": _hero(hero), "type": "cast", "slot": slot}
    if at is None:
        return order
    if isinstance(at, int):
        order["target"] = at
    elif isinstance(at, dict) and "id" in at:
        order["target"] = at["id"]
    elif isinstance(at, dict):
        order["x"], order["y"] = at["x"], at["y"]
    else:
        order["x"], order["y"] = at[0], at[1]
    return order


def stop(hero: Any) -> dict:
    return {"hero": _hero(hero), "type": "stop"}


def recall(hero: Any) -> dict:
    """Teleport home after standing still for five seconds."""
    return {"hero": _hero(hero), "type": "recall"}


def level(hero: Any, slot: int | str) -> dict:
    """Spend a skill point (only needed with ``auto_level = False``)."""
    return {"hero": _hero(hero), "type": "level", "slot": slot}


def buy(hero: Any, item: str) -> dict:
    """Buy an item at the fountain or while dead (only needed with ``auto_buy = False``)."""
    return {"hero": _hero(hero), "type": "buy", "item": item}


def action(hero: Any, a: Sequence[int]) -> dict:
    """An action of the ML interface for one hero: [kind, direction, distance, target]."""
    return {"hero": _hero(hero), "type": "action", "action": [int(v) for v in a]}


# --- Geometry ---------------------------------------------------------------------------------------------------
def _xy(p: Any) -> tuple[float, float]:
    return (p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1])


def dist(a: Any, b: Any) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return math.hypot(ax - bx, ay - by)


def towards(src: Any, dst: Any, d: float, overshoot: bool = False) -> tuple[float, float]:
    """The point ``d`` units from ``src`` in the direction of ``dst`` (not past it unless ``overshoot``)."""
    ax, ay = _xy(src)
    bx, by = _xy(dst)
    n = math.hypot(bx - ax, by - ay)
    if n < 1e-6 or (not overshoot and n <= d):
        return bx, by
    return ax + (bx - ax) / n * d, ay + (by - ay) / n * d


def away(src: Any, threat: Any, d: float) -> tuple[float, float]:
    """The point ``d`` units from ``src``, directly away from ``threat``."""
    ax, ay = _xy(src)
    bx, by = _xy(threat)
    n = math.hypot(ax - bx, ay - by)
    if n < 1e-6:
        return ax - d * 0.7071, ay + d * 0.7071
    return ax + (ax - bx) / n * d, ay + (ay - by) / n * d


def nearest(p: Any, items: Iterable[Any]) -> Any:
    """The entry closest to ``p`` (None for an empty list)."""
    best, bd = None, float("inf")
    for o in items:
        d = dist(p, o)
        if d < bd:
            best, bd = o, d
    return best


def within(p: Any, items: Iterable[Any], r: float) -> list:
    return [o for o in items if dist(p, o) <= r]


def lead(src: Any, target: dict, speed: float, delay: float = 0.0) -> tuple[float, float]:
    """Where something moving at (vx, vy) will be when a shot of ``speed`` fired now from ``src`` reaches it."""
    sx, sy = _xy(src)
    vx, vy = target.get("vx", 0.0), target.get("vy", 0.0)
    x, y = target["x"], target["y"]
    for _ in range(3):
        t = delay + (math.hypot(x - sx, y - sy) / speed if speed > 0 else 0.0)
        x, y = target["x"] + vx * t, target["y"] + vy * t
    return x, y


def after_resist(amount: float, resist: float) -> float:
    """Damage left of ``amount`` after ``resist`` armor or magic resist."""
    return amount * (100.0 / (100.0 + max(0.0, resist)))


class Map:
    """Map queries built from ``info["map"]``: lanes as polylines and the walkability grid.

    Lane positions are counted from the Blue base; with ``mirror`` on (the default) that is your own base on
    either side.
    """

    def __init__(self, info: dict) -> None:
        m = info["map"]
        self.size: float = m["size"]
        self.lanes: list[list[tuple[float, float]]] = [[(p[0], p[1]) for p in lane["points"]] for lane in m["lanes"]]
        self._cum: list[list[float]] = []
        for pts in self.lanes:
            cum = [0.0]
            for a, b in zip(pts, pts[1:]):
                cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            self._cum.append(cum)
        self.lane_length: list[float] = [c[-1] for c in self._cum]
        self._n: int = m["grid"]["n"]
        self._cell: float = m["grid"]["cell"]
        self._blocked: str = m["grid"]["blocked"]

    def lane_point(self, lane: int, s: float) -> tuple[float, float]:
        """The point ``s`` units along a lane (0 top, 1 mid, 2 bot)."""
        pts, cum = self.lanes[lane], self._cum[lane]
        s = min(max(s, 0.0), cum[-1])
        for i in range(len(pts) - 1):
            if s <= cum[i + 1]:
                seg = cum[i + 1] - cum[i]
                t = (s - cum[i]) / seg if seg > 0 else 0.0
                return pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t
        return pts[-1]

    def lane_progress(self, lane: int, x: float, y: float) -> float:
        """How far along a lane the lane point nearest to (x, y) is."""
        pts, cum = self.lanes[lane], self._cum[lane]
        best, best_d = 0.0, float("inf")
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            dx, dy = bx - ax, by - ay
            n2 = dx * dx + dy * dy
            t = 0.0 if n2 == 0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / n2))
            d = math.hypot(x - (ax + dx * t), y - (ay + dy * t))
            if d < best_d:
                best_d, best = d, cum[i] + math.sqrt(n2) * t
        return best

    def nearest_lane(self, x: float, y: float) -> int:
        return min(range(len(self.lanes)), key=lambda lane: dist((x, y), self.lane_point(lane, self.lane_progress(lane, x, y))))

    def is_walkable(self, x: float, y: float) -> bool:
        cx, cy = int(x // self._cell), int(y // self._cell)
        if cx < 0 or cy < 0 or cx >= self._n or cy >= self._n:
            return False
        return self._blocked[cy * self._n + cx] == "0"


# --- The bot and the loop ---------------------------------------------------------------------------------------
class Bot:
    """Base class for bots. Override ``act``; everything else is optional."""

    #: What to call the bot in logs.
    name: str | None = None
    #: Which forms of the game state to receive: "view" (the structured view), "obs" (the ML observation), or both.
    observations: Sequence[str] = ("view",)
    #: Simulation ticks between decisions (30 ticks = 1 second; the rules give the allowed range).
    decision_ticks: int = 6
    #: Mirror the map for Red, so your own base is always bottom-left.
    mirror: bool = True
    #: Let heroes spend skill points / shop along their build / attack when idle by themselves.
    #: A list of five booleans sets it per hero.
    auto_level: bool | Sequence[bool] = True
    auto_buy: bool | Sequence[bool] = True
    auto_attack: bool | Sequence[bool] = False

    #: Set before ``init``: the static game description, and (from ``start`` on) the side: 0 Blue, 1 Red.
    info: dict = {}
    team: int = -1

    def init(self, info: dict) -> None:
        """Called once before the game. ``info`` holds the map, heroes, abilities, items and the rules."""

    def start(self, team: int) -> None:
        """The game begins."""

    def act(self, state: dict) -> Any:
        """One decision. Return a list of orders, ``{"orders": [...], "actions": [...]}``, or None."""
        raise NotImplementedError

    def end(self, result: dict) -> None:
        """The game is over: ``result["result"]`` is "win", "loss" or "draw"."""


def _decode(b64: str, dtype: str, rows: int) -> Any:
    raw = base64.b64decode(b64)
    if _np is not None:
        return _np.frombuffer(raw, dtype=dtype).reshape(rows, -1)
    import array

    flat = array.array("f" if dtype == "<f4" else "B")
    flat.frombytes(raw)
    if sys.byteorder != "little" and dtype == "<f4":
        flat.byteswap()
    n = len(flat) // rows
    return [list(flat[i * n:(i + 1) * n]) for i in range(rows)]


def _plain(v: Any) -> Any:
    """numpy arrays and scalars as plain lists and numbers, for JSON."""
    if _np is not None:
        if isinstance(v, _np.ndarray):
            return v.tolist()
        if isinstance(v, _np.generic):
            return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def run(bot: Bot) -> None:
    """Serve a bot to the match runner until the game ends. Call this as the last thing in your script."""
    out = sys.stdout.buffer
    stdin = sys.stdin.buffer
    sys.stdout = sys.stderr  # print() goes to the log; stdout belongs to the protocol

    def send(msg: dict) -> None:
        out.write(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
        out.flush()

    try:
        for line in stdin:
            if not line.strip():
                continue
            msg = json.loads(line)
            kind = msg.get("type")
            if kind == "hello":
                bot.info = msg["info"]
                bot.init(msg["info"])
                five = lambda v: list(v) if isinstance(v, (list, tuple)) else bool(v)  # noqa: E731
                send({
                    "type": "ready", "name": bot.name, "observations": list(bot.observations), "decisionTicks": int(bot.decision_ticks),
                    "mirror": bool(bot.mirror), "autoLevel": five(bot.auto_level), "autoBuy": five(bot.auto_buy), "autoAttack": five(bot.auto_attack),
                })
            elif kind == "start":
                bot.team = msg["team"]
                bot.start(msg["team"])
            elif kind == "step":
                state = msg.get("view") or {"tick": msg["tick"]}
                if "obs" in msg:
                    state["obs"] = _decode(msg["obs"], "<f4", 5)
                    state["mask"] = _decode(msg["mask"], "u1", 5)
                reply = bot.act(state)
                if reply is None:
                    send({"type": "act"})
                elif isinstance(reply, dict):
                    send({"type": "act", "orders": _plain(reply.get("orders")), "actions": _plain(reply.get("actions"))})
                else:
                    send({"type": "act", "orders": _plain(list(reply))})
            elif kind == "end":
                bot.end(msg)
                break
    except BrokenPipeError:
        pass
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

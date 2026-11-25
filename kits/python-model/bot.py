"""An agent that plays with a neural network: the ML interface, end to end.

Each decision it receives the fixed-size observation and action mask of its five heroes, runs the network on
them and answers with five actions [kind, direction, distance, target]. The network here is the one
python/train/export.py writes (``model.bin``), evaluated with numpy alone so the entry has no dependencies.

To enter your own model:

  * the same architecture, trained with python/train/ppo.py:
        aimoba export runs/<run>/latest.pt bots/<your-bot>        (replaces model.bin)
    If that is all you need, the "policy" kit does it without any code.
  * anything else: keep ``act`` and replace ``Network`` with your own inference (PyTorch, ONNX Runtime and numpy
    are installed on the league's machines; see docs/league/ml.md for the observation and action layout).

Observations are mirrored for Red and the hero is one-hot in each row, so one network plays every hero on
both sides.
"""
import json
import os
import struct

import numpy as np

from aimoba_agent import Bot, run

GREEDY = False  # True: always the most likely action. Sampling is what self-play policies are trained with.


class Network:
    """The actor exported by python/train/export.py: an MLP body, a head for the action kind, and direction /
    distance / target heads with one set of outputs per kind."""

    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"AIMP":
            raise ValueError(f"{path} is not an AIMOBA policy file")
        version, header_len = struct.unpack_from("<II", data, 4)
        header = json.loads(data[12:12 + header_len])
        offset = 12 + header_len
        tensors = {}
        for name, *shape in header["tensors"]:
            n = int(np.prod(shape))
            tensors[name] = np.frombuffer(data, dtype="<f2", count=n, offset=offset).astype(np.float32).reshape(shape)
            offset += n * 2
        self.header = header
        self.nvec = header["nvec"]
        self.hero_offset = header["hero_offset"]
        self.activation = np.tanh if header["activation"] == "tanh" else (lambda x: np.maximum(x, 0.0))
        self.body = [(tensors[f"body.{i * 2}.weight"], tensors[f"body.{i * 2}.bias"]) for i in range(header["layers"])]
        self.heads = {h: (tensors[f"{h}_head.weight"], tensors[f"{h}_head.bias"]) for h in ("kind", "dir", "dist", "target")}

    def act(self, obs, mask, action_targets, rng):
        """obs (5, obs_size), mask (5, mask_size) -> actions (5, 4)."""
        nk, nd, ns, nt = self.nvec
        h = obs
        for w, b in self.body:
            h = self.activation(h @ w.T + b)
        rows = np.arange(len(obs))

        def choose(logits, allowed=None):
            if allowed is not None:
                logits = np.where(allowed, logits, -1e9)
            if rng is None:
                return logits.argmax(-1)
            # Gumbel-max: a sample from softmax(logits) without computing the softmax.
            return (logits + rng.gumbel(size=logits.shape)).argmax(-1)

        def head(name, n, kind):
            w, b = self.heads[name]
            return (h @ w.T + b).reshape(len(obs), nk, n)[rows, kind]

        kind = choose(h @ self.heads["kind"][0].T + self.heads["kind"][1], mask[:, :nk] > 0)
        # A target slot must be occupied (the mask) and make sense for this hero's action kind.
        hero = obs[:, self.hero_offset:self.hero_offset + 5].argmax(-1)
        fits = (mask[:, nk:] > 0) & (action_targets[hero, kind] > 0)
        fits |= ~fits.any(-1, keepdims=True)  # the kind takes no target: the head is ignored
        target = choose(head("target", nt, kind), fits)
        return np.stack([kind, choose(head("dir", nd, kind)), choose(head("dist", ns, kind)), target], axis=1)


class ModelBot(Bot):
    observations = ("obs",)  # add "view" to also get the structured view (for scripted overrides, say)
    decision_ticks = 6       # the cadence the model was trained at

    def init(self, info):
        self.net = Network(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.bin"))
        ml = info["ml"]
        if self.net.header["obs_size"] != ml["obsSize"] or self.net.nvec != ml["actionNvec"]:
            raise ValueError("model.bin was trained for a different observation or action space than this game")
        self.action_targets = np.asarray(ml["actionTargets"], dtype=np.uint8)
        self.rng = None

    def start(self, team):
        self.rng = None if GREEDY else np.random.default_rng()

    def act(self, state):
        return {"actions": self.net.act(state["obs"], state["mask"], self.action_targets, self.rng)}


if __name__ == "__main__":
    run(ModelBot())

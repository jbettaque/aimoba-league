# Entering a trained model

There are three ways to enter a model, from no code at all to complete freedom.

| Way | Your entry's folder holds | Use it when |
| --- | --- | --- |
| `policy` | One file, `model.bin`, exported from the repository's trainer | Your network has the trainer's architecture (an MLP of any width and depth). |
| `python` | A Python script plus your weights | You have your own architecture, your own observation features, or want rules around the network. |
| `command` | Any program | You do inference in another language. See [Other languages](protocol.md). |

All three use the same interface to the game, described here. A model does not have to use it: a Python agent
can ask for the structured view instead (or as well) and build its own features from it, and it can answer with
plain orders. See [What a bot sees and can do](api.md).

## The quickest start: enter the model that ships with the game

```
./aimoba init my-model --kit policy
./aimoba bench my-model expert -n 20
./aimoba submit my-model
```

`init` copies the game's self-play network into `bots/my-model/model.bin`. Replace it with your own:

```
./aimoba export runs/<your run>/latest.pt my-model
```

## Training

`python/aimoba` is a vectorised environment over the game (the simulation runs on the CPU cores, one process
per core), and `python/train` holds trainers that use it with the policy on a GPU. The route that works is
imitation of the expert bot, then PPO against the scripted bots, then self-play. PPO from scratch spends tens
of millions of steps before it takes a single tower.

```
python -m venv .venv
.venv/bin/python -m pip install numpy mlflow
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130    # a CUDA build; on Windows: .venv\Scripts\python

# 1. learn from the expert bot
.venv/bin/python python/train/imitate.py --run imitate --updates 400
# 2. improve on it with PPO against the bots; the imitation term fades out over the first 300 updates
.venv/bin/python python/train/ppo.py --run finetune --resume runs/imitate/latest.pt --opponent easy,basic,expert --expert expert --bc-coef 0.5 --bc-updates 300 --critic-updates 20 --lr 1e-4 --ent 0.001 --gamma 0.997 --updates 600
# 3. self-play against earlier snapshots, with some games still against the expert bot
.venv/bin/python python/train/ppo.py --run league --resume runs/finetune/latest.pt --opponent expert --self-play-procs 16 --pool-every 100 --pool-size 20 --lr 1e-4 --ent 0.002 --gamma 0.997 --updates 2000

./aimoba export runs/league/latest.pt my-model
```

On a 12-core CPU with an RTX 3080 Ti the three stages took about 20 minutes, 40 minutes and 3 hours. Every
script explains its options with `--help`; `python/train/evaluate.py` plays a checkpoint against the bots or
another checkpoint over a fixed set of games. A run writes to `runs/<name>/`.

The training environment and a league match give a model exactly the same observations and apply its actions
the same way, at the same five decisions a second, so what you train is what plays. To use the environment
with your own trainer:

```python
from aimoba import AimobaParallelEnv          # run from python/, or pip install -e python

env = AimobaParallelEnv(num_procs=8, envs_per_proc=16, agents="blue", opponent="expert")
obs, mask = env.reset(seed=0)                 # obs (128, 5, 381), mask (128, 5, 30)
obs, reward, done, mask, infos = env.step(actions)    # actions (128, 5, 4)
```

To measure a model, play it against others with `./aimoba bench` or `./aimoba tournament`. They sample from the
policy as training does. Greedy play (always the most likely action) is stronger against the scripted bots for
some models and erratic between two models; set `"greedy": true` in `aimoba.json` to enter a policy that way.

## The observation

Ask for it with `observations: ['obs']`. Each decision you get, for each of your five heroes,

- `obs`: 381 floats, egocentric and normalised, mirrored for Red so that one network plays both sides,
- `mask`: 30 bytes: which of the 8 action kinds are possible now, then which of the 22 target slots are occupied.

In JavaScript `state.obs` is a `Float32Array` of 5 x 381 and `state.mask` a `Uint8Array` of 5 x 30; in Python
they are numpy arrays of shape (5, 381) and (5, 30). Row `i` belongs to hero `i`.

The observation is made of blocks, in this order (`info.ml.obsBlocks` gives the same list as data):

| Block | Size | What |
| --- | --- | --- |
| global | 12 | Time, kill difference, towers lost by each side, Wyrm stacks, which epic monsters and own jungle camps are up |
| self | 48 | Position, health, mana, level, gold, stats, each ability's rank, cooldown and readiness, status effects, which hero this is (one-hot at `info.ml.heroOffset`), attack timing, the order in progress |
| allies | 5 x 9 | Alive, relative position, health, mana, level, whether it is this hero, crowd control, respawn |
| enemies | 5 x 20 | Visible, alive, relative position, distance, health, level, crowd control, range, in range, velocity, whether it targets you, a dash in progress, time since it was last seen, respawn, whether a creep blocks your line of fire |
| creeps | 8 x 9 | The eight nearest enemy minions and monsters: position, health, whether your next attack kills it, in range, whether it targets you |
| allyMinions | 4 x 4 | The four nearest allied minions |
| enemyStructures | 3 x 8 | The three nearest enemy structures: position, health, vulnerable, whether you stand in its range, tier, whether it targets you |
| allyStructures | 2 x 4 | The two nearest allied structures |
| lanes | 6 | How far each side's front minion has pushed in each lane |
| projectiles | 3 x 12 | Enemy skillshots that could reach you, soonest first: position, direction, speed, width, range left, sideways miss distance, seconds until it arrives, whether it hits if you stand still |
| zones | 2 x 7 | Enemy ground effects about to go off near you |

Fog of war applies: an enemy hero you cannot see is reported where your team last saw it, with the age of
that sighting. The exact features are in `observe` in `src/sim/ml.ts`.

## The action

One action per hero and decision: four integers `[kind, direction, distance, target]`
(`MultiDiscrete([8, 64, 4, 22])`).

| Component | Values |
| --- | --- |
| kind | 0 do nothing (the running order carries on), 1 move, 2 attack, 3 to 6 cast Q, W, E, R, 7 recall |
| direction | 64 compass directions, for moving and for aiming |
| distance | 4 steps of the move or cast range |
| target | 0 to 4 enemy heroes, 5 to 12 the creeps of the observation, 13 to 15 its enemy structures, 16 to 20 allied heroes, 21 none |

Target slots line up with the observation blocks: target 6 is the second creep. An aimed spell with an enemy as
target is aimed at that unit, leading its movement; with target 21 it goes where direction and distance say.

Not every kind uses every component. `info.ml.actionHeads[hero][kind]` says which of direction, distance and
target a kind uses, and `info.ml.actionTargets[hero][kind][slot]` which target slots make sense for it. Mask the
kind with `mask[:8]` and the target with `mask[8:]` combined with `actionTargets`.

Return the actions from `act`:

```python
def act(self, state):
    actions = self.net(state["obs"], state["mask"])     # shape (5, 4)
    return {"actions": actions}
```

or per hero, mixed with ordinary orders: `action(hero, [1, 12, 3, 21])`.

Heroes level up and shop by themselves unless you turn that off, as in training.

## A Python agent with your own network

```
./aimoba init my-agent --kit python-model
```

gives a script that loads `model.bin` and evaluates it with numpy. Replace the `Network` class with your own
inference and put your weights next to it. On the league's machines an entry has one CPU core, no GPU, no
network access and a read-only folder, and Python has numpy, PyTorch (CPU) and ONNX Runtime installed (see
[Rules](rules.md) for the exact environment and limits). Other pure-Python packages can be copied into the
entry's folder.

Keep an eye on time: `./aimoba validate my-agent` reports how long your decisions take against the budget.
Loading the model counts toward the start-up limit, not the thinking time.

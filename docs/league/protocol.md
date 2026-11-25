# Other languages

A bot is a program that reads JSON messages on standard input and writes JSON messages on standard output, one
message per line. If your language can do that, it can play. The JavaScript and Python kits are thin wrappers
around exactly this.

## The entry

A folder with a manifest, `aimoba.json`:

```json
{ "name": "my-bot", "runtime": "command", "command": ["./my-bot", "--some-flag"] }
```

A program named with a path (`./my-bot`) is taken from the entry's folder; a bare name (`java`) from the
machine. On the league, `command` entries run in a plain Debian image: ship a statically linked binary, or
everything your program needs, in the folder.

Run it locally like any other bot: `./aimoba play bots/my-bot easy`.

## The conversation

All messages are single-line JSON objects with a `type`. Anything written to standard error is kept as the
bot's log. Nothing but protocol messages may be written to standard output.

```
runner -> bot   {"type":"hello","protocol":1,"info":{...}}
bot -> runner   {"type":"ready","observations":["view"],"decisionTicks":6}
runner -> bot   {"type":"start","team":0}
runner -> bot   {"type":"step","tick":0,"view":{...}}
bot -> runner   {"type":"act","orders":[{"hero":2,"type":"attack_move","x":6900,"y":1100}]}
runner -> bot   {"type":"step","tick":6,"view":{...}}
bot -> runner   {"type":"act"}
...
runner -> bot   {"type":"end","result":"win","reason":"core","summary":{...}}
```

| Message | Fields |
| --- | --- |
| `hello` | `protocol` (1) and `info`: the static game data and the rules (see [game data](api.md#game-data)). Answer within `info.rules.initMs` milliseconds. |
| `ready` | Your options, all optional: `name`, `observations`, `decisionTicks`, `mirror`, `autoLevel`, `autoBuy`, `autoAttack` (see [options](api.md#options)). |
| `start` | `team`: 0 Blue, 1 Red. No answer expected. |
| `step` | `tick`, and what you asked for: `view` (the [view](api.md#the-view)) and / or `obs` and `mask`. Answer every `step` with one `act`. |
| `act` | `orders`: a list of [orders](api.md#orders). And / or `actions`: five [ML actions](ml.md#the-action), `[[kind, direction, distance, target], ...]`. Both may be left out. |
| `end` | `result` (`win`, `loss`, `draw`), `reason`, `summary`. The program should exit; it is stopped shortly after. |

`obs` is base64 of 5 x 381 little-endian 32-bit floats, `mask` base64 of 5 x 30 bytes, one row per hero.

The game does not advance while the runner waits for an `act`, but the wait is on the clock; see
[Rules](rules.md#thinking-time). Flush standard output after every message.

## A complete bot

In shell-flavoured pseudocode:

```
read line                      # hello
print {"type":"ready"}
read line                      # start
loop:
  read line
  if type == "end": exit
  print {"type":"act","orders":[ ...one attack_move per living hero... ]}
```

`python/aimoba_agent.py` is a full implementation in 300 lines and a good template for a port.

## What ends a game early

| Reason | When |
| --- | --- |
| `init` | No `ready` within the start-up limit, or the program could not be started. |
| `crash` | The program exited before the game was over. |
| `timeout` | The thinking-time reserve ran out. |
| `protocol` | A line on standard output was not a JSON object, the wrong message type came back, or a message was longer than 1 MiB. |

Each of these forfeits the game. Invalid orders do not: they are dropped and noted in the log.

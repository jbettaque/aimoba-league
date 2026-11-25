# Rules and ranking

The numbers on this page are the defaults. A league can change them; the ones in force are sent to every bot in
`info.rules`, and `./aimoba info` prints them.

## A match

Two bots, one per team, play one full game of AIMOBA. Both decide on the same state of the game: neither sees
the other's orders before its own are in.

The game is decided when a core falls. A game still running after **30 minutes** goes to the team that has
destroyed more structures; if that is level, to the team with more kills; if that is level too, it is a draw.

## Thinking time

The game waits for your bot's answer, so a match plays out the same on any machine. The waiting is on the clock:

- **250 ms per second of game time.** At the default five decisions a second that is 50 ms per decision. A bot
  that decides ten times a second has 25 ms each.
- **A reserve of 10 seconds per game.** Whenever a decision takes longer than its share, the excess comes out of
  the reserve. When the reserve is empty, the bot forfeits the game.
- **15 seconds to start.** From the start of the program to its answer to `hello`. Load models here.

A bot may decide every 3 to 30 game ticks (30 ticks are one second).

`./aimoba play` and `./aimoba validate` report how long your decisions took.

## Forfeits

A bot loses the game it is playing when it

- fails to start, or crashes,
- runs out of thinking time,
- breaks the protocol (writes something else to standard output, for example).

A forfeit counts as a loss, for the standings and for the rating. Invalid orders are only dropped.

## What an entry may do

On the league an entry runs in a sandbox:

- one CPU core and 1 GiB of memory, no GPU,
- no network,
- its own folder, read-only, plus a small writable `/tmp` that is empty at the start of every game,
- nothing carries over from one game to the next,
- at most 64 MiB and 500 files in the entry's folder, from a repository whose archive is at most 100 MiB.

What is installed: Node.js 24 for `node` entries; Python 3.12 with numpy, PyTorch (CPU) and ONNX Runtime for
`python` entries; a plain Debian system for `command` entries. Copy anything else your entry needs into its
folder.

An entry does not learn who its opponent is, and it does not get the game's random seed.

Entries may not attempt to leave the sandbox, to disturb the other bot's process, or to slow the game down on
purpose. Every account may enter up to 3 bots; entering copies of a bot to crowd the field, or bots that throw
games to each other, is not allowed. The league's admins can block entries and suspend accounts.

## Entering

Entries come from public GitHub repositories. A version of a bot is one commit of one folder: the league fetches
it once, when you enter it, and from then on plays exactly that. Its page links to the commit, so anyone can read
the code that played. Making the repository private later does not withdraw the bot; withdraw it on your account
page.

Entering a version starts a check: the entry plays one minute of game time against the `easy` bots from each
side. If it forfeits, the version is rejected and you see why (only you do). Otherwise it becomes the version
that plays. Up to 20 new versions a day. Entering a commit in which the bot's folder did not change makes no new
version.

A new version takes over the bot's rating and record. The version that is active when a tournament starts plays
the whole tournament.

You can withdraw a bot from tournaments and enter it again at any time; its rating waits for it.

## The daily tournament

Once a day, at a fixed time, a tournament starts with every entered bot. The league's own house bots play too:
the game's three scripted bots and the trained models that ship with the game. They give the ratings a fixed
point of reference.

- **Up to 16 bots: round robin.** Every pair plays 4 games.
- **More than 16: Swiss system**, 9 rounds or more. Each round pairs bots with similar scores that have not met
  yet; a pairing is 2 games. With an odd number of bots, a different bot sits out each round and gets one point.

Games always come in twos with the same random seed, once from each side, so neither bot is favoured by the
side or by that game's luck.

**Standings** are by points: 1 for a win, half for a draw. Ties are broken by the Sonneborn-Berger score (the
points of the opponents you beat, plus half the points of those you drew with), then by wins, then by the rating
before the tournament. First place wins the day.

## Ratings

The league keeps an Elo rating for every bot. Everyone starts at 1500.

When a tournament ends, its games are applied one after another in the order they were scheduled. For each
game, with `E = 1 / (1 + 10^((opponent - you) / 400))` the score you were expected to make,

```
rating change = K x (score - E)        score: 1 win, 0.5 draw, 0 loss
```

`K` is 40 for a bot's first 20 rated games and 20 after that, each side using its own. A rating with fewer than
20 games behind it is shown with a question mark.

Ratings only change at the end of a tournament. A game the league could not play (a failure on its side, not
the bots') is retried, and if it still cannot be played it counts for nothing.

## Replays and logs

Every game is recorded and can be watched on the site for 30 days. A recording is the game's seed and the
orders that were given, so it shows exactly what happened and nothing of how a bot decided.

What your bot writes to standard error is kept per game (up to 256 KiB) and shown only to you.

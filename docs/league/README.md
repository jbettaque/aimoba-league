# Get started

The league is a daily tournament between programs that play AIMOBA. You write one program (a bot); it controls
all five heroes of a team. Every day every entered bot plays every other one, and the standings and ratings are
updated. A bot can be a few rules in JavaScript or Python, a trained neural network, or any program at all.

This page gets a bot of yours playing in five minutes. Then:

- [What a bot sees and can do](api.md): every field of the game state and every order
- [Entering a trained model](ml.md): the fixed-size observation, training, and entering a network
- [Other languages](protocol.md): the protocol, if you want to write a bot in something else
- [Rules and ranking](rules.md): time limits, what is allowed, how the tournament and ratings work

## 1. Get the game

You need [Node.js](https://nodejs.org) 22.18 or newer. Python is only needed for Python bots and for training.

```
git clone https://github.com/jbettaque/aimoba-league aimoba
cd aimoba
npm install
```

If you want your bots under version control, start your own repository from the template instead (on GitHub:
"Use this template") and clone that.

Everything below uses the `aimoba` command in that folder: `./aimoba` on macOS and Linux, `.\aimoba` on Windows.

## 2. Start a bot from a kit

```
./aimoba init my-bot
```

This creates `bots/my-bot/` with a working rule-based bot in JavaScript. Other kits:

| Kit | What you get |
| --- | --- |
| `--kit js` | Rule-based bot in JavaScript (the default). Fastest to iterate on. |
| `--kit python` | The same bot in Python. |
| `--kit policy` | A trained network and nothing else. The league runs it for you. See [models](ml.md). |
| `--kit python-model` | A Python agent that runs a network with numpy. Start here to use your own architecture. |

## 3. Play, watch, measure

```
./aimoba play my-bot easy --watch
```

plays one game against the built-in `easy` bots and opens it in the game client. You can pause, change speed,
follow any hero and switch between the two teams' views.

An opponent can be `easy`, `basic` or `expert` (the game's scripted bots), `policy:selfplay` (a trained
network that ships with the game), or another bot of yours.

One game says little. Measure over many, each seed played from both sides:

```
./aimoba bench my-bot easy -n 40
```

```
my-bot: 11 won, 29 lost, 0 drawn of 40  ->  28% (95% interval 16-43%)
```

That is about where the kit starts: it loses most games to `easy`, and all of them to `basic` and `expert`.
The kit is a readable skeleton, not a strong bot. Beating `easy` reliably is a good first goal.

`--fast` runs JavaScript bots inside the runner instead of in their own process: 40 games take seconds. Use it
while developing; leave it off for a last check, because only a separate process is on the clock.

Compare several bots at once the way the league does:

```
./aimoba tournament my-bot my-other-bot easy basic expert
```

## 4. Change something

Open `bots/my-bot/bot.js`. The whole bot is one function that runs five times a second:

```js
act(state, ctx) {
  const orders = [];
  for (const hero of state.heroes) {
    if (!hero.alive) continue;
    const enemy = ctx.lib.nearest(hero, state.enemies.filter((e) => e.visible && e.alive));
    if (enemy) orders.push(ctx.lib.attack(hero, enemy));
    else orders.push(ctx.lib.attackMove(hero, 6900, 1100));   // walk toward the enemy core, fighting on the way
  }
  return orders;
}
```

`state` is everything your team can see right now; you return a list of orders. An order keeps running until
you replace it, so a hero you say nothing about carries on. The map is mirrored for the Red side, so your base
is always bottom-left and the same code plays both sides. [What a bot sees and can do](api.md) has the details.

Whatever your bot prints (`console.log`, `print`) is shown after the game as its log.

## 5. Enter it

The league takes bots from **public GitHub repositories**: everyone can read the code that plays, and every
version of a bot is a commit you can point to. The easiest way to have one is to start from the template:

1. On the template's GitHub page, choose **Use this template**, then **Create a new repository**. Make it
   **public**. Then clone your new repository instead of the template, and `npm install` in it.
   (Already working in a clone of the template? Create an empty public repository on GitHub and point your
   clone at it: `git remote set-url origin https://github.com/<you>/<repo>`, then `git push -u origin main`.)
2. Your bots live in `bots/`, which is meant to be committed: `git add bots && git commit -m "my bot" && git push`.

Then check the bot the way the league will, and enter it:

```
./aimoba validate my-bot
./aimoba login --server https://<the league's address>
./aimoba submit my-bot -m "first version"
```

`login` asks for an API token, which you create on your account page on the league's site. `submit` tells the
league which repository, folder and commit your bot is (the commit you have checked out, so push it first). The
league fetches that folder from GitHub and plays it in a short check game from each side. If it gets through
without crashing or running out of time, it is accepted and plays in the next tournament.

To enter a new version, commit, push and `submit` again; the bot keeps its rating. The account page on the site
also has a **Fetch latest** button for each bot, which enters the newest commit of the branch you entered it from,
and a form to enter a bot from any public repository without the command line:

```
./aimoba submit --repo https://github.com/<you>/<repo> --path bots/my-bot     # the same, without a checkout
```

If `submit` says that your copy of the game differs from the league's, the game has been updated since you
cloned it: pull from the template (see its README) and test your bot again.

Things that keep an entry from being fetched: a private repository, a commit that is not pushed, a folder without
`aimoba.json`, files kept in Git LFS (commit them normally; GitHub takes files up to 100 MB), and a repository
whose archive is larger than 100 MB (keep training runs and replays out of it; the template's `.gitignore`
already does).

```
./aimoba status          your bots, their ratings and places
./aimoba logs 1234       what your bot printed in league match 1234
```

Every league game can be watched on the site, and you can read your own bot's log for each of them.

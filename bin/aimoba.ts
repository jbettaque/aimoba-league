#!/usr/bin/env node
/**
 * The AIMOBA league command line.
 *
 *   aimoba init my-bot                 start a bot from a kit
 *   aimoba play my-bot expert --watch  play one game and watch it
 *   aimoba bench my-bot basic -n 20    win rate over many games
 *   aimoba validate my-bot             check it the way the league will
 *   aimoba submit my-bot               enter it
 *
 * `aimoba help` lists everything.
 */
import { spawn, spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { BUILTIN_BOTS, EntryError, MANIFEST, REPO_ROOT, entryFiles, loadEntry, resolveAgent, validName, type AgentSpec } from '../src/agent/entry.ts';
import { DEFAULT_RULES, gameInfo, stepBudgetMs, type MatchRules } from '../src/agent/info.ts';
import { describeResult, runMatch, type MatchResult } from '../src/agent/match.ts';
import { SandboxError } from '../src/agent/protocol.ts';
import { MatchPool, defaultPoolSize } from '../src/agent/pool.ts';
import { defaultPython, dockerAvailable, prepareEntry, type SandboxConfig } from '../src/agent/sandbox.ts';
import { gameFingerprint } from '../src/agent/version.ts';
import { runLocalTournament } from '../src/league/local.ts';
import { wilson } from '../src/league/rating.ts';
import { parseRepo } from '../src/league/source.ts';

// --- Small helpers ---------------------------------------------------------------------------------------------
const tty = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code: string) => (s: string | number) => (tty ? `\x1b[${code}m${s}\x1b[0m` : String(s));
const bold = paint('1'), dim = paint('2'), green = paint('32'), red = paint('31'), yellow = paint('33'), cyan = paint('36');

class UsageError extends Error {}

interface Args { _: string[]; flags: Record<string, string | boolean> }

/** `--name value`, `--name=value`, `--flag`, `-n value`. `booleans` never take a value. */
function parseArgs(argv: string[], booleans: string[], short: Record<string, string> = {}): Args {
  const out: Args = { _: [], flags: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--') { out._.push(...argv.slice(i + 1)); break; }
    if (a.startsWith('--') || (a.startsWith('-') && a.length === 2 && short[a[1]])) {
      let name = a.startsWith('--') ? a.slice(2) : short[a[1]];
      let value: string | undefined;
      const eq = name.indexOf('=');
      if (eq >= 0) { value = name.slice(eq + 1); name = name.slice(0, eq); }
      if (booleans.includes(name)) out.flags[name] = value === undefined ? true : value !== 'false';
      else {
        value ??= argv[++i];
        if (value === undefined) throw new UsageError(`--${name} needs a value`);
        out.flags[name] = value;
      }
    } else out._.push(a);
  }
  return out;
}

const num = (v: string | boolean | undefined, dflt: number, name: string): number => {
  if (v === undefined || typeof v === 'boolean') return dflt;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new UsageError(`--${name} must be a number, got "${v}"`);
  return n;
};

const BOT_DIRS = ['bots', 'kits'];
const agent = (spec: string): AgentSpec => resolveAgent(spec, BOT_DIRS);
/** With --docker: check the images and mounts once, before any game is on the clock. */
async function prepare(specs: AgentSpec[], sandbox: SandboxConfig) {
  for (const s of specs) if (s.kind === 'entry') await prepareEntry(s.manifest, s.dir, sandbox);
}

function sandboxFrom(flags: Args['flags']): SandboxConfig {
  if (flags.docker) {
    const d = dockerAvailable();
    if (!d.ok) throw new UsageError(`--docker: Docker is not available (${d.detail})`);
    return { driver: 'docker' };
  }
  return { driver: 'process' };
}

function rulesFrom(flags: Args['flags']): Partial<MatchRules> {
  const r: Partial<MatchRules> = {};
  if (flags['max-time'] !== undefined) r.maxTime = num(flags['max-time'], DEFAULT_RULES.maxTime, 'max-time');
  if (flags['no-clock']) { r.bankMs = 1e12; r.initMs = 600_000; }
  return r;
}

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

function table(rows: (string | number)[][], align: string): string {
  const width = rows[0].map((_, c) => Math.max(...rows.map((r) => stripAnsi(String(r[c])).length)));
  return rows.map((r) => r.map((cell, c) => {
    const s = String(cell), pad = ' '.repeat(width[c] - stripAnsi(s).length);
    return align[c] === 'r' ? pad + s : s + pad;
  }).join('  ').trimEnd()).join('\n');
}
// eslint-disable-next-line no-control-regex
const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, '');

function openBrowser(url: string) {
  const [cmd, args] = process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]] : process.platform === 'darwin' ? ['open', [url]] : ['xdg-open', [url]];
  try { spawn(cmd, args as string[], { stdio: 'ignore', detached: true }).unref(); } catch { /* the URL is printed anyway */ }
}

function timing(r: MatchResult, rules: MatchRules): string {
  const rows: (string | number)[][] = [[dim('team'), dim('runs as'), dim('decisions'), dim('mean ms'), dim('max ms'), dim('budget ms'), dim('bank left'), dim('dropped orders')]];
  r.teams.forEach((t, i) => {
    if (t.runtime === 'builtin') return;
    const budget = t.decisions ? stepBudgetMs(rules, Math.round(r.ticks / t.decisions) || 6) : 0;
    rows.push([`${i === 0 ? 'blue' : 'red'} ${t.name}`, t.runtime, t.decisions, t.meanMs.toFixed(2), t.maxMs.toFixed(1), budget.toFixed(0), `${(t.bankLeftMs / 1000).toFixed(1)} s`, t.rejectedOrders ? yellow(t.rejectedOrders) : 0]);
  });
  return rows.length > 1 ? table(rows, 'llrrrrrr') : '';
}

function printLogs(r: MatchResult, lines: number) {
  r.teams.forEach((t, i) => {
    if (t.failure) console.log(red(`\n${i === 0 ? 'blue' : 'red'} ${t.name} forfeited (${t.failure.kind}): ${t.failure.message}`));
    const log = t.log.trimEnd();
    if (!log) return;
    const all = log.split('\n');
    console.log(dim(`\n--- log of ${t.name}${all.length > lines ? ` (last ${lines} of ${all.length} lines; --log N for more)` : ''} ---`));
    console.log(all.slice(-lines).join('\n'));
  });
}

// --- Commands --------------------------------------------------------------------------------------------------
const KITS: Record<string, { dir: string; about: string }> = {
  js: { dir: 'js-bot', about: 'rule-based bot in JavaScript (the quickest start)' },
  python: { dir: 'python-bot', about: 'rule-based bot in Python' },
  'python-model': { dir: 'python-model', about: 'Python agent that runs a trained network (numpy; swap in your own inference)' },
  policy: { dir: 'policy', about: 'a network exported by python/train/export.py, no code at all' },
};

async function cmdInit(argv: string[]) {
  const a = parseArgs(argv, ['force'], {});
  const name = a._[0];
  const kit = String(a.flags.kit ?? 'js');
  if (!name) throw new UsageError(`usage: aimoba init <name> [--kit ${Object.keys(KITS).join('|')}] [--dir bots] [--model selfplay|file.bin|checkpoint.pt]\n\nKits:\n${Object.entries(KITS).map(([k, v]) => `  ${k.padEnd(13)} ${v.about}`).join('\n')}`);
  if (!validName(name)) throw new UsageError(`"${name}" is not a valid bot name: use 3-40 lower-case letters, digits and single dashes`);
  if (BUILTIN_BOTS.includes(name)) throw new UsageError(`"${name}" is the name of a built-in bot`);
  if (!KITS[kit]) throw new UsageError(`unknown kit "${kit}" (have: ${Object.keys(KITS).join(', ')})`);
  const dest = resolve(String(a.flags.dir ?? 'bots'), name);
  if (existsSync(dest) && readdirSync(dest).length && !a.flags.force) throw new UsageError(`${dest} already exists (use --force to overwrite files in it)`);
  mkdirSync(dest, { recursive: true });
  cpSync(join(REPO_ROOT, 'kits', KITS[kit].dir), dest, { recursive: true });
  const manifest = JSON.parse(readFileSync(join(dest, MANIFEST), 'utf8'));
  manifest.name = name;
  delete manifest.description;
  writeFileSync(join(dest, MANIFEST), JSON.stringify(manifest, null, 2) + '\n');
  if (manifest.runtime === 'python') cpSync(join(REPO_ROOT, 'python', 'aimoba_agent.py'), join(dest, 'aimoba_agent.py'));
  if (kit === 'python-model' || kit === 'policy') installModel(String(a.flags.model ?? 'selfplay'), dest);
  const rel = relative(process.cwd(), dest) || '.';
  console.log(`${green('created')} ${rel} from the ${bold(kit)} kit\n`);
  console.log(`  ${cyan(`aimoba play ${rel} easy --watch`)}    play a game and watch it`);
  console.log(`  ${cyan(`aimoba bench ${rel} basic -n 20`)}    measure it over 20 games`);
  console.log(`  ${cyan(`aimoba validate ${rel}`)}             check it the way the league will`);
  console.log(`  ${cyan(`aimoba submit ${rel}`)}               enter it\n`);
  console.log(dim(`Start in ${join(rel, manifest.entry ?? MANIFEST)}. The API is described in docs/league/.`));
}

/** Put a network into an entry as model.bin: a model id from public/policies, a .bin file, or a training checkpoint (.pt). */
function installModel(model: string, dest: string) {
  const out = join(dest, 'model.bin');
  if (model.endsWith('.pt')) {
    if (!existsSync(model)) throw new UsageError(`${model} does not exist`);
    const r = spawnSync(defaultPython(), [join(REPO_ROOT, 'python', 'train', 'export.py'), resolve(model), '--id', 'model', '--out', dest, '--no-index'], { stdio: 'inherit' });
    if (r.status !== 0) throw new UsageError('exporting the checkpoint failed (it needs the Python environment with PyTorch, see README "Training on a GPU")');
    return;
  }
  const listed = join(REPO_ROOT, 'public', 'policies', `${model}.bin`);
  const src = existsSync(model) ? model : existsSync(listed) ? listed : null;
  if (!src) throw new UsageError(`no model "${model}": give a model id from public/policies, a .bin file or a .pt checkpoint`);
  cpSync(src, out);
}

async function cmdExport(argv: string[]) {
  const a = parseArgs(argv, [], {});
  const [checkpoint, dir] = a._;
  if (!checkpoint || !dir) throw new UsageError('usage: aimoba export <checkpoint.pt | model id | model.bin> <bot directory>   (writes model.bin into the entry)');
  const entry = loadEntry(existsSync(dir) ? dir : join('bots', dir));
  installModel(checkpoint, entry.dir);
  console.log(`${green('wrote')} ${relative(process.cwd(), join(entry.dir, 'model.bin'))}`);
}

async function cmdPlay(argv: string[]) {
  const a = parseArgs(argv, ['watch', 'fast', 'docker', 'no-clock', 'no-replay', 'json'], { o: 'out' });
  const [blue, redName] = a._;
  if (!blue || !redName) throw new UsageError('usage: aimoba play <blue> <red> [--seed N] [--watch] [--fast] [--out file] [--max-time S] [--log N] [--docker] [--no-clock] [--json]');
  const seed = num(a.flags.seed, Math.floor(Math.random() * 1e9) + 1, 'seed');
  const rules = { ...DEFAULT_RULES, ...rulesFrom(a.flags) };
  const specs = [agent(blue), agent(redName)];
  const playSandbox = sandboxFrom(a.flags);
  await prepare(specs, playSandbox);
  const quiet = a.flags.json === true;
  if (!quiet) process.stdout.write(`${bold(specs[0].name)} ${dim('(blue)')} vs ${bold(specs[1].name)} ${dim('(red)')}, seed ${seed}\n`);
  let last = 0;
  const r = await runMatch({
    blue: specs[0], red: specs[1], seed, rules, sandbox: playSandbox, inProcess: a.flags.fast === true,
    onProgress: quiet || !tty ? undefined : (g) => {
      if (g.tick - last < 30 * 30) return;
      last = g.tick;
      process.stdout.write(`\r${dim(`${clock(g.time)}  kills ${g.teamKills[0]}-${g.teamKills[1]}  structures ${g.towersLost[1]}-${g.towersLost[0]}`)}   `);
    },
  });
  let file: string | undefined;
  if (r.replay && !a.flags['no-replay']) {
    file = resolve(String(a.flags.out ?? join('replays', `${specs[0].name}-vs-${specs[1].name}-${seed}.replay.json`)));
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(r.replay));
  }
  if (quiet) {
    const { replay: _, ...rest } = r;
    console.log(JSON.stringify({ ...rest, replayFile: file }, null, 2));
    return;
  }
  if (tty) process.stdout.write('\r' + ' '.repeat(60) + '\r');
  const colour = r.winner === 0 ? green : r.winner === 1 ? red : yellow;
  console.log(colour(bold(describeResult(r))));
  const t = timing(r, rules);
  if (t) console.log('\n' + t);
  printLogs(r, num(a.flags.log, 15, 'log'));
  if (file) {
    console.log(`\nreplay: ${relative(process.cwd(), file)}  ${dim(`(aimoba watch ${relative(process.cwd(), file)})`)}`);
    if (a.flags.watch) await watch(file, num(a.flags.port, 5174, 'port'));
  }
  if (r.teams.some((x) => x.failure)) process.exitCode = 2;
}

async function watch(file: string, port: number) {
  const { createServer } = await import('vite');
  const data = readFileSync(file);
  const server = await createServer({
    root: REPO_ROOT, logLevel: 'error', cacheDir: join(REPO_ROOT, '.vite', 'watch'),
    server: { port, strictPort: false, host: '127.0.0.1' },
    plugins: [{
      name: 'aimoba-replay',
      configureServer(s) {
        s.middlewares.use('/__replay.json', (_req, res) => {
          res.setHeader('Content-Type', 'application/json');
          res.end(data);
        });
      },
    }],
  });
  await server.listen();
  const addr = server.httpServer?.address();
  const url = `http://127.0.0.1:${typeof addr === 'object' && addr ? addr.port : port}/?replay=/__replay.json`;
  console.log(`\nwatching at ${cyan(url)}  ${dim('(Ctrl+C to stop)')}`);
  openBrowser(url);
  await new Promise(() => {});
}

async function cmdWatch(argv: string[]) {
  const a = parseArgs(argv, [], {});
  const file = a._[0];
  if (!file) throw new UsageError('usage: aimoba watch <replay.json> [--port 5174]');
  if (!existsSync(file)) throw new UsageError(`${file} does not exist`);
  await watch(resolve(file), num(a.flags.port, 5174, 'port'));
}

async function cmdBench(argv: string[]) {
  const a = parseArgs(argv, ['fast', 'docker', 'no-clock', 'json'], { n: 'games', j: 'jobs' });
  const [first, second] = a._;
  if (!first || !second) throw new UsageError('usage: aimoba bench <bot> <opponent> [-n 20] [-j jobs] [--seed N] [--fast] [--max-time S] [--docker] [--replays dir]');
  const games = Math.max(2, Math.ceil(num(a.flags.games, 20, 'games') / 2) * 2);
  const seed0 = num(a.flags.seed, 1000, 'seed');
  const specs = [agent(first), agent(second)];
  const pool = new MatchPool(Math.min(games, num(a.flags.jobs, defaultPoolSize(), 'jobs')));
  const replays = typeof a.flags.replays === 'string' ? resolve(a.flags.replays) : null;
  const rules = rulesFrom(a.flags), sandbox = sandboxFrom(a.flags);
  await prepare(specs, sandbox);
  let wins = 0, losses = 0, draws = 0, forfeits = 0, failed = 0, done = 0, seconds = 0, blueWins = 0;
  const t0 = Date.now();
  const reasons: Record<string, number> = {};
  if (!a.flags.json) console.log(`${bold(specs[0].name)} vs ${bold(specs[1].name)}: ${games} games, each seed from both sides, ${pool.size} at a time`);
  try {
    await Promise.all(Array.from({ length: games }, async (_, i) => {
      // Games 2k and 2k+1 share a seed with the sides swapped.
      const mine = i % 2;
      const seed = seed0 + (i >> 1);
      let r: MatchResult;
      try {
        r = await pool.run({ blue: specs[mine], red: specs[1 - mine], seed, rules, sandbox, inProcess: a.flags.fast === true, record: replays !== null, replayFile: replays ? join(replays, `${seed}-${mine === 0 ? 'blue' : 'red'}.replay.json`) : undefined });
      } catch (err) {
        failed++;
        if (!a.flags.json) console.error(red(`\ngame ${i} could not be played: ${(err as Error).message.split('\n')[0]}`));
        return;
      }
      const won = r.winner < 0 ? null : r.winner === mine;
      if (won === null) draws++; else if (won) wins++; else losses++;
      if (r.winner === 0) blueWins++;
      if (r.teams[mine].failure) { forfeits++; reasons[`forfeit: ${r.teams[mine].failure!.message}`] = (reasons[`forfeit: ${r.teams[mine].failure!.message}`] ?? 0) + 1; }
      seconds += r.duration;
      done++;
      if (tty && !a.flags.json) process.stdout.write(`\r${dim(`${done}/${games}`)}  ${green(wins)} won  ${red(losses)} lost  ${yellow(draws)} drawn   `);
    }));
  } finally {
    pool.close();
  }
  const n = wins + losses + draws;
  const score = wins + draws / 2;
  const [lo, hi] = wilson(score, n);
  if (a.flags.json) {
    console.log(JSON.stringify({ bot: specs[0].name, opponent: specs[1].name, games: n, wins, losses, draws, forfeits, failed, score: n ? score / n : 0, ci95: [lo, hi], meanGameSeconds: n ? seconds / n : 0 }));
    return;
  }
  if (tty) process.stdout.write('\r' + ' '.repeat(60) + '\r');
  console.log(`${bold(specs[0].name)}: ${green(`${wins} won`)}, ${red(`${losses} lost`)}, ${yellow(`${draws} drawn`)} of ${n}  ->  ${bold(`${n ? ((score / n) * 100).toFixed(0) : 0}%`)} ${dim(`(95% interval ${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%)`)}`);
  console.log(dim(`mean game ${clock(n ? seconds / n : 0)}, blue won ${blueWins} of ${n}, ${((Date.now() - t0) / 1000).toFixed(0)} s`));
  if (forfeits) {
    console.log(red(`\n${specs[0].name} forfeited ${forfeits} game${forfeits === 1 ? '' : 's'}:`));
    for (const [why, k] of Object.entries(reasons)) console.log(red(`  ${k} x ${why}`));
    console.log(dim(`run "aimoba play ${first} ${second}" to see the log`));
  }
  if (replays) console.log(dim(`replays in ${relative(process.cwd(), replays)}`));
  if (failed || forfeits) process.exitCode = 2;
}

async function cmdTournament(argv: string[]) {
  const a = parseArgs(argv, ['fast', 'docker', 'no-clock', 'json'], { j: 'jobs' });
  if (a._.length < 2) throw new UsageError('usage: aimoba tournament <bot> <bot> [<bot> ...] [--games-per-pair 4] [-j jobs] [--seed N] [--fast] [--replays dir]');
  const specs = a._.map(agent);
  const ids = specs.map((s) => s.name);
  if (new Set(ids).size !== ids.length) throw new UsageError('two entrants have the same name');
  const pool = new MatchPool(num(a.flags.jobs, defaultPoolSize(), 'jobs'));
  const replays = typeof a.flags.replays === 'string' ? resolve(a.flags.replays) : null;
  const t0 = Date.now();
  try {
    const out = await runLocalTournament({
      entrants: specs.map((spec) => ({ id: spec.name, spec })), pool, seed: num(a.flags.seed, Math.floor(Math.random() * 1e9) + 1, 'seed'),
      format: { gamesPerPair: num(a.flags['games-per-pair'], 4, 'games-per-pair'), roundRobinMax: num(a.flags['round-robin-max'], 16, 'round-robin-max') },
      rules: rulesFrom(a.flags), sandbox: sandboxFrom(a.flags), inProcess: a.flags.fast === true,
      replayFile: replays ? (f) => join(replays, `${String(f.index).padStart(4, '0')}-${f.blue}-vs-${f.red}.replay.json`) : undefined,
      onGame: (_f, _r, done, total) => { if (tty && !a.flags.json) process.stdout.write(`\r${dim(`${done}/${total} games`)}   `); },
    });
    if (a.flags.json) {
      console.log(JSON.stringify({ format: out.format, standings: out.standings, games: out.games.map((g) => ({ ...g.fixture, winner: g.result?.winner ?? null, reason: g.result?.reason ?? 'void' })) }, null, 2));
      return;
    }
    if (tty) process.stdout.write('\r' + ' '.repeat(40) + '\r');
    console.log(`${bold('Tournament')} ${dim(`${out.format}, ${out.games.length} games, ${((Date.now() - t0) / 1000).toFixed(0)} s`)}\n`);
    const rows: (string | number)[][] = [[dim('#'), dim('bot'), dim('points'), dim('W'), dim('D'), dim('L'), dim('SB'), dim('rating'), dim('change')]];
    for (const s of out.standings) rows.push([s.rank, s.rank === 1 ? bold(s.id) : s.id, s.points, s.wins, s.draws, s.losses, s.sb.toFixed(1), Math.round(s.rating), (s.ratingChange >= 0 ? green : red)(`${s.ratingChange >= 0 ? '+' : ''}${Math.round(s.ratingChange)}`)]);
    console.log(table(rows, 'rlrrrrrrr'));
    const voids = out.games.filter((g) => !g.result).length;
    const forfeits = out.games.filter((g) => g.result && g.result.teams.some((t) => t.failure));
    if (voids) console.log(red(`\n${voids} games could not be played`));
    if (forfeits.length) {
      console.log(yellow(`\n${forfeits.length} games ended by forfeit:`));
      const seen = new Set<string>();
      for (const g of forfeits) for (const t of g.result!.teams) if (t.failure && !seen.has(t.name + t.failure.message)) { seen.add(t.name + t.failure.message); console.log(yellow(`  ${t.name}: ${t.failure.message}`)); }
    }
    if (replays) console.log(dim(`\nreplays in ${relative(process.cwd(), replays)}`));
  } finally {
    pool.close();
  }
}

/** Everything the league checks before an entry may play, run locally. */
async function cmdValidate(argv: string[]) {
  const a = parseArgs(argv, ['docker', 'fast'], {});
  const dir = a._[0];
  if (!dir) throw new UsageError('usage: aimoba validate <bot directory> [--docker]');
  const spec = agent(dir);
  if (spec.kind !== 'entry' || !existsSync(join(spec.dir, MANIFEST))) throw new UsageError(`${dir} is not an entry: it needs a directory with ${MANIFEST}`);
  const ok = (s: string) => console.log(`${green('ok')}    ${s}`);
  const bad = (s: string) => { console.log(`${red('fail')}  ${s}`); process.exitCode = 1; };
  ok(`${MANIFEST}: "${spec.manifest.name}", runtime ${spec.manifest.runtime}`);
  const { bytes, files } = entrySize(spec.dir);
  const limit = 64 * 1024 * 1024, fileLimit = 500;
  (bytes <= limit && files <= fileLimit ? ok : bad)(`size: ${files} files, ${(bytes / 1024).toFixed(0)} KiB (limits: ${fileLimit} files, ${limit / 1024 / 1024} MiB)`);
  // Where the league would fetch it from.
  const warn = (s: string) => console.log(`${yellow('warn')}  ${s}`);
  try {
    const src = localSource(spec.dir);
    ok(`source: github.com/${src.repo.owner}/${src.repo.repo}${src.path ? `, folder ${src.path}` : ''}, commit ${src.commit.slice(0, 7)}${src.branch ? ` on ${src.branch}` : ''}`);
    if (src.dirty.length) warn(`uncommitted changes in the folder (${src.dirty.length} file${src.dirty.length === 1 ? '' : 's'}): the league only sees what is committed and pushed`);
    if (!src.pushed) warn(`commit ${src.commit.slice(0, 7)} does not look pushed to ${src.remote}: run git push before you submit`);
    for (const f of entryFiles(spec.dir)) {
      const size = statSync(join(spec.dir, f)).size;
      if (size > 95 * 1024 * 1024) bad(`${f} is ${(size / 1048576).toFixed(0)} MB: GitHub refuses files over 100 MB`);
      else if (size < 300 && readFileSync(join(spec.dir, f), 'utf8').startsWith('version https://git-lfs.github.com/spec/')) bad(`${f} is a Git LFS pointer: the league cannot read files kept in LFS; commit the file itself`);
    }
  } catch (err) {
    if (!(err instanceof UsageError)) throw err;
    warn(`source: ${err.message}`);
  }
  // A short game from each side under the league's rules, against the easy bots.
  const rules: MatchRules = { ...DEFAULT_RULES, maxTime: 90 };
  for (const side of [0, 1]) {
    const easy = agent('easy');
    const r = await runMatch({ blue: side === 0 ? spec : easy, red: side === 0 ? easy : spec, seed: 4242 + side, rules, sandbox: sandboxFrom(a.flags), inProcess: a.flags.fast === true, record: false });
    const t = r.teams[side];
    const label = side === 0 ? 'as blue' : 'as red';
    if (t.failure) {
      bad(`${label}: forfeited (${t.failure.kind}): ${t.failure.message}`);
      if (t.log.trim()) console.log(dim(t.log.trimEnd().split('\n').slice(-20).map((l) => '        ' + l).join('\n')));
      continue;
    }
    const budget = stepBudgetMs(rules, Math.round(r.ticks / Math.max(1, t.decisions)) || 6);
    ok(`${label}: played ${clock(r.duration)} of game time, ${t.decisions} decisions, mean ${t.meanMs.toFixed(2)} ms, longest ${t.maxMs.toFixed(1)} ms (budget ${budget.toFixed(0)} ms each + ${rules.bankMs / 1000} s bank), started in ${t.initMs} ms`);
    if (t.meanMs > budget) bad(`${label}: the mean decision takes longer than its budget; over a full game the bank will run out`);
    else if (t.meanMs > budget * 0.6) console.log(`${yellow('warn')}  ${label}: decisions use ${Math.round((t.meanMs / budget) * 100)}% of the budget on this machine; the league's machines may be slower`);
    if (t.rejectedOrders) console.log(`${yellow('warn')}  ${label}: ${t.rejectedOrders} orders were dropped as invalid, e.g.\n${dim(t.log.split('\n').filter((l) => l.startsWith('[runner]')).slice(0, 3).map((l) => '        ' + l).join('\n'))}`);
  }
  console.log(process.exitCode ? red('\nnot ready to submit') : green(`\nready: aimoba submit ${dir}`));
}

function cmdInfo(argv: string[]) {
  const a = parseArgs(argv, ['json'], {});
  const info = gameInfo();
  if (a.flags.json) { console.log(JSON.stringify(info, null, 2)); return; }
  console.log(bold('Rules of a match'));
  const r = info.rules;
  console.log(`  game time limit ${r.maxTime / 60} min, then decided on structures, then kills`);
  console.log(`  thinking time ${r.msPerGameSecond} ms per game second (${stepBudgetMs(r, 6)} ms per decision at 5 decisions a second) + ${r.bankMs / 1000} s bank per game`);
  console.log(`  decisions every ${r.minDecisionTicks}-${r.maxDecisionTicks} ticks (${info.tickRate} ticks a second), start-up within ${r.initMs / 1000} s`);
  console.log(bold('\nHeroes') + dim('  (index in your team, lane, abilities with targeting and range)'));
  for (const h of info.heroes) console.log(`  ${h.index} ${h.id.padEnd(8)} ${h.role.padEnd(9)} ${['jungle', 'top', 'mid', 'bot'][h.lane + 1].padEnd(6)} ${h.abilities.map((ab) => `${ab.key} ${ab.name} (${ab.targeting}${ab.range ? ' ' + ab.range : ''})`).join(', ')}`);
  console.log(bold('\nML interface'));
  console.log(`  observation ${info.ml.obsSize} floats per hero, mask ${info.ml.maskSize}, action ${JSON.stringify(info.ml.actionNvec)} = [kind, direction, distance, target]`);
  console.log(dim('\n--json prints everything an agent receives in "hello" (map, items, unit stats, observation layout)'));
}

// --- The league ------------------------------------------------------------------------------------------------
const CRED_FILE = join(homedir(), '.aimoba', 'credentials.json');
interface Credentials { server: string; token: string; login: string }

function credentials(flags: Args['flags']): Credentials {
  const envToken = process.env.AIMOBA_TOKEN;
  const saved: Partial<Credentials> = existsSync(CRED_FILE) ? JSON.parse(readFileSync(CRED_FILE, 'utf8')) : {};
  const server = String(flags.server ?? process.env.AIMOBA_SERVER_URL ?? saved.server ?? '').replace(/\/+$/, '');
  const token = envToken ?? saved.token;
  if (!server || !token) throw new UsageError('not logged in: run "aimoba login --server https://your-league.example"');
  return { server, token, login: saved.login ?? '' };
}

async function api(c: { server: string; token?: string }, method: string, path: string, body?: Buffer | object, type?: string): Promise<any> {
  let res: Response;
  try {
    res = await fetch(c.server + path, {
      method,
      headers: { ...(c.token ? { Authorization: `Bearer ${c.token}` } : {}), ...(body ? { 'Content-Type': type ?? 'application/json' } : {}), Accept: 'application/json' },
      body: body === undefined ? undefined : Buffer.isBuffer(body) ? new Uint8Array(body) : JSON.stringify(body),
    });
  } catch (err) {
    throw new UsageError(`could not reach ${c.server}: ${(err as Error).cause ? String(((err as Error).cause as Error).message ?? (err as Error).cause) : (err as Error).message}`);
  }
  const text = await res.text();
  let data: any;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text.slice(0, 200) }; }
  if (!res.ok) throw new UsageError(`${c.server} said: ${data.error ?? res.statusText} (${res.status})`);
  return data;
}

const ask = (q: string) => new Promise<string>((res) => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  rl.question(q, (v) => { rl.close(); res(v.trim()); });
});

async function cmdLogin(argv: string[]) {
  const a = parseArgs(argv, [], {});
  const saved: Partial<Credentials> = existsSync(CRED_FILE) ? JSON.parse(readFileSync(CRED_FILE, 'utf8')) : {};
  const server = String(a.flags.server ?? process.env.AIMOBA_SERVER_URL ?? saved.server ?? '').replace(/\/+$/, '');
  if (!server) throw new UsageError('usage: aimoba login --server https://your-league.example [--token aim_...]');
  let token = typeof a.flags.token === 'string' ? a.flags.token : '';
  if (typeof a.flags.dev === 'string') {
    // A league run with AIMOBA_DEV_LOGIN=1 hands out accounts by name (for local testing).
    token = (await api({ server }, 'POST', '/auth/dev', { login: a.flags.dev, token: true })).token;
  }
  if (!token) {
    const url = `${server}/league/account`;
    console.log(`Create an API token on your account page and paste it here:\n  ${cyan(url)}\n`);
    openBrowser(url);
    token = await ask('token: ');
  }
  const me = await api({ server, token }, 'GET', '/api/me');
  mkdirSync(dirname(CRED_FILE), { recursive: true });
  writeFileSync(CRED_FILE, JSON.stringify({ server, token, login: me.user.login }, null, 2) + '\n', { mode: 0o600 });
  console.log(`${green('logged in')} as ${bold(me.user.login)} on ${server}`);
}

/** Files and bytes of an entry folder, counted as the league counts them. */
function entrySize(dir: string): { bytes: number; files: number } {
  const names = entryFiles(dir);
  return { bytes: names.reduce((n, name) => n + statSync(join(dir, name)).size, 0), files: names.length };
}

const git = (dir: string, ...args: string[]): string | null => {
  const r = spawnSync('git', args, { cwd: dir, encoding: 'utf8', windowsHide: true });
  return r.status === 0 ? r.stdout.trim() : null;
};

interface LocalSource {
  repo: { owner: string; repo: string };
  /** The entry's folder in the repository. */
  path: string;
  commit: string;
  /** The branch checked out, if any, and the remote it tracks. */
  branch: string | null;
  remote: string;
  /** Uncommitted changes in the entry's folder (they are not part of what the league fetches). */
  dirty: string[];
  /** Whether a remote branch contains the commit, as far as this checkout knows. */
  pushed: boolean;
}

/** Where the league would fetch the bot in `dir` from, worked out from its git checkout. Throws with what to do when it cannot be. */
function localSource(dir: string, remoteName?: string): LocalSource {
  if (git(dir, '--version') === null) throw new UsageError('git is not installed. The league takes bots from public GitHub repositories, so you need git to push yours.');
  const top = git(dir, 'rev-parse', '--show-toplevel');
  if (top === null) throw new UsageError(`${dir} is not in a git repository. The league takes bots from public GitHub repositories: put your bots in one (see "Enter it" in docs/league/README.md), push, and submit again.`);
  const commit = git(dir, 'rev-parse', 'HEAD');
  if (!commit) throw new UsageError(`the repository at ${top} has no commits yet: commit your bot and push it first`);
  const path = (git(dir, 'rev-parse', '--show-prefix') ?? '').replace(/\/+$/, '');
  const branch = git(dir, 'symbolic-ref', '--quiet', '--short', 'HEAD');
  const upstream = branch ? git(dir, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}') : null;
  const remote = remoteName ?? (upstream ? upstream.split('/')[0] : 'origin');
  const url = git(dir, 'remote', 'get-url', remote);
  if (!url) throw new UsageError(`the repository has no remote "${remote}". Create a public repository on GitHub, then: git remote add ${remote} https://github.com/<you>/<repo> && git push -u ${remote} ${branch ?? 'main'}`);
  const repo = parseRepo(url);
  if (!repo) throw new UsageError(`the remote "${remote}" (${url}) is not on GitHub. The league fetches bots from public GitHub repositories; push yours to one, or give it with --repo.`);
  const dirty = (git(dir, 'status', '--porcelain', '--', '.') ?? '').split('\n').filter(Boolean).map((l) => l.slice(3));
  const pushed = (git(dir, 'branch', '-r', '--contains', commit) ?? '').split('\n').some((l) => l.trim().startsWith(`${remote}/`));
  return { repo: { owner: repo.owner, repo: repo.repo }, path, commit, branch: upstream ? upstream.slice(remote.length + 1) : branch, remote, dirty, pushed };
}

async function cmdSubmit(argv: string[]) {
  const a = parseArgs(argv, ['no-wait'], { m: 'note' });
  const dir = a._[0];
  const repoFlag = typeof a.flags.repo === 'string' ? a.flags.repo : null;
  if (!dir && !repoFlag) {
    throw new UsageError(
      'usage: aimoba submit <bot directory> [-m "what changed"] [--no-wait]\n' +
      '       aimoba submit --repo <github repository> [--path <folder>] [--ref <branch | tag | commit>]\n\n' +
      'The league fetches your bot from its public GitHub repository. With a directory, it takes the commit you have\n' +
      'checked out there (push it first); with --repo, the folder and branch you name.',
    );
  }
  const c = credentials(a.flags);
  // A bot tuned against one version of the game may behave differently on another: say so before it is entered.
  const league = await api(c, 'GET', '/api/config');
  if (league.game && league.game !== gameFingerprint()) {
    console.log(yellow(`Your copy of the game differs from the league's (yours ${gameFingerprint()}, the league's ${league.game}).`));
    console.log(yellow(`Update it${league.templateUrl ? ` from ${league.templateUrl}` : ''} (git pull) and test your bot again; what you measured here may not hold there.`));
  }
  const note = typeof a.flags.note === 'string' ? a.flags.note : undefined;
  let body: Record<string, string | undefined>;
  if (repoFlag) {
    body = { repo: repoFlag, path: typeof a.flags.path === 'string' ? a.flags.path : undefined, ref: typeof a.flags.ref === 'string' ? a.flags.ref : undefined, note };
    console.log(`asking the league to fetch ${bold(repoFlag)}${body.path ? ` (${body.path})` : ''}${body.ref ? ` at ${body.ref}` : ''}`);
  } else {
    const entry = loadEntry(existsSync(dir) ? dir : join('bots', dir));
    const src = localSource(entry.dir, typeof a.flags.remote === 'string' ? a.flags.remote : undefined);
    const at = `github.com/${src.repo.owner}/${src.repo.repo}${src.path ? `/${src.path}` : ''} at ${src.commit.slice(0, 7)}`;
    if (src.dirty.length) console.log(yellow(`Uncommitted changes in this folder are not entered (${src.dirty.slice(0, 3).join(', ')}${src.dirty.length > 3 ? ', ...' : ''}). Commit and push them first if they belong in.`));
    if (!src.pushed) console.log(yellow(`Commit ${src.commit.slice(0, 7)} does not look pushed to ${src.remote}. Run git push first (if you just did, ignore this).`));
    console.log(`entering ${bold(entry.manifest.name)} from ${at}`);
    body = { repo: `https://github.com/${src.repo.owner}/${src.repo.repo}`, path: src.path, ref: src.commit, track: src.branch ?? undefined, name: entry.manifest.name, note };
  }
  const out = await api(c, 'POST', '/api/versions', body);
  const where = `${c.server}/league/bots/${out.bot.owner}/${out.bot.name}`;
  if (out.unchanged) {
    console.log(`no change: version ${bold(out.version.number)} of ${out.bot.name} already has these files (${out.version.status === 'active' ? 'it plays' : out.version.status})\n${cyan(where)}`);
    return;
  }
  console.log(`version ${bold(out.version.number)} fetched (commit ${String(out.version.commit).slice(0, 7)}); the league is checking it`);
  if (a.flags['no-wait']) { console.log(`follow it at ${cyan(where)}`); return; }
  for (let i = 0; i < 180; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const v = (await api(c, 'GET', `/api/bots/${out.bot.owner}/${out.bot.name}/versions/${out.version.number}`)).version;
    if (v.status === 'checking') continue;
    if (v.status === 'active') {
      console.log(`${green('accepted')}: version ${v.number} plays in the next tournament\n${cyan(where)}`);
    } else {
      console.log(red(`rejected: ${v.checkMessage ?? v.status}`));
      if (v.checkLog) console.log(dim(v.checkLog.trimEnd().split('\n').slice(-30).join('\n')));
      process.exitCode = 1;
    }
    return;
  }
  console.log(yellow(`still being checked; follow it at ${where}`));
}

async function cmdStatus(argv: string[]) {
  const a = parseArgs(argv, ['json'], {});
  const c = credentials(a.flags);
  const me = await api(c, 'GET', '/api/me');
  if (a.flags.json) { console.log(JSON.stringify(me, null, 2)); return; }
  console.log(`${bold(me.user.login)} on ${c.server}`);
  if (!me.bots.length) { console.log(dim('no bots yet: aimoba submit <bot directory>')); return; }
  const rows: (string | number)[][] = [[dim('bot'), dim('rank'), dim('rating'), dim('games'), dim('W-D-L'), dim('version'), dim('status')]];
  for (const b of me.bots) rows.push([b.name, b.rank ?? '-', Math.round(b.rating), b.games, `${b.wins}-${b.draws}-${b.losses}`, b.version ? `v${b.version.number}` : '-', b.enabled ? (b.version?.status ?? 'no version') : 'withdrawn']);
  console.log(table(rows, 'lrrrlll'));
  if (me.next) console.log(dim(`\nnext tournament: ${new Date(me.next).toLocaleString()}`));
}

async function cmdLogs(argv: string[]) {
  const a = parseArgs(argv, [], {});
  const id = a._[0];
  if (!id) throw new UsageError('usage: aimoba logs <match id>   (your own bot\'s log of a league match)');
  const out = await api(credentials(a.flags), 'GET', `/api/matches/${id}/log`);
  for (const l of out.logs) console.log(dim(`--- ${l.bot} (${l.side}) ---`) + '\n' + (l.log.trimEnd() || dim('(empty)')));
}

/** The league server is part of the main repository only; the public template has just the entrant's side. */
const SERVER = join(REPO_ROOT, 'src', 'league', 'server', 'main.ts');
const HAS_SERVER = existsSync(SERVER);

async function cmdServe(argv: string[]) {
  const { main } = await import(pathToFileURL(SERVER).href);
  await main(argv);
}

const HELP = `${bold('aimoba')}: build a bot for AIMOBA, test it, enter it in the league

${bold('Build and test')}
  init <name> [--kit js|python|python-model|policy]   start a bot in bots/<name>
  export <checkpoint.pt> <bot directory>              put a trained network into an entry as model.bin
  play <blue> <red> [--watch] [--seed N] [--fast]     one game; saves a replay
  bench <bot> <opponent> [-n 20] [--fast]             many games from both sides, with a win rate
  tournament <bot> <bot> ... [--games-per-pair 4]     a local tournament with standings and ratings
  watch <replay.json>                                 open a replay in the game client
  validate <bot directory>                            the league's checks, run here
  info [--json]                                       rules, heroes and the data an agent receives

${bold('The league')}
  login --server <url>                                store an API token for a league
  submit <bot directory> [-m note]                    enter the commit you have checked out (from its public GitHub repository)
  status                                              your bots and their ratings
  logs <match id>                                     what your bot printed in a league match${HAS_SERVER ? `
  serve                                               run a league (see docs/hosting.md)` : ''}

An agent is: ${BUILTIN_BOTS.join(', ')} (the game's scripted bots), policy:<id> (a model in public/policies),
a bot directory (bots/<name>), or a .js / .ts / .py / .bin file.
--fast runs JavaScript bots inside the runner (quicker, but without the clock or the sandbox).
--docker runs entries the way a public league does. Docs: docs/league/README.md`;

const COMMANDS: Record<string, (argv: string[]) => Promise<void> | void> = {
  init: cmdInit, export: cmdExport, play: cmdPlay, watch: cmdWatch, bench: cmdBench, tournament: cmdTournament, validate: cmdValidate, info: cmdInfo,
  login: cmdLogin, submit: cmdSubmit, status: cmdStatus, logs: cmdLogs,
};
if (HAS_SERVER) COMMANDS.serve = cmdServe;

const [command, ...rest] = process.argv.slice(2);
if (!command || command === 'help' || command === '--help' || command === '-h') {
  console.log(HELP);
} else if (!COMMANDS[command]) {
  console.error(`unknown command "${command}"\n\n${HELP}`);
  process.exitCode = 1;
} else {
  try {
    await COMMANDS[command](rest);
  } catch (err) {
    if (err instanceof UsageError || err instanceof EntryError || err instanceof SandboxError) {
      console.error(red(err.message));
      process.exitCode = 1;
    } else throw err;
  }
}

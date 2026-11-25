/**
 * Entries: what a competitor hands in, and how the name of an agent on a command line is resolved.
 *
 * An entry is a directory with a manifest, `aimoba.json`:
 *
 *   { "name": "my-bot", "runtime": "node", "entry": "bot.js" }
 *
 *   runtime "node"     `entry` is a JavaScript or TypeScript module whose default export is a bot (see host.ts)
 *   runtime "python"   `entry` is a Python script that calls `aimoba_agent.run(...)`
 *   runtime "policy"   `entry` is a network exported by python/train/export.py; it plays without any code
 *   runtime "command"  `command` is a program and its arguments, speaking the agent protocol on stdin / stdout
 *
 * On a command line an agent can also be named directly: `easy`, `basic`, `expert` (the scripted bots of the
 * game), `policy:<id>` (a model in public/policies), a `.bin` file, a `.js` / `.ts` / `.py` file, or a directory
 * with a manifest.
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const MANIFEST = 'aimoba.json';
export const RUNTIMES = ['node', 'python', 'policy', 'command'] as const;
export type Runtime = (typeof RUNTIMES)[number];
export const BUILTIN_BOTS = ['easy', 'basic', 'expert'];

export interface Manifest {
  name: string;
  runtime: Runtime;
  /** The bot module, script or model file, relative to the entry's directory. */
  entry?: string;
  /** runtime "command": the program and its arguments. A relative program path is relative to the entry's directory. */
  command?: string[];
  /** runtime "policy": play the most likely action instead of sampling one. */
  greedy?: boolean;
  description?: string;
}

/** An agent ready to be started: a scripted bot of the game, or an entry on disk. */
export type AgentSpec =
  | { kind: 'builtin'; bot: string; name: string }
  | { kind: 'entry'; dir: string; manifest: Manifest; name: string };

export class EntryError extends Error {}

const NAME_RE = /^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$/;
/** A bot's name: 3-40 characters, lower-case letters, digits and dashes. */
export const validName = (s: unknown): s is string => typeof s === 'string' && NAME_RE.test(s) && !s.includes('--');

/** Whether `file` stays inside `dir` (no `..`, no absolute path, no drive letter). */
export function inside(dir: string, file: string): boolean {
  if (!file || isAbsolute(file) || /^[a-zA-Z]:/.test(file)) return false;
  const rel = relative(resolve(dir), resolve(dir, file));
  return rel !== '' && !rel.startsWith('..') && !isAbsolute(rel);
}

/** Check a manifest's shape. `dir` (when given) is where its files must exist. Returns the cleaned manifest or throws EntryError. */
export function checkManifest(raw: unknown, dir?: string): Manifest {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new EntryError(`${MANIFEST} must be a JSON object`);
  const m = raw as Record<string, unknown>;
  if (!validName(m.name)) throw new EntryError(`${MANIFEST}: "name" must be 3-40 characters: lower-case letters, digits and single dashes (got ${JSON.stringify(m.name)})`);
  if (typeof m.runtime !== 'string' || !RUNTIMES.includes(m.runtime as Runtime)) throw new EntryError(`${MANIFEST}: "runtime" must be one of ${RUNTIMES.join(', ')} (got ${JSON.stringify(m.runtime)})`);
  const out: Manifest = { name: m.name, runtime: m.runtime as Runtime };
  if (typeof m.description === 'string') out.description = m.description.slice(0, 280);
  if (out.runtime === 'command') {
    if (!Array.isArray(m.command) || !m.command.length || m.command.some((c) => typeof c !== 'string' || !c)) throw new EntryError(`${MANIFEST}: runtime "command" needs "command": ["program", "arg", ...]`);
    out.command = m.command as string[];
  } else {
    if (typeof m.entry !== 'string' || !m.entry) throw new EntryError(`${MANIFEST}: runtime "${out.runtime}" needs "entry", the file to run`);
    const file = m.entry.split('\\').join('/');
    if (dir !== undefined) {
      if (!inside(dir, file)) throw new EntryError(`${MANIFEST}: "entry" must be a file inside the entry's directory (got ${JSON.stringify(m.entry)})`);
      const full = join(dir, file);
      if (!existsSync(full) || !statSync(full).isFile()) throw new EntryError(`${MANIFEST}: entry file ${JSON.stringify(m.entry)} does not exist`);
    }
    const ext = extname(file).toLowerCase();
    const ok: Record<string, string[]> = { node: ['.js', '.mjs', '.ts', '.mts'], python: ['.py'], policy: ['.bin'] };
    if (!ok[out.runtime].includes(ext)) throw new EntryError(`${MANIFEST}: a "${out.runtime}" entry must be a ${ok[out.runtime].join(' / ')} file (got ${JSON.stringify(m.entry)})`);
    out.entry = file;
    if (out.runtime === 'policy' && m.greedy === true) out.greedy = true;
  }
  return out;
}

/** Read and check the manifest of an entry directory. */
export function loadEntry(dir: string): AgentSpec & { kind: 'entry' } {
  const file = join(dir, MANIFEST);
  if (!existsSync(file)) throw new EntryError(`${dir} has no ${MANIFEST}`);
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(file, 'utf8'));
  } catch (err) {
    throw new EntryError(`${file} is not valid JSON: ${(err as Error).message}`);
  }
  const manifest = checkManifest(raw, dir);
  return { kind: 'entry', dir: resolve(dir), manifest, name: manifest.name };
}

const slug = (s: string) => {
  const v = s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40);
  return v.length >= 3 ? v : `bot-${v}`.slice(0, 40);
};

/** Models listed in public/policies/index.json: [id, file, greedy]. */
function listedPolicies(): { id: string; file: string; greedy?: boolean }[] {
  const index = join(REPO_ROOT, 'public', 'policies', 'index.json');
  return existsSync(index) ? JSON.parse(readFileSync(index, 'utf8')) : [];
}

/**
 * Turn what a user typed into an agent. `bases` are extra directories in which a bare name is looked up
 * (the CLI passes `bots/`). Throws EntryError with a message that says what would have worked.
 */
export function resolveAgent(spec: string, bases: string[] = []): AgentSpec {
  if (BUILTIN_BOTS.includes(spec)) return { kind: 'builtin', bot: spec, name: spec };
  if (spec.startsWith('policy:')) {
    const id = spec.slice(7);
    const p = listedPolicies().find((m) => m.id === id);
    if (!p) throw new EntryError(`there is no model "${id}" in public/policies (have: ${listedPolicies().map((m) => m.id).join(', ') || 'none'})`);
    return { kind: 'entry', dir: join(REPO_ROOT, 'public', 'policies'), manifest: { name: slug(id), runtime: 'policy', entry: p.file, greedy: p.greedy }, name: id };
  }
  const candidates = [spec, ...bases.map((b) => join(b, spec))];
  for (const c of candidates) {
    if (!existsSync(c)) continue;
    const st = statSync(c);
    if (st.isDirectory()) return loadEntry(c);
    const ext = extname(c).toLowerCase();
    const runtime: Runtime | null = ['.js', '.mjs', '.ts', '.mts'].includes(ext) ? 'node' : ext === '.py' ? 'python' : ext === '.bin' ? 'policy' : null;
    if (basename(c) === MANIFEST) return loadEntry(dirname(c));
    if (!runtime) throw new EntryError(`do not know how to run ${c}: expected a directory with ${MANIFEST}, or a .js / .ts / .py / .bin file`);
    const name = slug(basename(c, ext));
    return { kind: 'entry', dir: resolve(dirname(c)), manifest: { name, runtime, entry: basename(c) }, name };
  }
  const local = bases.flatMap((b) => (existsSync(b) ? readdirSync(b).filter((d) => existsSync(join(b, d, MANIFEST))) : []));
  throw new EntryError(
    `unknown agent "${spec}". Use one of: ${BUILTIN_BOTS.join(', ')}; policy:<id> (${listedPolicies().map((m) => m.id).join(', ')}); ` +
    `a bot directory or file${local.length ? ` (found: ${local.join(', ')})` : ''}.`,
  );
}

/** All files of an entry directory, relative paths with forward slashes, skipping what never belongs in a submission. */
export function entryFiles(dir: string): string[] {
  const skip = new Set(['node_modules', '.git', '__pycache__', '.venv', 'venv', '.idea', '.vscode', 'replays', '.DS_Store']);
  const out: string[] = [];
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (skip.has(e.name) || e.name.endsWith('.pyc') || e.name.endsWith('.replay.json')) continue;
      const full = join(d, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.isFile()) out.push(relative(dir, full).split(sep).join('/'));
    }
  };
  walk(dir);
  return out.sort();
}

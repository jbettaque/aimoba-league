/**
 * How an entry's program is started, and what it is allowed to touch.
 *
 *   driver "process"  runs it as a plain child process of the runner. For your own code on your own machine:
 *                     nothing stops it from reading your files or using the network. (Node entries are started
 *                     under Node's permission model, which blocks writes, child processes and worker threads, so
 *                     the most common ways a bot would fail on the league show up locally. That is a convenience,
 *                     not a security boundary.)
 *   driver "docker"   runs it in a throw-away container: no network, read-only file system with the entry mounted
 *                     at /agent, one CPU, capped memory and process count, no capabilities, an unprivileged user.
 *                     This is what a public league must use.
 *
 * Either way the child gets a minimal environment: none of the runner's own variables (tokens, secrets) reach it.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync } from 'node:fs';
import { isAbsolute, join, resolve } from 'node:path';
import { REPO_ROOT, inside, type Manifest } from './entry.ts';
import { SandboxError } from './protocol.ts';

export interface DockerConfig {
  /** The docker (or podman) executable. */
  bin: string;
  /** Images per runtime. "command" entries run in `base`. Build them with docker/build-images.sh. */
  images: { node: string; python: string; base: string };
  cpus: number;
  memoryMb: number;
  pids: number;
  /** Size of the writable /tmp inside the container. */
  tmpfsMb: number;
  /** Extra `docker run` arguments, e.g. ["--runtime=runsc"] to run the containers under gVisor. */
  extraArgs: string[];
}

export interface SandboxConfig {
  driver: 'process' | 'docker';
  /** Executables for the "process" driver. Defaults: this Node; AIMOBA_PYTHON, the repo's .venv, or python on PATH. */
  node?: string;
  python?: string;
  /** "process" driver: start Node entries under Node's permission model (default true). */
  nodePermission?: boolean;
  docker?: Partial<DockerConfig>;
}

export const DEFAULT_DOCKER: DockerConfig = {
  bin: 'docker',
  images: { node: 'aimoba-agent-node', python: 'aimoba-agent-python', base: 'aimoba-agent-base' },
  cpus: 1,
  memoryMb: 1024,
  pids: 128,
  tmpfsMb: 64,
  extraArgs: [],
};

export const HOST_SCRIPT = join(REPO_ROOT, 'src', 'agent', 'host.ts');
/** Where the images keep this repository's agent code and where the entry is mounted. */
const IMAGE_HOST_SCRIPT = '/opt/aimoba/src/agent/host.ts';
const IMAGE_ENTRY_DIR = '/agent';

export function defaultPython(): string {
  if (process.env.AIMOBA_PYTHON) return process.env.AIMOBA_PYTHON;
  for (const p of [join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'), join(REPO_ROOT, '.venv', 'bin', 'python')]) if (existsSync(p)) return p;
  return process.platform === 'win32' ? 'python' : 'python3';
}

/** The variables a child process needs to start at all, and nothing else from the runner's environment. */
function baseEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const k of ['PATH', 'Path', 'SystemRoot', 'SYSTEMROOT', 'windir', 'TEMP', 'TMP', 'TMPDIR', 'LANG', 'LC_ALL', 'HOME', 'USERPROFILE', 'PATHEXT', 'COMSPEC']) {
    const v = process.env[k];
    if (v !== undefined) env[k] = v;
  }
  return env;
}

/** Settings every agent gets: unbuffered UTF-8 pipes and one thread for numeric libraries (the league gives one CPU). */
const AGENT_ENV: Record<string, string> = {
  PYTHONUNBUFFERED: '1',
  PYTHONIOENCODING: 'utf-8',
  PYTHONDONTWRITEBYTECODE: '1',
  OMP_NUM_THREADS: '1',
  MKL_NUM_THREADS: '1',
  OPENBLAS_NUM_THREADS: '1',
  NUMEXPR_NUM_THREADS: '1',
  NODE_NO_WARNINGS: '1',
  AIMOBA_AGENT: '1',
};

export interface Launched {
  child: ChildProcess;
  /** Stop it for good (kills the container too). Safe to call more than once. */
  kill(): void;
  /** What was run, for error messages. */
  describe: string;
}

/** The command line for an entry inside its directory (`dir` as the program will see it). */
function commandFor(m: Manifest, dir: string, node: string, python: string, hostScript: string, permission: string[]): string[] {
  const posix = dir.startsWith('/');
  const at = (file: string) => (posix ? `${dir}/${file}` : join(dir, file));
  switch (m.runtime) {
    case 'node': return [node, ...permission, hostScript, at(m.entry!)];
    case 'python': return [python, '-u', at(m.entry!)];
    case 'command': {
      const [prog, ...rest] = m.command!;
      // A program named with a path is taken from the entry's directory; a bare name comes from PATH.
      const local = (prog.includes('/') || prog.includes('\\')) && !isAbsolute(prog) && inside('.', prog);
      return [local ? at(prog.split('\\').join('/')) : prog, ...rest];
    }
    default: throw new Error(`runtime "${m.runtime}" is not started as a program`);
  }
}

export function dockerArgs(m: Manifest, dir: string, cfg: DockerConfig, name: string): string[] {
  const image = m.runtime === 'node' ? cfg.images.node : m.runtime === 'python' ? cfg.images.python : cfg.images.base;
  const env = { ...AGENT_ENV, HOME: '/tmp', TMPDIR: '/tmp' };
  return [
    'run', '--rm', '-i', '--init', '--name', name,
    '--network', 'none',
    '--cpus', String(cfg.cpus), '--memory', `${cfg.memoryMb}m`, '--memory-swap', `${cfg.memoryMb}m`, '--pids-limit', String(cfg.pids),
    '--read-only', '--tmpfs', `/tmp:rw,noexec,nosuid,nodev,size=${cfg.tmpfsMb}m`,
    '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
    '--ulimit', 'nofile=1024:1024', '--ulimit', 'core=0', '--log-driver', 'none', '--stop-timeout', '1',
    '--workdir', IMAGE_ENTRY_DIR, '--volume', `${resolve(dir)}:${IMAGE_ENTRY_DIR}:ro`,
    ...Object.entries(env).flatMap(([k, v]) => ['--env', `${k}=${v}`]),
    ...cfg.extraArgs,
    image,
    ...commandFor(m, IMAGE_ENTRY_DIR, 'node', 'python3', IMAGE_HOST_SCRIPT, []),
  ];
}

/** Start an entry's program. The caller owns the returned child's stdio. */
export function launch(m: Manifest, dir: string, sandbox: SandboxConfig = { driver: 'process' }): Launched {
  if (sandbox.driver === 'docker') {
    const cfg: DockerConfig = { ...DEFAULT_DOCKER, ...sandbox.docker, images: { ...DEFAULT_DOCKER.images, ...sandbox.docker?.images } };
    const name = `aimoba-${randomBytes(8).toString('hex')}`;
    const args = dockerArgs(m, dir, cfg, name);
    const child = spawn(cfg.bin, args, { stdio: ['pipe', 'pipe', 'pipe'], env: baseEnv(), windowsHide: true });
    let killed = false;
    return {
      child,
      describe: `${cfg.bin} run ${args[args.indexOf('--volume') + 1]} ...`,
      kill() {
        if (killed) return;
        killed = true;
        // Removing the container by name stops it; killing the CLI alone would leave it running.
        try { spawn(cfg.bin, ['rm', '--force', name], { stdio: 'ignore', windowsHide: true, detached: true }).unref(); } catch { /* the container is already gone */ }
        try { child.kill(); } catch { /* already exited */ }
      },
    };
  }
  const node = sandbox.node ?? process.execPath;
  const python = sandbox.python ?? defaultPython();
  const permission = m.runtime === 'node' && (sandbox.nodePermission ?? true)
    ? ['--permission', `--allow-fs-read=${join(REPO_ROOT, 'src')}${process.platform === 'win32' ? '\\' : '/'}*`, `--allow-fs-read=${resolve(dir)}${process.platform === 'win32' ? '\\' : '/'}*`, `--allow-fs-read=${join(REPO_ROOT, 'package.json')}`]
    : [];
  const [cmd, ...args] = commandFor(m, resolve(dir), node, python, HOST_SCRIPT, permission);
  const env = { ...baseEnv(), ...AGENT_ENV, PYTHONPATH: join(REPO_ROOT, 'python') };
  const child = spawn(cmd, args, { cwd: resolve(dir), stdio: ['pipe', 'pipe', 'pipe'], env, windowsHide: true });
  let killed = false;
  return {
    child,
    describe: [cmd, ...args].join(' '),
    kill() {
      if (killed) return;
      killed = true;
      try {
        // On Windows a plain kill leaves grandchildren behind; taskkill takes the whole tree.
        if (process.platform === 'win32' && child.pid) spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
        else child.kill('SIGKILL');
      } catch { /* already exited */ }
    },
  };
}

/**
 * Make sure the sandbox can run an entry before it is played: the image exists and the entry's folder can be
 * mounted. With the "process" driver there is nothing to check. With Docker this starts a container that does
 * nothing, trying a few times, and throws SandboxError with the daemon's message if it keeps failing. (Docker
 * Desktop needs a moment the first time a host folder is mounted; a missing image is the usual real cause.)
 */
export async function prepareEntry(m: Manifest, dir: string, sandbox: SandboxConfig = { driver: 'process' }): Promise<void> {
  if (sandbox.driver !== 'docker' || m.runtime === 'policy') return;
  const cfg: DockerConfig = { ...DEFAULT_DOCKER, ...sandbox.docker, images: { ...DEFAULT_DOCKER.images, ...sandbox.docker?.images } };
  const image = m.runtime === 'node' ? cfg.images.node : m.runtime === 'python' ? cfg.images.python : cfg.images.base;
  let detail = '';
  for (let attempt = 0; attempt < 4; attempt++) {
    if (attempt) await new Promise((r) => setTimeout(r, 1000 * attempt));
    const name = `aimoba-prep-${randomBytes(6).toString('hex')}`;
    const args = ['run', '--rm', '--name', name, '--network', 'none', '--read-only', '--volume', `${resolve(dir)}:${IMAGE_ENTRY_DIR}:ro`, ...cfg.extraArgs, image, 'true'];
    const result = await new Promise<{ code: number | null; err: string }>((done) => {
      const child = spawn(cfg.bin, args, { stdio: ['ignore', 'ignore', 'pipe'], env: baseEnv(), windowsHide: true });
      let err = '';
      const timer = setTimeout(() => { try { child.kill(); } catch { /* gone */ } }, 30_000);
      child.stderr!.on('data', (c: Buffer) => { if (err.length < 4000) err += c.toString('utf8'); });
      child.on('error', (e) => { clearTimeout(timer); done({ code: -1, err: e.message }); });
      child.on('exit', (code) => { clearTimeout(timer); done({ code, err }); });
    });
    if (result.code === 0) {
      sweepContainers(sandbox);
      return;
    }
    detail = result.err.trim().split('\n').filter(Boolean).slice(0, 2).join(' ') || `docker exited with code ${result.code}`;
  }
  sweepContainers(sandbox);
  throw new SandboxError(`the sandbox cannot run entries with image "${image}": ${detail}`);
}

/**
 * Remove this league's containers that were created but never ran, or have exited: `docker run --rm` only cleans
 * up after a container that actually started. Containers that are running are left alone.
 */
export function sweepContainers(sandbox: SandboxConfig) {
  if (sandbox.driver !== 'docker') return;
  const bin = sandbox.docker?.bin ?? DEFAULT_DOCKER.bin;
  try {
    const list = spawnSync(bin, ['ps', '--all', '--quiet', '--filter', 'name=^aimoba-', '--filter', 'status=created', '--filter', 'status=exited', '--filter', 'status=dead'], { encoding: 'utf8', timeout: 15_000, windowsHide: true });
    const ids = (list.stdout ?? '').split(/\s+/).filter(Boolean);
    if (ids.length) spawnSync(bin, ['rm', '--force', ...ids], { stdio: 'ignore', timeout: 30_000, windowsHide: true });
  } catch { /* the daemon is not reachable; nothing to sweep */ }
}

/** Whether the docker driver can work on this machine: the CLI answers and the daemon is up. */
export function dockerAvailable(bin = 'docker'): { ok: boolean; detail: string } {
  try {
    const r = spawnSync(bin, ['version', '--format', '{{.Server.Version}}'], { encoding: 'utf8', timeout: 10_000, windowsHide: true });
    if (r.status === 0) return { ok: true, detail: `docker ${r.stdout.trim()}` };
    return { ok: false, detail: (r.stderr || r.stdout || 'docker did not answer').trim().split('\n')[0] };
  } catch (err) {
    return { ok: false, detail: (err as Error).message };
  }
}

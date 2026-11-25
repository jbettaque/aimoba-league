/**
 * Runs a JavaScript bot as an agent process: `node src/agent/host.ts path/to/bot.js`.
 *
 * Speaks the agent protocol (protocol.ts) on stdin / stdout and calls the bot's functions (bot.ts). Anything the
 * bot prints with `console.log` goes to stderr, which the match runner keeps as the bot's log; stdout belongs to
 * the protocol.
 */
import { createInterface } from 'node:readline';
import { format } from 'node:util';
import { BotRunner, loadBot } from './bot.ts';
import type { RunnerMessage } from './protocol.ts';

const log = (...args: unknown[]) => process.stderr.write(format(...args) + '\n');
for (const k of ['log', 'info', 'debug', 'warn', 'error', 'trace'] as const) console[k] = log;

const send = (msg: unknown) => process.stdout.write(JSON.stringify(msg) + '\n');
const die = (err: unknown): never => {
  log(err instanceof Error ? (err.stack ?? err.message) : String(err));
  process.exit(1);
};
process.on('uncaughtException', die);
process.on('unhandledRejection', die);

const file = process.argv[2];
if (!file) die(new Error('usage: node host.ts <bot module>'));

let runner: BotRunner | null = null;
// Messages are handled strictly one after another, also when the bot's functions are async.
let chain: Promise<void> = Promise.resolve();

async function handle(msg: RunnerMessage) {
  switch (msg.type) {
    case 'hello':
      runner = new BotRunner(await loadBot(file), log);
      send(await runner.init(msg.info));
      break;
    case 'start':
      await runner!.start(msg.team);
      break;
    case 'step':
      send(await runner!.act(msg));
      break;
    case 'end':
      await runner?.end({ result: msg.result, reason: msg.reason, summary: msg.summary });
      process.exit(0);
  }
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', (line) => {
  if (!line) return;
  chain = chain.then(() => handle(JSON.parse(line) as RunnerMessage)).catch(die);
});
rl.on('close', () => { chain.then(() => process.exit(0)); });

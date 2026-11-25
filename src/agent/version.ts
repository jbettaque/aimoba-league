/**
 * A fingerprint of the game as a bot experiences it: the simulation and the bot interface.
 *
 * Entrants develop against their own copy of the game (the template repository) while the league plays on its
 * copy. If the two differ, a bot tuned at home meets different rules in the league. Both sides compute this
 * fingerprint from their own sources; `aimoba submit` compares them and says so when they differ.
 *
 * Line endings are ignored, so a Windows checkout and a Linux one of the same commit agree.
 */
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { REPO_ROOT } from './entry.ts';

let cached: string | null = null;

export function gameFingerprint(): string {
  if (cached) return cached;
  const files = [
    ...readdirSync(join(REPO_ROOT, 'src', 'sim')).filter((f) => f.endsWith('.ts')).sort().map((f) => join('src', 'sim', f)),
    ...['view.ts', 'info.ts', 'protocol.ts', 'match.ts'].map((f) => join('src', 'agent', f)),
  ];
  const hash = createHash('sha256');
  for (const f of files) {
    hash.update(f.split('\\').join('/'));
    hash.update(readFileSync(join(REPO_ROOT, f), 'utf8').split('\r\n').join('\n'));
  }
  cached = hash.digest('hex').slice(0, 12);
  return cached;
}

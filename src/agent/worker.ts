/** A match worker of `MatchPool`: plays the matches its parent sends, one at a time. */
import { mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { gzipSync } from 'node:zlib';
import { runMatch } from './match.ts';
import type { MatchJob } from './pool.ts';

process.on('message', async (job: MatchJob) => {
  try {
    const { replayFile, ...opts } = job;
    const result = await runMatch(opts);
    if (replayFile && result.replay) {
      const text = JSON.stringify(result.replay);
      mkdirSync(dirname(replayFile), { recursive: true });
      // Written under another name first, so a reader never finds half a file.
      writeFileSync(replayFile + '.tmp', replayFile.endsWith('.gz') ? gzipSync(text) : text);
      renameSync(replayFile + '.tmp', replayFile);
      delete result.replay;
    }
    process.send!({ ok: true, result });
  } catch (err) {
    process.send!({ ok: false, error: String((err as Error)?.stack ?? err) });
  }
});
process.on('disconnect', () => process.exit(0));

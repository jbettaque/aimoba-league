/** Renders the sound bank off the main thread and posts each sound back as soon as it is ready. */
import { BANK, renderSound } from './bank.ts';

const post = (self as unknown as Worker).postMessage.bind(self);
for (const spec of BANK) {
  const takes = renderSound(spec);
  post({ name: spec.name, takes }, takes.flatMap((t) => t.chans.map((c) => c.buffer as ArrayBuffer)));
}

import { defineConfig } from 'vite';

// The game client: `npm run dev` to play, and what `./aimoba watch` opens replays in.
export default defineConfig({
  base: './',
  build: { target: 'es2022', outDir: 'dist' },
  server: { host: true },
});

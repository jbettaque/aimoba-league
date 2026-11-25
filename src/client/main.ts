/** Browser entry point: menus, game loop, input, camera and event-driven effects. */
import { DT, MAGIC, PHYS, type Command, type GameEvent, type Unit } from '../sim/defs.ts';
import { Game } from '../sim/game.ts';
import { HEROES } from '../sim/heroes.ts';
import { MAP_SIZE } from '../sim/map.ts';
import { makeReplay, replayConfig, type Replay } from '../sim/ml.ts';
import { PolicyDriver, loadPolicy, type PolicyNet } from '../sim/policy.ts';
import { SoundDirector, type SoundCtx } from './audio/director.ts';
import { audio, type AudioSettings } from './audio/engine.ts';
import { Fx } from './fx.ts';
import { Hud, scoreboardHtml } from './hud.ts';
import { iconUrl } from './icons.ts';
import { Renderer, TEAM_COLORS, type View } from './render.ts';
import { Showcase } from './showcase.ts';
import '@fontsource-variable/fraunces/full.css';
import '@fontsource/alegreya-sans/400.css';
import '@fontsource/alegreya-sans/400-italic.css';
import '@fontsource/alegreya-sans/500.css';
import '@fontsource/alegreya-sans/700.css';
import '@fontsource/alegreya-sans/800.css';

type Mode = 'menu' | 'play' | 'spectate' | 'replay';

const canvas = document.getElementById('game') as HTMLCanvasElement;
const menuEl = document.getElementById('menu')!;
const overlayEl = document.getElementById('overlay')!;
const hudEl = document.getElementById('hud')!;
const renderer = new Renderer(canvas);
const fx = new Fx();
const portraits = HEROES.map((_, i) => renderer.portrait(i, 256));
const showcase = new Showcase(renderer.assets);
const sound = new SoundDirector(audio);
audio.init();

let mode: Mode = 'menu';
let game: Game;
let playerSlot = -1;
let followSlot = 2;
let speed = 1;
let acc = 0;
let camLock = true;
let ended = false;
let overlayOpen = false;
let replay: Replay | null = null;
let replayIdx = 0;
let selHero = 2;
let selTeam = 0;
/** Who plays your allies and your enemies: a scripted bot ('easy' | 'basic' | 'expert') or the id of a trained model. */
let selAllies = 'basic';
let selEnemies = 'basic';
/** The trained models at work in the current game (none in replays: their orders are part of the recording). */
let drivers: PolicyDriver[] = [];
let hintTimer = 0;
const view: View = { team: -1, focusId: -1, hoverId: -1, aimSlot: -1, amove: false, mouseX: 0, mouseY: 0 };
const mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2, right: false, drag: false, lastMove: 0, pressX: 0, pressY: 0, pressTarget: -1 };
const keys = new Set<string>();

const hud = new Hud(hudEl, renderer, {
  levelUp: (slot) => levelAbility(slot),
  buy: (item) => {
    const u = me();
    if (u && game.canBuy(u, item)) { send({ t: 'buy', item }); sound.ui('buy'); } else sound.ui('deny');
  },
  castKey: (slot) => { if (mode === 'play') castAbility(slot, true); },
  minimap: (x, y, right) => {
    if (right && mode === 'play') { send({ t: 'move', x, y }); return; }
    camLock = false;
    followSlot = -1;
    renderer.cam.x = x;
    renderer.cam.y = y;
  },
  follow: (slot) => {
    if (mode === 'play') { const u = game.heroes[slot]; camLock = false; renderer.cam.x = u.x; renderer.cam.y = u.y; return; }
    followSlot = slot;
  },
  speed: (m) => { setSpeed(m); sound.ui('click'); },
  vision: (t) => { view.team = t; hud.markVision(t); sound.ui('click'); },
  recall: () => send({ t: 'recall' }),
  sound: (name) => sound.ui(name),
});

// --- Trained models ---------------------------------------------------------------------------
/** One entry of public/policies/index.json (written by python/train/export.py). */
interface ModelInfo { id: string; label: string; blurb: string; file: string; greedy?: boolean }
const BOT_LABEL: Record<string, string> = { easy: 'Easy', basic: 'Normal', expert: 'Expert' };
let models: ModelInfo[] = [];
const nets = new Map<string, PolicyNet>();
const netLoads = new Map<string, Promise<PolicyNet>>();
const modelsReady: Promise<void> = fetch('./policies/index.json')
  .then((r) => (r.ok ? r.json() : []))
  .then((list: ModelInfo[]) => { models = list; })
  .catch(() => {});
const isBotKind = (kind: string) => kind in BOT_LABEL;
const aiName = (kind: string) => (isBotKind(kind) ? `${BOT_LABEL[kind]} bots` : `${models.find((m) => m.id === kind)?.label ?? kind} model`);
function loadNet(id: string): Promise<PolicyNet> {
  let load = netLoads.get(id);
  if (!load) {
    const info = models.find((m) => m.id === id);
    load = (info ? fetch(`./policies/${info.file}`) : Promise.reject(new Error(`there is no model "${id}"`)))
      .then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.arrayBuffer(); })
      .then((buf) => { const net = loadPolicy(buf); nets.set(id, net); return net; });
    load.catch(() => netLoads.delete(id));
    netLoads.set(id, load);
  }
  return load;
}
/** A new match: `myTeam`'s heroes are played by the allies choice, the others by the enemies choice, `player` (a slot or -1) by a human. */
function newMatch(seed: number, myTeam: number, player: number): Game {
  const kind = (s: number) => ((s < 5 ? 0 : 1) === myTeam ? selAllies : selEnemies);
  const slots = Array.from({ length: 10 }, (_, s) => s);
  const scripted = (s: number) => s !== player && isBotKind(kind(s));
  const g = new Game({ seed, bots: slots.map(scripted), manual: slots.map((s) => s === player), botKind: slots.map((s) => (scripted(s) ? kind(s) : 'basic')), recordEvents: true, recordCommands: true });
  drivers = [...new Set([selAllies, selEnemies])].filter((k) => !isBotKind(k))
    .map((k) => new PolicyDriver(g, nets.get(k)!, slots.filter((s) => s !== player && kind(s) === k), { greedy: models.find((m) => m.id === k)?.greedy }));
  return g;
}

function me(): Unit | undefined {
  return playerSlot >= 0 ? game.heroes[playerSlot] : undefined;
}
function focusUnit(): Unit | undefined {
  if (mode === 'play') return me();
  return followSlot >= 0 ? game.heroes[followSlot] : undefined;
}
function send(c: Command) {
  if (mode === 'play' && playerSlot >= 0 && !ended) game.command(playerSlot, c);
}
/** Spend a skill point, with a sound that says whether it took. */
function levelAbility(slot: number) {
  const u = me();
  if (!u) return;
  sound.ui(game.canLevel(u, slot) ? 'rank' : 'deny');
  send({ t: 'level', slot });
}
function hint(text: string) {
  hud.setHint(text);
  clearTimeout(hintTimer);
  hintTimer = window.setTimeout(() => hud.setHint(''), 1300);
}
function setSpeed(m: number) {
  speed = m;
  hud.markSpeed(m);
}

// --- Starting games -----------------------------------------------------------------
function startDemo() {
  mode = 'menu';
  game = new Game({ seed: (Math.random() * 1e9) | 0, recordEvents: true });
  for (let i = 0; i < 30 * 50; i++) { game.step(); game.events.length = 0; } // skip the quiet opening
  playerSlot = -1;
  followSlot = 2;
  view.team = -1;
  view.aimSlot = -1;
  speed = 1;
  ended = false;
  fx.clear();
  sound.reset();
  drivers = [];
  hudEl.classList.add('hidden');
  menuEl.classList.remove('hidden');
  renderer.setZoom(0.85);
  renderer.cam.x = game.heroes[2].x;
  renderer.cam.y = game.heroes[2].y;
}

function startGame(m: 'play' | 'spectate' | 'replay', opts: { hero?: number; team?: number; seed?: number; replay?: Replay } = {}) {
  if (m !== 'replay') {
    // Fetch the chosen models first (a couple of megabytes each, once).
    const missing = [selAllies, selEnemies].filter((k) => !isBotKind(k) && !nets.has(k));
    if (missing.length) {
      Promise.all(missing.map(loadNet)).then(() => startGame(m, opts), (e) => alert('Could not load the model: ' + (e as Error).message));
      return;
    }
  }
  mode = m;
  drivers = [];
  ended = false;
  acc = 0;
  fx.clear();
  sound.reset();
  sound.ui('start');
  closeOverlay();
  const seed = opts.seed ?? ((Math.random() * 1e9) | 0);
  if (m === 'play') {
    enterFullscreen();
    playerSlot = (opts.team ?? 0) * 5 + (opts.hero ?? 2);
    game = newMatch(seed, opts.team ?? 0, playerSlot);
    view.team = opts.team ?? 0;
    camLock = true;
    setSpeed(1);
    replay = null;
  } else if (m === 'replay' && opts.replay) {
    replay = opts.replay;
    replayIdx = 0;
    game = new Game({ ...replayConfig(replay), recordEvents: true });
    playerSlot = -1;
    const agent = replay.bots.findIndex((b) => !b);
    followSlot = agent >= 0 ? agent : 2;
    view.team = -1;
    setSpeed(1);
  } else {
    game = newMatch(seed, selTeam, -1);
    playerSlot = -1;
    followSlot = 2;
    view.team = -1;
    setSpeed(1);
    replay = null;
  }
  view.aimSlot = -1;
  view.amove = false;
  const f = focusUnit();
  if (f) { renderer.cam.x = f.x; renderer.cam.y = f.y; }
  renderer.setZoom(1);
  hud.setMode(m, m === 'replay' && replay?.meta?.teams ? `${replay.meta.teams[0]} vs ${replay.meta.teams[1]}` : undefined);
  hud.markVision(view.team);
  menuEl.classList.add('hidden');
  hudEl.classList.remove('hidden');
  if (m === 'play') hud.announce('Destroy the enemy Core', 'neutral');
  else if (m === 'spectate') hud.announce(`${aiName(selTeam === 0 ? selAllies : selEnemies)} <small>vs</small> ${aiName(selTeam === 0 ? selEnemies : selAllies)}`, 'neutral');
}

// --- Menu ---------------------------------------------------------------------------------
const LANE_PHRASE = ['the top lane', 'the mid lane', 'the bottom lane'];
/** Keyboard handling for the menu (hero keys, arrows, Enter); set by buildMenu. */
let menuKey: (k: string) => boolean = () => false;
/** Redraws the menu's choices after they changed from outside (models listed, URL options); set by buildMenu. */
let menuRefresh: () => void = () => {};

function buildMenu() {
  menuEl.innerHTML = `
    <header class="m-top">
      <div class="m-brand"><span class="m-word">Aimoba</span><span class="m-sub">Clash of the Aether Cores</span></div>
      <nav class="m-nav">
        <button id="btn-spectate" title="Watch these two teams play without you">Watch a match</button>
        <button id="btn-replay">Load a replay</button>
        <button id="btn-help">How to play</button>
        <input type="file" id="replay-file" accept=".json,application/json" hidden>
      </nav>
    </header>
    <main class="m-main">
      <section class="m-stage" aria-label="Selected hero">
        <div class="m-pool"></div>
        <p class="m-turn">Drag to turn</p>
      </section>
      <section class="m-dossier">
        <div class="m-who"></div>
        <ul class="m-kit"></ul>
        <div class="m-setup">
          <div class="m-field"><span class="m-label" id="lbl-side">Your side</span>
            <div class="m-pennants" role="radiogroup" aria-labelledby="lbl-side">
              <button class="m-pennant" role="radio" data-team="0">Blue</button>
              <button class="m-pennant" role="radio" data-team="1">Red</button>
            </div>
          </div>
          <div class="m-field"><label class="m-label" for="sel-allies">Your allies</label>
            <select class="m-select" id="sel-allies" data-side="allies"></select>
          </div>
          <div class="m-field"><label class="m-label" for="sel-enemies">Your enemies</label>
            <select class="m-select" id="sel-enemies" data-side="enemies"></select>
          </div>
        </div>
        <p class="m-note" aria-live="polite"></p>
        <button class="m-play" id="btn-play"><span class="m-play-text"></span><kbd>Enter</kbd></button>
      </section>
    </main>
    <footer class="m-roster" role="radiogroup" aria-label="Choose your hero">${HEROES.map((h, i) => `
      <button class="m-pick" role="radio" data-hero="${i}" style="--c:${h.color}">
        <span class="m-face"><img src="${portraits[i]}" alt=""></span>
        <span class="m-pick-name">${h.name}</span>
        <span class="m-pick-role">${h.role}</span>
        <kbd>${i + 1}</kbd>
      </button>`).join('')}</footer>`;
  const q = <T extends HTMLElement = HTMLElement>(sel: string) => menuEl.querySelector(sel) as T;
  const all = (sel: string) => [...menuEl.querySelectorAll<HTMLElement>(sel)];
  q('.m-stage').prepend(showcase.canvas);

  // Scripted bots first, then the trained models once their list has arrived.
  const fillAi = () => {
    const html = `<optgroup label="Scripted bots">${Object.entries(BOT_LABEL).map(([k, label]) => `<option value="${k}">${label}</option>`).join('')}</optgroup>` +
      (models.length ? `<optgroup label="Trained models">${models.map((m) => `<option value="${m.id}">${m.label}</option>`).join('')}</optgroup>` : '');
    for (const sel of all('.m-select')) sel.innerHTML = html;
  };
  fillAi();

  let shown = -1;
  const refresh = () => {
    const h = HEROES[selHero];
    menuEl.style.setProperty('--c', h.color);
    menuEl.dataset.team = String(selTeam);
    for (const c of all('.m-pick')) { const on = Number(c.dataset.hero) === selHero; c.classList.toggle('on', on); c.setAttribute('aria-checked', String(on)); }
    for (const b of all('.m-pennant')) { const on = Number(b.dataset.team) === selTeam; b.classList.toggle('on', on); b.setAttribute('aria-checked', String(on)); }
    for (const sel of all('.m-select') as HTMLSelectElement[]) {
      const kind = sel.dataset.side === 'allies' ? selAllies : selEnemies;
      sel.value = kind;
      sel.classList.toggle('m-select-model', !isBotKind(kind));
      sel.title = models.find((x) => x.id === kind)?.blurb ?? '';
    }
    const note = (kind: string, who: string) => { const m = models.find((x) => x.id === kind); return m ? `<span><b>${who}: ${m.label}.</b> ${m.blurb}</span>` : ''; };
    q('.m-note').innerHTML = selAllies === selEnemies ? note(selAllies, 'Both teams') : note(selAllies, 'Allies') + note(selEnemies, 'Enemies');
    q('.m-play-text').textContent = `Play ${h.name}`;
    showcase.set(h.id, selTeam);
    if (shown === selHero) return;
    shown = selHero;
    // The dossier is rebuilt, which also restarts its entrance animation.
    q('.m-who').innerHTML = `
      <h1 class="m-name">${h.name}</h1>
      <p class="m-title">${h.title}</p>
      <p class="m-blurb"><b>${h.role} for ${h.lane < 0 ? 'the jungle' : LANE_PHRASE[h.lane]}.</b> ${h.blurb}</p>`;
    q('.m-kit').innerHTML = `
      <li><span class="m-key m-key-passive">Passive</span><img src="${iconUrl('sun', h.color)}" alt=""><div><b>${h.passive.name}</b><span>${h.passive.desc}</span></div></li>
      ${h.abilities.map((a, i) => `<li><span class="m-key">${'QWER'[i]}</span><img src="${iconUrl(a.icon, h.color)}" alt=""><div><b>${a.name}</b><span>${a.desc}</span></div></li>`).join('')}`;
  };
  // Each hero answers with a note of their own (a pentatonic scale, so any order sounds like a tune).
  const pick = (i: number) => { selHero = (i + HEROES.length) % HEROES.length; sound.ui('pick', { rate: 2 ** ([0, 2, 4, 7, 9][selHero % 5] / 12) }); refresh(); };
  const play = () => startGame('play', { hero: selHero, team: selTeam });

  for (const c of all('.m-pick')) {
    c.addEventListener('click', () => pick(Number(c.dataset.hero)));
    c.addEventListener('dblclick', play);
  }
  for (const b of all('.m-pennant')) b.addEventListener('click', () => { selTeam = Number(b.dataset.team); sound.ui('click'); refresh(); });
  for (const sel of all('.m-select') as HTMLSelectElement[]) sel.addEventListener('change', () => {
    const kind = sel.value;
    if (sel.dataset.side === 'allies') selAllies = kind; else selEnemies = kind;
    if (!isBotKind(kind)) loadNet(kind).catch(() => {}); // start fetching now; startGame reports a failure
    sound.ui('click');
    sel.blur(); // hand the keyboard back to the menu (hero keys, Enter)
    refresh();
  });
  q('#btn-play').addEventListener('click', play);
  q('#btn-spectate').addEventListener('click', () => startGame('spectate'));
  q('#btn-help').addEventListener('click', showHelp);
  const file = q<HTMLInputElement>('#replay-file');
  q('#btn-replay').addEventListener('click', () => file.click());
  file.addEventListener('change', async () => {
    const f = file.files?.[0];
    if (!f) return;
    try {
      const r = JSON.parse(await f.text()) as Replay;
      if (r.version !== 1 || !Array.isArray(r.commands)) throw new Error('not an AIMOBA replay');
      startGame('replay', { replay: r });
    } catch (err) {
      alert('Could not load replay: ' + (err as Error).message);
    }
    file.value = '';
  });
  menuKey = (k) => {
    if (k >= '1' && k <= String(HEROES.length)) pick(Number(k) - 1);
    else if (k === 'arrowleft') pick(selHero - 1);
    else if (k === 'arrowright') pick(selHero + 1);
    else if (k === 'enter' && !(document.activeElement instanceof HTMLButtonElement)) play();
    else if (k === 'm') audio.set({ muted: !audio.settings.muted });
    else return false;
    return true;
  };
  menuRefresh = () => { fillAi(); refresh(); };
  refresh();
}

// --- Overlays -----------------------------------------------------------------------------
function openOverlay(html: string) {
  overlayEl.innerHTML = `<div class="dialog panel">${html}</div>`;
  if (!overlayOpen) sound.ui('open');
  overlayEl.classList.remove('hidden');
  overlayOpen = true;
}
function closeOverlay() {
  if (overlayOpen) sound.ui('close');
  overlayEl.classList.add('hidden');
  overlayOpen = false;
}
const HELP_HTML = `
  <div class="help-cols">
    <div><h4>Controls</h4><div class="help-grid">
      <span><kbd>Right-click</kbd></span><span>Move / attack the unit under the cursor</span>
      <span><kbd>Q</kbd> <kbd>W</kbd> <kbd>E</kbd> <kbd>R</kbd></span><span>Hold to aim, release to cast at the cursor</span>
      <span><kbd>Ctrl</kbd>+<kbd>Q…R</kbd></span><span>Spend a skill point (or click the + above an ability)</span>
      <span><kbd>A</kbd> then click</span><span>Attack-move</span>
      <span><kbd>S</kbd></span><span>Stop</span>
      <span><kbd>B</kbd></span><span>Recall to base (5s channel)</span>
      <span><kbd>P</kbd></span><span>Open the shop (buy at the fountain or while dead)</span>
      <span><kbd>Tab</kbd></span><span>Scoreboard</span>
      <span><kbd>Y</kbd> / <kbd>Space</kbd></span><span>Toggle camera lock / centre on your hero</span>
      <span><kbd>Wheel</kbd></span><span>Zoom</span>
      <span><kbd>M</kbd> / <kbd>Esc</kbd></span><span>Mute / menu</span>
    </div></div>
    <div><h4>How to win</h4><div class="help-grid" style="max-width:380px">
      <span>1</span><span>Last-hit enemy minions for gold, stay near fights for experience.</span>
      <span>2</span><span>Spend gold in the shop on items; level up abilities as you gain levels.</span>
      <span>3</span><span>Towers shoot you unless your minions are tanking. Push with your wave.</span>
      <span>4</span><span>Jungle camps give gold and buffs. The river Wyrm empowers your whole team; the Titan grants a siege buff.</span>
      <span>5</span><span>Destroy the outer and inner tower of a lane, then the two Guardians, then the enemy Core.</span>
    </div></div>
  </div>`;
const VOLUMES: [Exclude<keyof AudioSettings, 'muted'>, string][] = [['master', 'Master'], ['effects', 'Effects'], ['music', 'Music'], ['ambience', 'Ambience']];
function showHelp() {
  openOverlay(`<h3>How to play</h3>${HELP_HTML}<button class="btn" id="ov-close">Got it</button>`);
  overlayEl.querySelector('#ov-close')!.addEventListener('click', closeOverlay);
}
function showPause() {
  openOverlay(`<h3>${mode === 'play' ? 'Paused' : 'Menu'}</h3>
    <div class="menu-row"><button class="btn primary" id="ov-resume">Resume</button></div>
    <div class="volumes${audio.settings.muted ? ' muted' : ''}">${VOLUMES.map(([key, label]) => `
      <label><span>${label}</span><input type="range" min="0" max="100" step="1" data-vol="${key}" value="${Math.round(audio.settings[key] * 100)}"><output>${Math.round(audio.settings[key] * 100)}</output></label>`).join('')}
    </div>
    <div class="menu-row"><button class="btn" id="ov-help">Controls</button>${replay ? '' : '<button class="btn" id="ov-save">Save replay</button>'}<button class="btn" id="ov-mute">${audio.settings.muted ? 'Unmute' : 'Mute'} sound</button><button class="btn" id="ov-quit">Quit to menu</button></div>`);
  overlayEl.querySelector('#ov-resume')!.addEventListener('click', closeOverlay);
  overlayEl.querySelector('#ov-help')!.addEventListener('click', showHelp);
  overlayEl.querySelector('#ov-save')?.addEventListener('click', saveReplay);
  overlayEl.querySelector('#ov-mute')!.addEventListener('click', () => { audio.set({ muted: !audio.settings.muted }); showPause(); });
  overlayEl.querySelector('#ov-quit')!.addEventListener('click', () => { closeOverlay(); startDemo(); });
  for (const input of overlayEl.querySelectorAll<HTMLInputElement>('[data-vol]')) {
    input.addEventListener('input', () => {
      audio.set({ [input.dataset.vol!]: Number(input.value) / 100 });
      input.nextElementSibling!.textContent = input.value;
    });
    // Let go of the effects slider and hear the level you chose.
    input.addEventListener('change', () => { if (input.dataset.vol !== 'music' && input.dataset.vol !== 'ambience') sound.ui('pick'); });
  }
}
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
/** How a recorded match was decided, when not by a core: [with a winner, as a draw]. */
const REASONS: Record<string, [string, string]> = {
  time: ['Time limit: ahead on structures or kills', 'Time limit: level on structures and kills'],
  draw: ['', 'Time limit: level on structures and kills'],
  crash: ['{loser} crashed', 'Both sides crashed'],
  timeout: ['{loser} ran out of thinking time', 'Both sides ran out of thinking time'],
  protocol: ['{loser} broke the protocol', 'Both sides broke the protocol'],
  init: ['{loser} failed to start', 'Neither side started'],
};

function showEnd() {
  ended = true;
  const w = game.winner;
  let title: string, cls: string, note = '';
  if (mode === 'play') {
    const won = w === me()!.team;
    title = w < 0 ? 'Draw' : won ? 'Victory' : 'Defeat';
    cls = won ? 'win' : 'lose';
  } else if (replay?.meta) {
    // A recorded match: named teams, and the result as it was scored (which a time limit or a forfeit can decide).
    const r = replay.meta.result ?? w, names = replay.meta.teams ?? ['Blue', 'Red'];
    title = r < 0 ? 'Draw' : `${esc(names[r])} wins`;
    cls = 'win';
    const why = REASONS[replay.meta.reason ?? ''];
    if (why) note = `<div class="sub">${r < 0 ? why[1] : why[0].replace('{loser}', esc(names[1 - r]))}</div>`;
  } else {
    title = w < 0 ? 'Draw' : w === 0 ? 'Blue wins' : 'Red wins';
    cls = 'win';
  }
  setTimeout(() => {
    if (!ended) return;
    openOverlay(`<h2 class="${cls}">${title}</h2>${note}<div class="sub">Game length ${Math.floor(game.time / 60)}:${String(Math.floor(game.time % 60)).padStart(2, '0')}</div>
      <div id="scoreboard" style="position:static;transform:none;display:flex;flex-direction:column;gap:14px">${scoreboardHtml(game, portraits)}</div>
      <div class="menu-row"><button class="btn primary" id="ov-again">${mode === 'play' ? 'Play again' : 'Watch another'}</button>${replay ? '' : '<button class="btn" id="ov-save">Save replay</button>'}<button class="btn" id="ov-quit">Main menu</button></div>`);
    const m = mode;
    overlayEl.querySelector('#ov-again')!.addEventListener('click', () => (m === 'play' ? startGame('play', { hero: selHero, team: selTeam }) : m === 'replay' && replay ? startGame('replay', { replay }) : startGame('spectate')));
    overlayEl.querySelector('#ov-save')?.addEventListener('click', saveReplay);
    overlayEl.querySelector('#ov-quit')!.addEventListener('click', () => { closeOverlay(); startDemo(); });
  }, 1800);
}
function saveReplay() {
  const blob = new Blob([JSON.stringify(makeReplay(game))], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `aimoba-${game.seed}.replay.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- Input ----------------------------------------------------------------------------------
/** Render interpolation factor of the last frame, so picking matches what is on screen. */
let drawAlpha = 1;
/** Where a unit is drawn right now (it trails the simulation by up to one tick). */
function drawnPos(u: Unit): [number, number] {
  if (Math.abs(u.x - u.px) + Math.abs(u.y - u.py) >= 300) return [u.x, u.y];
  return [u.px + (u.x - u.px) * drawAlpha, u.py + (u.y - u.py) * drawAlpha];
}
/** Pick tolerance around a unit's body: at least 14 screen pixels, whatever the zoom. */
const pickSlack = () => Math.max(25, 14 / renderer.cam.zoom);

/**
 * The unit under the cursor, whose ground point is (x, y). Distance is measured from the
 * cursor ray to the unit's upright body, so pointing at a model picks it. A unit whose body
 * contains the cursor wins over one that is merely close (heroes first among those), otherwise
 * the nearest within `slack` world units.
 */
function unitAt(x: number, y: number, pred: (u: Unit) => boolean, slack = pickSlack()): Unit | undefined {
  let best: Unit | undefined, bestScore = Infinity;
  for (const u of game.units) {
    if (!u.alive || !pred(u)) continue;
    const [ux, uy] = drawnPos(u);
    const d = renderer.pickDistance(u, ux, uy, x, y) - u.radius;
    if (d >= slack) continue;
    const score = d <= 0 ? (u.kind === 'hero' ? -2e6 : -1e6) + d : d;
    if (score < bestScore) { bestScore = score; best = u; }
  }
  return best;
}
/** What a right-click at (x, y) would attack: the same rule drives the hover highlight. */
function attackTargetAt(x: number, y: number, team: number): Unit | undefined {
  return unitAt(x, y, (o) => o.team !== team && visible(o) && (o.vulnerable || (o.kind !== 'tower' && o.kind !== 'core')));
}
const visible = (u: Unit) => view.team < 0 || game.canSee(view.team, u);

function castAbility(slot: number, fromClick = false) {
  const u = me();
  if (!u || !u.hero) return;
  const err = game.castError(u, slot);
  if (err) { hint(err[0].toUpperCase() + err.slice(1)); sound.ui('deny'); return; }
  const ab = HEROES[u.hero.def].abilities[slot];
  let x = view.mouseX, y = view.mouseY, id = -1;
  if (fromClick && (ab.targeting === 'dir' || ab.targeting === 'point')) { view.aimSlot = slot; hint('Click to cast, right-click to cancel'); return; }
  if (ab.targeting === 'enemy' || ab.targeting === 'enemyHero') {
    const ok = (t: Unit) => t.team !== u.team && visible(t) && t.kind !== 'tower' && t.kind !== 'core' && (ab.targeting === 'enemy' || t.kind === 'hero');
    const t = unitAt(x, y, ok, fromClick ? 1e9 : 260);
    if (!t || (fromClick && Math.hypot(t.x - u.x, t.y - u.y) > ab.range + 400)) { hint('No target'); sound.ui('deny'); return; }
    id = t.id;
  } else if (ab.targeting === 'ally') {
    const t = fromClick || keys.has('alt') ? undefined : unitAt(x, y, (t) => t.team === u.team && t.kind === 'hero', 220);
    id = t ? t.id : u.id;
  } else if (ab.targeting === 'self') { x = u.x; y = u.y; }
  send({ t: 'cast', slot, x, y, id });
}

function rightClick() {
  const u = me();
  if (!u) return;
  const t = attackTargetAt(view.mouseX, view.mouseY, u.team);
  mouse.pressX = mouse.x;
  mouse.pressY = mouse.y;
  mouse.pressTarget = t ? t.id : -1;
  if (t) {
    send({ t: 'attack', id: t.id });
    fx.ring(t.x, t.y, t.radius + 40, t.radius + 5, '#ff6b5e', 0.35, 6);
  } else {
    send({ t: 'move', x: view.mouseX, y: view.mouseY });
    fx.ring(view.mouseX, view.mouseY, 46, 8, '#7dffa0', 0.35, 6);
  }
  mouse.lastMove = performance.now();
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
window.addEventListener('mousemove', (e) => {
  if (mouse.drag) renderer.panPixels(e.movementX, e.movementY);
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});
canvas.addEventListener('mousedown', (e) => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
  [view.mouseX, view.mouseY] = renderer.toWorld(e.clientX, e.clientY);
  if (mode === 'play') {
    if (e.button === 2) {
      if (view.aimSlot >= 0 || view.amove) { view.aimSlot = -1; view.amove = false; return; }
      mouse.right = true;
      rightClick();
    } else if (e.button === 0) {
      if (view.amove) { send({ t: 'amove', x: view.mouseX, y: view.mouseY }); fx.ring(view.mouseX, view.mouseY, 46, 8, '#ff6b5e', 0.35, 6); view.amove = false; }
      else if (view.aimSlot >= 0) { const s = view.aimSlot; view.aimSlot = -1; castAbility(s); }
    }
  } else if (mode !== 'menu' && e.button === 0) {
    const t = unitAt(view.mouseX, view.mouseY, (o) => o.kind === 'hero', 40);
    if (t) followSlot = t.hero!.slot;
    else { followSlot = -1; mouse.drag = true; }
  }
});
window.addEventListener('mouseup', (e) => {
  if (e.button === 2) mouse.right = false;
  if (e.button === 0) mouse.drag = false;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (mode !== 'menu') renderer.setZoom(renderer.zoomFactor * (e.deltaY > 0 ? 0.92 : 1.08));
}, { passive: false });
window.addEventListener('resize', () => renderer.resize());
window.addEventListener('blur', () => { keys.clear(); mouse.right = false; mouse.drag = false; });

// Ctrl+W (level up W) closes the tab: browsers ignore preventDefault on it. The only real fix is the
// Keyboard Lock API, which Chromium honours only while the page is in element fullscreen — so play
// matches go fullscreen and lock the ability keys (Esc too, so it still opens pause; hold Esc to leave
// fullscreen). Outside fullscreen / in other browsers, beforeunload at least asks before closing.
const LOCK_KEYS = ['KeyQ', 'KeyW', 'KeyE', 'KeyR', 'Escape'];
const kbd = (navigator as Navigator & { keyboard?: { lock(codes?: string[]): Promise<void>; unlock(): void } }).keyboard;
function enterFullscreen() {
  if (document.fullscreenElement || !document.documentElement.requestFullscreen) return;
  document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
}
document.addEventListener('fullscreenchange', () => {
  if (!kbd) return;
  if (document.fullscreenElement) kbd.lock(LOCK_KEYS).catch(() => {});
  else kbd.unlock();
});
window.addEventListener('beforeunload', (e) => {
  if (mode === 'play' && !ended) { e.preventDefault(); e.returnValue = ''; }
});

const SLOT_KEYS: Record<string, number> = { q: 0, w: 1, e: 2, r: 3 };
window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (mode === 'menu') {
    if (k === 'escape' && overlayOpen) closeOverlay();
    else if (!overlayOpen && !e.repeat && menuKey(k)) e.preventDefault();
    return;
  }
  if (k === 'tab') { e.preventDefault(); hud.showScoreboard(true); return; }
  if (k === 'escape') {
    if (overlayOpen) { if (!ended) closeOverlay(); }
    else if (hud.isShopOpen()) hud.toggleShop(false);
    else if (view.aimSlot >= 0 || view.amove) { view.aimSlot = -1; view.amove = false; }
    else showPause();
    return;
  }
  if (overlayOpen) return;
  if (k === 'm') { audio.set({ muted: !audio.settings.muted }); hint(audio.settings.muted ? 'Sound off' : 'Sound on'); return; }
  if (e.repeat) return;
  keys.add(k === ' ' ? 'space' : k);
  if (mode === 'play') {
    if (k in SLOT_KEYS) {
      e.preventDefault();
      const slot = SLOT_KEYS[k];
      const u = me();
      if (!u) return;
      if (e.ctrlKey || e.metaKey) { levelAbility(slot); return; }
      const ab = HEROES[u.hero!.def].abilities[slot];
      const err = game.castError(u, slot);
      if (err) { hint(err[0].toUpperCase() + err.slice(1)); sound.ui('deny'); return; }
      if (ab.targeting === 'self') castAbility(slot);
      else { view.aimSlot = slot; view.amove = false; }
    } else if (k === 'a') { view.amove = true; view.aimSlot = -1; }
    else if (k === 's') send({ t: 'stop' });
    else if (k === 'b') send({ t: 'recall' });
    else if (k === 'p') hud.toggleShop();
    else if (k === 'y') { camLock = !camLock; hint(camLock ? 'Camera locked' : 'Camera unlocked'); sound.ui('click'); }
  } else {
    if (k >= '0' && k <= '9') followSlot = k === '0' ? 9 : Number(k) - 1;
    else if (k === ' ') setSpeed(speed === 0 ? 1 : 0);
    else if (k === '+' || k === '=') setSpeed(Math.min(8, speed === 0 ? 1 : speed * 2));
    else if (k === '-') setSpeed(Math.max(0.5, speed / 2));
  }
});
window.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  keys.delete(k === ' ' ? 'space' : k);
  if (k === 'tab') hud.showScoreboard(false);
  if (mode === 'play' && k in SLOT_KEYS && view.aimSlot === SLOT_KEYS[k] && !overlayOpen) {
    view.aimSlot = -1;
    castAbility(SLOT_KEYS[k]);
  }
});

// --- Simulation events -> effects ---------------------------------------------------------
/** What the sound director needs to know about the current view (one object, refreshed in place). */
const sctx: SoundCtx = {
  game: undefined!, mode: 'menu', focus: undefined, camX: 0, camY: 0, paused: false,
  shown: (x, y) => view.team < 0 || renderer.inSight(game, view.team, x, y),
};
function soundCtx(): SoundCtx {
  sctx.game = game;
  sctx.mode = mode;
  sctx.focus = focusUnit();
  sctx.camX = renderer.cam.x;
  sctx.camY = renderer.cam.y;
  sctx.paused = mode !== 'menu' && overlayOpen && (mode === 'play' || ended);
  return sctx;
}

function handleEvent(ev: GameEvent) {
  sound.event(ev, soundCtx());
  const x = ev.x ?? 0, y = ev.y ?? 0;
  const focus = focusUnit();
  const fid = focus ? focus.id : -1;
  const shown = (px: number, py: number) => view.team < 0 || renderer.inSight(game, view.team, px, py);
  const near = () => Math.max(0, 1 - Math.hypot(x - renderer.cam.x, y - renderer.cam.y) / 2600);
  switch (ev.type) {
    case 'dmg': {
      const t = game.unit(ev.id!), s = ev.other! >= 0 ? game.unit(ev.other!) : undefined;
      if (!shown(x, y)) return;
      const amt = Math.round(ev.amount!);
      const involvesFocus = ev.id === fid || ev.other === fid;
      const heroInvolved = (t && t.kind === 'hero') || (s && s.kind === 'hero');
      if (ev.slot !== 0 && (involvesFocus || heroInvolved)) fx.burst(x, y, ev.dtype === MAGIC ? '#c9a2ff' : ev.dtype === PHYS ? '#ffd9a8' : '#fff', 4, 260, 7, 0.25);
      if (amt >= 2 && (mode === 'play' ? involvesFocus : heroInvolved && t && t.kind === 'hero')) {
        const mine = ev.id === fid;
        fx.text(x, y, String(amt), mine ? '#ff6b5e' : ev.dtype === MAGIC ? '#d6b3ff' : ev.dtype === PHYS ? '#ffe2b8' : '#ffffff', amt > 250 ? 24 : amt > 90 ? 19 : 15);
      }
      break;
    }
    case 'heal':
      if (ev.amount! >= 8 && shown(x, y)) { fx.text(x, y, '+' + Math.round(ev.amount!), '#7af08a', 16); }
      break;
    case 'gold':
      if (ev.id === fid) fx.text(x, y - 30, '+' + Math.round(ev.amount!) + 'g', '#ffd84d', 16, 1.1);
      break;
    case 'cast': {
      const u = game.unit(ev.id!);
      if (!u || !u.hero || !shown(x, y)) return;
      fx.ring(x, y, 30, 110, HEROES[u.hero.def].color, 0.35, 7);
      break;
    }
    case 'levelup':
      if (!shown(x, y)) return;
      fx.ring(x, y, 20, 170, '#ffe58a', 0.7, 10);
      fx.burst(x, y, '#ffe58a', 18, 380, 9, 0.7);
      if (ev.id === fid) fx.text(x, y - 50, 'LEVEL UP', '#ffe58a', 20, 1.4);
      break;
    case 'respawn':
      if (shown(x, y)) { fx.ring(x, y, 160, 30, '#bfe8ff', 0.6, 10); if (ev.id === fid && mode === 'play') camLock = true; }
      break;
    case 'death': {
      if (!shown(x, y)) return;
      const col = TEAM_COLORS[ev.team ?? 2];
      if (ev.name === 'minion') fx.burst(x, y, col, 8, 240, 8, 0.4);
      else if (ev.name === 'monster') { fx.burst(x, y, '#e0c06a', 22, 420, 12, 0.6); fx.ring(x, y, 20, 150, '#e0c06a', 0.5); }
      else if (ev.name === 'hero') {
        fx.burst(x, y, col, 40, 620, 13, 0.9);
        fx.burst(x, y, '#ffffff', 16, 380, 8, 0.6);
        fx.ring(x, y, 30, 260, col, 0.7, 14);
        fx.disc(x, y, 150, col, 0.5);
      } else {
        fx.shards(x, y, '#6b7280', 40, 700, 26);
        fx.burst(x, y, col, 60, 900, 18, 1.2);
        fx.ring(x, y, 40, 520, '#fff', 0.9, 20);
        fx.disc(x, y, 320, col, 0.8);
        fx.shake = Math.max(fx.shake, 26 * near());
      }
      break;
    }
    case 'kill':
      hud.onEvent(ev, game);
      if (mode === 'play' && ev.other === fid && ev.id !== fid) hud.announce('Enemy slain!', 'good');
      break;
    case 'announce':
      hud.onEvent(ev, game);
      break;
    case 'gameover':
      if (mode !== 'menu') showEnd();
      break;
    case 'fx': {
      if (!shown(x, y)) return;
      const r = ev.r ?? 100;
      const v = near();
      switch (ev.name) {
        case 'shield': fx.ring(x, y, r * 1.6, r * 0.8, '#d6ecff', 0.4, 8); break;
        case 'buff': fx.ring(x, y, 30, 130, '#ffe29a', 0.45, 8); fx.burst(x, y, '#ffe29a', 10, 300, 8, 0.5); break;
        case 'quake': fx.ring(x, y, 40, r, '#d9a441', 0.45, 18); fx.disc(x, y, r, '#8a6220', 0.35); fx.shards(x, y, '#7a6a4a', 14, 520, 14); fx.shake = Math.max(fx.shake, 9 * v); break;
        case 'telegraph': fx.disc(x, y, r, '#ffb347', 0.5); fx.ring(x, y, r, r * 0.9, '#ffd98a', 0.5, 8); break;
        case 'cataclysm': fx.ring(x, y, 60, r * 1.1, '#fff1c2', 0.6, 24); fx.disc(x, y, r, '#d9a441', 0.5); fx.shards(x, y, '#8a7a5a', 30, 800, 20); fx.burst(x, y, '#ffd98a', 40, 800, 14, 0.8); fx.shake = Math.max(fx.shake, 24 * v); break;
        case 'impact': fx.burst(x, y, '#fff1c2', 16, 480, 11, 0.4); fx.ring(x, y, 20, 130, '#fff1c2', 0.3, 10); fx.shake = Math.max(fx.shake, 7 * v); break;
        case 'smoke': fx.burst(x, y, '#5b3fa0', 30, 320, 26, 0.9); fx.burst(x, y, '#1d1433', 20, 220, 30, 1.1); break;
        case 'slash': fx.line(x - 70, y - 50, x + 70, y + 50, '#e6d9ff', 0.25, 12); fx.line(x - 70, y + 50, x + 70, y - 50, '#b794ff', 0.3, 10); fx.burst(x, y, '#b794ff', 10, 400, 8, 0.3); break;
        case 'execute': fx.line(x - 110, y - 80, x + 110, y + 80, '#fff', 0.35, 18); fx.line(x - 110, y + 80, x + 110, y - 80, '#ff5d7a', 0.4, 16); fx.ring(x, y, 30, 220, '#ff5d7a', 0.5, 14); fx.burst(x, y, '#ff5d7a', 30, 700, 12, 0.6); fx.shake = Math.max(fx.shake, 12 * v); break;
        case 'burst': fx.burst(x, y, '#ff8a3d', 22, 520, 13, 0.45); fx.ring(x, y, 20, 120, '#ffcf8a', 0.3, 10); break;
        case 'explosion': fx.disc(x, y, r, '#ff7a2e', 0.45); fx.ring(x, y, r * 0.3, r * 1.1, '#ffd08a', 0.45, 16); fx.burst(x, y, '#ff8a3d', 36, 700, 16, 0.7); fx.shake = Math.max(fx.shake, 10 * v); break;
        case 'starfall': fx.disc(x, y, r, '#ffb347', 0.9); fx.ring(x, y, r * 0.2, r * 1.25, '#fff3c4', 0.7, 28); fx.ring(x, y, r * 0.1, r * 0.8, '#ff7a2e', 0.9, 18); fx.burst(x, y, '#ff8a3d', 80, 1100, 20, 1.1); fx.burst(x, y, '#fff3c4', 30, 700, 12, 0.7); fx.shards(x, y, '#5a3a2a', 24, 900, 18); fx.shake = Math.max(fx.shake, 30 * v); break;
        case 'snare': fx.ring(x, y, 150, 50, '#ffb347', 0.4, 12); fx.burst(x, y, '#ffb347', 12, 300, 9, 0.4); break;
        case 'spark': fx.burst(x, y, '#6ee7ae', 10, 420, 8, 0.3); break;
        case 'thunder': fx.disc(x, y, r, '#aef6ff', 0.45); fx.ring(x, y, 30, r * 1.1, '#ffffff', 0.5, 18); fx.burst(x, y, '#7df9ff', 46, 900, 14, 0.7); for (let i = 0; i < 5; i++) { const a = i * 1.2566; fx.line(x, y, x + Math.cos(a) * r, y + Math.sin(a) * r, '#ffffff', 0.3, 9); } fx.shake = Math.max(fx.shake, 16 * v); break;
        case 'flash': fx.burst(x, y, '#fff3c4', 18, 460, 11, 0.4); fx.ring(x, y, 20, 140, '#ffd166', 0.35, 10); break;
        case 'heal': fx.ring(x, y, 110, 40, '#7af08a', 0.5, 9); fx.burst(x, y, '#9dffac', 12, 240, 9, 0.7); break;
        case 'beam': fx.line(x, y, ev.x2 ?? x, ev.y2 ?? y, '#fff3c4', 0.35, 12); break;
        case 'chain': fx.line(x, y, ev.x2 ?? x, ev.y2 ?? y, '#aef6ff', 0.22, 7); fx.burst(ev.x2 ?? x, ev.y2 ?? y, '#7df9ff', 8, 360, 8, 0.3); break;
        case 'sanctuary': fx.disc(x, y, r, '#ffd166', 0.7); fx.ring(x, y, 60, r, '#fff3c4', 0.8, 20); fx.ring(x, y, 30, r * 0.7, '#ffd166', 1.0, 12); fx.burst(x, y, '#fff3c4', 50, 800, 12, 1); break;
        case 'recall': fx.ring(x, y, 120, 50, '#9be0ff', 0.5, 8); break;
        case 'teleport': fx.ring(x, y, 30, 220, '#9be0ff', 0.5, 14); fx.burst(x, y, '#bfe8ff', 26, 520, 11, 0.6); break;
      }
      break;
    }
  }
}

// --- Frame loop ---------------------------------------------------------------------------------
let last = performance.now();
function frame(now: number) {
  const dt = Math.max(0, Math.min(0.1, (now - last) / 1000)); // the first rAF timestamp can predate `last`
  last = now;
  const simSpeed = mode === 'menu' ? 1 : overlayOpen && mode === 'play' ? 0 : ended ? 0 : speed;
  acc += dt * simSpeed;
  let steps = 0;
  const maxSteps = Math.max(6, Math.ceil(simSpeed * 4));
  while (acc >= DT && steps < maxSteps) {
    if (replay) {
      const cmds = replay.commands;
      while (replayIdx < cmds.length && cmds[replayIdx][0] <= game.tick + 1) { game.command(cmds[replayIdx][1], cmds[replayIdx][2]); replayIdx++; }
    }
    for (const d of drivers) d.tick();
    game.step();
    for (const ev of game.events) handleEvent(ev);
    game.events.length = 0;
    acc -= DT;
    steps++;
    if (game.over) {
      if (mode === 'menu') { startDemo(); break; }
      break;
    }
    if (replay && game.tick >= replay.endTick) {
      // The recording ends here although no core fell: a match stopped by a forfeit.
      if (!ended) showEnd();
      break;
    }
  }
  if (acc > DT * 3) acc = 0;
  const alpha = simSpeed === 0 || game.over ? 1 : Math.min(1, acc / DT);
  drawAlpha = alpha;

  // Camera.
  const cam = renderer.cam;
  const f = focusUnit();
  const pan = (1500 / renderer.zoomFactor) * dt;
  if (mode === 'menu') {
    if (f) { cam.x += (f.x - cam.x) * Math.min(1, dt * 1.2); cam.y += (f.y - cam.y) * Math.min(1, dt * 1.2); }
  } else {
    const following = mode === 'play' ? camLock || keys.has('space') : followSlot >= 0;
    if (following && f) {
      const fx0 = f.px + (f.x - f.px) * alpha, fy0 = f.py + (f.y - f.py) * alpha;
      const jump = Math.hypot(fx0 - cam.x, fy0 - cam.y) > 1500;
      const k = jump ? 1 : Math.min(1, dt * 10);
      cam.x += (fx0 - cam.x) * k;
      cam.y += (fy0 - cam.y) * k;
    } else if (!overlayOpen) {
      const edge = 14;
      if (keys.has('arrowleft') || (mouse.x < edge && document.hasFocus())) cam.x -= pan;
      if (keys.has('arrowright') || (mouse.x > renderer.w - edge && document.hasFocus())) cam.x += pan;
      if (keys.has('arrowup') || (mouse.y < edge && document.hasFocus())) cam.y -= pan;
      if (keys.has('arrowdown') || (mouse.y > renderer.h - edge && document.hasFocus())) cam.y += pan;
    }
  }
  renderer.clampCam();

  [view.mouseX, view.mouseY] = renderer.toWorld(mouse.x, mouse.y);
  view.focusId = f && mode !== 'menu' ? f.id : -1;
  const meTeam = f ? f.team : 0;
  view.hoverId = -1;
  if (mode !== 'menu' && !overlayOpen) {
    // In play the highlight is exactly what a right-click would attack.
    const h = mode === 'play' && f
      ? attackTargetAt(view.mouseX, view.mouseY, meTeam)
      : unitAt(view.mouseX, view.mouseY, (o) => visible(o) && o !== f && o.kind !== 'tower' && o.kind !== 'core');
    if (h) view.hoverId = h.id;
  }
  if (mode === 'play' && mouse.right && now - mouse.lastMove > 140 && !overlayOpen && f) {
    // Holding right-click: keep attacking a unit under the cursor (and catch new ones it passes
    // over); otherwise walk to the cursor. A press that started on a target keeps attacking it
    // until the mouse actually moves away, so a fleeing target cannot drag you onto the ground.
    const t = view.hoverId >= 0 ? game.unit(view.hoverId) : undefined;
    const o = f.hero!.order;
    const held = mouse.pressTarget >= 0 && Math.hypot(mouse.x - mouse.pressX, mouse.y - mouse.pressY) < 12;
    if (t) {
      if (o.type !== 'attack' || o.targetId !== t.id) { send({ t: 'attack', id: t.id }); mouse.pressTarget = t.id; }
      mouse.lastMove = now;
    } else if (!held) {
      mouse.pressTarget = -1;
      send({ t: 'move', x: view.mouseX, y: view.mouseY });
      mouse.lastMove = now;
    }
  }
  document.body.classList.toggle('aiming', view.aimSlot >= 0 || view.amove);

  fx.update(dt);
  sound.frame(dt, soundCtx());
  // Sounds carry a little beyond the edge of the screen, however far the camera is zoomed.
  audio.frame(cam.x, cam.y, Math.max(1800, Math.min(4500, (1.7 * renderer.w) / 2 / cam.zoom)));
  if (mode === 'menu') showcase.frame(dt);
  renderer.draw(game, view, alpha, dt, fx);
  if (mode !== 'menu') hud.update(game, f, view.team, replay ? Math.min(1, game.tick / Math.max(1, replay.endTick)) : 0);
  requestAnimationFrame(frame);
}

// --- Boot --------------------------------------------------------------------------------------
buildMenu();
startDemo();
const params = new URLSearchParams(location.search);
const pm = params.get('mode') ?? params.get('auto'); // ?auto=spectate | ?auto=play (alias of ?mode=)
const seedParam = params.get('seed') ? Number(params.get('seed')) : undefined;
if (params.get('zoom')) setTimeout(() => renderer.setZoom(Number(params.get('zoom'))), 0);
// ?bots= sets who plays both teams, ?allies= / ?enemies= one of them: easy, basic, expert or the id of a trained model.
modelsReady.then(() => {
  const known = (k: string | null): k is string => !!k && (isBotKind(k) || models.some((m) => m.id === k));
  if (known(params.get('bots'))) selAllies = selEnemies = params.get('bots')!;
  if (known(params.get('allies'))) selAllies = params.get('allies')!;
  if (known(params.get('enemies'))) selEnemies = params.get('enemies')!;
  menuRefresh();
  // A failed download is reported by startGame, which tries again.
  return Promise.all([selAllies, selEnemies].filter((k) => !isBotKind(k)).map(loadNet)).catch(() => {});
}).then(() => {
  if (pm === 'play') {
    selHero = Number(params.get('hero') ?? 2);
    selTeam = Number(params.get('team') ?? 0);
    startGame('play', { hero: selHero, team: selTeam, seed: seedParam });
  } else if (pm === 'spectate') {
    selTeam = Number(params.get('team') ?? 0);
    startGame('spectate', { seed: seedParam });
    if (params.get('speed')) setSpeed(Number(params.get('speed')));
    if (params.get('follow')) followSlot = Number(params.get('follow'));
  } else if (params.get('replay')) {
    fetch(params.get('replay')!).then((r) => r.json()).then((r: Replay) => startGame('replay', { replay: r })).catch((e) => alert('Could not load replay: ' + e));
  }
  menuRefresh();
});
// Debug / automation handle.
(window as any).aimoba = { get game() { return game; }, get drivers() { return drivers; }, renderer, startGame, setSpeed, view, MAP_SIZE, audio };
requestAnimationFrame(frame);

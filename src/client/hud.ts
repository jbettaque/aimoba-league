/** DOM heads-up display: ability bar, stats, shop, scoreboard, minimap, kill feed. */
import {
  BUILDS_INTO, ITEMS, ITEM_BY_ID, MAX_ITEMS, MAX_LEVEL, buildDone, buildTarget, canAfford, isComponent, itemPrice, nextPurchase, recipeUse, xpForLevel,
  type GameEvent, type Unit,
} from '../sim/defs.ts';
import type { Game } from '../sim/game.ts';
import { HEROES } from '../sim/heroes.ts';
import { CAMPS, MAP_SIZE } from '../sim/map.ts';
import { iconUrl } from './icons.ts';
import { TEAM_COLORS, type Renderer } from './render.ts';

const SHOP_FILTERS = [['all', 'Everything'], ['attack', 'Attack'], ['magic', 'Magic'], ['defense', 'Defense'], ['utility', 'Utility']];
/** Which shop filters an item appears under, from the stats it gives. */
function shopCats(id: string): string[] {
  const s = ITEM_BY_ID[id].stats, out: string[] = [];
  if (s.ad || s.as || s.lifesteal || s.armorPen || s.onHit) out.push('attack');
  if (s.ap || s.magicPen || s.spellSlow) out.push('magic');
  if (s.hp || s.armor || s.mr || s.hpRegen) out.push('defense');
  if (s.ms || s.haste || s.mana || s.manaRegen || s.healPower) out.push('utility');
  return out;
}

export interface HudCallbacks {
  levelUp(slot: number): void;
  buy(item: string): void;
  castKey(slot: number): void;
  minimap(x: number, y: number, rightButton: boolean): void;
  follow(slot: number): void;
  speed(mult: number): void;
  vision(team: number): void;
  recall(): void;
  /** Play an interface sound ('open', 'close', 'click', ...). */
  sound(name: string): void;
}

const KEYS = ['Q', 'W', 'E', 'R'];
const $ = <T extends HTMLElement = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
const fmtTime = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

export class Hud {
  root: HTMLElement;
  cb: HudCallbacks;
  private r: Renderer;
  private mm: HTMLCanvasElement;
  private mmBg: HTMLCanvasElement;
  private portraits: string[];
  private cache = new Map<HTMLElement, string>();
  private shopOpen = false;
  private scoreOpen = false;
  private mode: 'play' | 'spectate' | 'replay' = 'play';
  private lastFocus = -2;
  private announceTimer = 0;
  private abilityEls: HTMLElement[] = [];
  private itemEls: HTMLElement[] = [];
  private frameEls: HTMLElement[] = [];
  private tooltip: HTMLElement;
  private shopSig = '';
  private shopSel: string | null = null;
  /** True while the selection should keep following the recommended next purchase. */
  private shopAuto = true;
  private shopFilter = 'all';
  private shopGold = -1;
  private shopCan = false;
  private shopUnit: Unit | null = null;
  private scoreSig = '';

  constructor(root: HTMLElement, renderer: Renderer, cb: HudCallbacks) {
    this.root = root;
    this.cb = cb;
    this.r = renderer;
    this.portraits = HEROES.map((_, i) => renderer.portrait(i, 128));
    root.innerHTML = `
      <div id="topbar">
        <div class="frames" id="frames-blue"></div>
        <div id="score">
          <div class="team blue"><span class="tw" id="tw-blue"></span><span class="kills" id="kills-blue">0</span></div>
          <div id="clock">0:00</div>
          <div class="team red"><span class="kills" id="kills-red">0</span><span class="tw" id="tw-red"></span></div>
        </div>
        <div class="frames" id="frames-red"></div>
      </div>
      <div id="spec-bar" class="panel">
        <span class="label" id="spec-label">Spectating</span>
        <div class="seg" id="speed-seg">${[0, 0.5, 1, 2, 4, 8].map((s) => `<button data-speed="${s}">${s === 0 ? '❚❚' : s + '×'}</button>`).join('')}</div>
        <div class="seg" id="vision-seg"><button data-vision="-1">All vision</button><button data-vision="0">Blue</button><button data-vision="1">Red</button></div>
        <div id="replay-progress"><div></div></div>
      </div>
      <div id="killfeed"></div>
      <div id="announce"></div>
      <div id="death-overlay"><div class="big">You have fallen</div><div class="sub">Back in <b id="respawn-t">0</b>s &nbsp;·&nbsp; you can shop while dead <kbd>P</kbd></div></div>
      <div id="bottom">
        <div id="hero-panel" class="panel">
          <div id="portrait-wrap"><div id="portrait"><img alt=""></div><span id="hero-level">1</span></div>
          <div id="hero-mid">
            <div id="buffs"></div>
            <div id="abilities"></div>
            <div class="bar hp"><div class="fill"></div><div class="shield"></div><span></span></div>
            <div class="bar mana"><div class="fill"></div><span></span></div>
            <div class="bar xp"><div class="fill"></div></div>
          </div>
          <div id="stats"></div>
          <div id="inv">
            <div id="items"></div>
            <div id="gold-row"><button id="shop-btn" title="Shop (P)"><span class="coin"></span><b id="gold">0</b></button><button id="recall-btn" title="Recall to base (B)">Recall <kbd>B</kbd></button></div>
          </div>
        </div>
      </div>
      <div id="minimap-wrap" class="panel"><canvas id="minimap" width="260" height="260"></canvas></div>
      <div id="shop" class="panel hidden"></div>
      <div id="scoreboard" class="panel hidden"></div>
      <div id="tooltip" class="panel hidden"></div>
      <div id="hint"></div>`;
    this.tooltip = $('#tooltip', root);
    this.mm = $<HTMLCanvasElement>('#minimap', root);
    this.mmBg = document.createElement('canvas');
    this.mmBg.width = this.mmBg.height = 260;
    this.mmBg.getContext('2d')!.drawImage(renderer.ground, 0, 0, 260, 260);

    // Ability buttons.
    const ab = $('#abilities', root);
    for (let s = 0; s < 4; s++) {
      const el = document.createElement('div');
      el.className = 'ability';
      el.innerHTML = `<div class="icon"></div><div class="cd"></div><span class="cdt"></span><kbd>${KEYS[s]}</kbd><div class="pips"></div><button class="plus" title="Level up (Ctrl+${KEYS[s]})">+</button>`;
      el.addEventListener('mouseenter', () => this.showAbilityTip(s, el));
      el.addEventListener('mouseleave', () => this.tooltip.classList.add('hidden'));
      $('.plus', el).addEventListener('click', (e) => { e.stopPropagation(); cb.levelUp(s); });
      el.addEventListener('click', () => cb.castKey(s));
      ab.appendChild(el);
      this.abilityEls.push(el);
    }
    const items = $('#items', root);
    for (let i = 0; i < 6; i++) {
      const el = document.createElement('div');
      el.className = 'item';
      el.addEventListener('mouseenter', () => this.showItemTip(el.dataset.item ?? '', el));
      el.addEventListener('mouseleave', () => this.tooltip.classList.add('hidden'));
      items.appendChild(el);
      this.itemEls.push(el);
    }
    // Team frames.
    for (let slot = 0; slot < 10; slot++) {
      const el = document.createElement('div');
      el.className = 'frame';
      el.innerHTML = `<img src="${this.portraits[slot % 5]}" alt=""><div class="fhp"><div></div></div><span class="flvl"></span><span class="fdead"></span>`;
      el.title = `${HEROES[slot % 5].name} ${HEROES[slot % 5].title}`;
      el.addEventListener('click', () => cb.follow(slot));
      $(slot < 5 ? '#frames-blue' : '#frames-red', root).appendChild(el);
      this.frameEls.push(el);
    }
    $('#shop-btn', root).addEventListener('click', () => this.toggleShop());
    const shop = $('#shop', root);
    shop.addEventListener('click', (e) => this.shopClick(e, false));
    shop.addEventListener('dblclick', (e) => this.shopClick(e, true));
    shop.addEventListener('contextmenu', (e) => this.shopClick(e, true));
    $('#recall-btn', root).addEventListener('click', () => cb.recall());
    $('#speed-seg', root).addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) cb.speed(Number(b.dataset.speed));
    });
    $('#vision-seg', root).addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) cb.vision(Number(b.dataset.vision));
    });
    // Minimap interaction.
    let dragging = false;
    const mmEvent = (e: MouseEvent, right: boolean) => {
      const rect = this.mm.getBoundingClientRect();
      cb.minimap(((e.clientX - rect.left) / rect.width) * MAP_SIZE, ((e.clientY - rect.top) / rect.height) * MAP_SIZE, right);
    };
    this.mm.addEventListener('mousedown', (e) => { e.stopPropagation(); if (e.button === 0) { dragging = true; mmEvent(e, false); } else if (e.button === 2) mmEvent(e, true); });
    window.addEventListener('mousemove', (e) => { if (dragging) mmEvent(e, false); });
    window.addEventListener('mouseup', () => (dragging = false));
    this.mm.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  private set(el: HTMLElement, text: string) {
    if (this.cache.get(el) !== text) { this.cache.set(el, text); el.textContent = text; }
  }
  private cls(el: HTMLElement, name: string, on: boolean) {
    if (el.classList.contains(name) !== on) el.classList.toggle(name, on);
  }
  private style(el: HTMLElement, prop: string, val: string) {
    const key = prop + ':' + val;
    const k2 = el as any;
    if (k2['__' + prop] !== key) { k2['__' + prop] = key; el.style.setProperty(prop, val); }
  }

  /** `label`: what to call a replay in the spectator bar (the two teams of a recorded match). */
  setMode(mode: 'play' | 'spectate' | 'replay', label?: string) {
    this.mode = mode;
    this.root.dataset.mode = mode;
    this.shopOpen = false;
    this.scoreOpen = false;
    $('#shop', this.root).classList.add('hidden');
    $('#scoreboard', this.root).classList.add('hidden');
    $('#killfeed', this.root).innerHTML = '';
    $('#announce', this.root).innerHTML = '';
    $('#spec-label', this.root).textContent = mode === 'replay' ? (label ?? 'Replay') : 'Spectating';
    this.lastFocus = -2;
    this.shopSig = this.scoreSig = '';
    this.setHint('');
  }

  setHint(text: string) {
    const el = $('#hint', this.root);
    el.textContent = text;
    el.classList.toggle('show', text !== '');
  }

  markSpeed(mult: number) {
    for (const b of this.root.querySelectorAll<HTMLElement>('#speed-seg button')) b.classList.toggle('on', Number(b.dataset.speed) === mult);
  }
  markVision(team: number) {
    for (const b of this.root.querySelectorAll<HTMLElement>('#vision-seg button')) b.classList.toggle('on', Number(b.dataset.vision) === team);
  }

  // --- Tooltips -------------------------------------------------------------------
  private tipFocus: Unit | null = null;
  private showTip(html: string, anchor: HTMLElement) {
    const t = this.tooltip;
    t.innerHTML = html;
    t.classList.remove('hidden');
    const r = anchor.getBoundingClientRect();
    const tw = t.offsetWidth, th = t.offsetHeight;
    t.style.left = Math.max(8, Math.min(window.innerWidth - tw - 8, r.left + r.width / 2 - tw / 2)) + 'px';
    t.style.top = Math.max(8, r.top - th - 12) + 'px';
  }
  private showAbilityTip(slot: number, el: HTMLElement) {
    const u = this.tipFocus;
    if (!u || !u.hero) return;
    const ab = HEROES[u.hero.def].abilities[slot];
    const rank = Math.max(1, u.hero.ranks[slot]);
    const cd = (ab.cd[Math.min(ab.cd.length, rank) - 1] * 100) / (100 + u.haste);
    this.showTip(`<div class="tt-title">${ab.name} <kbd>${KEYS[slot]}</kbd></div>
      <div class="tt-meta"><span class="mana">${ab.cost[Math.min(ab.cost.length, rank) - 1]} mana</span><span>${cd.toFixed(1)}s cooldown</span>${ab.range > 0 ? `<span>${ab.range} range</span>` : ''}</div>
      <div class="tt-desc">${ab.desc}</div>
      <div class="tt-foot">Rank ${u.hero.ranks[slot]} / ${ab.cd.length}${u.hero.ranks[slot] === 0 ? ' — not learned yet' : ''}</div>`, el);
  }
  private showItemTip(id: string, el: HTMLElement) {
    const it = ITEM_BY_ID[id];
    if (!it) return;
    const names = (ids: string[]) => ids.map((i) => ITEM_BY_ID[i].name).join(', ');
    const foot = it.from ? `Built from ${it.from.map((i) => ITEM_BY_ID[i].name).join(' + ')}` : isComponent(id) ? `Builds into ${names(BUILDS_INTO[id])}` : '';
    this.showTip(`<div class="tt-title">${it.name}</div><div class="tt-meta"><span class="gold">${it.cost} gold</span></div><div class="tt-desc">${it.desc}${it.passive ? `<br><b>${it.passive[0]}.</b> ${it.passive[1]}` : ''}</div>${foot ? `<div class="tt-foot">${foot}</div>` : ''}`, el);
  }

  // --- Shop / scoreboard -------------------------------------------------------------
  toggleShop(force?: boolean) {
    if (this.mode !== 'play') return;
    const open = force ?? !this.shopOpen;
    if (open !== this.shopOpen) this.cb.sound(open ? 'open' : 'close');
    this.shopOpen = open;
    $('#shop', this.root).classList.toggle('hidden', !this.shopOpen);
    this.shopSig = '';
    if (this.shopOpen) { this.shopAuto = true; this.shopSel = null; }
  }
  isShopOpen() { return this.shopOpen; }
  showScoreboard(on: boolean) {
    if (this.scoreOpen === on) return;
    this.scoreOpen = on;
    $('#scoreboard', this.root).classList.toggle('hidden', !on);
    this.scoreSig = '';
  }

  /** Handle a click inside the shop (delegated: the shop's DOM is rebuilt as the inventory changes). */
  private shopClick(e: MouseEvent, buyTile: boolean) {
    const t = e.target as HTMLElement;
    const tile = t.closest<HTMLElement>('[data-item]');
    if (buyTile) {
      if (tile) { e.preventDefault(); this.cb.buy(tile.dataset.item!); }
      return;
    }
    const buy = t.closest<HTMLElement>('[data-buy]');
    if (buy) { if (buy.dataset.buy) this.cb.buy(buy.dataset.buy); return; }
    if (t.closest('.shop-close')) { this.toggleShop(false); return; }
    const filter = t.closest<HTMLElement>('[data-filter]');
    if (filter) { this.shopFilter = filter.dataset.filter!; this.shopSig = ''; this.cb.sound('click'); return; }
    if (tile && this.shopUnit) {
      // Selecting only swaps the highlight and the detail pane, so a double-click still lands on the same tile.
      if (this.shopSel !== tile.dataset.item) this.cb.sound('click');
      this.shopSel = tile.dataset.item!;
      this.shopAuto = false;
      const root = $('#shop', this.root);
      for (const el of root.querySelectorAll<HTMLElement>('.tile, .step')) el.classList.toggle('sel', el.dataset.item === this.shopSel);
      this.renderShopDetail(this.shopUnit);
    }
  }

  private renderShop(g: Game, u: Unit) {
    const h = u.hero!;
    const canShop = !u.alive || g.inFountain(u);
    const hero = HEROES[h.def], build = h.build;
    const target = buildTarget(h.items, build);
    const step = canShop ? nextPurchase(h.items, h.gold, build) : null;
    this.shopUnit = u;
    this.shopCan = canShop;
    // Until the player picks something themselves, the selection follows the recommended next purchase.
    if (this.shopAuto || !this.shopSel) this.shopSel = step ?? target ?? build[build.length - 1];
    const afford = ITEMS.filter((i) => canAfford(h.items, h.gold, i.id)).map((i) => i.id).join(',');
    const sig = `${h.items.join(',')}|${canShop}|${this.shopSel}|${this.shopFilter}|${afford}|${step}`;
    if (sig !== this.shopSig) {
      this.shopSig = sig;
      const price = (id: string) => itemPrice(h.items, id);
      const tile = (id: string) => {
        const it = ITEM_BY_ID[id];
        const comp = isComponent(id);
        const count = h.items.filter((x) => x === id).length;
        const owned = !comp && count > 0;
        const ok = !owned && canShop && canAfford(h.items, h.gold, id);
        const p = price(id);
        return `<button class="tile ${owned ? 'owned' : ''} ${ok ? 'ok' : ''} ${id === this.shopSel ? 'sel' : ''} ${build.includes(id) ? 'rec' : ''}" data-item="${id}" style="--ic:${it.color}">
          <span class="tile-art"><img src="${iconUrl(it.glyph, it.color)}" alt=""></span>
          <span class="tile-name">${it.name}</span>
          ${owned ? '<span class="tile-price owned">Owned</span>' : `<span class="tile-price">${p < it.cost ? `<s>${it.cost}</s> ` : ''}${p}</span>`}
          ${comp && count ? `<span class="tile-count">${count} owned</span>` : ''}
        </button>`;
      };
      const steps = build.map((id, i) => {
        const it = ITEM_BY_ID[id];
        const done = buildDone(h.items, id);
        return `<button class="step ${done ? 'done' : ''} ${id === target ? 'target' : ''} ${id === this.shopSel ? 'sel' : ''}" data-item="${id}" style="--ic:${it.color}" title="${it.name}">
          <span class="step-art"><img src="${iconUrl(it.glyph, it.color)}" alt=""></span>
          <span class="step-n">${done ? '✓' : i + 1}</span>
          ${id === target ? `<i class="step-bar"><i data-live="bar" data-price="${price(id)}"></i></i>` : ''}
        </button>`;
      }).join('<i class="step-link"></i>');
      let next: string;
      if (!target) next = '<p class="shop-next-note">Build complete. Anything else is up to you.</p>';
      else if (!canShop) next = `<p class="shop-next-note">Next up: <b>${ITEM_BY_ID[target].name}</b></p>`;
      else if (step) next = `<button class="shop-buy-next" data-buy="${step}"><span>Buy ${ITEM_BY_ID[step].name}</span><b><span class="coin"></span>${price(step)}</b></button>${step !== target ? `<p class="shop-next-note">a part of ${ITEM_BY_ID[target].name}</p>` : ''}`;
      else next = `<p class="shop-next-note"><b>${ITEM_BY_ID[target].name}</b> needs <span data-live="need" data-price="${price(target)}"></span> more gold</p>`;
      const show = (i: { id: string }) => this.shopFilter === 'all' || shopCats(i.id).includes(this.shopFilter);
      const finished = ITEMS.filter((i) => !isComponent(i.id) && show(i)), parts = ITEMS.filter((i) => isComponent(i.id) && show(i));
      const el = $('#shop', this.root);
      const scroll = el.querySelector('.shop-catalog')?.scrollTop ?? 0;
      el.innerHTML = `
        <header class="shop-head">
          <h2>Armory</h2>
          <span class="shop-slots">${h.items.length} of ${MAX_ITEMS} item slots used</span>
          <span class="shop-gold"><span class="coin"></span><b data-live="gold"></b></span>
          <button class="shop-close" title="Close the shop">Close <kbd>P</kbd></button>
        </header>
        ${canShop ? '' : '<p class="shop-away">You can look around, but buying only works at your fountain or while you are dead. Press <kbd>B</kbd> to recall.</p>'}
        <section class="shop-path">
          <div class="shop-path-text"><h3>${hero.name}'s build</h3><p>Seven items, bought in this order.</p></div>
          <div class="steps">${steps}</div>
          <div class="shop-next">${next}</div>
        </section>
        <div class="shop-body">
          <section class="shop-catalog">
            <nav class="shop-filters">${SHOP_FILTERS.map(([id, label]) => `<button data-filter="${id}" class="${id === this.shopFilter ? 'on' : ''}">${label}</button>`).join('')}</nav>
            ${finished.length ? `<h3>Finished items</h3><div class="tiles">${finished.map((i) => tile(i.id)).join('')}</div>` : ''}
            ${parts.length ? `<h3>Parts</h3><div class="tiles">${parts.map((i) => tile(i.id)).join('')}</div>` : ''}
            <p class="shop-tip">Double-click or right-click an item to buy it straight away.</p>
          </section>
          <aside class="shop-detail"></aside>
        </div>`;
      const cat = el.querySelector('.shop-catalog');
      if (cat) cat.scrollTop = scroll;
      this.renderShopDetail(u);
      this.shopGold = -1;
    }
    // Cheap per-frame updates: gold, "needs N more" and progress bars.
    const gold = Math.floor(h.gold);
    if (gold !== this.shopGold) {
      this.shopGold = gold;
      for (const el of $('#shop', this.root).querySelectorAll<HTMLElement>('[data-live]')) {
        const p = Number(el.dataset.price);
        if (el.dataset.live === 'gold') el.textContent = String(gold);
        else if (el.dataset.live === 'need') el.textContent = String(Math.max(0, p - gold));
        else el.style.width = `${Math.min(100, (gold / Math.max(1, p)) * 100)}%`;
      }
    }
  }

  /** The right-hand pane: everything about the selected item, and the button that buys it. */
  private renderShopDetail(u: Unit) {
    const h = u.hero!, id = this.shopSel!, it = ITEM_BY_ID[id];
    const comp = isComponent(id);
    const owned = !comp && h.items.includes(id);
    const p = itemPrice(h.items, id);
    const used = recipeUse(h.items, id);
    const full = h.items.length - used.length + 1 > MAX_ITEMS;
    const chip = (c: string, have: boolean) => `<button class="chip ${have ? 'have' : ''}" data-item="${c}" style="--ic:${ITEM_BY_ID[c].color}"><img src="${iconUrl(ITEM_BY_ID[c].glyph, ITEM_BY_ID[c].color)}" alt=""><span>${ITEM_BY_ID[c].name}</span><b>${have ? 'Owned' : ITEM_BY_ID[c].cost}</b></button>`;
    let recipe = '';
    if (it.from) {
      const have = used.map((i) => h.items[i]);
      const chips = it.from.map((c) => { const k = have.indexOf(c); if (k >= 0) have.splice(k, 1); return chip(c, k >= 0); });
      const fee = it.cost - it.from.reduce((t, c) => t + ITEM_BY_ID[c].cost, 0);
      recipe = `<h4>Made from</h4><div class="chips">${chips.join('')}</div><p class="detail-fee">plus ${fee} gold to combine. Parts you own come off the price.</p>`;
    } else if (comp) {
      recipe = `<h4>Builds into</h4><div class="chips">${BUILDS_INTO[id].map((f) => `<button class="chip" data-item="${f}" style="--ic:${ITEM_BY_ID[f].color}"><img src="${iconUrl(ITEM_BY_ID[f].glyph, ITEM_BY_ID[f].color)}" alt=""><span>${ITEM_BY_ID[f].name}</span></button>`).join('')}</div>`;
    }
    let action: string;
    if (owned) action = '<button class="detail-buy" disabled>You own this</button>';
    else if (!this.shopCan) action = '<button class="detail-buy" disabled>Buy at the fountain</button>';
    else if (full) action = '<button class="detail-buy" disabled>No free item slot</button>';
    else if (h.gold < p) action = `<button class="detail-buy" disabled>Needs <span data-live="need" data-price="${p}"></span> more gold</button><i class="detail-bar"><i data-live="bar" data-price="${p}"></i></i>`;
    else action = `<button class="detail-buy" data-buy="${id}"><span>Buy</span><b><span class="coin"></span>${p}</b></button>`;
    $('.shop-detail', this.root).innerHTML = `
      <div class="detail-top" style="--ic:${it.color}">
        <span class="detail-art"><img src="${iconUrl(it.glyph, it.color)}" alt=""></span>
        <div><h3>${it.name}</h3><p class="detail-price"><span class="coin"></span>${p < it.cost ? `<s>${it.cost}</s> ` : ''}${p}${h.build.includes(id) ? ` <em>in ${HEROES[h.def].name}'s build</em>` : ''}</p></div>
      </div>
      <ul class="detail-stats">${it.desc.split(', ').map((d) => `<li>${d}</li>`).join('')}</ul>
      ${it.passive ? `<p class="detail-passive"><b>${it.passive[0]}</b>${it.passive[1]}</p>` : ''}
      ${recipe}
      <div class="detail-action">${action}</div>`;
    this.shopGold = -1;
  }

  private renderScoreboard(g: Game) {
    const sig = g.heroes.map((u) => { const h = u.hero!; return `${h.level},${h.kills},${h.deaths},${h.assists},${h.cs},${Math.floor(h.totalGold / 50)},${h.items.length}`; }).join('|');
    if (sig === this.scoreSig) return;
    this.scoreSig = sig;
    $('#scoreboard', this.root).innerHTML = scoreboardHtml(g, this.portraits);
  }

  // --- Events ---------------------------------------------------------------------------
  onEvent(ev: GameEvent, g: Game) {
    if (ev.type === 'kill') {
      const victim = g.unit(ev.id!);
      const killer = ev.other !== undefined && ev.other >= 0 ? g.unit(ev.other) : undefined;
      if (!victim || !victim.hero) return;
      const feed = $('#killfeed', this.root);
      const row = document.createElement('div');
      const kTeam = killer ? killer.team : 1 - victim.team;
      row.className = 'kf ' + (kTeam === 0 ? 'blue' : 'red');
      const kImg = killer && killer.hero ? `<img src="${this.portraits[killer.hero.def]}" alt=""><b>${HEROES[killer.hero.def].name}</b>` : `<b>${killer && killer.kind === 'tower' ? 'Tower' : killer ? 'Minions' : 'Executed'}</b>`;
      row.innerHTML = `${kImg}<span class="sword">⚔</span><b>${HEROES[victim.hero.def].name}</b><img src="${this.portraits[victim.hero.def]}" alt="">`;
      feed.appendChild(row);
      while (feed.children.length > 5) feed.firstChild!.remove();
      setTimeout(() => row.classList.add('fade'), 6000);
      setTimeout(() => row.remove(), 7000);
    } else if (ev.type === 'announce') {
      this.announce(ev.text ?? '', ev.team === 0 ? 'blue' : ev.team === 1 ? 'red' : 'neutral');
    }
  }

  announce(text: string, cls = 'neutral') {
    const el = $('#announce', this.root);
    el.innerHTML = `<div class="ann ${cls}">${text}</div>`;
    clearTimeout(this.announceTimer);
    this.announceTimer = window.setTimeout(() => (el.innerHTML = ''), 3600);
  }

  // --- Per-frame update -----------------------------------------------------------------
  update(g: Game, focus: Unit | undefined, viewTeam: number, replayFrac: number) {
    const root = this.root;
    this.set($('#clock', root), fmtTime(g.time));
    this.set($('#kills-blue', root), String(g.teamKills[0]));
    this.set($('#kills-red', root), String(g.teamKills[1]));
    this.set($('#tw-blue', root), `♜ ${g.towersLost[1]}`);
    this.set($('#tw-red', root), `${g.towersLost[0]} ♜`);
    if (this.mode === 'replay') this.style($('#replay-progress > div', root), 'width', (replayFrac * 100).toFixed(1) + '%');

    for (let slot = 0; slot < 10; slot++) {
      const u = g.heroes[slot], h = u.hero!, el = this.frameEls[slot];
      this.cls(el, 'dead', !u.alive);
      this.cls(el, 'sel', focus === u);
      this.style(el.querySelector('.fhp > div') as HTMLElement, 'width', (u.alive ? (u.hp / u.maxHp) * 100 : 0).toFixed(0) + '%');
      this.set(el.querySelector('.flvl') as HTMLElement, String(h.level));
      this.set(el.querySelector('.fdead') as HTMLElement, u.alive ? '' : String(Math.ceil(h.respawn)));
    }

    this.cls(root, 'no-focus', !focus);
    this.tipFocus = focus ?? null;
    if (focus && focus.hero) {
      const u = focus, h = u.hero!, def = HEROES[h.def];
      if (this.lastFocus !== u.id) {
        this.lastFocus = u.id;
        ($('#portrait img', root) as HTMLImageElement).src = this.portraits[h.def];
        this.abilityEls.forEach((el, s) => {
          ($('.icon', el) as HTMLElement).style.backgroundImage = `url(${iconUrl(def.abilities[s].icon, def.color)})`;
          $('.pips', el).innerHTML = '<i></i>'.repeat(def.abilities[s].cd.length);
        });
        this.shopSig = '';
      }
      this.set($('#hero-level', root), String(h.level));
      const hpBar = $('.bar.hp', root), manaBar = $('.bar.mana', root);
      const total = Math.max(u.maxHp, u.hp + u.shield);
      this.style($('.fill', hpBar), 'width', ((u.hp / total) * 100).toFixed(1) + '%');
      this.style($('.shield', hpBar), 'width', ((u.shield / total) * 100).toFixed(1) + '%');
      this.style($('.shield', hpBar), 'left', ((u.hp / total) * 100).toFixed(1) + '%');
      this.set($('span', hpBar), `${Math.ceil(u.hp)} / ${Math.round(u.maxHp)}`);
      this.style($('.fill', manaBar), 'width', ((u.maxMana ? u.mana / u.maxMana : 0) * 100).toFixed(1) + '%');
      this.set($('span', manaBar), `${Math.floor(u.mana)} / ${Math.round(u.maxMana)}`);
      this.style($('.bar.xp .fill', root), 'width', (h.level >= MAX_LEVEL ? 100 : (h.xp / xpForLevel(h.level)) * 100).toFixed(1) + '%');
      this.set($('#gold', root), String(Math.floor(h.gold)));
      const playable = this.mode === 'play';
      this.abilityEls.forEach((el, s) => {
        const ab = def.abilities[s], rank = h.ranks[s];
        const cd = h.cds[s];
        const maxCd = rank > 0 ? (ab.cd[rank - 1] * 100) / (100 + u.haste) : 1;
        this.cls(el, 'unlearned', rank === 0);
        this.cls(el, 'oom', rank > 0 && cd <= 0 && u.mana < ab.cost[rank - 1]);
        this.cls(el, 'ready', rank > 0 && cd <= 0 && u.alive && u.mana >= ab.cost[rank - 1]);
        this.cls(el, 'can-level', playable && g.canLevel(u, s));
        this.style($('.cd', el), '--p', cd > 0 ? Math.min(1, cd / maxCd).toFixed(3) : '0');
        this.set($('.cdt', el), cd > 0 ? (cd < 1 ? cd.toFixed(1) : String(Math.ceil(cd))) : '');
        const pips = $('.pips', el).children;
        for (let i = 0; i < pips.length; i++) this.cls(pips[i] as HTMLElement, 'on', i < rank);
      });
      const st = $('#stats', root);
      const stats = `<div title="Attack damage"><i class="s-ad"></i>${Math.round(u.ad)}</div><div title="Ability power"><i class="s-ap"></i>${Math.round(u.ap)}</div>
        <div title="Armor"><i class="s-ar"></i>${Math.round(u.armor)}</div><div title="Magic resist"><i class="s-mr"></i>${Math.round(u.mr)}</div>
        <div title="Attacks per second"><i class="s-as"></i>${u.as.toFixed(2)}</div><div title="Move speed"><i class="s-ms"></i>${Math.round(u.ms)}</div>
        <div class="kda" title="Kills / Deaths / Assists">${h.kills} / ${h.deaths} / ${h.assists}</div><div class="kda" title="Creep score">${h.cs} CS</div>`;
      if (this.cache.get(st) !== stats) { this.cache.set(st, stats); st.innerHTML = stats; }
      this.itemEls.forEach((el, i) => {
        const id = h.items[i] ?? '';
        if (el.dataset.item !== id) {
          el.dataset.item = id;
          el.style.backgroundImage = id ? `url(${iconUrl(ITEM_BY_ID[id].glyph, ITEM_BY_ID[id].color)})` : '';
        }
      });
      // Buff chips.
      const names: Record<string, [string, string]> = {
        might: ['Might', '#ff6b57'], insight: ['Insight', '#5aa9ff'], titan: ["Titan's Wrath", '#c084fc'], stun: ['Stunned', '#ffe66d'], root: ['Rooted', '#ffb347'],
        slow: ['Slowed', '#74c0fc'], stealth: ['Stealth', '#b197fc'], shield: ['Shield', '#e8f3ff'], as: ['Frenzy', '#ffd166'], haste: ['Haste', '#96f2d7'],
        empower: ['Empowered', '#ffa94d'], resist: ['Fortified', '#ced4da'], dot: ['Burning', '#ff7a2e'],
      };
      let bh = '';
      const seen = new Set<string>();
      for (const b of u.buffs) {
        const n = names[b.kind];
        if (!n || seen.has(b.kind)) continue;
        seen.add(b.kind);
        bh += `<span style="--c:${n[1]}">${n[0]}${b.t > 3 ? ` <small>${Math.ceil(b.t)}s</small>` : ''}</span>`;
      }
      if (g.wyrmStacks[u.team] > 0) bh += `<span style="--c:#35c9ae">Wyrm ×${g.wyrmStacks[u.team]}</span>`;
      const be = $('#buffs', root);
      if (this.cache.get(be) !== bh) { this.cache.set(be, bh); be.innerHTML = bh; }
      const dead = playable && !u.alive;
      this.cls($('#death-overlay', root), 'show', dead);
      if (dead) this.set($('#respawn-t', root), String(Math.ceil(h.respawn)));
      this.cls($('#shop-btn', root), 'pulse', playable && (g.inFountain(u) || !u.alive) && nextPurchase(h.items, h.gold, h.build) !== null);
      if (this.shopOpen) this.renderShop(g, u);
    } else {
      this.cls($('#death-overlay', root), 'show', false);
    }
    if (this.scoreOpen) this.renderScoreboard(g);
    this.drawMinimap(g, focus, viewTeam);
  }

  private drawMinimap(g: Game, focus: Unit | undefined, viewTeam: number) {
    const c = this.mm.getContext('2d')!;
    const S = this.mm.width, k = S / MAP_SIZE;
    c.drawImage(this.mmBg, 0, 0);
    c.fillStyle = 'rgba(0,0,0,0.25)';
    c.fillRect(0, 0, S, S);
    const sees = (u: Unit) => viewTeam < 0 || g.canSee(viewTeam, u);
    for (let i = 0; i < CAMPS.length; i++) {
      if (g.campAlive[i] <= 0) continue;
      const cp = CAMPS[i];
      const big = cp.side === 2;
      c.fillStyle = cp.type === 'might' ? '#ff6b57' : cp.type === 'insight' ? '#5aa9ff' : cp.type === 'wyrm' ? '#35c9ae' : cp.type === 'titan' ? '#c084fc' : '#d8c27a';
      c.beginPath(); c.arc(cp.x * k, cp.y * k, big ? 5 : 3, 0, 6.283); c.fill();
    }
    for (const u of g.units) {
      if (!u.alive || u.kind !== 'minion' || !sees(u)) continue;
      c.fillStyle = TEAM_COLORS[u.team];
      c.fillRect(u.x * k - 1.5, u.y * k - 1.5, 3, 3);
    }
    for (const s of g.structures) {
      if (!s.alive) continue;
      const x = s.x * k, y = s.y * k, r = s.kind === 'core' ? 8 : 5;
      c.fillStyle = TEAM_COLORS[s.team];
      c.strokeStyle = '#0a2124';
      c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(x, y - r); c.lineTo(x + r, y); c.lineTo(x, y + r); c.lineTo(x - r, y); c.closePath(); c.fill(); c.stroke();
      if (g.tick - s.lastDamageTick < 45 && Math.floor(g.time * 4) % 2 === 0) { c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.arc(x, y, r + 5, 0, 6.283); c.stroke(); }
    }
    for (const u of g.heroes) {
      if (!u.alive || !sees(u)) continue;
      const x = u.x * k, y = u.y * k;
      c.fillStyle = HEROES[u.hero!.def].color;
      c.strokeStyle = u === focus ? '#8dffa0' : TEAM_COLORS[u.team];
      c.lineWidth = 2.5;
      c.beginPath(); c.arc(x, y, 6.5, 0, 6.283); c.fill(); c.stroke();
      c.fillStyle = '#0a2124';
      c.font = "800 9px 'Alegreya Sans', system-ui";
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(HEROES[u.hero!.def].name[0], x, y + 0.5);
    }
    // The camera's footprint on the ground (a trapezoid under the perspective camera).
    c.strokeStyle = 'rgba(244,236,214,0.9)';
    c.lineWidth = 1.5;
    c.beginPath();
    for (const [vx, vy] of this.r.viewCorners()) c.lineTo(vx * k, vy * k);
    c.closePath();
    c.stroke();
  }
}

export function scoreboardHtml(g: Game, portraits: string[]): string {
  const row = (u: Unit) => {
    const h = u.hero!;
    const def = HEROES[h.def];
    const items = [0, 1, 2, 3, 4, 5].map((i) => (h.items[i] ? `<img src="${iconUrl(ITEM_BY_ID[h.items[i]].glyph, ITEM_BY_ID[h.items[i]].color)}" title="${ITEM_BY_ID[h.items[i]].name}" alt="">` : '<i></i>')).join('');
    return `<tr class="${u.alive ? '' : 'dead'}"><td class="who"><img src="${portraits[h.def]}" alt=""><div><b>${def.name}${h.isBot ? '' : ` <span class="you">${h.autoAttack ? 'YOU' : 'AGENT'}</span>`}</b><small>${def.role}</small></div></td>
      <td>${h.level}</td><td class="kda">${h.kills} / ${h.deaths} / ${h.assists}</td><td>${h.cs}</td><td class="g">${(h.totalGold / 1000).toFixed(1)}k</td><td>${(h.dmgDealt / 1000).toFixed(1)}k</td><td class="its">${items}</td></tr>`;
  };
  const table = (team: number) => `<table class="sb ${team === 0 ? 'blue' : 'red'}"><thead><tr><th>${team === 0 ? 'Blue team' : 'Red team'} · ${g.teamKills[team]} kills</th><th>Lv</th><th>K / D / A</th><th>CS</th><th>Gold</th><th>Dmg</th><th>Items</th></tr></thead>
    <tbody>${g.heroes.filter((u) => u.team === team).map(row).join('')}</tbody></table>`;
  return table(0) + table(1);
}

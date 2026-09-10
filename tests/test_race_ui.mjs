// 一覧・詳細に足した6つの表示を検証する（2026-09-10）。
//   #1 結果オーバーレイ（着順・配当）  #2 上位3頭ストリップ  #3 能力差/平均
//   #4 オッズの土台バッジ              #5 脚質カウント        #6 表示列チューザ
//
// 🔴 固定したいこと:
//   ・配当が取れていない行を「0円」と書かない（欠損は「-」）
//   ・隊列の**位置**を出さない（2026-08-17/08-26 に天井を測って打ち切った領域）
//   ・列を隠しても「印/番/馬名/RL/AI勝率/AI複勝」は残る
//
// 実行: node tests/test_race_ui.mjs
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

const store = {};
const ctx = vm.createContext({ console, Math, JSON, document: undefined, window: {},
  localStorage: { getItem: k => (k in store ? store[k] : null),
                  setItem: (k, v) => { store[k] = String(v); } },
  fetch: () => Promise.reject(new Error('no net')),
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0, addEventListener: () => {} });
try { vm.runInContext(js, ctx); } catch (e) { /* DOM依存の初期化は無視 */ }

const { _finBadge, _payoutText, _styleCounts, _abilitySpread, _anchorChip,
        _raceSummaryBar, _top3Strip, _colBar, _tblClass, _resultsFor } = ctx;
assert.ok(_top3Strip, 'ヘルパーが読めていない');
// _RESULTS は var なのでコンテキストのプロパティ。ここから差し替えられる
const __setResults = o => { ctx._RESULTS = o; };

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

const race = {
  race_id: '20260906_06_10', r: 10, name: 'テストS',
  horses: [
    { n: 1, name: 'アルファ', rl_rank: 1, odds: 3.2, score: 8.4, style: '逃げ' },
    { n: 2, name: 'ブラボー', rl_rank: 2, odds: 5.1, score: 7.1, style: '先行' },
    { n: 3, name: 'チャーリー', rl_rank: 3, odds: 9.9, score: 6.0, style: '差し' },
    { n: 4, name: 'デルタ',   rl_rank: 4, odds: 22.0, score: 4.2, style: '差し' },
    { n: 5, name: 'エコー',   rl_rank: 5, odds: 60.0, score: 2.9, style: '追込' },
  ],
};
const results = { races: { '20260906_06_10': {
  '1': { place: 1, tan: 320, fuku: 150 },
  '3': { place: 2, tan: null, fuku: 240 },
  '4': { place: 3, tan: null, fuku: null },   // 🔴 3着なのに配当が欠損している行
  '2': { place: 7, tan: null, fuku: null },
} } };

console.log('■ #1 結果オーバーレイ');
t('着順メダルが出る', () => {
  assert.ok(_finBadge(1).includes('🥇'));
  assert.ok(_finBadge(2).includes('🥈'));
  assert.ok(_finBadge(3).includes('🥉'));
  assert.ok(_finBadge(7).includes('7着'));
  assert.strictEqual(_finBadge(null), '');
});

t('🔴 配当の欠損を「0円」と書かない', () => {
  assert.ok(_payoutText({ place: 1, tan: 320, fuku: 150 }).includes('単320'));
  assert.ok(_payoutText({ place: 2, tan: null, fuku: 240 }).includes('複240'));
  const miss = _payoutText({ place: 3, tan: null, fuku: null });
  assert.ok(miss.includes('配当-'), '欠損が「配当-」になっていない');
  assert.ok(!/0/.test(miss.replace(/[^0-9]/g, '')), '0 が配当として出ている');
  assert.strictEqual(_payoutText({ place: 7, tan: null, fuku: null }), '');
});

t('レース前は結果を持たない（列ごと出ない）', () => {
  __setResults(null);
  assert.strictEqual(_resultsFor(race), null);
});

t('結果が来たら race_id で引ける', () => {
  __setResults(results);
  assert.strictEqual(_resultsFor(race)['1'].place, 1);
  assert.strictEqual(_resultsFor({ race_id: 'nope' }), null);
});

console.log('■ #2 上位3頭ストリップ');
t('RL1-3 だけを順に出す', () => {
  __setResults(null);
  const h = _top3Strip(race, 'k');
  assert.ok(h.includes('アルファ') && h.includes('ブラボー') && h.includes('チャーリー'));
  assert.ok(!h.includes('デルタ'), 'RL4以下が混ざっている');
  assert.ok(h.indexOf('アルファ') < h.indexOf('ブラボー'));
});

t('結果があれば着順・配当が重なる', () => {
  __setResults(results);
  const h = _top3Strip(race, 'k');
  assert.ok(h.includes('🥇') && h.includes('単320'));
  assert.ok(h.includes('🥈') && h.includes('複240'));
});

console.log('■ #3 能力差 / 平均');
t('最大−最小と平均を返す', () => {
  const a = _abilitySpread(race);
  assert.ok(Math.abs(a.spread - (8.4 - 2.9)) < 1e-9);
  assert.ok(Math.abs(a.avg - (8.4 + 7.1 + 6.0 + 4.2 + 2.9) / 5) < 1e-9);
});

t('score が無ければ null（作り話をしない）', () => {
  assert.strictEqual(_abilitySpread({ horses: [{ n: 1 }, { n: 2 }] }), null);
});

console.log('■ #4 オッズの土台バッジ');
t('直前オッズ反映後は緑', () => {
  const r = { ...race, _update_time: '14:52', _anchor: 'odds' };
  assert.ok(_anchorChip(r).includes('直前オッズ反映'));
});

t('生成時のままなら「生成時オッズ」', () => {
  assert.ok(_anchorChip(race).includes('生成時オッズ'));
});

t('🔴 オッズが1件も無ければ「オッズ未取得」', () => {
  const r = { horses: race.horses.map(h => ({ ...h, odds: 0 })) };
  assert.ok(_anchorChip(r).includes('オッズ未取得'));
});

console.log('■ #5 脚質カウント（位置は出さない）');
t('過去走の多数決を数える', () => {
  const c = _styleCounts(race);   // vm realm のオブジェクトなので deepStrictEqual は使えない
  assert.strictEqual(c['逃げ'], 1);
  assert.strictEqual(c['先行'], 1);
  assert.strictEqual(c['差し'], 2);
  assert.strictEqual(c['追込'], 1);
});

t('🔴 隊列の「位置」を描いていない（2026-08-17/08-26 で打ち切った領域）', () => {
  assert.ok(!/隊列マップ/.test(html), '隊列マップを実装している');
  const bar = _raceSummaryBar(race);
  assert.ok(bar.includes('逃1 先1 差2 追1'), 'カウントが出ていない');
  assert.ok(!/先頭|後方/.test(bar), '位置を描いている');
});

console.log('■ #6 表示列チューザ');
t('既定は全列オン', () => {
  delete store['keiba_hidden_cols'];
  assert.strictEqual(_tblClass().trim(), 'htbl');
  assert.strictEqual((_colBar().match(/class="on"/g) || []).length, 5);
});

t('隠した列がクラスになる', () => {
  store['keiba_hidden_cols'] = JSON.stringify(['pop', 'val']);
  const c = _tblClass();
  assert.ok(c.includes('h-pop') && c.includes('h-val'));
  assert.ok(!c.includes('h-odds'));
});

t('CSS が th/td の両方を隠す', () => {
  const css = html.replace(/\s+/g, ' ');
  for (const k of ['pop', 'odds', 'mk', 'solo', 'val']) {
    assert.ok(css.includes('.htbl.h-' + k + ' th.col-' + k), 'th 側が無い: ' + k);
    assert.ok(css.includes('.htbl.h-' + k + ' td.col-' + k), 'td 側が無い: ' + k);
  }
});

t('AI単を隠してもRL列は残る（別クラスになっていること）', () => {
  assert.ok(html.includes('class="col-solo"'), 'AI単が col-rl のままだとRLごと消える');
});

t('🔴 印・番・馬名・AI勝率・AI複勝は隠せない（常時表示）', () => {
  const keys = COLSKEYS(html);
  for (const bad of ['name', 'um', 'rl']) assert.ok(!keys.includes(bad), bad + ' が隠せてしまう');
});
function COLSKEYS(h) {
  const m = h.match(/const COLS = \[(.*?)\];/s);
  return [...m[1].matchAll(/\['(\w+)'/g)].map(x => x[1]);
}

console.log(`\n✅ ${pass} テスト通過`);

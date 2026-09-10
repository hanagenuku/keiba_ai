// 画面の「AI勝率 / AI複勝 / 必要オッズ」がEV用の較正済み確率を出すことを検証する。
//
// 経緯（2026-09-11）:
//   2026-09-10 に AI勝率を softmax(T=3.5) から作って画面に出したが、
//   OOS 24,993頭で測ると ECE 0.0757 で壊れていた（cal_prob は 0.0094）。
//   ability経路に Isotonic を通した版は ECE 0.0080 だったので、そちらに差し替える。
//
// 🔴 固定したいこと（退行すると数字が静かに嘘をつく）:
//   ① 表示は ev_tan_pct / ev_fuku_pct であって tan_pct / fuku_pct ではない
//   ② 較正器の値が無ければ「-」。**softmax で代用しない**（クライアント計算を持たない）
//   ③ EV列は「必要オッズ = 1.2 / EV用勝率」であって EV ではない
//   ④ 必要オッズに届いても**色を付けない**（買い推奨として見せない）
//   ⑤ 順位用の量は sim_ev という別名で、RL順位・買い目はそちらのまま
//   ⑥ 直前オッズを押しても EV用確率は動かない
//
// 実行: node tests/test_ai_win_rate.mjs
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

const ctx = vm.createContext({ console, Math, JSON, document: undefined, window: {},
                               localStorage: { getItem: () => null, setItem: () => {} },
                               fetch: () => Promise.reject(new Error('no net')),
                               setInterval: () => 0, clearInterval: () => {},
                               setTimeout: () => 0, addEventListener: () => {} });
try { vm.runInContext(js, ctx); } catch (e) { /* DOM依存の初期化は無視 */ }

const { updateOddsAndEV } = ctx;
assert.ok(updateOddsAndEV, 'updateOddsAndEV が読めていない');

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

const AM = [1.2, -0.4, 0.7, -1.1, 0.05, 2.0, -0.9, 0.33, -0.15, 1.5, -2.2, 0.9];
const mkRace = (odds) => ({ horses: AM.map((m, i) => ({
  n: i + 1, ability_margin: m, tan_pct: 8.0, fuku_pct: 25.0,
  ev_tan_pct: 5.0 + i, ev_fuku_pct: 15.0 + i, need_odds: Math.round(1200 / (5 + i)) / 10,
  odds: odds ? odds[i] : null, pop: i + 1, rl_rank: i + 1,
})) });
const fresh = (odds) => Object.fromEntries(odds.map((o, i) => [String(i + 1), { tansho: o }]));

console.log('■ ① 表示はEV用の確率を読む');
t('勝率・複勝列が ev_tan_pct / ev_fuku_pct を読む', () => {
  assert.ok(/const _aiT = h\.ev_tan_pct, _aiF = h\.ev_fuku_pct;/.test(html),
            '表示が ev_* を見ていない');
  assert.ok(html.includes('>AI勝率</th>') && html.includes('>AI複勝</th>'));
});

t('順位用の値はツールチップに残す（消さない）', () => {
  assert.ok(/順位用（市場込み・softmax）の勝率/.test(html));
});

console.log('■ ② 値が無ければ数字を作らない');
t('🔴 クライアント側で確率を計算し直す関数を持たない', () => {
  assert.strictEqual(ctx._ensureAiProbs, undefined,
                     '_ensureAiProbs が残っている（softmaxでの代用が復活する）');
  assert.ok(!html.includes('_ensureAiProbs('), '_ensureAiProbs の呼び出しが残っている');
});

t('ev_tan_pct が無ければ "-"（tan_pct で代用しない）', () => {
  assert.ok(!/h\.ev_tan_pct\s*!=\s*null\s*\?\s*h\.ev_tan_pct\s*:\s*h\.tan_pct/.test(html),
            '順位用の勝率にフォールバックしている');
});

console.log('■ ③④ 必要オッズ列');
t('ヘッダが「必要ｵｯｽﾞ」で、EV ではない', () => {
  assert.ok(html.includes('>必要ｵｯｽﾞ</th>'), 'ヘッダが必要オッズになっていない');
  assert.ok(!/>EV<\/th>/.test(html), 'EV 列が残っている');
});

t('セルが need_odds を読む', () => {
  assert.ok(/const evVal = h\.need_odds;/.test(html));
});

t('🔴 必要オッズに届いても色を付けない（買い推奨として見せない）', () => {
  assert.ok(/val-met/.test(html), '達成クラスが無い');
  const m = html.match(/\.htbl td\.val-met\{([^}]*)\}/);
  assert.ok(m, 'val-met の CSS が無い');
  assert.ok(!/color:#(2|1|0)[0-9a-f]{5}/i.test(m[1]) || /color:#333/.test(m[1]),
            '緑系の色が付いている: ' + m[1]);
  assert.ok(/font-weight/.test(m[1]), '太字だけの中立表示になっていない');
});

t('実測値がツールチップに書いてある（改善しないことを明示）', () => {
  assert.ok(/73\.7%/.test(html) && /73\.1%/.test(html),
            '「超えても改善しない」実測値が書かれていない');
});

console.log('■ ⑤⑥ 順位用との分離');
t('sim_ev が順位用の量として分離されている', () => {
  assert.ok(/h\.sim_ev = \(h\.odds/.test(html), 'sim_ev の算出が無い');
  assert.ok(!/\bh\.ev\b/.test(html), 'h.ev が残っている（期待値と誤読される）');
});

t('⑥ 直前オッズを押しても EV用確率と必要オッズは動かない', () => {
  const o = [2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 25, 30];
  const r = mkRace(o);
  const before = r.horses.map(h => [h.ev_tan_pct, h.ev_fuku_pct, h.need_odds]);
  updateOddsAndEV(r, fresh([30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]));
  r.horses.forEach((h, i) => {
    assert.deepStrictEqual([h.ev_tan_pct, h.ev_fuku_pct, h.need_odds], before[i]);
  });
});

t('順位用（tan_pct）と RL順位は従来どおり市場で動く', () => {
  const odds = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1.1];
  const r = mkRace(odds);
  updateOddsAndEV(r, fresh(odds));
  const rl1 = r.horses.find(h => h.rl_rank === 1);
  assert.strictEqual(rl1.n, 12, 'RL1 は市場込みの勝率で決まるべき');
  assert.ok(r.horses.every(h => h.sim_ev != null), 'sim_ev が計算されていない');
});

console.log(`\n✅ ${pass} テスト通過`);

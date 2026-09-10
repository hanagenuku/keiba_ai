// 画面の勝率列が「市場ゼロのAI勝率」を出すことを検証する。
//
// 経緯（2026-09-10 ユーザー要望）:
//   「勝率表示は市場オッズから算出されているように見える。AI予想の勝率で表示してほしい」
// 残差学習は raw_margin = base_margin(市場人気) + ability_margin(AI) なので、
// 従来の tan_pct は市場を土台にした値だった。base_margin をフラット（全馬同値）に
// 置き換えたAI単独の勝率を ai_tan_pct として出し、画面はそちらを表示する。
//
// 🔴 固定したいこと（退行するとユーザーの要望が静かに戻る）:
//   ① 表示が ai_tan_pct であって tan_pct ではない
//   ② ai_tan_pct が**オッズに依存しない**（直前オッズを押しても動かない）
//   ③ EV は従来どおり市場込みの tan_pct から作る（AI勝率×オッズは3度否定・回収57.9%）
//   ④ RL順位は従来どおり市場込みの勝率で決まる（買い目・軸が動かないこと）
//   ⑤ JS と Python(app_json._build_solo_probs) の値が一致する
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

const ctx = vm.createContext({ console, Math, document: undefined, window: {},
                               localStorage: { getItem: () => null, setItem: () => {} },
                               fetch: () => Promise.reject(new Error('no net')),
                               setInterval: () => 0, clearInterval: () => {},
                               setTimeout: () => 0, addEventListener: () => {} });
try { vm.runInContext(js, ctx); } catch (e) { /* DOM依存の初期化は無視 */ }

const { _ensureAiProbs, updateOddsAndEV } = ctx;
assert.ok(_ensureAiProbs, '_ensureAiProbs が無い');

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

// 本番と同じ形の馬辞書（ability_margin は残差モデルが必ず入れる）
const AM = [1.2, -0.4, 0.7, -1.1, 0.05, 2.0, -0.9, 0.33, -0.15, 1.5, -2.2, 0.9];
const mkRace = (odds) => ({ horses: AM.map((m, i) => ({
  n: i + 1, ability_margin: m, tan_pct: 8.0, fuku_pct: 25.0,
  odds: odds ? odds[i] : null, pop: i + 1, rl_rank: i + 1,
})) });
// updateOddsAndEV(race, freshOdds) は {馬番: {tansho}} を取る
const fresh = (odds) => Object.fromEntries(odds.map((o, i) => [String(i + 1), { tansho: o }]));

console.log('■ AI勝率の算出');
t('全馬に ai_tan_pct / ai_fuku_pct が付く', () => {
  const r = mkRace(); _ensureAiProbs(r);
  r.horses.forEach(h => {
    assert.ok(h.ai_tan_pct != null && h.ai_fuku_pct != null);
  });
});

t('勝率はレース内で合計100%になる', () => {
  const r = mkRace(); _ensureAiProbs(r);
  const s = r.horses.reduce((a, h) => a + h.ai_tan_pct, 0);
  assert.ok(Math.abs(s - 100) < 0.6, `合計 ${s}`);
});

t('ability_margin が高い馬ほど AI勝率が高い', () => {
  const r = mkRace(); _ensureAiProbs(r);
  const best = r.horses[AM.indexOf(Math.max(...AM))];
  const worst = r.horses[AM.indexOf(Math.min(...AM))];
  assert.ok(best.ai_tan_pct > worst.ai_tan_pct);
  assert.strictEqual(best.ai_tan_pct, Math.max(...r.horses.map(h => h.ai_tan_pct)));
});

t('🔴 オッズを変えても AI勝率は動かない（市場ゼロであることの検査）', () => {
  const a = mkRace([1.2, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]);
  const b = mkRace([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1.2]);
  _ensureAiProbs(a); _ensureAiProbs(b);
  a.horses.forEach((h, i) => assert.strictEqual(h.ai_tan_pct, b.horses[i].ai_tan_pct));
});

t('ability_margin が1頭でも欠けたら列を作らない（非残差モデル）', () => {
  const r = mkRace(); r.horses[3].ability_margin = null;
  _ensureAiProbs(r);
  assert.ok(r.horses.every(h => h.ai_tan_pct == null));
});

t('サーバが入れた値を上書きしない', () => {
  const r = mkRace(); r.horses.forEach(h => { h.ai_tan_pct = 1.1; h.ai_fuku_pct = 2.2; });
  _ensureAiProbs(r);
  assert.ok(r.horses.every(h => h.ai_tan_pct === 1.1));
});

console.log('■ 表示（勝率列が AI勝率 であること）');
t('① 勝率列は ai_tan_pct を読む', () => {
  assert.ok(/const _aiT = h\.ai_tan_pct/.test(html), '表示が ai_tan_pct を見ていない');
  assert.ok(/const _aiF = h\.ai_fuku_pct/.test(html));
  assert.ok(html.includes('>AI勝率</th>'), 'ヘッダが AI勝率 になっていない');
  assert.ok(html.includes('>AI複勝</th>'));
});

t('市場込みの値はツールチップに残る（消さない）', () => {
  assert.ok(/市場込みモデルの勝率/.test(html));
});

console.log('■ 🔴 買い目・EV・RL順位が動いていないこと');
t('③ EV は従来どおり tan_pct（市場込み）から作る', () => {
  const o = [2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 25, 30];
  const r = mkRace(o);
  _ensureAiProbs(r);
  updateOddsAndEV(r, fresh(o));
  // updateOddsAndEV は tan_pct を市場込みで振り直したうえで EV を作る
  r.horses.forEach(h => {
    const want = Math.round((h.tan_pct / 100) * h.odds * 1000) / 1000;
    assert.strictEqual(h.ev, want, `EV が tan_pct 由来でない (#${h.n})`);
  });
});

t('④ RL順位は市場込みの勝率で決まる（AI勝率ではない）', () => {
  // 馬番12(ability 0.9)より馬番6(ability 2.0)の方がAI評価は上だが、
  // 馬番12に極端に短いオッズを与えると市場込みの勝率では逆転する。
  const odds = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1.1];
  const r = mkRace(odds);
  _ensureAiProbs(r);
  updateOddsAndEV(r, fresh(odds));
  const byAi = [...r.horses].sort((a, b) => b.ai_tan_pct - a.ai_tan_pct)[0];
  const rl1  = r.horses.find(h => h.rl_rank === 1);
  assert.strictEqual(byAi.n, 6, 'AI勝率1位は ability 最大の馬');
  assert.notStrictEqual(rl1.n, byAi.n, 'RL1が AI勝率1位と同じ＝市場が効いていない');
  assert.strictEqual(rl1.n, 12, 'RL1 は市場込みの勝率で決まるべき');
});

t('② 直前オッズを押しても AI勝率は変わらない', () => {
  const r = mkRace([2.0, 4.0, 6.0, 8.0, 10, 12, 14, 16, 18, 20, 25, 30]);
  _ensureAiProbs(r);
  const before = r.horses.map(h => h.ai_tan_pct);
  updateOddsAndEV(r, fresh([30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]));
  r.horses.forEach((h, i) => assert.strictEqual(h.ai_tan_pct, before[i]));
});

console.log(`\n✅ ${pass} テスト通過`);

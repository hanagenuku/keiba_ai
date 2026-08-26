// 直前オッズ取得後の再同期が、人気順位(Zipf近似)ではなく**実オッズの大きさ**で
// アンカーを作ることを検証する。
//
// 経緯: ユーザーは締め切り直前まで待ち、直前オッズボタンを何度も押す。
// ところが再同期は取得したオッズを昇順に並べて順位だけ残し、大きさを捨てていた。
// 2026-08-25 に3,940レース・53,715頭で測ると、その時点で
//   人気順位アンカー AUC 0.8226 → 実オッズアンカー 0.8292（+0.0066・3窓とも同符号）
// ⚠ 事前登録の基準②（平均+0.01以上）は未達。ユーザーの判断で採用した。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_odds_anchor_resync.mjs
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

const { _baseMarginFromOdds, _oddsBookIsUsable, _baseMarginFromPopularity,
        updateOddsAndEV } = ctx;
assert.ok(_baseMarginFromOdds, '_baseMarginFromOdds が無い');

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

console.log('■ アンカーの計算');
t('レース内で合計1に正規化される（控除率が落ちる）', () => {
  const bm = _baseMarginFromOdds([2.0, 4.0, 4.0]);
  const p = bm.map(m => 1 / (1 + Math.exp(-m)));
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1.0) < 1e-9);
  assert.ok(Math.abs(p[0] - 0.5) < 1e-9);
});

t('控除率が違う盤でも同じアンカー値になる', () => {
  const a = _baseMarginFromOdds([2.0, 4.0, 4.0]);    // Σ(1/o)=1.00
  const b = _baseMarginFromOdds([2.5, 5.0, 5.0]);    // Σ(1/o)=0.80
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-12));
});

t('オッズが短いほどアンカーが高い', () => {
  const bm = _baseMarginFromOdds([1.5, 10.0, 30.0]);
  assert.ok(bm[0] > bm[1] && bm[1] > bm[2]);
});

console.log('■ 盤の妥当性検査（壊れた盤で大きさを使わない）');
t('正常な盤は使う', () => {
  assert.strictEqual(_oddsBookIsUsable([2.5, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]), true);
});

t('2026-08-15 の壊れ方（全16頭が2.1倍）は弾く', () => {
  assert.strictEqual(_oddsBookIsUsable(new Array(16).fill(2.1)), false);
});

t('オッズ欠損がある盤は弾く（正規化が狂うため）', () => {
  assert.strictEqual(_oddsBookIsUsable([2.5, null, 8.0, 12.0]), false);
  assert.strictEqual(_oddsBookIsUsable([2.5, undefined, 8.0, 12.0]), false);
});

t('1.0倍未満が混じる盤は弾く', () => {
  assert.strictEqual(_oddsBookIsUsable([0.1, 5.0, 8.0, 12.0]), false);
});

console.log('■ 再同期の統合');
const mkRace = odds => ({
  horses: odds.map((o, i) => ({
    n: i + 1, odds: o, ability_margin: [0.9, 0.3, -0.2, -0.5, -0.8][i],
    rl_rank: i + 1, tan_pct: 0, fuku_pct: 0,
  })),
});
const fresh = odds => Object.fromEntries(
  odds.map((o, i) => [String(i + 1), { tansho: o, fukusho: o / 3 }]));

t('正常な盤では実オッズアンカーを使う', () => {
  const odds = [2.6, 4.2, 7.5, 14.0, 30.0];   // Σ(1/o)≈0.90
  const race = mkRace(odds);
  updateOddsAndEV(race, fresh(odds));
  assert.strictEqual(race._anchor, 'odds');
  const s = race.horses.reduce((a, h) => a + h.tan_pct, 0);
  assert.ok(s > 95 && s < 105, `勝率合計が不正: ${s}`);
});

t('壊れた盤では従来どおり人気順位に落ちる', () => {
  const odds = [2.1, 2.1, 2.1, 2.1, 2.1];     // Σ(1/o)≈2.38
  const race = mkRace(odds);
  updateOddsAndEV(race, fresh(odds));
  assert.strictEqual(race._anchor, 'rank');
});

t('オッズが欠けたら従来どおり人気順位に落ちる', () => {
  const odds = [2.6, 4.2, 7.5, 14.0, 30.0];
  const race = mkRace(odds);
  race.horses[2].odds = null;
  const f = fresh(odds); delete f['3'];
  updateOddsAndEV(race, f);
  assert.strictEqual(race._anchor, 'rank');
});

t('順位ではなくオッズの大きさが効いている（同順位でも値が変わる）', () => {
  // 順位は同じだが2番手のオッズだけ大きく違う2つの盤。
  // 順位アンカーなら結果は同一になるはず。
  const a = mkRace([2.0, 2.1, 20.0, 25.0, 30.0]);
  const b = mkRace([2.0, 12.0, 20.0, 25.0, 30.0]);
  updateOddsAndEV(a, fresh([2.0, 2.1, 20.0, 25.0, 30.0]));
  updateOddsAndEV(b, fresh([2.0, 12.0, 20.0, 25.0, 30.0]));
  assert.strictEqual(a._anchor, 'odds');
  assert.strictEqual(b._anchor, 'odds');
  assert.notStrictEqual(a.horses[1].tan_pct, b.horses[1].tan_pct,
    '2番手のオッズを2.1→12.0に変えても勝率が動かない＝順位しか見ていない');
});

console.log('■ Python 実装との突合');
// scripts/eval_odds_anchor.py の _odds_to_base_margin が同じ入力に対して返す値。
// 🔑 同じ数字を tests/test_odds_anchor.py 側にも置いて突き合わせている。
//    どちらかの実装がずれたら、必ずどちらかのテストが落ちる。
//    （評価はPythonで測り、本番はJSで動くので、片方だけ直る事故を防ぐ）
const PY_ODDS = [2.6, 4.2, 7.5, 14.0, 30.0, 1.4, 88.8];
const PY_BM = [-1.1392798311068162, -1.7338976161530684, -2.388546827627736,
               -3.054422777178042, -3.841398484036717, -0.19958335343326103,
               -4.940700063500772];
t('_odds_to_base_margin と同じ値を返す', () => {
  const got = _baseMarginFromOdds(PY_ODDS);
  got.forEach((v, i) => assert.ok(Math.abs(v - PY_BM[i]) < 1e-12,
    `i=${i}: JS ${v} vs Py ${PY_BM[i]}`));
});

console.log(`\n${pass} tests passed`);

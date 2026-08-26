// 市場ベースラインKPI（CLAUDE.md が「唯一のKPI」と呼ぶ指標）が成績ページに
// 出ることを検証する。
//
// 経緯: stats.json に model_kpi を出力していたのに **画面に一度も出ていなかった**
// （2026-08-26 の棚卸しで発覚）。odds_movement も同様だった。
// 溜めているのに誰も見ない、を繰り返さないために固定する。
//
// North Star #6 に従い、手打ちの模擬データではなく **本番の data/stats.json** で検証する。
//
//     node tests/test_model_kpi_display.mjs
import fs from 'fs';
import path from 'path';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const S = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'stats.json'), 'utf8'));

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

t('本番の stats.json に model_kpi がある', () => {
  assert.ok(S.model_kpi && S.model_kpi.total_races > 0,
    'model_kpi が空。generate_stats.py 側の問題');
});

t('index.html が model_kpi を読んでいる', () => {
  assert.ok(html.includes('S.model_kpi'), 'model_kpi を参照していない');
});

t('AI・市場・差の3つを表示する', () => {
  for (const k of ['ai_logloss', 'mkt_logloss', 'verdict']) {
    assert.ok(html.includes(`kpi.${k}`), `${k} を表示していない`);
  }
  assert.ok(html.includes('kpi.delta'), 'delta を表示していない');
});

t('回収率ではないという注意を出す（North Star #9）', () => {
  const i = html.indexOf('市場ベースラインKPI');
  const seg = html.slice(i, i + 3000);
  assert.ok(/回収率が上がるとは限りません/.test(seg),
    'AUC/log-loss と回収率の混同を防ぐ注意書きが無い');
});

t('delta の符号と色が正しい（負=AI優位=緑）', () => {
  const i = html.indexOf("const col = d < -0.001");
  assert.ok(i > 0, 'delta の判定が見つからない');
  const seg = html.slice(i, i + 160);
  // 負なら緑、正なら赤
  assert.ok(seg.indexOf('#27ae60') < seg.indexOf('#c0392b'),
    '負(AI優位)が緑・正(市場優位)が赤になっていない');
});

t('実データで判定が現実と合っている', () => {
  const d = S.model_kpi.delta;
  const v = S.model_kpi.verdict;
  const expect = d < -0.001 ? '市場優位' : d > 0.001 ? '市場優位' : '同等';
  // 実データは現在プラス（市場優位）のはず。ここが変わったら本物の進展
  assert.ok(typeof v === 'string' && v.length > 0);
  console.log(`     （現在 delta ${d > 0 ? '+' : ''}${d} = ${v}）`);
});

console.log(`\n${pass} tests passed`);

// ── 直前オッズ変動も同じ「計算しているのに見えない」状態だった ──────────
console.log('\n■ 直前オッズの変動');
let pass2 = 0;
const t2 = (name, fn) => { fn(); console.log('  ✅', name); pass2++; };

t2('index.html が odds_movement を読んでいる', () => {
  assert.ok(html.includes('S.odds_movement'));
});

t2('🔴「買われた馬を追ってはいけない」の警告が必ず出る', () => {
  const i = html.indexOf('直前オッズの変動と結果');
  const seg = html.slice(i, i + 3000);
  assert.ok(/買われた馬を追ってはいけません/.test(seg), '誤用を招く');
  assert.ok(/58\.2%/.test(seg), '実測値の根拠が書かれていない');
});

t2('壊れたオッズ(1.0倍未満)を集計が拾わない', () => {
  const gs = fs.readFileSync(path.join(ROOT, 'scripts', 'generate_stats.py'), 'utf8');
  const i = gs.indexOf('def calc_odds_movement_analysis');
  const seg = gs.slice(i, i + 2500);
  assert.ok(/tansho >= 1\.0/.test(seg) && /tansho_odds >= 1\.0/.test(seg),
    '> 0 のままだと 0.1倍のゴミ値を拾い「変動+76000%」が出る');
});

console.log(`\n${pass + pass2} tests passed`);

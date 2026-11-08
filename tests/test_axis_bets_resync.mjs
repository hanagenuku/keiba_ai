// 直前オッズ取得後の「軸1頭ベース買い目の組み直し」が、サーバ側
// (src/betting/axis_bets.py + pl_exact.py + app_json._format_axis_bets) と
// **同じ数字**を出すことを確かめる。
//
// 🔴 なぜこのテストが要るか
// 2026-08-09 に、クライアント側で買い目を作り直す recalcGumbelBets() を
// 取り消した。理由の第一が「サーバ側とロジックが二重になって実際に食い違った」
// （サーバは軸#12のフォーメーション13点＋ワイド3点＋馬連8点を出していたのに、
//  ボタンを押すとRL1-5ボックス10点に置き換わっていた）。
// 2026-11-08 に、直前オッズで軸(RL1)が入れ替わるのに買い目が朝の軸のまま
// 凍結されていた問題（実例 福島12R）を直すため組み直しを戻した。
// 二重実装が黙って食い違うのを防ぐ唯一の手段がこの突合テストなので、
// **JS か Python のどちらかを変えたら必ずここが落ちる**ようにしてある。
//
// フィクスチャは tests/make_axis_bets_crosscheck.py が生成する
// （Python が同じ入力で出した買い目を固定してある）。手で書き換えないこと。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     python tests/make_axis_bets_crosscheck.py   # 入力を変えたときだけ
//     node tests/test_axis_bets_resync.mjs
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

const { recalcAxisBets, buildAxisBetsJS, formatAxisBetsJS, _axRound,
        _axTicketProbs, _axWeightsFromRatings, _axExactPayouts } = ctx;
for (const [n, f] of Object.entries({ recalcAxisBets, buildAxisBetsJS,
                                      formatAxisBetsJS, _axRound,
                                      _axTicketProbs, _axWeightsFromRatings,
                                      _axExactPayouts })) {
  assert.strictEqual(typeof f, 'function', `${n} が index.html から取れない`);
}

const fx = JSON.parse(fs.readFileSync(
  path.join(ROOT, 'tests', 'fixtures', 'axis_bets_crosscheck.json'), 'utf8'));

const EPS = 1e-9;
let checks = 0;

function near(a, b, where) {
  assert.strictEqual(typeof a, 'number', `${where}: JS側が数値でない (${a})`);
  assert.strictEqual(typeof b, 'number', `${where}: Python側が数値でない (${b})`);
  assert.ok(Math.abs(a - b) <= EPS,
    `${where}: JS ${a} / Python ${b} （差 ${Math.abs(a - b)}）`);
  checks++;
}

// ⚠ vm.createContext で作ったオブジェクトは別realmなので prototype が違い、
//    assert.deepStrictEqual が中身が同じでも落ちる。JSON化して比べる。
function same(a, b, where) {
  assert.strictEqual(JSON.stringify(a), JSON.stringify(b),
    `${where}: JS と Python が食い違う\n  JS     ${JSON.stringify(a)}`
    + `\n  Python ${JSON.stringify(b)}`);
  checks++;
}

// ── 1. 買い目の生出力が Python と一致する ────────────────────────────────
for (const c of fx.cases) {
  // rating を直接与える（base_margin の作り方は test_odds_anchor_resync.mjs
  // が別に担保しているので、ここでは軸1頭ベースの組み立てだけを見る）。
  const horses = c.horses.map(h => ({
    n: h.n, name: h.name, rl_rank: h.rl_rank, odds: h.odds,
    ability_margin: h.rating,
  }));
  const race = { horses, gumbel_bets: [], _ax_bms: horses.map(() => 0) };

  const ok = recalcAxisBets(race);
  const exp = c.axis_bets;
  assert.ok(ok, `${c.name}: recalcAxisBets が組み直せなかった`);

  // recalcAxisBets の中で作られた生出力をもう一度同じ手順で作って比べる
  const probs = _axTicketProbs(
    _axWeightsFromRatings(horses.map(h => h._ax_rating)), horses.map(h => h.n));
  const winOdds = {};
  for (const h of horses) if (h._ax_odds > 0) winOdds[h.n] = h._ax_odds;
  const ab = buildAxisBetsJS(horses, probs, _axExactPayouts(winOdds));

  for (const bt of ['win', 'wide', 'quinella', 'trio']) {
    same(ab[bt].map(b => String(b.key)), exp[bt].map(b => String(b.key)),
         `${c.name} / ${bt} の買い目と順序`);
    for (let i = 0; i < ab[bt].length; i++) {
      near(ab[bt][i].prob, exp[bt][i].prob, `${c.name} / ${bt}[${i}] の確率`);
      near(ab[bt][i].odds, exp[bt][i].odds, `${c.name} / ${bt}[${i}] の推定配当`);
      near(ab[bt][i].ev, exp[bt][i].ev, `${c.name} / ${bt}[${i}] の EV`);
      same(ab[bt][i].amount, exp[bt][i].amount, `${c.name} / ${bt}[${i}] の金額`);
      same(ab[bt][i].label, exp[bt][i].label, `${c.name} / ${bt}[${i}] のラベル`);
    }
  }

  const es = exp.summary;
  same(ab.summary.axis, es.axis, `${c.name} / 軸`);
  same(ab.summary.total_points, es.total_points, `${c.name} / 合計点数`);
  same(ab.summary.total_amount, es.total_amount, `${c.name} / 合計金額`);
  for (const k of ['quinella_syn_odds', 'trio_syn_odds', 'trio_points',
                   'payout_min', 'payout_max']) {
    if (es[k] === undefined) {
      assert.strictEqual(ab.summary[k], undefined,
        `${c.name} / summary.${k}: Pythonに無いのにJSにある`);
      checks++;
    } else {
      near(ab.summary[k], es[k], `${c.name} / summary.${k}`);
    }
  }
  if (es.trio_legs === undefined) {
    assert.strictEqual(ab.summary.trio_legs, undefined,
      `${c.name} / trio_legs: Pythonに無いのにJSにある`);
    checks++;
  } else {
    same(ab.summary.trio_legs, es.trio_legs, `${c.name} / 三連複の列`);
  }

  // ── 2. 画面に出る表示文字列まで一致する ───────────────────────────────
  const fmt = formatAxisBetsJS(ab, horses);
  same(fmt.length, c.formatted.length, `${c.name} / 表示行数`);
  for (let i = 0; i < fmt.length; i++) {
    const a = fmt[i], b = c.formatted[i];
    for (const k of ['tag', 'label', 'horse', 'est', 'amt']) {
      same(a[k], b[k], `${c.name} / 表示[${i}].${k}`);
    }
    if (b.combos) same(a.combos, b.combos, `${c.name} / 表示[${i}].combos`);
    if (b.nums) same(a.nums, b.nums, `${c.name} / 表示[${i}].nums`);
    if (b.legs) same(a.legs, b.legs, `${c.name} / 表示[${i}].legs`);
    if (b.syn_odds !== undefined) near(a.syn_odds, b.syn_odds,
                                       `${c.name} / 表示[${i}].syn_odds`);
    if (b.mates) {
      same(a.mates.map(m => m.n), b.mates.map(m => m.n),
           `${c.name} / 表示[${i}].mates の馬番`);
      for (let j = 0; j < b.mates.length; j++) {
        near(a.mates[j].odds, b.mates[j].odds,
             `${c.name} / 表示[${i}].mates[${j}].odds`);
        near(a.mates[j].prob, b.mates[j].prob,
             `${c.name} / 表示[${i}].mates[${j}].prob`);
      }
    }
  }
}

// ── 3. 組み直せない条件では何もせず朝の買い目を残す ──────────────────────
{
  const morning = [{ tag: 'tan', label: '単勝（軸）', horse: '#1 A' }];
  // ability_margin が欠けている（残差モデルでない）
  const r1 = { horses: [{ n: 1, rl_rank: 1, odds: 2.0 }],
               gumbel_bets: morning, _ax_bms: [0] };
  assert.strictEqual(recalcAxisBets(r1), false, 'ability_margin なしで組み直した');
  same(r1.gumbel_bets, morning, 'ability_margin なしで朝の買い目を壊した');

  // _ax_bms が無い（updateOddsAndEV が再同期をスキップした）
  const r2 = { horses: [{ n: 1, rl_rank: 1, odds: 2.0, ability_margin: 0.5 },
                        { n: 2, rl_rank: 2, odds: 9.0, ability_margin: -0.5 },
                        { n: 3, rl_rank: 3, odds: 12.0, ability_margin: -0.9 }],
               gumbel_bets: morning };
  assert.strictEqual(recalcAxisBets(r2), false, '_ax_bms なしで組み直した');
  same(r2.gumbel_bets, morning, '_ax_bms なしで朝の買い目を壊した');

  // オッズが2頭しか取れていない
  const r3 = { horses: [{ n: 1, rl_rank: 1, odds: 2.0, ability_margin: 0.5 },
                        { n: 2, rl_rank: 2, odds: 9.0, ability_margin: -0.5 },
                        { n: 3, rl_rank: 3, odds: 0, ability_margin: -0.9 }],
               gumbel_bets: morning, _ax_bms: [0, 0, 0] };
  assert.strictEqual(recalcAxisBets(r3), false, 'オッズ2頭で組み直した');
  same(r3.gumbel_bets, morning, 'オッズ2頭で朝の買い目を壊した');

  // 軸(RL1)のオッズが取れていない（取消等）。ここは guard を通り抜けて
  // buildAxisBetsJS まで行くが、軸を含むペアの推定配当が1つも作れないので
  // total_points が 0 になる。空の買い目で朝の買い目を上書きしてはいけない。
  const r3b = { horses: [
      { n: 1, name: 'A', rl_rank: 1, odds: 0,    ability_margin: 0.9 },
      { n: 2, name: 'B', rl_rank: 2, odds: 3.0,  ability_margin: 0.4 },
      { n: 3, name: 'C', rl_rank: 3, odds: 9.0,  ability_margin: -0.3 },
      { n: 4, name: 'D', rl_rank: 4, odds: 14.0, ability_margin: -0.7 }],
    gumbel_bets: morning, _ax_bms: [0, 0, 0, 0] };
  assert.strictEqual(recalcAxisBets(r3b), false, '軸のオッズ欠損で組み直した');
  same(r3b.gumbel_bets, morning, '軸のオッズ欠損で朝の買い目を壊した');

  // rl_rank が無い（サーバが順位を出していない古い latest.json）
  const r3c = { horses: [
      { n: 1, rl_rank: null, odds: 3.0,  ability_margin: 0.4 },
      { n: 2, rl_rank: null, odds: 9.0,  ability_margin: -0.3 },
      { n: 3, rl_rank: null, odds: 14.0, ability_margin: -0.7 }],
    gumbel_bets: morning, _ax_bms: [0, 0, 0] };
  assert.strictEqual(recalcAxisBets(r3c), false, 'rl_rank なしで組み直した');
  same(r3c.gumbel_bets, morning, 'rl_rank なしで朝の買い目を壊した');

  // サーバが軸1頭ベース買い目を出していないレース（gumbel_bets が無い）
  const r4 = { horses: [{ n: 1, rl_rank: 1, odds: 2.0, ability_margin: 0.5 },
                        { n: 2, rl_rank: 2, odds: 9.0, ability_margin: -0.5 },
                        { n: 3, rl_rank: 3, odds: 12.0, ability_margin: -0.9 }],
               _ax_bms: [0, 0, 0] };
  assert.strictEqual(recalcAxisBets(r4), false, 'gumbel_bets が無いのに作った');
  assert.strictEqual(r4.gumbel_bets, undefined, 'gumbel_bets を勝手に作った');
}

// ── 4. 軸が入れ替わったら買い目の軸も入れ替わる（今回の defect の回帰） ──
{
  // 朝: #9 が1番人気3.9倍でRL1。直前: 8.5倍まで流れて #6 がRL1。
  // 実例は 2026-11-08 福島12R。
  const mk = (odds9, odds6) => ({
    horses: [
      { n: 6, name: 'F', rl_rank: 2, odds: odds6, ability_margin: 0.40 },
      { n: 9, name: 'I', rl_rank: 1, odds: odds9, ability_margin: 0.45 },
      { n: 1, name: 'A', rl_rank: 3, odds: 9.0, ability_margin: -0.30 },
      { n: 2, name: 'B', rl_rank: 4, odds: 14.0, ability_margin: -0.60 },
      { n: 3, name: 'C', rl_rank: 5, odds: 21.0, ability_margin: -0.95 },
      { n: 4, name: 'D', rl_rank: 6, odds: 45.0, ability_margin: -1.50 },
    ],
    gumbel_bets: [], _ax_bms: [0, 0, 0, 0, 0, 0],
  });
  const morningRace = mk(3.9, 4.6);
  assert.ok(recalcAxisBets(morningRace));
  assert.strictEqual(morningRace._ax_axis, 9, '朝の軸が #9 でない');

  // 直前: #9 が流れて #6 が被される（rl_rank は updateOddsAndEV が振り直す）
  const lateRace = mk(8.5, 3.1);
  lateRace.horses.find(h => h.n === 9).rl_rank = 2;
  lateRace.horses.find(h => h.n === 6).rl_rank = 1;
  assert.ok(recalcAxisBets(lateRace));
  assert.strictEqual(lateRace._ax_axis, 6, '直前オッズで軸が #6 に移っていない');
  assert.ok(lateRace.gumbel_bets.length > 0, '買い目が空になった');
  assert.ok(lateRace.gumbel_bets[0].horse.startsWith('#6 '),
            `単勝の軸が #6 でない: ${lateRace.gumbel_bets[0].horse}`);
  for (const e of lateRace.gumbel_bets) {
    if (e.axis !== undefined) {
      assert.strictEqual(e.axis, 6, `${e.label} の軸が #6 でない`);
    }
  }
  checks += 6;
}

// ── 5. Python の round() と同じ丸め（ちょうど半分は偶数側） ───────────────
same(_axRound(1.25, 1), 1.2, '_axRound(1.25,1) が偶数丸めになっていない');
same(_axRound(1.35, 1), 1.4, '_axRound(1.35,1)');
same(_axRound(0.125, 2), 0.12, '_axRound(0.125,2) が偶数丸めになっていない');
same(_axRound(2.5, 0), 2.0, '_axRound(2.5,0) が偶数丸めになっていない');
same(_axRound(3.5, 0), 4.0, '_axRound(3.5,0)');

console.log(`✅ 軸1頭ベース買い目の Python↔JS 突合: ${checks}件の比較が一致 `
            + `(${fx.cases.length}ケース)`);

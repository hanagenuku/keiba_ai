// 「直前オッズ取得」を押したとき、軸1頭ベース買い目（gumbel_bets）が
// **サーバと同じ規則で**組み直されることを検証する。
//
// 🔴 このファイルは 2026-11-08 に意味が反転した。
// 旧版（test_odds_refresh_keeps_bets.mjs）は「押しても買い目を作り変えない」
// ことを固定していた。2026-08-09 の実害がひどかったためで、その内容は今も
// 読む価値がある——サーバは新潟5Rに
//   三連複(13点) 軸#12 / 2列目 #3 #11 / 3列目 8頭
//   ワイド(3点) #12-#3・#11・#10 ／ 馬連(8点) 合成3.8倍
// を出していたのに、ボタンを押した瞬間クライアント側 recalcGumbelBets() が
// 全部作り直し、**RL1〜5（＝人気1〜5）のボックス10点**に置き換わっていた。
// 単勝・ワイド・馬連は「除外」の取り消し線になった。さらにその選別根拠のEVは、
// 三連複だけ cal_prob（複勝＝3着内確率）を単勝勝率の位置に入れており
// 「確率」が3.5を超えていた（EV8.81の正体）。
//
// 🔴 しかし「作り変えない」は別の実害を生んでいた。直前オッズでAI+市の1位
// （＝軸）が入れ替わっても買い目は朝の軸のまま凍結されていた。
// 実例 2026-11-08 福島12R: 朝11:06 は #9 が3.9倍・1番人気でRL1だったが、
// 15:56 には8.5倍・4番人気まで流れてRL4になり、RL1は #6 に移っていた。
// それでも買い目の軸は #9 のままだった。
//
// → 2026-11-08 に組み直しを戻した。ただし 2026-08-09 の形には戻さない:
//    ① 選び方はサーバ側 axis_bets.py の移植（軸=RL1 / 中穴=単勝7〜30倍 /
//       合成オッズ3.0）。EVによる選別は持ち込まない
//    ② 確率はモンテカルロではなく Plackett-Luce の厳密式なので、
//       tests/test_axis_bets_resync.mjs が Python と 1e-9 まで突合できる
// このファイルが固定するのは「押したあとの形」側（②〜⑤は旧版のまま）。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_odds_refresh_rebuilds_bets.mjs
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

// 2026-08-09 新潟5R の実データ（latest.json より。RL順＝人気順が一致するレース）
const HORSES = [
  { n: 12, name: 'シュヴァルツシルト',  rl_rank: 1,  pop: 1,  odds: 3.8,   tan_pct: 21.9, fuku_pct: 56.1, cal_prob: 0.6544, ability_margin: 0.9 },
  { n: 2,  name: 'ベルゼビュート',      rl_rank: 2,  pop: 2,  odds: 5.1,   tan_pct: 15.7, fuku_pct: 44.2, cal_prob: 0.5301, ability_margin: 0.5 },
  { n: 9,  name: 'アンバサダネージュ',  rl_rank: 3,  pop: 3,  odds: 5.9,   tan_pct: 11.7, fuku_pct: 35.0, cal_prob: 0.4457, ability_margin: 0.3 },
  { n: 3,  name: 'サトノエスケープ',    rl_rank: 4,  pop: 4,  odds: 8.5,   tan_pct: 7.2,  fuku_pct: 22.9, cal_prob: 0.2552, ability_margin: 0.1 },
  { n: 11, name: 'ゼファーテソーロ',    rl_rank: 5,  pop: 5,  odds: 9.8,   tan_pct: 6.6,  fuku_pct: 21.1, cal_prob: 0.2217, ability_margin: 0.0 },
  { n: 10, name: 'スリーコーズ',        rl_rank: 6,  pop: 6,  odds: 12.6,  tan_pct: 5.9,  fuku_pct: 18.9, cal_prob: 0.189,  ability_margin: -0.1 },
  { n: 6,  name: 'ベアアッコチャン',    rl_rank: 7,  pop: 8,  odds: 17.3,  tan_pct: 5.0,  fuku_pct: 16.2, cal_prob: 0.162,  ability_margin: -0.2 },
  { n: 15, name: 'ペッパーミル',        rl_rank: 8,  pop: 10, odds: 24.8,  tan_pct: 4.4,  fuku_pct: 14.4, cal_prob: 0.144,  ability_margin: -0.3 },
  { n: 8,  name: 'プルシャプラ',        rl_rank: 9,  pop: 7,  odds: 16.4,  tan_pct: 4.1,  fuku_pct: 13.5, cal_prob: 0.135,  ability_margin: -0.4 },
  { n: 1,  name: 'ブリングライト',      rl_rank: 10, pop: 9,  odds: 18.6,  tan_pct: 3.8,  fuku_pct: 12.5, cal_prob: 0.125,  ability_margin: -0.5 },
  { n: 5,  name: 'マジンタクシー',      rl_rank: 13, pop: 11, odds: 26.0,  tan_pct: 2.9,  fuku_pct: 9.7,  cal_prob: 0.097,  ability_margin: -0.6 },
];

// サーバ（axis_bets.py）が実際に出した買い目
const SERVER_BETS = [
  { tag: 'tan', label: '単勝（軸）', horse: '#12 シュヴァルツシルト', est: '3.8倍', amt: '¥100' },
  { tag: 'wide', label: 'ワイド(3点)', axis: 12, horse: '#12 - #3・#11・#10',
    mates: [{ n: 3 }, { n: 11 }, { n: 10 }], est: '6.2〜8.8倍', amt: '¥300' },
  { tag: 'umaren', label: '馬連(8点)', axis: 12, horse: '#12 - #3・#11・#10・#6・#15・#1・#5・#8',
    mates: [{ n: 3 }, { n: 11 }, { n: 10 }, { n: 6 }, { n: 15 }, { n: 1 }, { n: 5 }, { n: 8 }],
    est: '17.0〜49.5倍', amt: '¥800', syn_odds: 3.8 },
  { tag: 'sanfuku', label: '三連複(13点)', trio_type: 'formation',
    legs: [[12], [3, 11], [2, 3, 6, 8, 9, 10, 11, 15]],
    nums: [2, 3, 6, 8, 9, 10, 11, 12, 15],
    combos: ['2-3-12', '3-9-12', '2-11-12', '9-11-12', '3-11-12', '3-6-12', '10-11-12',
             '11-12-15', '6-11-12', '3-10-12', '8-11-12', '3-12-15', '3-8-12'],
    est: '¥2,210〜¥13,680', syn_odds: 4.2, amt: '¥1300' },
];

function makeRace() {
  return {
    race_id: '20260809_04_05', r: 5, name: '3歳以上1勝クラス', rec: true,
    honmei: { n: 12, name: 'シュヴァルツシルト', odds: 3.8, style: '先行' },
    horses: JSON.parse(JSON.stringify(HORSES)),
    bets: [], gumbel_bets: JSON.parse(JSON.stringify(SERVER_BETS)),
  };
}

// 11:28時点の直前オッズ（朝から動いている）
const FRESH_ODDS = {};
for (const h of HORSES) {
  FRESH_ODDS[h.n] = { tansho: Math.round(h.odds * 1.25 * 10) / 10,
                      fukusho: Math.round(h.odds * 0.35 * 10) / 10 };
}

function run() {
  const race = makeRace();
  const log = [];
  const ctx = {
    console: { log: (...a) => log.push(a.join(' ')) },
    localStorage: { _d: {}, getItem(k){return this._d[k]??null;}, setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];} },
    document: {
      visibilityState: 'visible',
      // 末尾の loadData() 起動で触られるだけなので、素通しのスタブでよい
      getElementById: () => ({ innerHTML: '', textContent: '', style: {},
                               classList: { add(){}, remove(){} } }),
      querySelectorAll: () => [],
      addEventListener: () => {}, removeEventListener: () => {},
    },
    window: { addEventListener: () => {}, removeEventListener: () => {} },
    alert: m => log.push('ALERT:' + m),
    fetch: async () => ({ ok: true, json: async () => ({ status: 'ok', odds: FRESH_ODDS, updated_at: '11:28' }) }),
    Date, Promise, JSON, Math, String, Number, Object, Array, Error, Set, Map,
    parseInt, parseFloat, isNaN, encodeURIComponent, setTimeout, clearTimeout,
    setInterval: () => 1, clearInterval: () => {},
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(js
    + '\n;globalThis.__race = null;'
    + '\n;findRace = function(){ return globalThis.__race; };'
    + '\n;redrawRace = function(){};'
    + '\n;globalThis.__fetchFreshOdds = fetchFreshOdds;', ctx);
  ctx.__race = race;
  return { ctx, race, log };
}

const { ctx, race, log } = run();
const before = JSON.stringify(race.gumbel_bets);

await ctx.__fetchFreshOdds('k', { textContent: '', disabled: false });

// ── ① 買い目が組み直されていること（2026-11-08 福島12R の回帰テスト） ─────
assert.strictEqual(race._ax_rebuilt, true,
  '直前オッズ取得で軸1頭ベース買い目が組み直されていない');
// 朝の買い目は捨てずに残す（何が変わったか後から言えるようにする）
assert.strictEqual(JSON.stringify(race._morning_gumbel_bets), before,
  '朝の買い目が _morning_gumbel_bets に残っていない');
// このデータは全馬のオッズを一律1.25倍にしてあるので順位は動かない。
// よって軸は #12 のまま（＝組み直しても軸の規則がRL1であること）。
assert.strictEqual(race._ax_axis, 12, '組み直し後の軸が RL1(#12) でない');
const rl1 = race.horses.find(h => h.rl_rank === 1);
assert.strictEqual(race._ax_axis, rl1.n, '軸が RL1 と一致していない');
// 中穴帯(7〜30倍)の顔ぶれは一律1.25倍で変わるので、買い目は朝と同一では
// ないはず（同一なら組み直しが効いていない）。
assert.notStrictEqual(JSON.stringify(race.gumbel_bets), before,
  'オッズが動いたのに買い目が朝と1字も変わっていない');
// 全部の相手が軸とのペアであること（ボックスに化けていない）
for (const b of race.gumbel_bets) {
  if (b.axis !== undefined) {
    assert.strictEqual(b.axis, 12, b.label + ' の軸が #12 でない');
    for (const m of (b.mates || [])) {
      assert.notStrictEqual(m.n, 12, b.label + ' の相手に軸自身が入っている');
    }
  }
}

// ── ①-b ◎本命と買い目の軸が一致すること ─────────────────────────────────
// PR #326 で ◎ だけ直前オッズに追従させたため、◎ と買い目の軸が食い違う
// 状態を一度作ってしまった（◎は新しいRL1・買い目は朝の軸）。両方そろえる。
assert.strictEqual(race.honmei.n, race._ax_axis,
  `◎(#${race.honmei.n}) と買い目の軸(#${race._ax_axis}) が食い違っている`);
assert.ok(race.gumbel_bets[0].horse.startsWith('#' + race.honmei.n + ' '),
  '単勝の軸が ◎ と違う: ' + race.gumbel_bets[0].horse);

// ── ② 三連複がフォーメーションのまま（RL1-5ボックスに化けていない） ────────
const trio = race.gumbel_bets.find(b => b.tag === 'sanfuku');
assert.strictEqual(trio.trio_type, 'formation', '三連複がボックスに化けている');
// ⚠ vm.createContext のオブジェクトは別realmで prototype が違うため
//    deepStrictEqual は中身が同じでも落ちる。JSON化して比べる。
assert.strictEqual(JSON.stringify(trio.legs[0]), '[12]', '三連複の軸が変わっている');
// 点数は中穴帯の顔ぶれで変わるが、A-中穴2-(RL2..9＋中穴) の形は保つ。
assert.ok(trio.combos.length >= 4 && trio.combos.length <= 24,
  `三連複の点数が構造から外れている: ${trio.combos.length}点`);
for (const c of trio.combos) {
  assert.ok(c.split('-').map(Number).includes(12),
    `三連複に軸#12を含まない点がある: ${c}`);
}
assert.strictEqual(trio.legs[1].length, 2, '三連複の2列目が中穴2頭でない');
// 事故の形そのもの: RL1〜5のボックス（12,2,9,3,11の10点）になっていないこと
const box5 = [2, 3, 9, 11, 12];
assert.ok(!(trio.nums.length === 5 && box5.every(n => trio.nums.includes(n))),
  'RL1〜5（＝人気1〜5）のボックスに置き換わっている');

// ── ③ 単勝・ワイド・馬連が「除外」されず残っていること ────────────────────
for (const tag of ['tan', 'wide', 'umaren']) {
  assert.ok(race.gumbel_bets.some(b => b.tag === tag), tag + ' が消えている');
}

// ── ④ EVがどの買い目にも出ないこと ───────────────────────────────────────
// EVは買い目の根拠にならないと2026-07-05・07-30に2回否定済み。
// 三連複のEVは cal_prob（複勝確率）を単勝勝率の位置に入れた壊れた値だった。
for (const b of race.gumbel_bets) {
  assert.strictEqual(b.ev, undefined, b.tag + ' にEVが付いている');
}

// ── ⑤ 馬表のオッズ・勝率は更新されていること（表示更新まで止めていない） ──
const axis = race.horses.find(h => h.n === 12);
assert.notStrictEqual(axis.odds, 3.8, '直前オッズが馬表に反映されていない');
assert.strictEqual(race._update_time, '11:28', '更新時刻が入っていない');
assert.strictEqual(race._odds_refreshed, true, '_odds_refreshed が立っていない');

console.log('✅ test_odds_refresh_rebuilds_bets: 6群すべて通過');

// 直前オッズ取得後、◎本命（race.honmei）が振り直した RL順位1位に追従することを検証する。
//
// 経緯: 2026-11-08 福島12R。朝11:06 は #9 ジェムシリカ 3.9倍・1番人気でRL1だったが、
// 15:56 には 8.5倍・4番人気まで流れてRL4になり、RL1は #6 ディオデルマーレに移っていた。
// updateOddsAndEV は tan_pct / fuku_pct / rl_rank を振り直していたのに race.honmei だけ
// 更新していなかったため、◎と行ハイライトが4位の馬に貼り付いたまま残り、
// AI+市列のツールチップ「◎が付いている馬がこの列の1位です」が嘘になっていた。
// 2026-07-26② で勝率・複勝率・EVの据え置きを直したときの漏れ。
//
// ⚠ 買い目は honmei を読まない（recalcGumbelBets は rl_rank を読む）。表示だけの話。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_honmei_resync.mjs
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

const { updateOddsAndEV } = ctx;
assert.ok(updateOddsAndEV, 'updateOddsAndEV が読めていない');

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

// 朝の盤: #9 が最有力（ability_margin 最大かつ最低オッズ）、#6 が2番手。
const mkRace = odds => ({
  honmei: { n: 9, name: 'ジェムシリカ', odds: 3.9, score: 8.4, style: '逃げ' },
  horses: [
    { n: 9, name: 'ジェムシリカ',    ability_margin:  0.55, style: '逃げ' },
    { n: 6, name: 'ディオデルマーレ', ability_margin:  0.50, style: '先行' },
    { n: 4, name: 'チュウワメロディー', ability_margin: 0.10, style: '差し' },
    { n: 1, name: 'マイネルユーゲント', ability_margin: -0.30, style: '差し' },
    { n: 5, name: 'ディーズメンフィス', ability_margin: -0.60, style: '追込' },
  ].map((h, i) => Object.assign(h, { odds: odds[i], rl_rank: i + 1,
                                     tan_pct: 0, fuku_pct: 0 })),
});
const fresh = pairs => Object.fromEntries(
  Object.entries(pairs).map(([n, o]) => [n, { tansho: o, fukusho: o / 3 }]));

console.log('■ ◎本命の再同期');

t('オッズが動いてRL1が入れ替わったら ◎ も移る（福島12Rの再現）', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  // #9 が 3.9 → 8.5 に流れ、#6 が 4.6 → 3.1 に買われる
  updateOddsAndEV(race, fresh({ 9: 8.5, 6: 3.1, 4: 5.4, 1: 6.9, 5: 13.1 }));
  const rl1 = race.horses.find(h => h.rl_rank === 1);
  assert.strictEqual(rl1.n, 6, `RL1 が #6 でない: #${rl1.n}`);
  assert.strictEqual(race.honmei.n, 6,
    `◎ が朝の #${race.honmei.n} のまま。RL1(#${rl1.n}) に追従していない`);
  assert.strictEqual(race.honmei.name, 'ディオデルマーレ');
});

t('◎のオッズも朝の値のまま残さない', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  updateOddsAndEV(race, fresh({ 9: 8.5, 6: 3.1, 4: 5.4, 1: 6.9, 5: 13.1 }));
  assert.strictEqual(race.honmei.odds, 3.1,
    `◎本命帯のオッズが古い: ${race.honmei.odds}`);
});

t('朝の◎が誰だったかは _morning_honmei に残る', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  updateOddsAndEV(race, fresh({ 9: 8.5, 6: 3.1, 4: 5.4, 1: 6.9, 5: 13.1 }));
  assert.strictEqual(race._morning_honmei, 9);
});

t('順位が変わらなければ ◎ も変わらない', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  updateOddsAndEV(race, fresh({ 9: 3.8, 6: 4.7, 4: 6.1, 1: 8.1, 5: 14.2 }));
  assert.strictEqual(race.honmei.n, 9);
});

t('ability_margin が欠ける馬が居れば再同期しない（◎も朝のまま）', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  race.horses[1].ability_margin = null;   // 非残差モデル等
  updateOddsAndEV(race, fresh({ 9: 8.5, 6: 3.1, 4: 5.4, 1: 6.9, 5: 13.1 }));
  assert.strictEqual(race.honmei.n, 9, '再同期しない経路で ◎ を動かしてはいけない');
});

t('honmei が無いレースでも落ちない', () => {
  const race = mkRace([3.9, 4.6, 6.0, 8.0, 14.0]);
  delete race.honmei;
  updateOddsAndEV(race, fresh({ 9: 8.5, 6: 3.1, 4: 5.4, 1: 6.9, 5: 13.1 }));
  assert.ok(!race.honmei);
});

console.log(`\n${pass} tests passed`);

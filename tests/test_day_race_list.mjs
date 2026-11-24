// 当日分カードの「記事に載せたレースの結果」を検証する。
//
// 経緯: 2026-11-14 ユーザー指摘「何度もお願いしているが、その日の全ての推奨レースを
// まとめた結果表示をしてくれ」。成績ページは券種ごとの集計（的中率・回収率）しか
// 出しておらず、**どのレースが当たってどのレースが外れたか**が読めなかった。
//
// 🔴 ここで固定するのは数字ではなく表示の規則:
//   1. 当日分カードの中（累計より前）に出ること。累計の下の折りたたみではない。
//   2. ★ が推奨レース（conf >= conf_strong）。★無しは 9〜11R の推奨漏れ。
//   3. 払戻は ¥、不的中は ×、評価不能（発売無し・払戻未取得）は — の3状態を区別する。
//      ここを × に丸めると「買ったのに外れた」と読めてしまう。
//   4. 結果がまだ無い日（settled=false）には出さない（0件表示で外れたと誤読させない）。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_day_race_list.mjs
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

let captured = '';
const ctx = vm.createContext({
  console, Math,
  document: { getElementById: () => ({ set innerHTML(v) { captured = v; },
                                       get innerHTML() { return captured; } }),
              querySelector: () => null, querySelectorAll: () => [] },
  window: {}, localStorage: { getItem: () => null, setItem: () => {} },
  fetch: () => Promise.reject(new Error('no net')),
  setInterval: () => 0, clearInterval: () => {}, setTimeout: () => 0,
  addEventListener: () => {},
});
try { vm.runInContext(js, ctx); } catch (e) { /* DOM依存の初期化は無視 */ }
assert.ok(ctx.renderStats, 'renderStats が読めていない');

const BETS = ['単勝', '複勝', '馬連', 'ワイド', '三連複'];

const row = (venue, num, rec, over) => Object.assign({
  date: '2026-11-14', venue, num, cls: '2勝', conf: rec ? 70 : 52,
  rec: rec ? 1 : 0, main: num >= 9 ? 1 : 0, rl: [8, 4, 3],
  axis: { num: 8, name: 'マディソンガール', pop: 1, place: 1 },
  result: [8, 4, 3],
  pays: { 単勝: 270, 複勝: 130, 馬連: 0, ワイド: null, 三連複: 0 },
}, over || {});

const stats = (over) => ({
  scope_accuracy: {
    conf_strong: 66, main_nums: [9, 10, 11], bet_types: BETS,
    period: { from: '2026-11-14', to: '2026-11-14' }, races: 2,
    excluded: { no_axis: 0, no_conf: 0 },
    // カードは groups が空だと「データ蓄積中」になるので、最小の1グループを置く
    groups: [{ label: '全レース', races: 2, bets: [], total: {} }],
    by_venue: [], monthly: [], daily: [], recent: [],
    latest_day: Object.assign({
      date: '2026-11-14', venues: ['京都'], races: 12, scope_races: 2,
      settled: true, groups: [],
      race_list: [row('京都', 8, true),
                  row('京都', 9, false, { axis: { num: 10, name: 'ダンツシタン', pop: 1, place: 4 },
                                          pays: { 単勝: 0, 複勝: 0, 馬連: 0, ワイド: null, 三連複: 0 } })],
    }, over || {}),
  },
});

const render = (S) => { captured = ''; ctx.renderStats(S); return captured; };

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

t('当日分カードの中に出る（累計より前）', () => {
  const h = render(stats());
  const i = h.indexOf('記事に載せたレースの結果');
  assert.ok(i > 0, '表が出ていない');
  assert.ok(i < h.indexOf('📊 累計'), '累計より後ろに出ている');
});

t('信頼度66以上の行に★が付き、それ以外には付かない', () => {
  const h = render(stats());
  const seg = h.slice(h.indexOf('記事に載せたレースの結果'));
  const table = seg.slice(0, seg.indexOf('</table>'));   // 脚注の★を拾わないため
  const rows = table.split('<tr').slice(2, 4);           // ヘッダ行を除いた2行
  assert.ok(rows[0].includes('★'), '推奨レースに★が無い');
  assert.ok(!rows[1].includes('★'), '推奨漏れに★が付いている');
});

t('◎の馬番・馬名・人気・着順を出す', () => {
  const h = render(stats());
  assert.ok(h.includes('マディソンガール'), '◎の馬名が無い');
  assert.ok(h.includes('(1番人気)'), '人気が無い');
  assert.ok(h.includes('>1着<'), '◎の着順が無い');
});

t('払戻・不的中・評価不能を区別する', () => {
  const h = render(stats());
  const seg = h.slice(h.indexOf('記事に載せたレースの結果'));
  assert.ok(seg.includes('¥270'), '払戻が出ていない');
  assert.ok(seg.includes('×'), '不的中が × になっていない');
  assert.ok(seg.includes('—'), '評価不能が — になっていない');
});

t('本数を見出しに出す（★は信頼度66以上の本数）', () => {
  // 🟢 2026-11-24 ユーザー指示「記事の対象レースに合わせて」。
  //    並ぶのは記事に載せたレースなので、見出しは「計 N R」が主。
  const h = render(stats());
  assert.ok(/計 2R/.test(h), '見出しの本数が違う: ' + h.match(/計 [^<]*/));
  assert.ok(/★=信頼度66以上 1R/.test(h), '★の本数が出ていない: ' + h.match(/★=[^<]*/));
});

t('結果がまだ無い日には出さない', () => {
  const h = render(stats({ settled: false }));
  assert.ok(!h.includes('記事に載せたレースの結果'), '結果待ちの日に表を出している');
  assert.ok(h.includes('結果待ち'), '結果待ちの表示が消えている');
});

t('一覧が空なら表そのものを出さない', () => {
  const h = render(stats({ race_list: [] }));
  assert.ok(!h.includes('記事に載せたレースの結果'), '空の表を出している');
});

t('閾値はデータ側の conf_strong を出す（画面で作らない）', () => {
  const S = stats();
  S.scope_accuracy.conf_strong = 72;
  const h = render(S);
  assert.ok(h.includes('信頼度 72 以上'), '閾値が画面側で固定されている');
});

console.log(`\n${pass} tests passed`);

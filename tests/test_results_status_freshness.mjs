// 成績ページの「最終実行」が、その回の実行時刻を出すか検証する。
//
// 2026-08-09 の実害: 20:10に日曜結果を取得して成功しているのに、成績ページの
// 「最終実行」が 09:49 JST（その日の朝のrefresh）のままだった。
// 原因は stats.json の results_status.workflow_run_at が **必ず1回前の値**に
// なること。generate_stats.py が workflow_status.json を読む時点では、
// その回の「Write status」ステップがまだ走っていないため。
// ユーザーからは「本日の結果取得失敗してます」と見えた。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_results_status_freshness.mjs
//
// 旧コード（stats.json の値をそのまま表示していた版）では失敗する。
import fs from 'fs'; import path from 'path'; import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

// 2026-08-09 に実際に起きた形をそのまま使う
const STALE_RUN_AT = '2026-08-09T00:49:48Z';   // 朝09:49のrefresh（stats.jsonに残る値）
const LIVE_RUN_AT  = '2026-08-09T11:20:40Z';   // 夜20:20の日曜結果（実際の最終実行）

const STATS = {
  results_status: {
    last_date: '2026-08-09', races: 32,
    venues: [{ name: '新潟', races: 11 }, { name: '中京', races: 11 }, { name: '札幌', races: 10 }],
    workflow_run_at: STALE_RUN_AT, workflow_result: 'success',
  },
  weekly_roi: [], rl_accuracy: {}, model_kpi: {},
};

function run({ liveAvailable }) {
  let rendered = '';
  const page = { set innerHTML(v) { rendered = v; }, get innerHTML() { return rendered; } };
  const ctx = {
    console: { log(){}, warn(){}, error(){} },
    Date, Promise, JSON, Math, String, Number, Object, Array, Error, Set, Map,
    parseInt, parseFloat, isNaN, isFinite, encodeURIComponent,
    setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    document: {
      getElementById: id => id === 'page-stats' ? page : ({ textContent: '', style: {}, parentElement: { title: '' } }),
      querySelectorAll: () => [], addEventListener() {}, removeEventListener() {},
      visibilityState: 'visible',
    },
    window: { addEventListener() {}, removeEventListener() {} },
    fetch: async (url) => {
      const u = String(url);
      if (u.includes('workflow_status')) {
        return liveAvailable
          ? { ok: true, json: async () => ({ status: 'success', workflow: 'sunday-results', updated_at: LIVE_RUN_AT }) }
          : { ok: false };
      }
      return { ok: true, json: async () => STATS };
    },
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(js + '\n;globalThis.__loadStats = loadStats;', ctx);
  return ctx.__loadStats().then(() => rendered);
}

// ── ① ライブの workflow_status.json があれば、そちらの時刻が出る ──────────
const a = await run({ liveAvailable: true });
assert.ok(a.includes('2026-08-09 20:20 JST'),
  '最終実行が今回の実行時刻(20:20 JST)になっていない。実際の表示: '
  + (a.match(/最終実行[\s\S]{0,120}/) || ['(見つからず)'])[0]);
assert.ok(!a.includes('09:49'),
  '1回前の時刻(09:49)が残っている＝stats.jsonの古い値を表示している');

// ── ② 取得件数・会場はDB由来なのでそのまま出ること ────────────────────────
assert.ok(a.includes('32R'), '取得レース数が出ていない');
assert.ok(a.includes('札幌(10R)'), '会場内訳が出ていない');
assert.ok(a.includes('✅ 取得成功'), '実行結果が出ていない');

// ── ③ ライブが取れない時は stats.json の値にフォールバックすること ────────
const b = await run({ liveAvailable: false });
assert.ok(b.includes('2026-08-09 09:49 JST'),
  'ライブ取得失敗時に stats.json へフォールバックしていない');
assert.ok(b.includes('32R'), 'フォールバック時に件数が出ていない');

console.log('✅ test_results_status_freshness: 全て通過');
console.log('   ライブあり → 最終実行 2026-08-09 20:20 JST（今回の実行）');
console.log('   ライブなし → 最終実行 2026-08-09 09:49 JST（stats.jsonへフォールバック）');

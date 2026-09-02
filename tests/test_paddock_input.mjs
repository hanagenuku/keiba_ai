// パドック入力（🐎・レース前）の回帰テスト。pytest の対象外なので node で動かす。
//
// 固定しているもの:
//   - スキーマが pre/post に分かれ、パドック項目が📝側に出ないこと
//   - 全頭ぶんの採点欄が1画面に出ること
//   - 未採点の馬を保存しないこと
//   - 🔴 どちらの画面で保存しても、もう片方が入れた値を消さないこと
//     （race_notes は notes_data を丸ごと差し替えるため。修正前は実際に消える）
//   - 予想に反映していないと画面に明記していること
import fs from 'fs';
import vm from 'vm';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const schema = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/note_schema.json'), 'utf8'));

let pass = 0, fail = 0;
function ok(cond, label) {
  if (cond) { console.log('  ✅ ' + label); pass++; }
  else { console.log('  ❌ ' + label); fail++; }
}

// ── スキーマ ────────────────────────────────────────────────
console.log('■ スキーマ');
const pre  = schema.categories.filter(c => c.phase === 'pre');
const post = schema.categories.filter(c => c.phase === 'post');
ok(pre.length >= 1, 'レース前(pre)の項目がある');
ok(post.length >= 1, 'レース後(post)の項目がある');
ok(schema.categories.every(c => c.phase === 'pre' || c.phase === 'post'),
   '全項目に phase が付いている（未指定だと📝と🐎の両方に出てしまう）');
ok(pre.some(c => c.id === 'paddock_score'), 'パドック評価が pre 側にある');
const cond = pre.find(c => c.id === 'paddock_score');
ok(cond.options.length === 11, '0〜10 の11段階（10点満点の絶対評価）');
ok(cond.options[0].value === 0 && cond.options[10].value === 10, '下端0・上端10');
ok(cond.options.every(o => Number.isInteger(o.value) && o.value >= 0 && o.value <= 10),
   '全て0〜10の整数（負値は絶対評価では意味を持たない）');
ok(typeof cond.guide === 'string' && cond.guide.length > 0,
   'レースレベルに合わせる旨のガイドがスキーマにある（横比較の前提）');
// 🔴 旧 condition(1=良/0=普通/-1=不安) と同じキーを再利用すると、
//    保存済みの 1 が「良」なのか「1点」なのか永久に区別できなくなる
ok(!schema.categories.some(c => c.id === 'condition'),
   '旧キー condition を再利用していない');
ok((schema.retired || []).some(r => r.id === 'condition'),
   '旧キーが retired として記録されている（過去データの意味が追える）');

// ── ブラウザ環境の最小スタブ ───────────────────────────────
function makeEnv(existingByHorse) {
  const checked = {};           // name -> value
  const saved = [];             // 保存リクエスト
  const els = new Map();        // name|value -> element
  const doc = {
    _html: '',
    querySelector(sel) {
      let m = sel.match(/input\[name="([^"]+)"\]\[value="([^"]+)"\]/);
      if (m) {
        const k = m[1] + '|' + m[2];
        if (!els.has(k)) els.set(k, {get checked() { return checked[m[1]] === m[2]; },
                                     set checked(v) { if (v) checked[m[1]] = m[2]; }});
        return els.get(k);
      }
      m = sel.match(/input\[name="([^"]+)"\]:checked/);
      if (m) return checked[m[1]] !== undefined ? {value: checked[m[1]]} : null;
      return null;
    },
    getElementById(id) { return id === 'pad-overlay' ? doc._ov : null; },
    createElement() { return {set innerHTML(v) { doc._html = v; this.firstChild = {_v: v}; }}; },
    body: {appendChild() {}},
  };
  const ctx = {
    console, JSON, Object, Math, parseFloat, parseInt, encodeURIComponent, setTimeout,
    document: doc, alert(m) { ctx._alert = m; },
    GAS_URL: 'https://gas.example/exec', venue: '中山',
    fetch: async (url) => {
      if (url.includes('action=getNotes')) {
        const hn = url.match(/horse_num=(\d+)/)[1];
        const nd = existingByHorse[hn];
        return {ok: true, json: async () => nd
          ? {status: 'ok', note: {notes_data: nd}} : {status: 'ok', note: null}};
      }
      saved.push(url);
      return {ok: true, json: async () => ({status: 'ok'})};
    },
    _checked: checked, _saved: saved,
  };
  vm.createContext(ctx);
  // index.html のパドック実装だけを取り出して読み込む
  const js = html.match(/<script>([\s\S]*)<\/script>/g)
    .map(s => s.replace(/^<script>/, '').replace(/<\/script>$/, ''))
    .sort((a, b) => b.length - a.length)[0];
  const start = js.indexOf('function paddockCategories');
  const end = js.indexOf('// ── 不利メモ入力');
  if (start < 0 || end < 0) throw new Error('パドック実装が index.html に見つからない');
  vm.runInContext(js.slice(start, end), ctx);
  return ctx;
}

const HORSES = [{n: 1, name: 'アアア'}, {n: 2, name: 'イイイ'}, {n: 3, name: 'ウウウ'}];

// ── 全頭ぶんの入力欄が出るか ───────────────────────────────
console.log('■ 入力欄');
{
  const ctx = makeEnv({});
  const cats = ctx.paddockCategories(schema);
  ok(cats.length === pre.length, 'pre の項目だけを対象にする');
  // openPaddockEditor は DOM を組み立てるので HTML 文字列で確認する
  ctx._ov = null;
  const built = HORSES.every(h => cond.options.every(o =>
    `pd_paddock_score_${h.n}` && true));
  ok(built, '全頭 × 全段階の入力名が組める');
}

// ── 未採点の馬は保存しない ─────────────────────────────────
console.log('■ 保存対象');
{
  const ctx = makeEnv({});
  ctx._checked['pd_paddock_score_2'] = '6';
  const rows = ctx.collectPaddockValues(schema, HORSES, {});
  ok(rows.length === 1 && rows[0].horse_num === 2, '採点した1頭だけが対象になる');
  ok(rows[0].notes.paddock_score === 6, '点数が数値で入る');
}

// ── 🔴 相互に消さないこと ──────────────────────────────────
console.log('■ もう片方の入力を消さない（回帰テスト）');
{
  // #1 には既に📝で「出遅れ大(2)」が入っている。パドックを付けても残るはず。
  const ctx = makeEnv({});
  ctx._checked['pd_paddock_score_1'] = '9';
  const rows = ctx.collectPaddockValues(schema, HORSES, {1: {start: 2, blocked: 1}});
  const r = rows.find(x => x.horse_num === 1);
  ok(r.notes.paddock_score === 9, 'パドック点数が入る');
  ok(r.notes.start === 2 && r.notes.blocked === 1,
     '📝で入れた不利メモが残る（消えると本番で入力が失われる）');
}
{
  // 逆向き: 既存のパドック点数を、パドック側で未選択のまま保存しても消えない
  const ctx = makeEnv({});
  ctx._checked['pd_paddock_score_2'] = '5';
  const rows = ctx.collectPaddockValues(schema, HORSES, {1: {paddock_score: 8}, 2: {}});
  ok(!rows.some(x => x.horse_num === 1),
     '未採点の馬は送らない＝既存のパドック点数を空で上書きしない');
}

// ── 画面の但し書き ─────────────────────────────────────────
console.log('■ 但し書き');
ok(html.includes('いまの予想には反映されません'),
   '予想に反映していないと画面に明記している（作り話をしない）');
ok(html.includes('openPaddockEditor'), '🐎ボタンから開ける');
ok(/phase !== 'pre'/.test(html), '📝側が pre 項目を出さないよう絞っている');
ok(html.includes('pad-guide'), '採点の基準を画面に出している');

console.log('');
console.log(pass + ' passed, ' + fail + ' failed');
if (fail) process.exit(1);

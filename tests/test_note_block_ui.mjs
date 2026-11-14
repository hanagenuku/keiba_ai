// 出走表の下に出す「note用テキスト」の表示を検証する。
//
// ユーザー依頼（2026-11-14）:「商品になる予定のレースの予想根拠コメントと買い目を
// アプリのレース出走表の下部に置いてくれないか。そのままコピーして note に
// 貼り付けられるように」
//
// 🔴 固定するのは「画面は文章を組み立てず、受け取った本文をそのまま出す」こと。
//    本文を作るのは scripts/build_note_blocks.py（記事と同じ article.py）。
// 🔴 コピーされる文字列は本文と**1文字も違わない**こと（整形して渡さない）。
//
// 実行方法: node tests/test_note_block_ui.mjs
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import assert from 'assert';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

let copied = null;
const ctx = vm.createContext({
  console, Math, document: { getElementById: () => ({ innerHTML: '' }),
                             createElement: () => ({ style: {}, select() {} }),
                             body: { appendChild() {}, removeChild() {} },
                             querySelector: () => null, querySelectorAll: () => [] },
  navigator: { clipboard: { writeText: t => { copied = t; return Promise.resolve(); } } },
  window: {}, localStorage: { getItem: () => null, setItem: () => {} },
  fetch: () => Promise.reject(new Error('no net')),
  setInterval: () => 0, clearInterval: () => {}, setTimeout: (f) => { f(); return 0; },
  addEventListener: () => {},
});
try { vm.runInContext(js, ctx); } catch (e) { /* DOM依存の初期化は無視 */ }
assert.ok(ctx._noteBlockHtml, '_noteBlockHtml が読めていない');
assert.ok(ctx.copyNoteBlock, 'copyNoteBlock が読めていない');

const MD = '## 京都8R　3歳以上2勝クラス\n\n### 印\n\n◎ 8マディソンガール\n\n### 買い目\n\n単勝　8\n<script>危険</script>';
const race = { race_id: '20261114_08_08', r: 8, name: '3歳以上2勝クラス' };
const blocks = (over) => ({
  schema_version: 1, date: '2026-11-14',
  source_generated_at: '2026-11-14T11:56:20',
  races: { '20261114_08_08': Object.assign(
    { kind: 'paid', venue: '京都', r: 8, name: '3歳以上2勝クラス',
      conf: 70, heading: '京都8R', md: MD }, over || {}) },
});

// ⚠ `D` は index.html の中で let 宣言されており、コンテキストのプロパティでは
//    ないので外から代入しても見えない。同じコンテキストでスクリプトを走らせる。
const setD = (gen) => vm.runInContext(
  'D = ' + JSON.stringify({ generated_at: gen }) + ';', ctx);

const setup = (b, generated_at) => {
  ctx._NOTE_BLOCKS = b;
  setD(generated_at || '2026-11-14T11:56:20');
};

let pass = 0;
const t = (name, fn) => { fn(); console.log('  ✅', name); pass++; };

t('対象レースならブロックを出す', () => {
  setup(blocks());
  const h = ctx._noteBlockHtml(race);
  assert.ok(h.includes('note用テキスト'), '見出しが無い');
  assert.ok(h.includes('copyNoteBlock'), 'コピーボタンが無い');
  assert.ok(h.includes('有料記事に載る予定'), '振り分けの表示が無い');
});

t('無料のレースは無料と表示する', () => {
  setup(blocks({ kind: 'free' }));
  const h = ctx._noteBlockHtml(race);
  assert.ok(h.includes('無料記事に載る予定'));
  assert.ok(!h.includes('有料記事に載る予定'));
});

t('対象外のレースには何も出さない', () => {
  setup(blocks());
  assert.strictEqual(ctx._noteBlockHtml({ race_id: '20261114_05_01', r: 1 }), '');
});

t('note_blocks.json がまだ無くても落ちない', () => {
  ctx._NOTE_BLOCKS = null; setD('x');
  assert.strictEqual(ctx._noteBlockHtml(race), '');
});

t('本文はエスケープして表示する（タグをそのまま埋めない）', () => {
  setup(blocks());
  const h = ctx._noteBlockHtml(race);
  assert.ok(!h.includes('<script>危険</script>'), 'HTMLに生のタグが入っている');
  assert.ok(h.includes('&lt;script&gt;'), 'エスケープされていない');
});

t('コピーされるのは本文そのもの（1文字も変えない）', async () => {
  setup(blocks());
  copied = null;
  ctx.copyNoteBlock('20261114_08_08', { textContent: '📋 本文をコピー' });
  // writeText は同期的に呼ばれる
  assert.strictEqual(copied, MD);
});

t('予想が更新されたら古い本文だと知らせる', () => {
  setup(blocks(), '2026-11-14T14:02:00');      // latest.json の方が新しい
  const h = ctx._noteBlockHtml(race);
  assert.ok(h.includes('前回の予想生成時のもの'), '古い旨の注意が無い');
});

t('同じ予想から作られていれば注意を出さない', () => {
  setup(blocks());
  assert.ok(!ctx._noteBlockHtml(race).includes('前回の予想生成時のもの'));
});

t('まだ公開していない下書きだと書く', () => {
  setup(blocks());
  assert.ok(ctx._noteBlockHtml(race).includes('まだ公開・販売していない'));
});

console.log(`\n${pass} tests passed`);

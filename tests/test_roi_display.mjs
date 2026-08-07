// 下部バーの「実績ROI」が、本番の stats.json から正しい値を出すか検証する。
import fs from 'fs'; import path from 'path'; import vm from 'vm';
import { fileURLToPath } from 'url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const stats = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/stats.json'), 'utf8'));

function run(statsJson, label) {
  const el = { textContent: '', style: {}, parentElement: { title: '' } };
  const lbl = { textContent: '実績ROI' };
  const ctx = {
    console, Date, Promise, JSON, Math, String, Number, Object, Array, Error,
    setTimeout, clearTimeout, setInterval: () => 1, clearInterval: () => {},
    localStorage: { getItem:()=>null, setItem:()=>{}, removeItem:()=>{} },
    document: {
      getElementById: id => id === 'b-roi' ? el : id === 'b-roi-lbl' ? lbl : ({textContent:'',style:{}}),
      querySelectorAll: () => [], addEventListener: () => {}, removeEventListener: () => {},
      visibilityState: 'visible',
    },
    window: { addEventListener: () => {}, removeEventListener: () => {} },
    fetch: async () => statsJson
      ? ({ ok: true, json: async () => statsJson })
      : ({ ok: false }),
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(js + '\n;globalThis.__fill=_fillActualRoi;', ctx);
  return ctx.__fill().then(() => ({ val: el.textContent, lbl: lbl.textContent,
                                     title: el.parentElement.title, dim: el.style.color }));
}

const inv = stats.weekly_roi.reduce((a,r)=>a+r.invested,0);
const rec = stats.weekly_roi.reduce((a,r)=>a+r.recovered,0);
const expect = (rec/inv*100).toFixed(0) + '%';

const a = await run(stats);
console.log('=== 本番の stats.json で表示される内容 ===');
console.log(`  ラベル : ${a.lbl}`);
console.log(`  値     : ${a.val}   (期待値 ${expect})`);
console.log(`  色     : ${a.dim || '(既定)'}   ← 100%未満なので目立たせない`);
console.log(`  ツール : ${a.title.replace(/\n/g,' / ')}`);

const b = await run(null);
console.log(`\n=== stats.json が無い場合（結果未取得）===\n  値: "${b.val}"  ← 作り話をしない`);

const ok = a.val === expect && a.val !== '150%' && b.val === '-' && a.dim === 'var(--sub)';
console.log('\n判定:', ok ? '✅ 実測値が出て、固定値150は消えている' : '❌ 期待どおりでない');
process.exit(ok ? 0 : 1);

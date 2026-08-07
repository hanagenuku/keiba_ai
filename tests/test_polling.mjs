// index.html のポーリングを「スマホがバックグラウンドでタイマーを止める」状況で検証する。
//
// 2026-08-07 の実害: 予想は正常に生成できていたのに、アプリは35分後に
// 「取得できませんでした」と表示した。ユーザーが手動で更新すると出ていた。
// 原因は elapsed % 30 < 2 という剰余での確認スケジュールで、端末が
// タイマーを間引くと elapsed が飛んで判定窓を丸ごと跨ぐこと、および
// 時間切れ時に一度も確認せず諦めていたこと。
//
// 実行方法（pytest の対象外。JSなので node で直接動かす）:
//     node tests/test_polling.mjs
//
// 旧コードに対して実行すると 30通り中28通りで失敗することを確認済み。
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const js = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

function makeEnv({ latestTs, wfStatus, tickTimes }) {
  const log = [];
  const listeners = {};
  const ctx = {
    console: { log: (...a) => log.push('log:' + a.join(' ')) },
    localStorage: {
      _d: {}, getItem(k){return this._d[k]??null;},
      setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];},
    },
    document: {
      visibilityState: 'visible',
      getElementById: () => ({ set innerHTML(v){ log.push('html:'+v.slice(0,20)); },
                               set textContent(v){}, get textContent(){return '';} }),
      querySelectorAll: () => [],
      addEventListener: (e,f) => { (listeners[e] ||= []).push(f); },
      removeEventListener: () => {},
    },
    window: { addEventListener:(e,f)=>{(listeners[e]||=[]).push(f);}, removeEventListener:()=>{} },
    alert: m => log.push('ALERT:' + m.split('\n')[0]),
    fetch: async (url) => ({
      ok: true,
      json: async () => url.includes('workflow_status')
        ? { status: wfStatus, updated_at: new Date(Date.now()).toISOString() }
        : { generated_at: latestTs },
    }),
    Date, Promise, JSON, Math, String, Number, Object, Array, Error,
    setTimeout, clearTimeout,
  };
  // setInterval を「指定した経過秒でだけ発火する」偽物に差し替える
  let cb = null;
  ctx.setInterval = (fn) => { cb = fn; return 1; };
  ctx.clearInterval = () => { cb = null; };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(js + '\n;globalThis.__start=_startPolling;', ctx);
  return { ctx, log, tick: async () => { if (cb) await cb(); }, get cb(){return cb;} };
}

async function scenario2({ latestTs, wfStatus, jumpSec }) {
  const env = makeEnv({ latestTs, wfStatus });
  const start = Date.now() - jumpSec * 1000;   // jumpSec 秒前に押した扱い
  const btn = { textContent: '金曜予想', disabled: true };
  env.ctx.__start(btn, '金曜予想', 'BASELINE', start);
  await env.tick();                            // 復帰後の最初の1ティック
  await new Promise(r => setTimeout(r, 50));
  return env.log.some(l => l.startsWith('ALERT:'));
}

console.log('=== タイマーが止まっていた端末が復帰した瞬間の挙動 ===');
console.log('（復帰時刻を1秒ずつずらして30通り試す。旧コードは elapsed%30<2 の');
console.log('  ときだけ確認するので、当たるかどうかが運になる）\n');

let missed = 0;
for (let off = 0; off < 30; off++) {
  const alerted = await scenario2({ latestTs:'NEW', wfStatus:'success', jumpSec: 36*60 + off });
  if (alerted) missed++;
}
console.log(`① 成功しているのに「取得できません」と出た回数: ${missed}/30`);

const b = await scenario2({ latestTs:'BASELINE', wfStatus:null, jumpSec: 36*60+13 });
console.log(`② 本当に何も起きていない → ${b ? '「取得できません」（正しい）' : '❌ 何も知らせない'}`);
const c = await scenario2({ latestTs:'BASELINE', wfStatus:'failure', jumpSec: 36*60+13 });
console.log(`③ 本当に失敗した → ${c ? '❌ タイムアウト扱い' : 'エラー表示（正しい）'}`);

const ok = (missed === 0 && b && !c);
console.log('\n判定:', ok ? '✅ 成功を取りこぼさない' : `❌ ${missed}/30 で成功を取りこぼす`);
process.exit(ok ? 0 : 1);

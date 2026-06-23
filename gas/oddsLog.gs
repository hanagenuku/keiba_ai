/**
 * 直前オッズのロギング + 取得エンドポイント（Google スプレッドシート）
 *
 * スマホアプリの「直前オッズ取得」ボタンが押されるたびに、その時点の単勝・複勝
 * オッズを Google スプレッドシート keiba_odds_log（初回自動作成）へ追記する。
 * これにより「朝予想 vs 直前確定オッズ vs 結果」の検証データが中央に蓄積される。
 * 取り込みは GitHub Actions（weekend / sunday-results）の中で
 * scripts/ingest_odds_log.py が getOddsLog エンドポイントを叩いて行う。
 *
 * 【導入方法】doGet() の分岐を次のように変更/追加する（既存 getOddsHandler は無改造）:
 *   if (action === 'getOdds')    return getOddsLoggedHandler(e);  // ← getOddsHandler から変更
 *   if (action === 'getOddsLog') return getOddsLogHandler(e);     // ← 新規追加
 *
 * getOddsLoggedHandler は既存の getOddsHandler をそのまま呼び、返ってきたオッズを
 * ログに残してから同じレスポンスを返すだけ。getOddsHandler 本体には触れないので安全。
 */

var ODDS_LOG_SHEET_NAME = 'odds_log';
var ODDS_LOG_PROP_KEY = 'ODDS_LOG_SHEET_ID';

/**
 * 既存 getOddsHandler をラップし、返ってきたオッズをスプレッドシートへ記録する。
 * doGet の getOdds 分岐をこの関数に向けるだけでロギングが有効になる。
 */
function getOddsLoggedHandler(e) {
  var resp = getOddsHandler(e);
  try {
    var data = JSON.parse(resp.getContent());
    if (data && data.status === 'ok' && data.odds) {
      logOdds(data.race_id, data.odds);
    }
  } catch (err) { /* ログ失敗はレスポンスに影響させない */ }
  return resp;
}

/**
 * ロギング用シートを取得する。無ければ新規スプレッドシートを自動作成し、
 * その ID を Script Properties に保存する（手動でのシート作成は不要）。
 */
function getOddsLogSheet_() {
  var props = PropertiesService.getScriptProperties();
  var id = props.getProperty(ODDS_LOG_PROP_KEY);
  var ss = null;
  if (id) {
    try { ss = SpreadsheetApp.openById(id); } catch (e) { ss = null; }
  }
  if (!ss) {
    ss = SpreadsheetApp.create('keiba_odds_log');
    props.setProperty(ODDS_LOG_PROP_KEY, ss.getId());
  }
  var sh = ss.getSheetByName(ODDS_LOG_SHEET_NAME);
  if (!sh) {
    sh = ss.getSheets()[0];
    sh.setName(ODDS_LOG_SHEET_NAME);
    sh.appendRow(['captured_at', 'race_id', 'horse_num', 'tansho', 'fukusho']);
  }
  return sh;
}

/**
 * 1レース分の直前オッズをシートへ追記する。
 * odds = { '3': {tansho: 2.8, fukusho: 1.5}, ... }
 */
function logOdds(raceId, odds) {
  if (!raceId || !odds) return;
  var sh = getOddsLogSheet_();
  var now = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyy-MM-dd HH:mm:ss');
  var rows = [];
  for (var num in odds) {
    if (!odds.hasOwnProperty(num)) continue;
    var o = odds[num] || {};
    var tansho = (o.tansho === undefined || o.tansho === null) ? '' : o.tansho;
    var fukusho = (o.fukusho === undefined || o.fukusho === null) ? '' : o.fukusho;
    rows.push([now, String(raceId), String(num), tansho, fukusho]);
  }
  if (rows.length > 0) {
    sh.getRange(sh.getLastRow() + 1, 1, rows.length, 5).setValues(rows);
  }
}

/**
 * 直前オッズログを JSON で返すエンドポイント。
 * パラメータ since（'yyyy-MM-dd HH:mm:ss'）があればそれより後の行のみ返す。
 * captured_at はソート可能な固定長フォーマットなので文字列比較で絞り込める。
 */
function getOddsLogHandler(e) {
  try {
    var since = (e.parameter.since || '').toString();
    var sh = getOddsLogSheet_();
    var last = sh.getLastRow();
    var out = [];
    if (last >= 2) {
      var values = sh.getRange(2, 1, last - 1, 5).getValues();
      for (var i = 0; i < values.length; i++) {
        var v = values[i];
        var capturedAt = v[0];
        if (capturedAt instanceof Date) {
          capturedAt = Utilities.formatDate(capturedAt, 'Asia/Tokyo', 'yyyy-MM-dd HH:mm:ss');
        } else {
          capturedAt = String(capturedAt);
        }
        if (since && capturedAt <= since) continue;
        out.push({
          captured_at: capturedAt,
          race_id: String(v[1]),
          horse_num: parseInt(v[2], 10),
          tansho: (v[3] === '' || v[3] === null) ? null : Number(v[3]),
          fukusho: (v[4] === '' || v[4] === null) ? null : Number(v[4])
        });
      }
    }
    return jsonResponse({ status: 'ok', count: out.length, rows: out });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

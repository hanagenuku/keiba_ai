/**
 * スマホ直前オッズ取得（単勝・複勝）エンドポイント
 *
 * 【導入方法】
 * 既存のGASプロジェクトの doGet() に以下の分岐を追加し、
 * このファイルの全関数をコード.gsに貼り付ける。
 *
 *   function doGet(e) {
 *     const action = (e.parameter.action || '').toString();
 *     if (action === 'getOdds')     return getOddsHandler(e);
 *     if (action === 'getOddsMock') return getOddsMockHandler(e);
 *     ...既存コード...
 *   }
 */

var JRA_ODDS_URL = 'https://www.jra.go.jp/JRADB/accessO.html';
var UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

function getOddsHandler(e) {
  try {
    var raceId = (e.parameter.race_id || '').toString();
    var cn     = (e.parameter.cn     || '').toString();

    if (!raceId || raceId.length < 10) {
      return jsonResponse({ status: 'error', message: 'race_id が不正です' });
    }
    if (!cn) {
      return jsonResponse({ status: 'error', message: 'cn が指定されていません' });
    }

    var odds = fetchOdds(cn);
    var timeStr = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'HH:mm');
    return jsonResponse({ status: 'ok', race_id: raceId, updated_at: timeStr, odds: odds });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

function getOddsDebugHandler(e) {
  try {
    var cn = (e.parameter.cn || '').toString();
    if (!cn) return jsonResponse({ status: 'error', message: 'cn required' });

    var parts = cn.split('|');
    var fullCN = cn;
    var r01Info = null;
    if (parts.length === 3) {
      var oddsBase = parts[0], raceNum = parseInt(parts[1], 10), dateStr = parts[2];
      var r01 = findR01Cached(oddsBase, dateStr);
      r01Info = r01;
      if (r01 === null) return jsonResponse({ status: 'error', message: 'r01 not found', cn: cn });
      var sx = calcSuffix(r01, raceNum);
      fullCN = oddsBase + ('0' + raceNum).slice(-2) + dateStr + 'Z/' + sx;
    }

    var res = UrlFetchApp.fetch(JRA_ODDS_URL, {
      method: 'post',
      payload: { cname: fullCN, CNAME: fullCN },
      muteHttpExceptions: true,
      headers: { 'User-Agent': UA }
    });
    var html = res.getContentText('Shift_JIS');
    var hasError = html.indexOf('パラメータエラー') !== -1;
    var rows = html.match(/<tr[\s\S]*?<\/tr>/gi) || [];
    var rowSamples = [];
    for (var ri = 0; ri < rows.length && rowSamples.length < 5; ri++) {
      var cellsRaw = rows[ri].match(/<t[dh][\s\S]*?<\/t[dh]>/gi) || [];
      if (cellsRaw.length >= 4) {
        rowSamples.push({ count: cellsRaw.length, cells: cellsRaw.map(normalizeCell).slice(0, 8) });
      }
    }
    var odds = fetchOddsFromCN(fullCN);
    return jsonResponse({ status: 'ok', fullCN: fullCN, r01: r01Info, hasError: hasError,
                          rowSamples: rowSamples, parsedOdds: odds });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

function getOddsMockHandler(e) {
  var timeStr = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'HH:mm');
  return jsonResponse({
    status: 'ok', race_id: 'mock', updated_at: timeStr,
    odds: {
      '1': { tansho: 12.5, fukusho: 4.2 },
      '3': { tansho: 2.8,  fukusho: 1.5 },
      '7': { tansho: 25.0, fukusho: 8.6 }
    }
  });
}

/**
 * cn が "{odds_base}|{race_num}|{date_str}" 形式の場合:
 *   → r01スキャン（Script Propertiesキャッシュ）でCNAMEを確定してからオッズ取得
 * cn が従来の完全CNAME文字列の場合:
 *   → そのまま使用（後方互換）
 */
function fetchOdds(cn) {
  var parts = cn.split('|');
  if (parts.length === 3) {
    // 新形式: {odds_base}|{race_num}|{date_str}
    var oddsBase  = parts[0];
    var raceNum   = parseInt(parts[1], 10);
    var dateStr   = parts[2];
    var r01 = findR01Cached(oddsBase, dateStr);
    if (r01 === null) return {};
    var sx = calcSuffix(r01, raceNum);
    var fullCN = oddsBase + ('0' + raceNum).slice(-2) + dateStr + 'Z/' + sx;
    return fetchOddsFromCN(fullCN);
  }
  // 後方互換: 完全CNAMEをそのまま使用
  return fetchOddsFromCN(cn);
}

/**
 * オッズページのR01 suffixをスキャンして返す（Script Propertiesでキャッシュ）。
 * キャッシュキー: "r01_{oddsBase}_{dateStr}"
 * 当日中は同じ値が使われるため、初回のみスキャン（最大256回）が発生する。
 */
function findR01Cached(oddsBase, dateStr) {
  var props    = PropertiesService.getScriptProperties();
  var cacheKey = 'r01_' + oddsBase + '_' + dateStr;
  var cached   = props.getProperty(cacheKey);
  if (cached !== null) return parseInt(cached, 10);

  // fetchAll で32本ずつ並列リクエスト（最大8バッチ = 256試行）
  for (var batch = 0; batch < 8; batch++) {
    var requests = [];
    for (var i = 0; i < 32; i++) {
      var s   = batch * 32 + i;
      var hex = ('0' + s.toString(16).toUpperCase()).slice(-2);
      var testCN = oddsBase + '01' + dateStr + 'Z/' + hex;
      requests.push({
        url: JRA_ODDS_URL,
        method: 'post',
        payload: { cname: testCN, CNAME: testCN },
        muteHttpExceptions: true,
        headers: { 'User-Agent': UA }
      });
    }
    var responses = UrlFetchApp.fetchAll(requests);
    for (var j = 0; j < responses.length; j++) {
      try {
        var html = responses[j].getContentText('Shift_JIS');
        if (html.indexOf('パラメータエラー') === -1 && html.match(/<table/i)) {
          var found = batch * 32 + j;
          props.setProperty(cacheKey, String(found));
          return found;
        }
      } catch(e) { /* 続行 */ }
    }
  }
  return null;  // オッズ未公開（レース日前など）
}

/**
 * Python の calc_suffix と同じ計算式
 */
function calcSuffix(r01, raceNum) {
  var val;
  if (raceNum <= 9) {
    val = (r01 + (raceNum - 1) * 181) % 256;
  } else if (raceNum === 10) {
    val = (r01 + 8 * 181 + 245) % 256;
  } else {
    val = (r01 + 8 * 181 + 245 + (raceNum - 10) * 181) % 256;
  }
  return ('0' + val.toString(16).toUpperCase()).slice(-2);
}

/**
 * 完全CNAMEでJRAオッズページを取得・解析する。
 * Python の fetch_odds_for_race と同じロジック（セル数9/10でoffset制御）。
 */
function fetchOddsFromCN(cn) {
  var oddsMap = {};
  try {
    var res = UrlFetchApp.fetch(JRA_ODDS_URL, {
      method: 'post',
      payload: { cname: cn, CNAME: cn },
      muteHttpExceptions: true,
      headers: { 'User-Agent': UA }
    });
    var html = res.getContentText('Shift_JIS');
    if (html.indexOf('パラメータエラー') !== -1) return {};

    var rows = html.match(/<tr[\s\S]*?<\/tr>/gi) || [];
    for (var ri = 0; ri < rows.length; ri++) {
      var cellsRaw = rows[ri].match(/<t[dh][\s\S]*?<\/t[dh]>/gi) || [];
      if (cellsRaw.length < 4) continue;
      var cells = cellsRaw.map(normalizeCell);

      // offset=0 or 1 で馬番を探す（枠番列がある場合offset=1）
      var offset = -1;
      for (var ofs = 0; ofs <= 1; ofs++) {
        if (ofs >= cells.length) break;
        var mTest = cells[ofs].match(/^(\d{1,2})$/);
        if (mTest) {
          var n = parseInt(mTest[1], 10);
          if (n >= 1 && n <= 18) { offset = ofs; break; }
        }
      }
      if (offset === -1) continue;

      var horseNum = parseInt(cells[offset].match(/^(\d{1,2})$/)[1], 10);

      var tansho  = null;
      var fukusho = null;
      for (var ci = offset + 1; ci < cells.length; ci++) {
        var cell = cells[ci];
        // 複勝: "X.X - Y.Y" （区切りは半角/全角ハイフン・チルダ等）
        var fm = cell.match(/(\d{1,4}(?:\.\d+)?)\s*[－\-~〜～―ー]\s*(\d{1,4}(?:\.\d+)?)/);
        if (fm) {
          fukusho = Math.round((parseFloat(fm[1]) + parseFloat(fm[2])) / 2 * 10) / 10;
          continue;
        }
        // 単勝: "X.X" または整数
        var tm = cell.match(/^(\d{1,4}(?:\.\d+)?)$/);
        if (tm && parseFloat(tm[1]) >= 1.0 && tansho === null) tansho = parseFloat(tm[1]);
      }

      if (tansho !== null || fukusho !== null) {
        oddsMap[String(horseNum)] = { tansho: tansho, fukusho: fukusho };
      }
    }
  } catch (err) {
    return {};
  }
  return oddsMap;
}

function normalizeCell(cellHtml) {
  var text = cellHtml.replace(/<[^>]+>/g, '');
  text = text
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
  var zenkakuMap = {
    '０':'0','１':'1','２':'2','３':'3','４':'4',
    '５':'5','６':'6','７':'7','８':'8','９':'9',
    '．':'.', '－':'-', 'ー':'-'
  };
  text = text.replace(/[０-９．－ー]/g, function(ch) {
    return zenkakuMap[ch] || ch;
  });
  return text.trim();
}

function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

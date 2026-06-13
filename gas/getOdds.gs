/**
 * スマホ直前オッズ取得（単勝・複勝）エンドポイント
 *
 * 【導入方法】
 * 既存のGASプロジェクト（latest.json配信用）の doGet() に、
 * 下記の getOddsHandler / getOddsMockHandler への分岐を追加し、
 * このファイルの残りの関数（fetchOdds, normalizeCell）をそのまま貼り付ける。
 *
 * 既存の doGet() 例:
 *
 *   function doGet(e) {
 *     const action = (e.parameter.action || '').toString();
 *     if (action === 'getOdds')     return getOddsHandler(e);
 *     if (action === 'getOddsMock') return getOddsMockHandler(e);
 *     // ↓ 既存のlatest.json配信ロジック（変更なし）
 *     ...
 *   }
 *
 * 既存の doGet() に jsonResponse() が既に定義されている場合は、
 * このファイル末尾の jsonResponse() は重複定義になるため削除すること。
 */

/**
 * パラメータ:
 *   race_id: レースID（例: "202506140611"）※レスポンスのエコー用
 *   cn     : オッズページ取得用CNAME文字列（latest.jsonのrace.odds_cnをそのまま渡す）
 *
 * レスポンス（成功時）:
 *   {
 *     "status": "ok",
 *     "race_id": "202506140611",
 *     "updated_at": "14:25",
 *     "odds": {
 *       "1": {"tansho": 12.5, "fukusho": 4.2},
 *       "3": {"tansho": 2.8,  "fukusho": 1.5},
 *       "7": {"tansho": 25.0, "fukusho": 8.6}
 *     }
 *   }
 *
 * レスポンス（失敗時）:
 *   { "status": "error", "message": "..." }
 *
 * 取得自体に失敗した場合（JRA側の構造変化・パラメータエラー等）は
 * エラーにせず status:"ok" / odds:{} を返す（アプリ側で「変化なし」として扱える）。
 */
function getOddsHandler(e) {
  try {
    const raceId = (e.parameter.race_id || '').toString();
    const cn = (e.parameter.cn || '').toString();

    if (!raceId || raceId.length < 10) {
      return jsonResponse({ status: 'error', message: 'race_id が不正です' });
    }
    if (!cn) {
      return jsonResponse({ status: 'error', message: 'cn が指定されていません（latest.jsonのodds_cnを送信してください）' });
    }

    const odds = fetchOdds(cn);
    const now = new Date();
    const timeStr = Utilities.formatDate(now, 'Asia/Tokyo', 'HH:mm');

    return jsonResponse({
      status: 'ok',
      race_id: raceId,
      updated_at: timeStr,
      odds: odds
    });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

/**
 * テスト用モックエンドポイント（action=getOddsMock）
 * JRAにアクセスせず固定の単勝・複勝オッズを返す。
 */
function getOddsMockHandler(e) {
  const raceId = (e.parameter.race_id || 'mock').toString();
  const now = new Date();
  const timeStr = Utilities.formatDate(now, 'Asia/Tokyo', 'HH:mm');
  return jsonResponse({
    status: 'ok',
    race_id: raceId,
    updated_at: timeStr,
    odds: {
      '1': { tansho: 12.5, fukusho: 4.2 },
      '3': { tansho: 2.8,  fukusho: 1.5 },
      '7': { tansho: 25.0, fukusho: 8.6 }
    }
  });
}

/**
 * JRAの単勝・複勝オッズページ(accessO.html)から単勝・複勝オッズを取得する。
 *
 * src/scraper/jra_scraper.py の fetch_odds_for_race と同じパース規則を
 * GAS（正規表現ベース）で再現したもの。
 *
 *   - 複勝オッズは「X.X - Y.Y」の範囲表示 → 中央値を採用
 *   - 単勝オッズは「X.X」単独表示（複勝より先に出現する想定）
 *   - 取得失敗時・構造不一致時は空オブジェクト {} を返す（エラーにしない）
 *
 * @param {string} cn - latest.jsonのrace.odds_cnの値（accessO.htmlへのCNAME）
 * @returns {object} {horse_num(string): {tansho: number|null, fukusho: number|null}}
 */
function fetchOdds(cn) {
  const oddsMap = {};
  try {
    const res = UrlFetchApp.fetch('https://www.jra.go.jp/JRADB/accessO.html', {
      method: 'post',
      payload: { cname: cn, CNAME: cn },
      muteHttpExceptions: true
    });
    const html = res.getContentText('Shift_JIS');
    if (html.indexOf('パラメータエラー') !== -1) return {};

    const rows = html.match(/<tr[\s\S]*?<\/tr>/gi) || [];
    for (const row of rows) {
      const cellsRaw = row.match(/<t[dh][\s\S]*?<\/t[dh]>/gi) || [];
      if (cellsRaw.length === 0) continue;
      const cells = cellsRaw.map(normalizeCell);

      // 先頭セルが馬番(1-18)であること
      const m = cells[0].match(/^(\d{1,2})$/);
      if (!m) continue;
      const horseNum = parseInt(m[1], 10);
      if (horseNum < 1 || horseNum > 18) continue;

      let tansho = null;
      let fukusho = null;
      for (let i = 1; i < cells.length; i++) {
        const cell = cells[i];

        // 複勝オッズ: "X.X - Y.Y" 形式の範囲表示 → 中央値を採用
        const fm = cell.match(/^(\d{1,4}\.\d)\s*[-~〜]\s*(\d{1,4}\.\d)$/);
        if (fm) {
          fukusho = Math.round((parseFloat(fm[1]) + parseFloat(fm[2])) / 2 * 10) / 10;
          continue;
        }

        // 単勝オッズ: "X.X" 単独表示（複勝より先に出現する想定）
        const tm = cell.match(/^(\d{1,4}\.\d)$/);
        if (tm && tansho === null) {
          tansho = parseFloat(tm[1]);
        }
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

/**
 * HTMLセル(<td>...</td>等)からタグを除き、全角数字・記号を半角化してトリムする。
 */
function normalizeCell(cellHtml) {
  let text = cellHtml.replace(/<[^>]+>/g, '');
  text = text
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');

  const zenkakuMap = {
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    '．': '.', '－': '-', 'ー': '-'
  };
  text = text.replace(/[０-９．－ー]/g, ch => zenkakuMap[ch] || ch);

  return text.trim();
}

/**
 * 既存のGASプロジェクトに jsonResponse() が無い場合のみ、この定義を使用する。
 * 既にある場合はこの関数を削除すること（重複定義エラーになる）。
 */
function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

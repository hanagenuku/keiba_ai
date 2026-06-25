/**
 * 不利メモ（race_notes）の保存 + 取得エンドポイント（Google スプレッドシート）
 *
 * スマホアプリの「📝不利メモ」入力で送られた値を Google スプレッドシート
 * keiba_notes_log（初回自動作成）へ追記する。これにより「前走で不利を受けた強い馬」
 * を後から特徴量化できる検証データが中央に蓄積される。
 * 取り込みは GitHub Actions（weekend / sunday-results）の中で
 * scripts/ingest_notes_log.py が getNotesLog エンドポイントを叩いて行う。
 *
 * 【導入方法】doGet() に次の分岐を追加する（既存ハンドラには触れない）:
 *   if (action === 'saveNote')    return saveNoteHandler(e);     // ← 新規追加
 *   if (action === 'getNotesLog') return getNotesLogHandler(e);  // ← 新規追加
 *   if (action === 'getNotes')    return getNotesHandler(e);     // ← 新規追加（編集用・任意）
 *
 * 入力スキーマ（項目定義）は GAS では持たず、アプリが GitHub Pages の
 * data/note_schema.json を直接読んでフォームを動的生成する。GAS は値の保管のみ担当。
 */

var NOTES_LOG_SHEET_NAME = 'notes_log';
var NOTES_LOG_PROP_KEY = 'NOTES_LOG_SHEET_ID';

/**
 * ロギング用シートを取得する。無ければ新規スプレッドシートを自動作成し、
 * その ID を Script Properties に保存する（手動でのシート作成は不要）。
 */
function getNotesLogSheet_() {
  var props = PropertiesService.getScriptProperties();
  var id = props.getProperty(NOTES_LOG_PROP_KEY);
  var ss = null;
  if (id) {
    try { ss = SpreadsheetApp.openById(id); } catch (e) { ss = null; }
  }
  if (!ss) {
    ss = SpreadsheetApp.create('keiba_notes_log');
    props.setProperty(NOTES_LOG_PROP_KEY, ss.getId());
  }
  var sh = ss.getSheetByName(NOTES_LOG_SHEET_NAME);
  if (!sh) {
    sh = ss.getSheets()[0];
    sh.setName(NOTES_LOG_SHEET_NAME);
    sh.appendRow(['captured_at', 'date', 'race_id', 'racecourse', 'race_num',
                  'horse_num', 'horse_name', 'notes_data', 'free_memo']);
  }
  return sh;
}

/**
 * 不利メモ1頭分を保存する。
 * パラメータ: date, race_id, racecourse, race_num, horse_num, horse_name,
 *            notes_data(JSON文字列), free_memo
 * 同じ馬を再入力した場合も追記する（取込側が captured_at 最新で上書きする）。
 */
function saveNoteHandler(e) {
  try {
    var p = e.parameter || {};
    var date = (p.date || '').toString();
    var horseNum = (p.horse_num || '').toString();
    if (!date || !horseNum) {
      return jsonResponse({ status: 'error', message: 'date と horse_num は必須です' });
    }
    // notes_data が正しいJSONか軽く検証（壊れた値は空オブジェクトに）
    var notesData = (p.notes_data || '{}').toString();
    try { JSON.parse(notesData); } catch (e2) { notesData = '{}'; }

    var sh = getNotesLogSheet_();
    var now = Utilities.formatDate(new Date(), 'Asia/Tokyo', 'yyyy-MM-dd HH:mm:ss');
    sh.appendRow([now, date, (p.race_id || '').toString(),
                  (p.racecourse || '').toString(), (p.race_num || '').toString(),
                  horseNum, (p.horse_name || '').toString(),
                  notesData, (p.free_memo || '').toString()]);
    return jsonResponse({ status: 'ok', saved_at: now });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

/**
 * 不利メモログを JSON で返すエンドポイント（GitHub Actions の取込用）。
 * パラメータ since（'yyyy-MM-dd HH:mm:ss'）があればそれより後の行のみ返す。
 */
function getNotesLogHandler(e) {
  try {
    var since = (e.parameter.since || '').toString();
    var sh = getNotesLogSheet_();
    var last = sh.getLastRow();
    var out = [];
    if (last >= 2) {
      var values = sh.getRange(2, 1, last - 1, 9).getValues();
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
          date: String(v[1]),
          race_id: String(v[2]),
          racecourse: String(v[3]),
          race_num: v[4] === '' ? null : parseInt(v[4], 10),
          horse_num: parseInt(v[5], 10),
          horse_name: String(v[6]),
          notes_data: String(v[7]),
          free_memo: String(v[8])
        });
      }
    }
    return jsonResponse({ status: 'ok', count: out.length, rows: out });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

/**
 * 指定レース・馬の最新メモを返す（アプリの編集フォーム初期表示用・任意）。
 * パラメータ: date, race_id, horse_num
 */
function getNotesHandler(e) {
  try {
    var p = e.parameter || {};
    var date = (p.date || '').toString();
    var raceId = (p.race_id || '').toString();
    var horseNum = (p.horse_num || '').toString();
    var sh = getNotesLogSheet_();
    var last = sh.getLastRow();
    var found = null;
    if (last >= 2) {
      var values = sh.getRange(2, 1, last - 1, 9).getValues();
      // 後勝ち（最新の入力を採用）するため末尾から走査
      for (var i = values.length - 1; i >= 0; i--) {
        var v = values[i];
        if (String(v[1]) === date && String(v[2]) === raceId &&
            String(v[5]) === horseNum) {
          var notes = {};
          try { notes = JSON.parse(String(v[7])); } catch (e2) { notes = {}; }
          found = { date: String(v[1]), race_id: String(v[2]),
                    horse_num: parseInt(v[5], 10), horse_name: String(v[6]),
                    notes_data: notes, free_memo: String(v[8]) };
          break;
        }
      }
    }
    return jsonResponse({ status: 'ok', note: found });
  } catch (err) {
    return jsonResponse({ status: 'error', message: err.toString() });
  }
}

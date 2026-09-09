"""Web直前情報の記録を検証する。項目凍結を「約束」ではなく「機構」にするための層。

⚠ 研究段階のコード。本番の予想・買い目・DB書き込みには一切繋いでいない。
   本番に載せる時は src/ へ移し、tests/test_no_dead_wiring.py の対象にすること。

🔴 ここで弾く4つ:
  1. FIELDS.json に無いキー          → 後から項目を足すことを防ぐ
  2. enum に無い値                   → 分類の揺れを防ぐ
  3. captured_at >= post_time        → 発走後に記録することを防ぐ
  4. source_urls が空                → 「何も見ずに書いた」記録を防ぐ
"""
import hashlib
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS_PATH = os.path.join(HERE, 'FIELDS.json')
SHA_PATH = os.path.join(HERE, 'FIELDS.sha256')
PREFIX = 'web_'
# n_found の分母になる4項目（原文フィールドは数えない）
COUNTED = ('oikiri_grade', 'oikiri_note', 'stable_comment', 'condition_note')
EMPTY = ('なし', '不明', '')


class FrozenSchemaError(Exception):
    """FIELDS.json が凍結後に書き換えられた。"""


def load_fields(check_hash=True):
    """FIELDS.json を読む。凍結ハッシュと一致しなければ落とす。"""
    raw = open(FIELDS_PATH, 'rb').read()
    if check_hash and os.path.exists(SHA_PATH):
        want = open(SHA_PATH).read().strip()
        got = hashlib.sha256(raw).hexdigest()
        if want != got:
            raise FrozenSchemaError(
                f'FIELDS.json が凍結後に変更されています。\n'
                f'  凍結時 {want}\n  現在   {got}\n'
                f'項目を変えるなら schema_version を上げ、'
                f'それ以前の蓄積とは別データとして扱ってください（同じ列に混ぜない）。')
    return json.loads(raw.decode('utf-8'))


def _parse(ts):
    return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))


def validate(record, check_hash=True):
    """1頭ぶんの記録を検証し、エラーの一覧を返す（空なら合格）。

    record は web_ 接頭辞つきのフラットな dict を想定。
    """
    spec = load_fields(check_hash=check_hash)
    fields = {f['id']: f for f in spec['fields']}
    metas = {m['id']: m for m in spec['meta']}
    errs = []

    for k in record:
        if not k.startswith(PREFIX):
            errs.append(f'接頭辞 {PREFIX} が無いキー: {k}')
            continue
        bare = k[len(PREFIX):]
        if bare not in fields and bare not in metas:
            errs.append(f'FIELDS.json に無いキー: {bare}  ← 項目を後から足していないか')

    for mid, m in metas.items():
        if m.get('required') and record.get(PREFIX + mid) in (None, '', []):
            errs.append(f'必須メタが空: {mid}')

    for fid, f in fields.items():
        if f['type'] != 'enum':
            continue
        v = record.get(PREFIX + fid)
        if v is not None and v not in f['options']:
            errs.append(f'{fid} の値 {v!r} は選択肢に無い: {f["options"]}')

    ca, pt = record.get(PREFIX + 'captured_at'), record.get(PREFIX + 'post_time')
    if ca and pt:
        try:
            if _parse(ca) >= _parse(pt):
                errs.append(f'🔴 発走後の記録: captured_at {ca} >= post_time {pt}')
        except Exception as e:                      # noqa: BLE001
            errs.append(f'時刻の解析に失敗: {e}')

    urls = record.get(PREFIX + 'source_urls')
    if isinstance(urls, list) and len(urls) == 0:
        errs.append('source_urls が空。1件も見ずに記録することは無いはず')

    nf = record.get(PREFIX + 'n_found')
    if nf is not None:
        actual = sum(1 for c in COUNTED
                     if str(record.get(PREFIX + c, '') or '') not in EMPTY)
        if int(nf) != actual:
            errs.append(f'n_found が実際と食い違う: 記録 {nf} / 実際 {actual}')
    return errs


def merge_into_notes(existing_json, web_record):
    """既存の notes_data に web_* を足す。他経路のキーを消さない。

    🔴 race_notes は notes_data を丸ごと差し替える仕様なので、
       📝(不利メモ)・🐎(パドック) のキーを保ったままにする必要がある
       （2026-09-02 に同型の defect を潰している）。
    """
    base = dict(json.loads(existing_json) if existing_json else {})
    for k, v in web_record.items():
        if not k.startswith(PREFIX):
            raise ValueError(f'web_ 接頭辞が無いキーを混ぜようとしています: {k}')
        base[k] = v
    return json.dumps(base, ensure_ascii=False)

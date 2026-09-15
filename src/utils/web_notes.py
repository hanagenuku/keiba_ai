"""週末に web から集めた情報（`data/web_notes/*.json`）の読み書きと検証。

🔴 ここの内容は**予想に一切反映しない**。理由は `docs/web_info.md`。
   検証だけを先に作るのは、書き手が毎回ゼロから始まる別セッションの Claude で、
   スキーマ違反に人間もコードも気づけないままファイルだけ溜まるのを防ぐため。

⚠ 出典の無い値を作らせない、というのがこのモジュールの唯一の仕事。
   このプロジェクトでは「作り話の数字が画面に出ていた」事故が3回ある
   （ROI予測150%のべた書き / AI xx%バッジの逆向きの式 / 穴馬率が常に0%）。
"""
import json
import os

SCHEMA_FILE = 'web_note_schema.json'
NOTES_DIR = 'web_notes'

GOING_VALUES = ('良', '稍重', '重', '不良')
CONFIDENCE_VALUES = ('official', 'reported', 'inferred')
RACE_NOTE_KINDS = ('scratch', 'jockey_change', 'news', 'other')
# 特徴量に使ってよいのは公式発表だけ。報道・推測は記録どまり。
USABLE_CONFIDENCE = ('official',)


def load_schema(base_dir='.'):
    path = os.path.join(base_dir, 'data', SCHEMA_FILE)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_notes(base_dir='.', date=None):
    """記録を読む。date 省略時は全件を日付順で返す。"""
    d = os.path.join(base_dir, 'data', NOTES_DIR)
    if not os.path.isdir(d):
        return []
    files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
    if date:
        files = [f for f in files if f == f'{date}.json']
    out = []
    for f in files:
        with open(os.path.join(d, f), encoding='utf-8') as fh:
            out.append(json.load(fh))
    return out


def validate_web_note(note):
    """スキーマ違反を文字列のリストで返す（空なら合格）。"""
    errs = []
    if not isinstance(note, dict):
        return ['トップレベルがオブジェクトでない']

    for key in ('date', 'collected_at', 'phase', 'model_consumed'):
        if key not in note:
            errs.append(f'必須項目がない: {key}')
    if note.get('phase') != 'pre':
        errs.append("phase は 'pre' でなければならない（レース前に集めた情報の記録）")
    if note.get('model_consumed') is not False:
        errs.append('model_consumed は false でなければならない'
                    '（予想に反映しない記録であることの明示）')
    d = note.get('date')
    if isinstance(d, str) and (len(d) != 10 or d[4] != '-' or d[7] != '-'):
        errs.append(f'date は YYYY-MM-DD 形式: {d!r}')

    def _src(obj, where):
        if not obj.get('source'):
            errs.append(f'{where}: source が無い（出典の無い値を作らない）')
        c = obj.get('confidence')
        if c not in CONFIDENCE_VALUES:
            errs.append(f'{where}: confidence が {CONFIDENCE_VALUES} のいずれかでない: {c!r}')

    venues = note.get('venues') or {}
    if not isinstance(venues, dict):
        errs.append('venues はオブジェクトでなければならない')
        venues = {}
    for name, v in venues.items():
        if not isinstance(v, dict):
            errs.append(f'venues.{name}: オブジェクトでない')
            continue
        _src(v, f'venues.{name}')
        going = v.get('going') or {}
        if not isinstance(going, dict):
            errs.append(f'venues.{name}.going: オブジェクトでない')
            continue
        for surf, val in going.items():
            if surf not in ('芝', 'ダート'):
                errs.append(f'venues.{name}.going: 馬場は 芝/ダート のみ: {surf!r}')
            if val not in GOING_VALUES:
                errs.append(f'venues.{name}.going.{surf}: {GOING_VALUES} のいずれかでない: {val!r}')

    races = note.get('races') or []
    if not isinstance(races, list):
        errs.append('races は配列でなければならない')
        races = []
    for i, r in enumerate(races):
        if not isinstance(r, dict):
            errs.append(f'races[{i}]: オブジェクトでない')
            continue
        _src(r, f'races[{i}]')
        if r.get('kind') not in RACE_NOTE_KINDS:
            errs.append(f'races[{i}].kind が {RACE_NOTE_KINDS} のいずれかでない: {r.get("kind")!r}')
    return errs


def usable_going(note):
    """特徴量に使ってよい馬場状態だけを取り出す（＝公式発表のみ）。

    ⚠ いまはどこからも呼んでいない。繋ぐのは `docs/web_info.md` の手順で
       事前登録した基準を通してから。
    """
    out = {}
    for name, v in (note.get('venues') or {}).items():
        if v.get('confidence') not in USABLE_CONFIDENCE:
            continue
        for surf, val in (v.get('going') or {}).items():
            if val in GOING_VALUES:
                out[(name, surf)] = val
    return out

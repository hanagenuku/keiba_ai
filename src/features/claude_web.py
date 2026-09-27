"""Claude が週末の朝に web から集めた情報を、レース毎の**別建ての見解**として渡す層。

＝＝ この層が何をしないか（設計の核心。ここを崩すと6件目の撤回になる）＝＝

このプロジェクトでは「予想の出力を後から書き換える層」が**5件すべて撤回**されている
（市場補正レイヤー / error_tags週次補正 / rank_matrix_filter / AI xx%バッジ /
ROI予測150%）。したがってこの層は **model の数字を1つも変えない**:

  - `score` / `total` / `win_prob` / `cal_prob` / `rl_rank` / `fit_rank` / `bets` /
    `gumbel_bets` / `rec` / `conf` を**読まないし書かない**
  - 返すのは「Claude の序列・印・根拠・出典」だけで、画面では**AIの列とは別の枠**に出る
  - よって A/B が後から組める（AI単体 vs AI＋web を同じレースで並べて実配当で測れる）

⚠ 2026-09-26 の実測（N=1）: 別AIが web 情報でアプリの印を組み替えた案は、序列の
  精度が ρ +0.381 で**市場人気順 +0.548 より下**だった。2026-09-27 の23レース実測では
  web の能力指数とレース内順位の相関が 指数~市場 +0.825 に対し RL~市場 +0.627 で、
  **web指数のほうが市場に近い**＝独立な情報ではなかった。
  よって **この層に上乗せを期待しない**。作ってあるのは
  「AI単体 vs AI＋web」を**同じ条件で測れる形にする**ためで、採否は実配当で決める。

＝＝ 何が入ってよいか ＝＝

web で取れて DB に無い次元だけを入れる。実際に残っているのは2つ:
  1. **馬場状態・トラックバイアス**（`f_track_cond` は推論時 0.0=良 固定で、
     `going.py` は当日1レース目が終わるまで効かない）
  2. **調教タイム・陣営コメント**（2026-08-16 に「丸ごと欠けている唯一の次元」と確定）
市場オッズの言い換え（web の印の平均・人気分析）は**入れない**。

＝＝ 対象レース ＝＝

各会場の **10R・11R のみ**（`WEB_RACE_NUMS`）。理由は2つ:
  - web の記事が存在するのは特別・重賞だけで、未勝利・新馬には情報が無い
  - 2026-08-18 実測で **OPだけAIの上乗せがマイナス（−0.0039）** ＝ 情報がある母集団が
    「磨いても取れない」と測られた母集団と一致する。だから**広げない**
"""

import json
import os

SCHEMA_VERSION = 1

# 各会場でこのレース番号だけ Claude の見解を出す。広げないこと（docstring 参照）。
WEB_RACE_NUMS = (10, 11)

# ⚠ このファイル名は意図的に `data/claude_web_findings.json`。2026-09-15 に作った
#   「記録専用の層」（src/utils 配下）を守る grep ガード（tests の
#   TestItIsNotWiredIntoPrediction）が substring で当たるため、語が重なると
#   「ガードを弱めて通す」方向の変更を誘発する。別物なので名前を分けておく。
FINDINGS_FILENAME = os.path.join('data', 'claude_web_findings.json')

_VALID_MARKS = ('◎', '○', '▲', '△', '☆', '✕', '消')


def findings_path(base_dir):
    return os.path.join(base_dir or '.', FINDINGS_FILENAME)


def load_findings(base_dir):
    """`data/claude_web_findings.json` を読む。無い・壊れている・版が違うなら None。

    ⚠ 例外を握りつぶして黙って None にしない。読めなかった理由を必ず印字する。
    """
    path = findings_path(base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'⚠ claude_web_findings.json を読めませんでした（web見解なしで続行）: {e}')
        return None
    if not isinstance(data, dict):
        print('⚠ claude_web_findings.json の形式が不正（dictではない）。web見解なしで続行')
        return None
    ver = data.get('schema_version')
    if ver != SCHEMA_VERSION:
        print(f'⚠ claude_web_findings.json の schema_version={ver} は未対応'
              f'（期待 {SCHEMA_VERSION}）。web見解なしで続行')
        return None
    return data


def _clean_marks(raw, valid_nums):
    out, dropped = {}, []
    for k, v in (raw or {}).items():
        try:
            num = int(k)
        except (TypeError, ValueError):
            dropped.append(f'印のキー{k!r}が馬番として読めない')
            continue
        if num not in valid_nums:
            dropped.append(f'印の#{num}はこのレースに居ない')
            continue
        if v not in _VALID_MARKS:
            dropped.append(f'#{num}の印{v!r}は未定義')
            continue
        out[num] = v
    return out, dropped


def build_claude_view(race_id, race_num, horses, notes, race_date=None):
    """1レース分の Claude 見解を組み立てる。対象外・情報なし・古い場合は None。

    Args:
        race_id:   `20260927_06_11` 形式。
        race_num:  レース番号。`WEB_RACE_NUMS` 以外は必ず None を返す。
        horses:    そのレースの馬（`num` を持つ dict のリスト）。
        notes:     `load_findings()` の戻り値。
        race_date: `20260927` 形式。notes の date と一致しなければ None を返す
                   （**古い見解を使い回さない**。2026-08-25 の artifact 復元漏れと同型の
                   事故を防ぐ。前の開催の印が今日の画面に出るのが最悪）。

    Returns:
        画面用 dict、または None。**確率・買い目には一切触れない。**
    """
    if not notes or race_num not in WEB_RACE_NUMS:
        return None
    entry = (notes.get('races') or {}).get(race_id)
    if not isinstance(entry, dict):
        return None

    note_date = str(notes.get('date') or '')
    if race_date and note_date and note_date != str(race_date):
        print(f'⚠ claude_web_findings の date={note_date} が対象日 {race_date} と違うため '
              f'{race_id} の web見解は使いません（古い印を出さない）')
        return None

    sources = [s for s in (entry.get('sources') or []) if isinstance(s, str) and s.strip()]
    if not sources:
        # 出典の無い見解は通さない。source_registry と同じ思想＝
        # 「書くだけでは届かない。裏づけの実体が要る」
        print(f'⚠ {race_id} の web見解に出典(sources)が無いため使いません')
        return None

    valid_nums = {h.get('num') for h in horses if h.get('num') is not None}
    dropped = []

    order = []
    for v in (entry.get('order') or []):
        try:
            num = int(v)
        except (TypeError, ValueError):
            dropped.append(f'序列の{v!r}が馬番として読めない')
            continue
        if num not in valid_nums:
            dropped.append(f'序列の#{num}はこのレースに居ない')
            continue
        if num in order:
            dropped.append(f'序列の#{num}が重複')
            continue
        order.append(num)

    marks, mdrop = _clean_marks(entry.get('marks'), valid_nums)
    dropped += mdrop

    horse_notes = {}
    for k, v in (entry.get('horses') or {}).items():
        try:
            num = int(k)
        except (TypeError, ValueError):
            dropped.append(f'馬メモのキー{k!r}が馬番として読めない')
            continue
        if num not in valid_nums:
            dropped.append(f'馬メモの#{num}はこのレースに居ない')
            continue
        if not isinstance(v, dict):
            continue
        note = str(v.get('note') or '').strip()
        if not note:
            continue
        direction = v.get('direction')
        horse_notes[num] = {
            'note': note,
            'direction': direction if direction in ('up', 'down') else None,
            'sources': [s for s in (v.get('sources') or []) if isinstance(s, str)],
        }

    if not order and not marks and not horse_notes:
        return None

    web_ranks = {num: i + 1 for i, num in enumerate(order)}
    return {
        'order': order,
        'marks': marks,
        'web_ranks': web_ranks,
        'horse_notes': horse_notes,
        'going': entry.get('going') if isinstance(entry.get('going'), dict) else None,
        'summary': str(entry.get('summary') or '').strip() or None,
        'sources': sources,
        'collected_at': notes.get('collected_at'),
        # 全馬ぶん序列が付いたか。欠けたまま「序列」と名乗らせない
        'coverage': round(len(order) / len(valid_nums), 3) if valid_nums else 0.0,
        'dropped': dropped,
    }

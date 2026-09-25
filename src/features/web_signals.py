"""web 由来の情報を**上流の特徴量**として渡す層。

位置づけ
--------
`src/scraper/going.py`（当日の馬場状態）と同じ形。予想の出力を後から
書き換える「後付け補正」ではなく、`calc_features_for_xgb` に入る前の
入力として渡す。後付け層はこのプロジェクトで**5回とも撤回されている**。

学習/推論パリティ
-----------------
🔑 **レース内の正規化（順位・z値）をこのモジュールの1箇所だけで行う。**
学習側（`build_training_data`）と推論側（`weekend.py` の refresh）は
「生の指数を race に貼る」ところだけが違い、そこから先は同じ関数を通る。
値の作り方が2箇所に分かれると、片方だけ静かに既定値化する
（2026-07-30 の running_style / 2026-09-15 のペースモデルと同型の事故）。

⚠ 採用済みの情報源が無ければ**列を1つも作らない**。既定の挙動は現状と同一。
⚠ 指数の生値は特徴量にしない。記事ごとにスケールが動く可能性を排除できて
  いないため、レース内で正規化した量だけを渡す（`websig/CRITERIA.md`）。
⚠ `x_gap`（市場人気順位との差）は作らない。材料の `f_popularity` が学習CSVでは
  確定人気で、本番の朝の薄い人気より精度が高い＝本番に無い情報が混ざる
  （2026-09-25 に測定前に気づいて主から落とした）。
"""
import math
import os
import sqlite3

from src.utils.source_registry import adopted_sources

DB_REL = os.path.join('data', 'private', 'web_signals.db')


def _col_base(source):
    """列名に使える形にする（'bloodline-trackbias' → 'bloodline_trackbias'）。"""
    return ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in source)


def feature_cols_for(source):
    """その情報源が作る列名。台帳の feature_cols と一致させること。"""
    b = _col_base(source)
    return [f'f_ws_{b}_rank', f'f_ws_{b}_z']


def attach_web_signals(race, db_path, sources=None):
    """`race['web_signals'] = {source: {horse_num: 指数}}` を貼る。

    学習側・推論側の**両方**がこれを呼ぶ。読めなければ何も貼らない
    （情報が無いことと壊れていることを区別するため、例外は投げない）。
    """
    race_id = race.get('race_id')
    if not race_id or not db_path or not os.path.exists(db_path):
        return race
    try:
        conn = sqlite3.connect(db_path)
        try:
            if sources:
                ph = ','.join('?' * len(sources))
                rows = conn.execute(
                    'SELECT source, horse_num, index_value FROM web_signals '
                    f'WHERE race_id = ? AND source IN ({ph})',
                    (race_id, *sources)).fetchall()
            else:
                rows = conn.execute(
                    'SELECT source, horse_num, index_value FROM web_signals '
                    'WHERE race_id = ?', (race_id,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return race

    got = {}
    for src, num, val in rows:
        if num is None or val is None:
            continue
        got.setdefault(src, {})[int(num)] = float(val)
    if got:
        race['web_signals'] = got
    return race


def _ranks_and_z(values_by_num):
    """レース内の順位（同値は平均順位）と z 値。

    順位は 1 が最上位（指数が大きいほど上位）。
    """
    if not values_by_num:
        return {}, {}
    nums = list(values_by_num)
    vals = [values_by_num[n] for n in nums]

    # 同値は平均順位（1回の引きの大小で順位を捏造しないため。
    # 2026-09-18 Gate 0 で馬番タイブレークが計測器を汚した前例がある）
    order = sorted(nums, key=lambda n: -values_by_num[n])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and \
                values_by_num[order[j + 1]] == values_by_num[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = math.sqrt(var)
    zs = {n: ((values_by_num[n] - mean) / sd if sd > 0 else 0.0) for n in nums}
    return ranks, zs


def calc_web_signal_features(h, race, base_dir='.'):
    """採用済み情報源ぶんの特徴量を返す。採用ゼロなら空 dict。

    指数が無い馬は NaN（XGB は欠損方向を学習する）。
    """
    adopted = adopted_sources(base_dir)
    if not adopted:
        return {}

    attached = race.get('web_signals') or {}
    num = h.get('horse_num')
    try:
        num = int(num)
    except (TypeError, ValueError):
        num = None

    feats = {}
    for entry in adopted:
        src = entry.get('source')
        rank_col, z_col = feature_cols_for(src)
        values = attached.get(src) or {}
        if not values or num is None or num not in values:
            feats[rank_col] = float('nan')
            feats[z_col] = float('nan')
            continue
        ranks, zs = _ranks_and_z(values)
        feats[rank_col] = float(ranks[num])
        feats[z_col] = float(zs[num])
    return feats


def attach_for_adopted(race, base_dir='.'):
    """採用済み情報源がある場合だけ `race` に指数を貼る。

    🔑 学習側（`build_training_data`）と推論側（`weekend.py`）は**この関数を呼ぶ**。
    片方が別経路で貼ると、レース内正規化の母集団が変わって静かにズレる。

    採用ゼロなら何もしない（DBを開かないので毎レースのクエリも走らない）。
    """
    adopted = adopted_sources(base_dir)
    if not adopted:
        return race
    return attach_web_signals(
        race, os.path.join(base_dir, DB_REL),
        sources=[e.get('source') for e in adopted])

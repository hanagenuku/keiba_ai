"""
予測補正レイヤー

XGBoostの予測（cal_prob）を、直近の実績に基づいて補正する。
モデル自体は変更せず、予測の「信頼度」を調整する。

設計原則:
  - XGBoostは月1回再学習（安定層）
  - 補正レイヤーは毎週更新（適応層）
  - 指数平滑移動平均（EMA）でノイズを抑制
"""

import json
import sqlite3
from pathlib import Path

# EMAの平滑化係数（0に近いほど過去重視、1に近いほど直近重視）
EMA_ALPHA = 0.3

# 補正テーブルのデフォルト値（データ不足時）
DEFAULT_FACTOR = 1.0

# 最小サンプル数（これ未満のセルは補正しない）
MIN_SAMPLES = 10

# 補正係数の上下限（極端な補正を防ぐ）
FACTOR_MIN = 0.05
FACTOR_MAX = 5.0


def load_correction_table(base_dir):
    """
    correction_table.json を読み込む。
    ファイルがなければデフォルト（補正なし）を返す。

    Returns
    -------
    dict: {
        'rl_pop': {
            'RL上位_人気':   {'factor': 1.0,  'n': 150, 'actual_rate': 0.62},
            'RL上位_中人気': {'factor': 0.7,  'n': 80,  'actual_rate': 0.35},
            'RL上位_不人気': {'factor': 0.15, 'n': 40,  'actual_rate': 0.05},
            ...
        },
        'updated_at':    '2026-07-20',
        'total_samples': 3400
    }
    """
    path = Path(base_dir) / 'data' / 'correction_table.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'rl_pop': {}, 'updated_at': None, 'total_samples': 0}


def save_correction_table(table, base_dir):
    """correction_table.json を保存"""
    path = Path(base_dir) / 'data' / 'correction_table.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=2)


def classify_rl(rl_rank):
    """RL順位を3段階に分類"""
    if rl_rank <= 3:
        return 'RL上位'
    elif rl_rank <= 6:
        return 'RL中位'
    else:
        return 'RL下位'


def classify_pop(popularity):
    """人気を4段階に分類"""
    if popularity <= 3:
        return '人気'
    elif popularity <= 6:
        return '中人気'
    elif popularity <= 9:
        return '低人気'
    else:
        return '不人気'


def update_correction_table(base_dir, db_path, weeks=8):
    """
    race_predictions の直近N週間のデータから補正テーブルを更新する。
    既存テーブルがあれば EMA で平滑化する。

    Parameters
    ----------
    base_dir : str  データディレクトリ
    db_path : str   keiba.db のパス
    weeks : int     直近何週間分のデータを使うか
    """
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict

    JST = timezone(timedelta(hours=9))
    cutoff = (datetime.now(JST) - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT rl_rank, popularity, actual_place, fuku_prob
        FROM race_predictions
        WHERE actual_place IS NOT NULL
          AND date >= ?
    """, (cutoff,)).fetchall()
    conn.close()

    if len(rows) < MIN_SAMPLES * 3:
        print(f'⚠ データ不足（{len(rows)}件）。補正テーブルは更新しません。')
        return

    buckets = defaultdict(lambda: {'total': 0, 'hit': 0, 'pred_sum': 0.0})

    for r in rows:
        rl_group  = classify_rl(r['rl_rank'])
        pop_group = classify_pop(r['popularity'])
        key = f'{rl_group}_{pop_group}'

        buckets[key]['total'] += 1
        if r['actual_place'] is not None and r['actual_place'] <= 3:
            buckets[key]['hit'] += 1
        if r['fuku_prob'] is not None:
            buckets[key]['pred_sum'] += r['fuku_prob']

    old_table = load_correction_table(base_dir)
    new_rl_pop = {}

    for key, data in buckets.items():
        n = data['total']
        if n < MIN_SAMPLES:
            continue

        actual_rate    = data['hit'] / n
        predicted_rate = data['pred_sum'] / n if n > 0 else 0.01

        if predicted_rate > 0.001:
            new_factor = actual_rate / predicted_rate
        else:
            new_factor = DEFAULT_FACTOR

        new_factor = max(FACTOR_MIN, min(FACTOR_MAX, new_factor))

        old_entry = old_table.get('rl_pop', {}).get(key)
        if old_entry and old_entry.get('factor') is not None:
            smoothed = EMA_ALPHA * new_factor + (1 - EMA_ALPHA) * old_entry['factor']
        else:
            smoothed = new_factor

        new_rl_pop[key] = {
            'factor':         round(smoothed, 4),
            'n':              n,
            'actual_rate':    round(actual_rate, 4),
            'predicted_rate': round(predicted_rate, 4),
            'raw_factor':     round(new_factor, 4),
        }

    table = {
        'rl_pop':        new_rl_pop,
        'updated_at':    datetime.now(JST).strftime('%Y-%m-%d %H:%M'),
        'total_samples': len(rows),
        'weeks':         weeks,
        'ema_alpha':     EMA_ALPHA,
    }

    save_correction_table(table, base_dir)

    print(f'\n📊 補正テーブル更新（{len(rows)}件・直近{weeks}週間）')
    print(f'{"カテゴリ":<20} {"件数":>4} {"実績":>6} {"予測":>6} {"補正係数":>8}')
    print('-' * 50)
    for key in sorted(new_rl_pop.keys()):
        e = new_rl_pop[key]
        print(f'{key:<20} {e["n"]:>4} {e["actual_rate"]:>6.1%} '
              f'{e["predicted_rate"]:>6.1%} {e["factor"]:>8.3f}')

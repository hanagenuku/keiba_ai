"""
依頼4: 学習データ再生成スクリプト

全レースの絶対特徴量 + 相対特徴量を統一ロジックで計算し
data/horse_features.csv を再生成する。

使い方（Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.build_training_data import build_training_data
    build_training_data(BASE_DIR)
"""

import sqlite3
import os
import shutil
import json
from datetime import datetime


def _parse_date(date_str):
    """'2025-01-05' と '20260426' の両形式に対応。"""
    s = str(date_str).replace('-', '')[:8]
    try:
        return datetime.strptime(s, '%Y%m%d').date()
    except Exception:
        return None


def build_training_data(base_dir, output_csv='data/horse_features.csv',
                        backup_suffix='_old'):
    """
    history.db から全レースの特徴量を計算して horse_features.csv を生成する。

    Parameters
    ----------
    base_dir     : Google Drive 上のプロジェクトルート
    output_csv   : 出力先（base_dir からの相対パス）
    backup_suffix: 既存ファイルのバックアップ接尾辞
    """
    import pandas as pd
    from src.features.engine import (
        init_engine, calc_features_for_xgb, add_relative_features,
    )

    db_path  = os.path.join(base_dir, 'data', 'history.db')
    out_path = os.path.join(base_dir, output_csv)
    bak_path = out_path.replace('.csv', f'{backup_suffix}.csv')

    # ── 既存ファイルをバックアップ ──────────────────────────────────────
    if os.path.exists(out_path):
        shutil.copy2(out_path, bak_path)
        print(f'Backed up existing CSV → {bak_path}')

    # ── エンジン初期化 ──────────────────────────────────────────────────
    init_engine(base_dir)

    # ── レース一覧を取得（horse_historyをJOINして実頭数を確認） ──────────
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    race_sql = """
        SELECT r.race_id, r.date, r.racecourse, r.race_num, r.race_name,
               r.distance, r.surface, r.first_3f, r.last_3f,
               r.track_condition, r.race_class, r.pace_label,
               COUNT(h.id) AS n_horses
        FROM race_history r
        INNER JOIN horse_history h ON r.race_id = h.race_id
        GROUP BY r.race_id
        HAVING n_horses >= 3
        ORDER BY r.date, r.race_id
    """
    races = conn.execute(race_sql).fetchall()
    print(f'対象レース数: {len(races)}')

    horse_sql = """
        SELECT horse_name, horse_num, place, running_style,
               agari3f, jockey, trainer, corner_3, corner_4,
               distance, surface, racecourse, date, race_id,
               class_grade, agari_rank, field_size, margin,
               weight_load, sex, age, body_weight, body_weight_diff,
               bracket, win_odds, popularity, race_name,
               COALESCE(last_3f, agari3f) AS last_3f,
               time_diff_sec
        FROM horse_history
        WHERE race_id = ?
        ORDER BY horse_num
    """

    rows_out = []
    skipped  = 0

    for race_row in races:
        race_id = race_row['race_id']
        horses_db = conn.execute(horse_sql, (race_id,)).fetchall()
        if not horses_db:
            skipped += 1
            continue

        # race dict（engine.calc_features_for_xgb の引数形式に合わせる）
        race = {
            'race_id':        race_id,
            'date':           str(race_row['date']),
            'racecourse':     race_row['racecourse'] or '',
            'distance':       int(race_row['distance'] or 1600),
            'surface':        race_row['surface'] or '芝',
            'first_3f':       float(race_row['first_3f'] or 0),
            'race_class':     race_row['race_class'] or '1勝',
            'race_name':      race_row['race_name'] or '',
            'track_condition':race_row['track_condition'] or '良',
            'pace_label':     race_row['pace_label'] or 'mid',
            'horses':         [],
        }

        # 馬リスト構築（historyをSQLから構築）
        horse_objs = []
        for hdb in horses_db:
            h = {
                'name':          hdb['horse_name'],
                'horse_num':     hdb['horse_num'],
                'place':         int(hdb['place'] or 99),
                'running_style': hdb['running_style'] or '差し',
                'agari3f':       hdb['agari3f'],
                'last_3f':       hdb['last_3f'],
                'jockey':        hdb['jockey'] or '',
                'trainer':       hdb['trainer'] or '',
                'corner_3':      hdb['corner_3'],
                'weight_load':   float(hdb['weight_load'] or 56.0),
                'win_odds':      float(hdb['win_odds'] or 10.0),
                'age':           int(hdb['age'] or 4),
                'sex':           hdb['sex'] or '牡',
                'body_weight':   hdb['body_weight'],
                'history':       [],  # 過去走は別途取得
            }
            horse_objs.append(h)

        race['horses'] = horse_objs

        # 各馬の過去走履歴を取得（当該レース以前の最新10走）
        for h in horse_objs:
            hist_rows = conn.execute("""
                SELECT agari3f, COALESCE(last_3f, agari3f) AS last_3f,
                       place, corner_3, corner_4, distance, surface, racecourse,
                       date, race_id, race_name, agari_rank, field_size, margin,
                       running_style, class_grade, time_diff_sec
                FROM horse_history
                WHERE horse_name = ? AND date < ? AND place IS NOT NULL AND place < 99
                ORDER BY date DESC
                LIMIT 10
            """, (h['name'], str(race_row['date']))).fetchall()

            h['history'] = [dict(r) for r in hist_rows]

        # 絶対特徴量を計算
        all_xfeats = []
        valid_horses = []
        for h in horse_objs:
            try:
                xf = calc_features_for_xgb(h, race)
                all_xfeats.append(xf)
                valid_horses.append(h)
            except Exception as e:
                all_xfeats.append({})
                valid_horses.append(h)

        # 相対特徴量を一括計算（引き継ぎ書の真因2・3の修正）
        add_relative_features(all_xfeats)

        # 出力行を構築
        date_obj = _parse_date(race_row['date'])
        for h, xf in zip(valid_horses, all_xfeats):
            place = int(h.get('place', 99))
            row = {
                'race_id':    race_id,
                'date':       str(date_obj) if date_obj else str(race_row['date']),
                'horse_name': h['name'],
                'horse_num':  h['horse_num'],
                'place':      place,
                'is_fukusho': int(place <= 3),  # 複勝ラベル
                **xf,
            }
            rows_out.append(row)

    conn.close()

    df = pd.DataFrame(rows_out)

    # rank の最大値検証（引き継ぎ書「真因3」の修正確認）
    rank_cols = [c for c in df.columns if c.endswith('_rank') and not c.startswith('f_rl') and not c.startswith('f_cl')]
    if rank_cols:
        max_rank = df[rank_cols[0]].max()
        print(f'rank最大値（{rank_cols[0]}）: {max_rank}  ← 18頭立てなら18が最大のはず')

    df.to_csv(out_path, index=False)
    print(f'\n生成完了: {out_path}')
    print(f'  レース数: {len(races) - skipped}  スキップ: {skipped}')
    print(f'  行数: {len(df)}  列数: {len(df.columns)}')
    print(f'  is_fukusho=1: {df["is_fukusho"].sum()} ({df["is_fukusho"].mean()*100:.1f}%)')
    return df


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    build_training_data(base)

"""
学習データ再生成スクリプト

history.db から全レース×全馬の特徴量（絶対+相対）を統一ロジックで計算し
data/horse_features.csv を再生成する。

使い方（Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.build_training_data import build_training_data
    build_training_data(BASE_DIR)
"""

import sqlite3
import os
import shutil
from datetime import datetime


def _parse_date(date_str):
    """'2025-01-05' と '20260426' の両形式に対応。"""
    s = str(date_str or '').replace('-', '')[:8]
    try:
        return datetime.strptime(s, '%Y%m%d').date()
    except Exception:
        return None


def _get_history_before(conn, horse_name, before_date_str, limit=10):
    """
    before_date_str より前の過去走を取得し、engine が期待するキー名で返す。

    engine の各関数（f_rl, f_maturity, calc_features_for_xgb）が使うキー:
      last_3f / agari3f, race_class, track_condition, num_finishers,
      agari_rank, place, corner_3, distance, surface, racecourse,
      first_3f, margin, date, running_style
    """
    before_date = _parse_date(before_date_str)
    if before_date is None:
        return []

    try:
        rows = conn.execute("""
            SELECT h.agari3f, h.place, h.corner_3, h.distance, h.surface,
                   h.racecourse, h.date, h.race_id, h.running_style,
                   h.agari_rank, h.field_size, h.margin,
                   h.finish_time, h.time_diff_sec,
                   COALESCE(h.popularity, 0)                    AS popularity,
                   COALESCE(h.class_grade, r.race_class, '1勝') AS race_class,
                   COALESCE(r.track_condition, '良')             AS track_condition,
                   COALESCE(r.first_3f, 0.0)                    AS first_3f,
                   COALESCE(r.num_finishers, 0)                 AS num_finishers_r,
                   r.race_name
            FROM horse_history h
            LEFT JOIN race_history r ON h.race_id = r.race_id
            WHERE h.horse_name = ?
              AND h.place IS NOT NULL
              AND h.place < 99
            ORDER BY h.date DESC, h.id DESC
            LIMIT ?
        """, (horse_name, limit * 3)).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        row_date = _parse_date(row['date'])
        if row_date is None or row_date >= before_date:
            continue

        agari3f    = float(row['agari3f'] or 0.0)
        agari_rank = int(row['agari_rank'] or 0)

        # 出走頭数: num_finishers → field_size → race内COUNT
        n_fin = int(row['num_finishers_r'] or 0)
        if n_fin < 2:
            n_fin = int(row['field_size'] or 0)
        if n_fin < 2:
            try:
                r2 = conn.execute(
                    "SELECT COUNT(*) FROM horse_history WHERE race_id=?",
                    (row['race_id'],)
                ).fetchone()
                n_fin = int(r2[0] or 8)
            except Exception:
                n_fin = 8
        n_fin = max(n_fin, 2)

        # agari_rank が未記録の場合はレース内で計算
        if agari_rank <= 0 and agari3f > 0:
            try:
                vals = conn.execute(
                    "SELECT agari3f FROM horse_history WHERE race_id=? AND agari3f > 0",
                    (row['race_id'],)
                ).fetchall()
                vals = sorted([float(v[0]) for v in vals if v[0]])
                agari_rank = (vals.index(agari3f) + 1) if agari3f in vals else 0
            except Exception:
                agari_rank = 0

        agari3f_rank_pct = (agari_rank - 1) / max(n_fin - 1, 1) if agari_rank > 0 else 0.5

        results.append({
            # agari3f系 (両キー名を用意してどちらでも取れるようにする)
            "agari3f":          agari3f,
            "last_3f":          agari3f,
            # 着順・頭数
            "place":            int(row['place'] or 10),
            "finishers":        n_fin,
            "num_finishers":    n_fin,
            # 上がり順位
            "agari_rank":       agari_rank if agari_rank > 0 else None,
            "agari3f_rank_pct": round(agari3f_rank_pct, 3),
            # レース情報 (両キー名を用意)
            "race_class":       row['race_class'] or '1勝',
            "class":            row['race_class'] or '1勝',
            "track_condition":  row['track_condition'] or '良',
            "condition":        row['track_condition'] or '良',
            # タイム・コース
            "first_3f":         float(row['first_3f'] or 0.0),
            "finish_time":      float(row['finish_time']) if row['finish_time'] else None,
            "time_diff_sec":    float(row['time_diff_sec']) if row['time_diff_sec'] else None,
            "distance":         int(row['distance'] or 1600),
            "surface":          row['surface'] or '芝',
            "racecourse":       row['racecourse'] or '',
            "corner_3":         row['corner_3'],
            "margin":           float(row['margin'] or 0.0),
            # 市場評価（f_pop_last / f_pop_avg / f_beat_market_rate 用）
            "popularity":       int(row['popularity'] or 0),
            # メタ
            "date":             str(row['date'] or ''),
            "race_id":          row['race_id'],
            "running_style":    row['running_style'] or '',
            "race_name":        row['race_name'] or '',
        })

        if len(results) >= limit:
            break

    return results


def build_training_data(base_dir, output_csv='data/horse_features.csv',
                        backup_suffix='_old'):
    """
    history.db から全レースの特徴量を計算して horse_features.csv を再生成する。

    Parameters
    ----------
    base_dir     : プロジェクトルート（Google Drive のパス）
    output_csv   : 出力先（base_dir からの相対パス）
    backup_suffix: 既存ファイルのバックアップ接尾辞
    """
    import pandas as pd
    import src.features.engine as _eng
    from src.features.engine import (
        init_engine, calc_features_for_xgb, add_relative_features,
        build_member_level_cache,
    )

    db_path  = os.path.join(base_dir, 'data', 'history.db')
    out_path = os.path.join(base_dir, output_csv)
    bak_path = out_path.replace('.csv', f'{backup_suffix}.csv')

    # 既存ファイルをバックアップ
    if os.path.exists(out_path):
        shutil.copy2(out_path, bak_path)
        print(f'Backed up existing CSV → {bak_path}')

    # メンバーレベルキャッシュを必ず最新で再構築（新データ取り込み後は再生成が必要）
    # ── データリーク防止: train_xgb の val_start より前のデータのみを
    #    「その後の成績」として使う。これにより Val 期間のデータが Train の
    #    member_level 特徴量に漏れ込まない。
    # train_xgb のデフォルト val_start='2026-04-01' に合わせてカットオフを設定。
    # より厳密には train_end+1 日でよい（'2026-04-01' = train_end '2026-03-31' の翌日）。
    MEMBER_LEVEL_CUTOFF = '2026-04-01'
    ml_path = os.path.join(base_dir, 'data', 'member_level_cache.pkl')
    print(f'🔨 メンバーレベルキャッシュ再構築中... (cutoff={MEMBER_LEVEL_CUTOFF})')
    build_member_level_cache(base_dir, cutoff_date=MEMBER_LEVEL_CUTOFF)
    print(f'✅ メンバーレベルキャッシュ保存: {ml_path}')

    # エンジン初期化（jockey_dict / trainer_dict + キャッシュを読み込む）
    init_engine(base_dir)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # レース一覧（horse_history JOIN で実頭数を確認）
    races = conn.execute("""
        SELECT r.race_id, r.date, r.racecourse, r.distance, r.surface,
               r.first_3f, r.race_class, r.race_name, r.track_condition,
               r.pace_label,
               COUNT(h.id) AS n_horses
        FROM race_history r
        INNER JOIN horse_history h ON r.race_id = h.race_id
        GROUP BY r.race_id
        HAVING n_horses >= 3
        ORDER BY r.date, r.race_id
    """).fetchall()
    print(f'対象レース数: {len(races)}')

    rows_out = []
    skipped  = 0

    for idx, race_row in enumerate(races):
        if idx % 500 == 0:
            print(f'  処理中: {idx}/{len(races)} レース...', flush=True)

        race_id  = race_row['race_id']
        date_str = str(race_row['date'] or '')
        rc       = race_row['racecourse'] or ''
        surf     = race_row['surface'] or '芝'

        horses_db = conn.execute("""
            SELECT horse_name, horse_num, place, running_style,
                   agari3f, jockey, trainer, corner_3,
                   distance, surface, racecourse,
                   weight_load, sex, age, body_weight,
                   bracket, win_odds, popularity,
                   agari_rank, field_size
            FROM horse_history
            WHERE race_id = ?
            ORDER BY horse_num
        """, (race_id,)).fetchall()

        if not horses_db:
            skipped += 1
            continue

        race = {
            'race_id':         race_id,
            'date':            date_str,
            'racecourse':      rc,
            'distance':        int(race_row['distance'] or 1600),
            'surface':         surf,
            'first_3f':        float(race_row['first_3f'] or 0.0),
            'race_class':      race_row['race_class'] or '1勝',
            'race_name':       race_row['race_name'] or '',
            'track_condition': race_row['track_condition'] or '良',
            'pace_label':      race_row['pace_label'] or 'mid',
            'horses':          [],
        }

        horse_objs = []
        for hdb in horses_db:
            jn = (hdb['jockey'] or '').replace(' ', '').replace('　', '')
            tn = (hdb['trainer'] or '').replace(' ', '').replace('　', '')

            h = {
                'name':         hdb['horse_name'],
                'horse_num':    int(hdb['horse_num'] or 1),
                'place':        int(hdb['place'] or 99),
                'running_style':hdb['running_style'] or '差し',
                'agari3f':      hdb['agari3f'],
                'jockey':       hdb['jockey'] or '',
                'trainer':      hdb['trainer'] or '',
                'corner_3':     hdb['corner_3'],
                'weight_load':  float(hdb['weight_load'] or 56.0),
                'win_odds':     float(hdb['win_odds'] or 10.0),
                # 現走人気（f_popularity 用）。学習時は確定人気、
                # 予測時は calc_all が朝オッズから導出した人気が入る
                'popularity':   int(hdb['popularity'] or 0),
                'age':          int(hdb['age'] or 4),
                'sex':          hdb['sex'] or '牡',
                # init_engine 後の辞書から jockey_rate / trainer_rate を取得
                'jockey_rate': (
                    _eng._jockey_dict.get((jn, rc, surf)) or
                    _eng._jockey_dict.get((jn, '', '')) or
                    0.15
                ),
                'trainer_rate': _eng._trainer_dict.get(tn, 0.12),
                'history': [],
            }
            horse_objs.append(h)

        race['horses'] = horse_objs

        # 各馬の過去走を「このレースの日付より前」で取得
        for h in horse_objs:
            h['history'] = _get_history_before(conn, h['name'], date_str, limit=10)

        # 絶対特徴量を計算
        all_xfeats = []
        for h in horse_objs:
            try:
                xf = calc_features_for_xgb(h, race)
            except Exception:
                xf = {}
            all_xfeats.append(xf)

        # 相対特徴量を一括計算（引き継ぎ書 真因2・3 の修正）
        add_relative_features(all_xfeats)

        # 出力行を構築
        date_obj = _parse_date(date_str)
        for h, xf in zip(horse_objs, all_xfeats):
            place = int(h.get('place', 99))
            rows_out.append({
                'race_id':    race_id,
                'date':       str(date_obj) if date_obj else date_str,
                'horse_name': h['name'],
                'horse_num':  h['horse_num'],
                'place':      place,
                'is_fukusho': int(place <= 3),
                **xf,
            })

    conn.close()

    df = pd.DataFrame(rows_out)

    # 検証: rank の最大値が頭数に応じて変動しているか（真因3 修正確認）
    rank_cols = [c for c in df.columns if c.endswith('_rank')
                 and not c.startswith('f_rl') and not c.startswith('f_cl')]
    if rank_cols:
        print(f'rank最大値({rank_cols[0]}): {df[rank_cols[0]].max()}'
              f'  ← 18頭立てのレースがあれば18のはず（9クリップ廃止確認）')

    df.to_csv(out_path, index=False)
    print(f'\n✅ 生成完了: {out_path}')
    print(f'  レース数: {len(races) - skipped}  スキップ: {skipped}')
    print(f'  行数: {len(df)}  列数: {len(df.columns)}')
    print(f'  is_fukusho=1: {df["is_fukusho"].sum()} ({df["is_fukusho"].mean()*100:.1f}%)')
    return df


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    build_training_data(base)

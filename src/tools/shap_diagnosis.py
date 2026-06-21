"""
SHAP診断レポート

毎週末の結果取得後に実行し、「なぜ外したか」を自動分析する。
shap ライブラリが未インストールの場合でも外れパターン分析は動作する。
"""

import json
import os
import sqlite3


def generate_shap_report(base_dir, db_path, target_date=None):
    """
    指定日（デフォルト:直近の開催日）の全レースについて
    外れパターン分析を行い、診断レポートを生成する。

    Parameters
    ----------
    base_dir : str
    db_path : str   keiba.db のパス（race_predictions を含む）
    target_date : str (YYYY-MM-DD)  None なら直近

    Returns
    -------
    dict: レポートデータ
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if target_date is None:
        row = conn.execute(
            "SELECT MAX(date) FROM race_predictions WHERE actual_place IS NOT NULL"
        ).fetchone()
        target_date = row[0] if row else None

    if target_date is None:
        print('⚠ race_predictions に結果データがありません')
        conn.close()
        return None

    predictions = conn.execute("""
        SELECT *
        FROM race_predictions
        WHERE date = ? AND actual_place IS NOT NULL
        ORDER BY race_id, rl_rank
    """, (target_date,)).fetchall()
    conn.close()

    if not predictions:
        print(f'⚠ {target_date} のデータがありません')
        return None

    predictions = [dict(p) for p in predictions]

    # 大外し: RL上位（1-3位）なのに実着順 10位以下
    big_misses = [p for p in predictions
                  if p['rl_rank'] <= 3
                  and p['actual_place'] is not None
                  and p['actual_place'] >= 10]

    # 見逃し: RL下位（10位以下）なのに実着順 3位以内
    hidden_gems = [p for p in predictions
                   if p['rl_rank'] >= 10
                   and p['actual_place'] is not None
                   and p['actual_place'] <= 3]

    # RL上位が複勝内（3着以内）に来た率
    rl_top = [p for p in predictions if p['rl_rank'] <= 3]
    rl_hit  = [p for p in rl_top if p['actual_place'] is not None and p['actual_place'] <= 3]
    rl_hit_rate = len(rl_hit) / len(rl_top) if rl_top else 0.0

    report = {
        'date':          target_date,
        'total_races':   len(set(p['race_id'] for p in predictions)),
        'total_horses':  len(predictions),
        'big_misses':    len(big_misses),
        'hidden_gems':   len(hidden_gems),
        'rl_top_hit_rate': round(rl_hit_rate, 4),
        'miss_details':  [],
        'gem_details':   [],
    }

    for p in big_misses[:10]:
        report['miss_details'].append({
            'race':         f'{p.get("racecourse", "?")} R{p.get("race_num", "?")}',
            'horse':        p.get('horse_name', ''),
            'rl_rank':      p['rl_rank'],
            'actual_place': p['actual_place'],
            'popularity':   p.get('popularity', 99),
            'gap':          p.get('prediction_gap'),
        })

    for p in hidden_gems[:10]:
        report['gem_details'].append({
            'race':         f'{p.get("racecourse", "?")} R{p.get("race_num", "?")}',
            'horse':        p.get('horse_name', ''),
            'rl_rank':      p['rl_rank'],
            'actual_place': p['actual_place'],
            'popularity':   p.get('popularity', 99),
            'gap':          p.get('prediction_gap'),
        })

    # SHAP分析（オプション：shap未インストールでもスキップして続行）
    report['shap_available'] = False
    try:
        import shap  # noqa: F401
        import pickle
        import pandas as pd

        model_path = os.path.join(base_dir, 'data', 'xgb_fukusho_model.pkl')
        cols_path  = os.path.join(base_dir, 'data', 'xgb_feature_cols.json')
        if os.path.exists(model_path) and os.path.exists(cols_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(cols_path, 'r') as f:
                feature_cols = json.load(f)['feature_cols']
            report['shap_available'] = True
            # TODO: 外れ馬の特徴量を race_predictions から取得し SHAP 値を計算
            # （特徴量は build_training_data で horse_features.csv に保存済み）
    except ImportError:
        pass
    except Exception as _e:
        report['shap_error'] = str(_e)

    # レポート出力
    print(f'\n📋 SHAP診断レポート（{target_date}）')
    print(f'   全{report["total_races"]}レース・{report["total_horses"]}頭')
    print(f'   RL上位3頭の複勝的中率: {rl_hit_rate:.1%}')
    print(f'   大外し（RL上位→10着以下）: {report["big_misses"]}件')
    print(f'   見逃し（RL下位→3着以内）: {report["hidden_gems"]}件')

    if report['miss_details']:
        print('\n   【大外し詳細】')
        for m in report['miss_details']:
            print(f'   {m["race"]} {m["horse"]} '
                  f'RL{m["rl_rank"]}→{m["actual_place"]}着 '
                  f'（{m["popularity"]}番人気）')

    if report['gem_details']:
        print('\n   【見逃し詳細】')
        for g in report['gem_details']:
            print(f'   {g["race"]} {g["horse"]} '
                  f'RL{g["rl_rank"]}→{g["actual_place"]}着 '
                  f'（{g["popularity"]}番人気）')

    if not report['shap_available']:
        print('   ℹ SHAP: 未インストール（pip install shap で有効化可）')

    logs_dir = os.path.join(base_dir, 'data', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    report_path = os.path.join(logs_dir, f'shap_report_{target_date}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'   保存: {report_path}')

    return report

#!/usr/bin/env python3
"""日曜夜：結果取得・history.db保存・bet照合・週次ROI集計"""
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._session import create_session
from scripts.weekend import fetch_and_save_results
from src.betting.shadow import record_all_shadow_bets
from src.features.correction import update_correction_table
from src.features.engine import init_engine
from src.features.error_tags import process_weekly_error_tags
from src.tools.shap_diagnosis import generate_shap_report
from src.utils.db import (init_db, get_db_path, get_history_db_path,
                           backup_db, checkpoint_db)

JST = timezone(timedelta(hours=9))


def print_roi_breakdown(db_path, since_date):
    """直近1週間分のbetsを bet_type / racecourse 別に集計表示"""
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT b.date, b.race_id, b.bet_type, b.amount, b.is_hit, b.payout, "
        "r.racecourse FROM bets b "
        "LEFT JOIN races r ON r.id = b.race_id "
        "WHERE b.is_hit != -1 AND b.date >= ?",
        conn, params=(since_date,),
    )
    conn.close()

    if df.empty:
        print('📊 週次ROI: 対象データなし')
        return

    print(f'📊 週次ROI集計（{since_date}〜）  全{len(df)}件')
    for col in ['bet_type', 'racecourse']:
        print(f'\n--- {col}別 ---')
        g = df.groupby(col).agg(
            件数=('amount', 'count'),
            的中=('is_hit', 'sum'),
            投資=('amount', 'sum'),
            回収=('payout', 'sum'),
        )
        g['ROI%'] = (g['回収'] / g['投資'] * 100).round(1)
        print(g.to_string())


def main():
    db_path = get_db_path(ROOT)
    hist_path = get_history_db_path(ROOT)
    backup_db(db_path)
    backup_db(hist_path)

    init_db(ROOT)
    init_engine(ROOT)

    jst_now = datetime.now(JST)
    target_date = jst_now.strftime('%Y%m%d')

    sess = create_session()
    all_results = fetch_and_save_results(sess, hist_path, target_date)

    if all_results:
        record_all_shadow_bets(all_results, ROOT)

    # ② 補正テーブルを EMA で更新（race_predictions に4週分以上蓄積されてから有効）
    update_correction_table(ROOT, db_path, weeks=8)

    # ③ SHAP診断レポート生成
    jst_date = jst_now.strftime('%Y-%m-%d')
    generate_shap_report(ROOT, db_path, target_date=jst_date)

    # ④ エラータグ分類・蓄積（翌週予想の補正係数を自動更新）
    try:
        process_weekly_error_tags(ROOT, db_path, target_date=jst_date)
    except Exception as e:
        print(f'⚠ エラータグ処理失敗（予想には影響なし）: {e}')

    since_date = (jst_now - timedelta(days=7)).strftime('%Y-%m-%d')
    print_roi_breakdown(db_path, since_date)

    checkpoint_db(db_path)
    checkpoint_db(hist_path)


if __name__ == '__main__':
    main()

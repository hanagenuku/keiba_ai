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
from src.features.engine import init_engine
from src.features.error_tags import process_weekly_error_tags
from src.utils.db import compare_prediction_snapshots
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
    # 対象日は既定で当日だが、TARGET_DATE で過去日を指定できる。
    # ⚠ これが無いと**取りこぼした開催を後から取り直せない**。
    #   2026-08-02は新潟・札幌のスキャンが全滅して24レースを落としたが、
    #   翌日に再実行しても当日(月曜)を見に行って6秒で空振りに終わった。
    target_date = (os.environ.get('TARGET_DATE') or '').strip() \
        or jst_now.strftime('%Y%m%d')
    if not (len(target_date) == 8 and target_date.isdigit()):
        raise SystemExit(f'TARGET_DATE は YYYYMMDD で指定してください: {target_date!r}')
    if target_date != jst_now.strftime('%Y%m%d'):
        print(f'📅 対象日を指定して実行: {target_date}（当日ではありません）')

    sess = create_session()
    all_results = fetch_and_save_results(sess, hist_path, target_date)

    if all_results:
        # 実際に推奨・購入したレースを was_recommended=1 として記録する
        # （weekend.py の土曜側と同じロジック。従来はここが抜けており、
        # 日曜分のshadow_betsは推奨レースでも常に was_recommended=0 になっていた）
        _conn = sqlite3.connect(db_path)
        _rec_ids = {r[0] for r in _conn.execute(
            'SELECT DISTINCT race_id FROM bets WHERE date=?', (target_date,)).fetchall()}
        _conn.close()
        record_all_shadow_bets(all_results, ROOT, recommended_race_ids=_rec_ids)

    # ② SHAP診断レポート生成
    # 当日基準にすると TARGET_DATE 指定時に別の日を集計してしまうため、
    # 結果取得と同じ対象日から作る
    jst_date = f'{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}'
    generate_shap_report(ROOT, db_path, target_date=jst_date)

    # ③ エラータグ分類・蓄積（翌週予想の補正係数を自動更新）
    try:
        process_weekly_error_tags(ROOT, db_path, target_date=jst_date)
    except Exception as e:
        print(f'⚠ エラータグ処理失敗（予想には影響なし）: {e}')

    # ④ 前夜の予想 vs 当日refresh の突合
    # 🔑 prediction_snapshots は 2026-07-27⑩ から溜め続けていたが、
    #    compare_prediction_snapshots() を呼ぶコードが無く **一度も測られて
    #    いなかった**（2026-08-26 の棚卸しで発覚。8,464行が眠っていた）。
    #    溜めるだけで見ない、を繰り返さないよう週次で必ずログに出す。
    # ⚠ 2026-08-26 の初回実測（276レース）では的中率 22.8%→34.4% と上がる一方、
    #    単勝回収率は 92.4%→88.0% と下がった。当てやすさと儲けやすさは別
    #    （North Star #9）。この数字を「refreshで儲かる」と読まないこと。
    try:
        cmp_ = compare_prediction_snapshots(db_path=db_path)
        if cmp_:
            print(f'\n📸 前夜 vs 当日refresh  {cmp_["n_races"]}レース {cmp_["n_horses"]}頭')
            print(f'  RL順位が変わった馬: {cmp_["rank_changed"]}頭')
            print(f'  AI本命が入れ替わったレース: {cmp_["fav_changed"]}')
            print(f'  1着になったのは  前夜のRL1 {cmp_["initial_rl1_win"]} / '
                  f'朝のRL1 {cmp_["refresh_rl1_win"]}')
            print('  ⚠ 的中率の話。回収率は別（2026-08-26実測: 92.4%→88.0%）')
        else:
            print('📸 前夜 vs 当日refresh: 両時点が揃ったデータなし')
    except Exception as e:
        print(f'⚠ スナップショット突合に失敗（結果取得には影響なし）: {e}')

    since_date = (jst_now - timedelta(days=7)).strftime('%Y-%m-%d')
    print_roi_breakdown(db_path, since_date)

    checkpoint_db(db_path)
    checkpoint_db(hist_path)


if __name__ == '__main__':
    main()

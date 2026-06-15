#!/usr/bin/env python3
"""土曜夜/日曜夜：結果取得・history.db保存・bet照合（土曜夜は翌日予想も生成）"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._session import create_session
from src.features.engine import init_engine
from src.utils.db import (init_db, get_db_path, get_history_db_path,
                           backup_db, checkpoint_db,
                           save_race_db, save_bets_db,
                           save_history_db, save_results_db, check_and_update_bets)
from src.betting.make_bets import init_betting, make_bets, log_bet_simulation
from src.betting.ev_filter import ability_first_loose
from src.betting.app_json import to_app_json
from src.scraper.jra_scraper import fetch_races_on_date, fetch_results
from src.tools.bias import analyze_bias, build_avg_bias

JST = timezone(timedelta(hours=9))

BANKROLL = 100_000
TOP_N_RACES = 6
FUKU_AMT = 500
TAN_AMT = 300
WIDE_AMT = 300
REN_AMT = 300
TAN2_AMT = 300
SAN_AMT = 300

BIAS_PATH = os.path.join(ROOT, 'data', 'track_bias_latest.json')
APP_PATH = os.path.join(ROOT, 'data', 'latest.json')


def fetch_and_save_results(sess, hist_path, target_date):
    print(f'📅 本日結果取得: {target_date}')

    all_results = fetch_results(sess, target_date)
    print(f'📋 取得完了: {len(all_results)}レース')

    if all_results:
        surf_counts = {}
        for r in all_results:
            s = r.get('surface', '?')
            surf_counts[s] = surf_counts.get(s, 0) + 1
        print(f'   コース内訳: {surf_counts}')

        save_history_db(all_results, ROOT)
        save_results_db(all_results, ROOT)

        import sqlite3
        conn = sqlite3.connect(hist_path)
        dt_max = conn.execute('SELECT MAX(date) FROM race_history').fetchone()[0]
        rc = conn.execute('SELECT COUNT(*) FROM race_history').fetchone()[0]
        hc = conn.execute('SELECT COUNT(*) FROM horse_history').fetchone()[0]
        conn.close()
        print(f'   📊 history.db: {rc:,}レース / {hc:,}出走 / 最新: {dt_max}')

    chk = check_and_update_bets(all_results, ROOT)
    print(f'【照合結果】 {chk["hit"]}/{chk["total"]}的中  '
          f'投資¥{chk["invested"]:,} / 回収¥{chk["recovered"]:,}  '
          f'ROI {chk["roi"]:.1f}%')
    for d in chk['details']:
        print(d)

    return all_results


def predict_next_day(sess, hist_path, avg_bias, jst_now):
    next_date = (jst_now + timedelta(days=1)).strftime('%Y%m%d')
    print(f'📅 取得日: {next_date}（翌日）')

    races = fetch_races_on_date(sess, next_date, hist_path)
    print(f'📋 取得レース: {len(races)}R')

    surf_counts = {}
    for r in races:
        s = r.get('surface', '?')
        surf_counts[s] = surf_counts.get(s, 0) + 1
    print(f'   コース内訳: {surf_counts}')

    selected = ability_first_loose(races, avg_bias, top_n=TOP_N_RACES)
    print(f'⭐ 厳選: {len(selected)}レース')

    total_inv = 0
    for i, c in enumerate(selected, 1):
        race = c['race']
        top1 = c['top1']
        bets = make_bets(c)
        invest = sum(b['amount'] for b in bets)
        total_inv += invest

        print(f'【{i}】{race["racecourse"]} R{race["race_num"]:02d} {race.get("race_name", "")}'
              f'  {race["distance"]}m{race["surface"]}  {race.get("num_horses", 0)}頭')
        print(f'  ◎ #{top1["num"]} {top1["name"]}  {top1.get("win_odds", 0):.1f}倍  スコア:{top1["total"]:.2f}')
        for b in bets:
            print(f'  {b["type"]} #{b["nums"][0]} ¥{b["amount"]:,}  EV:{b["ev"]:.2f}')
        print(f'  投資: ¥{invest:,}')

        save_race_db(race, ROOT)
        save_bets_db(race['date'], race['id'], bets, ROOT)
        log_bet_simulation(race['date'], c, ROOT)

    print(f'💰 投資合計: ¥{total_inv:,}')

    jst_now = datetime.now(JST)
    app_data = to_app_json(selected, races, avg_bias, jst_now, day_type='sunday')
    os.makedirs(os.path.dirname(APP_PATH), exist_ok=True)
    with open(APP_PATH, 'w', encoding='utf-8') as f:
        json.dump(app_data, f, ensure_ascii=False, indent=2)
    print(f'✅ アプリJSON保存: {APP_PATH}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['saturday', 'sunday'], required=True)
    args = parser.parse_args()

    db_path = get_db_path(ROOT)
    hist_path = get_history_db_path(ROOT)
    backup_db(db_path)
    backup_db(hist_path)

    init_db(ROOT)
    init_engine(ROOT)
    init_betting(ROOT, bankroll=BANKROLL,
                  fuku_amt=FUKU_AMT, tan_amt=TAN_AMT, wide_amt=WIDE_AMT,
                  ren_amt=REN_AMT, tan2_amt=TAN2_AMT, san_amt=SAN_AMT)

    jst_now = datetime.now(JST)
    target_date = jst_now.strftime('%Y%m%d')

    sess = create_session()
    all_results = fetch_and_save_results(sess, hist_path, target_date)

    if args.mode == 'saturday':
        # バイアス分析
        bias_by_course = analyze_bias(all_results)
        print('【トラックバイアス分析】')
        for rc, b in bias_by_course.items():
            print(f'  {rc}: {b["summary"]}  '
                  f'内外:{b["inner_outer"]:+.2f} ペース:{b["pace_bias"]:+.2f} '
                  f'時計:{b["track_speed"]:+.2f}  ({b["race_count"]}R)')

        prev_bias = None
        if os.path.exists(BIAS_PATH):
            with open(BIAS_PATH, encoding='utf-8') as f:
                prev_bias = json.load(f)

        avg_bias = build_avg_bias(bias_by_course, prev_bias)
        print(f'【全体バイアス】{avg_bias["summary"]}')
        print(f'  内外:{avg_bias["inner_outer"]:+.2f}  ペース:{avg_bias["pace_bias"]:+.2f}  '
              f'時計:{avg_bias["track_speed"]:+.2f}')

        with open(BIAS_PATH, 'w', encoding='utf-8') as f:
            json.dump(avg_bias, f, ensure_ascii=False, indent=2)
        print(f'✅ バイアス保存: {BIAS_PATH}')

        # 翌日（日曜）予想
        predict_next_day(sess, hist_path, avg_bias, jst_now)

    checkpoint_db(db_path)
    checkpoint_db(hist_path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""金曜夜：土曜レース取得・予想・買い目生成・latest.json更新"""
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
                           save_race_db, save_bets_db, save_race_predictions)
from src.betting.make_bets import init_betting, make_bets, log_bet_simulation
from src.betting.ev_filter import select_quality_races, build_market_odds_from_races
from src.betting.app_json import to_app_json
from src.features.engine import calc_all
from src.scraper.jra_scraper import fetch_races_on_date

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


def main():
    db_path = get_db_path(ROOT)
    hist_path = get_history_db_path(ROOT)
    backup_db(db_path)
    backup_db(hist_path)

    init_db(ROOT)
    init_engine(ROOT)
    init_betting(ROOT, bankroll=BANKROLL,
                  fuku_amt=FUKU_AMT, tan_amt=TAN_AMT, wide_amt=WIDE_AMT,
                  ren_amt=REN_AMT, tan2_amt=TAN2_AMT, san_amt=SAN_AMT)

    avg_bias = None
    if os.path.exists(BIAS_PATH):
        with open(BIAS_PATH, encoding='utf-8') as f:
            avg_bias = json.load(f)
        print(f'📊 バイアス({avg_bias.get("date", "")}): {avg_bias.get("summary", "フラット")}')
    else:
        print('📊 バイアス: なし（フラット想定）')

    jst_now = datetime.now(JST)
    weekday = jst_now.weekday()  # 4=金 5=土 6=日
    if weekday == 4:
        target_date = (jst_now + timedelta(days=1)).strftime('%Y%m%d')
    else:
        target_date = jst_now.strftime('%Y%m%d')

    print(f'📅 取得日: {target_date}  ({"月火水木金土日"[weekday]}曜)')

    sess = create_session()
    races = fetch_races_on_date(sess, target_date, hist_path)
    print(f'📋 取得レース: {len(races)}R')

    if not races:
        print('⚠ レースが見つかりません（開催日ではないか、出馬表未掲載）')
        checkpoint_db(get_db_path(ROOT))
        checkpoint_db(hist_path)
        return

    surf_counts = {}
    for r in races:
        s = r.get('surface', '?')
        surf_counts[s] = surf_counts.get(s, 0) + 1
    print(f'   コース内訳: {surf_counts}')
    if len(surf_counts) == 1 and 'ダート' in surf_counts and len(races) > 6:
        print('⚠ 警告: 全レースがダート判定されています。surfaceパーサーにバグの可能性あり')

    # 全レース・全馬のRL予測を先にまとめて保存（結果との乖離学習用）。
    # 厳選レースだけでなく全レースを残すことで、土曜分も race_predictions に蓄積され、
    # 補正テーブル（correction_table.json）が土日フルのデータで更新される。
    print('💾 全レース予測を race_predictions に保存中...')
    for race in races:
        scored_all = calc_all(race, avg_bias)
        if scored_all:
            save_race_predictions(race, scored_all, ROOT)
    print(f'   {len(races)}レース完了')

    selected = select_quality_races(races, avg_bias)
    print(f'⭐ 厳選: {len(selected)}レース')
    if not selected:
        print('⚠ 厳選レースなし（品質閾値を満たすレースがありません）')

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
    market_odds_map = build_market_odds_from_races(races)
    app_data = to_app_json(selected, races, avg_bias, jst_now,
                           day_type='saturday', market_odds_map=market_odds_map)
    os.makedirs(os.path.dirname(APP_PATH), exist_ok=True)
    with open(APP_PATH, 'w', encoding='utf-8') as f:
        json.dump(app_data, f, ensure_ascii=False, indent=2)
    print(f'✅ アプリJSON保存: {APP_PATH}')

    checkpoint_db(db_path)
    checkpoint_db(hist_path)


if __name__ == '__main__':
    main()

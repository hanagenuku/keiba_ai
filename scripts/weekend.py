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
                           save_history_db, save_results_db, check_and_update_bets,
                           save_race_predictions, update_prediction_results)
from src.betting.make_bets import init_betting, make_bets, log_bet_simulation
from src.betting.ev_filter import select_quality_races, build_market_odds_from_races
from src.features.engine import calc_all
from src.betting.app_json import to_app_json
from src.betting.shadow import record_all_shadow_bets
from src.scraper.jra_scraper import (
    fetch_races_on_date, fetch_results, fetch_odds_map, apply_odds_to_races,
)
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
        updated = update_prediction_results(all_results, ROOT)
        print(f'   race_predictions 更新: {updated}件')

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


def _already_generated(app_path, mode_type, today_str):
    """latest.json が今日の same type で既に生成済みかどうかを確認する。"""
    if not os.path.exists(app_path):
        return False
    try:
        with open(app_path, encoding='utf-8') as f:
            data = json.load(f)
        gen_at = data.get('generated_at', '')
        gen_type = data.get('type', '')
        gen_date = gen_at[:10]  # 'YYYY-MM-DD'
        return gen_date == today_str and gen_type == mode_type
    except Exception:
        return False


def predict_next_day(sess, hist_path, avg_bias, jst_now, force=False):
    next_date = (jst_now + timedelta(days=1)).strftime('%Y%m%d')
    today_str = jst_now.strftime('%Y-%m-%d')

    if not force and _already_generated(APP_PATH, 'sunday', today_str):
        print(f'⏭ 日曜予想は本日({today_str})生成済みのためスキップ。'
              f'再生成するには --force オプションを使用してください。')
        return

    print(f'📅 取得日: {next_date}（翌日）')

    races = fetch_races_on_date(sess, next_date, hist_path)
    print(f'📋 取得レース: {len(races)}R')

    if not races:
        print(f'⚠️ {next_date} のレースが0件のため予想生成をスキップ（latest.json は上書きしません）')
        return

    # 専用オッズページ(accessO.html)から単勝オッズを取得し各馬に反映する。
    # 出馬表ページにはオッズが載らないため、これを行わないと win_odds=0 のまま
    # 予想が走り popularity 導出・バリュー表示・EV買い目が空になる。
    print('💴 オッズ取得中（専用オッズページ）...')
    market_odds_map = fetch_odds_map(sess, races)
    n_odds = apply_odds_to_races(races, market_odds_map)
    print(f'   オッズ反映: {n_odds}頭 / {len(races)}R')

    surf_counts = {}
    for r in races:
        s = r.get('surface', '?')
        surf_counts[s] = surf_counts.get(s, 0) + 1
    print(f'   コース内訳: {surf_counts}')

    # 全レース・全馬のRL予測を先にまとめて保存（結果との比較用）
    print(f'💾 全レース予測を race_predictions に保存中...')
    for race in races:
        scored_all = calc_all(race, avg_bias)
        if scored_all:
            save_race_predictions(race, scored_all, ROOT)
    print(f'   {len(races)}レース完了')

    selected = select_quality_races(races, avg_bias)
    print(f'⭐ 厳選: {len(selected)}レース'
          + ('（推奨レースなし）' if not selected else ''))

    total_inv = 0
    for i, c in enumerate(selected, 1):
        race = c['race']
        top1 = c['top1']
        scored = c.get('scored', [])
        scored_by_num = {h.get('num', h.get('horse_num')): h for h in scored}
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
        save_bets_db(race['date'], race['id'], bets, ROOT,
                     race=race, scored_by_num=scored_by_num)
        log_bet_simulation(race['date'], c, ROOT)
        save_race_predictions(race, scored, ROOT)

    print(f'💰 投資合計: ¥{total_inv:,}')

    jst_now = datetime.now(JST)
    # 専用オッズページで取れなかったレースは出馬表オッズ（win_odds）で補完する。
    _fallback = build_market_odds_from_races(races)
    for _rid, _om in _fallback.items():
        if not market_odds_map.get(_rid):
            market_odds_map[_rid] = _om
    app_data = to_app_json(selected, races, avg_bias, jst_now,
                           day_type='sunday', market_odds_map=market_odds_map)
    os.makedirs(os.path.dirname(APP_PATH), exist_ok=True)
    with open(APP_PATH, 'w', encoding='utf-8') as f:
        json.dump(app_data, f, ensure_ascii=False, indent=2)
    print(f'✅ アプリJSON保存: {APP_PATH}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['saturday', 'sunday'], required=True)
    parser.add_argument('--force', action='store_true',
                        help='当日予想済みでも強制的に再生成する')
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

        # 土曜分の shadow_bets 記録
        if all_results:
            import sqlite3 as _sq
            _conn = _sq.connect(get_db_path(ROOT))
            _rec_ids = {r[0] for r in _conn.execute(
                'SELECT DISTINCT race_id FROM bets WHERE date=?', (target_date,)).fetchall()}
            _conn.close()
            record_all_shadow_bets(all_results, ROOT,
                                   bias_data=avg_bias,
                                   recommended_race_ids=_rec_ids)

        # 翌日（日曜）予想
        predict_next_day(sess, hist_path, avg_bias, jst_now, force=args.force)

    checkpoint_db(db_path)
    checkpoint_db(hist_path)


if __name__ == '__main__':
    main()

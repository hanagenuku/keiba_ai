"""
全レースシャドウ記録モジュール。

結果取得時に「もし推奨外レースも全部買っていたら」のシミュレーション結果を
shadow_bets テーブルに記録する。

呼び出し元: scripts/sunday_results.py（および scripts/weekend.py）
タイミング: fetch_and_save_results() 後、init_engine() のキャッシュが
            まだ当日結果を含まない状態で呼ぶこと（post-hocな再現予測）。
"""
from src.utils.db import _connect, get_db_path
from src.features.engine import calc_all
from src.betting.ev_filter import classify_race_chaos


def record_all_shadow_bets(all_results, base_dir, bias_data=None,
                            recommended_race_ids=None):
    """
    当日全レースのシャドウベット結果を shadow_bets テーブルに記録する。

    Parameters
    ----------
    all_results : list of dict
        fetch_results() の戻り値。各要素に 'finishers', 'dividends', 'info' を含む。
    base_dir : str
        プロジェクトルート（keiba.db のパス解決に使う）。
    bias_data : dict, optional
        calc_all に渡すバイアスデータ。None の場合はフラット。
    recommended_race_ids : set of str, optional
        実際に推奨されたレースの race_id 集合。None の場合は全て非推奨扱い。
    """
    if not all_results:
        return

    db_path = get_db_path(base_dir)
    recommended_race_ids = recommended_race_ids or set()
    bias_data = bias_data or {}

    rows = []
    for result in all_results:
        try:
            row = _build_shadow_row(result, bias_data, recommended_race_ids)
            if row:
                rows.append(row)
        except Exception as e:
            rid = result.get('id', '?')
            print(f'  ⚠ shadow_bets 記録スキップ ({rid}): {e}')

    if not rows:
        print('  ⚠ shadow_bets: 記録対象なし')
        return

    conn = _connect(db_path)
    conn.executemany(
        '''INSERT OR IGNORE INTO shadow_bets (
            date, race_id, racecourse, race_num, race_class, num_horses,
            surface, distance, chaos_grade,
            rl1_num, rl1_name, rl1_win_prob, rl1_cal_prob,
            rl2_num, rl2_name, rl3_num, rl3_name,
            winner_num, winner_pop, winner_odds, second_num, third_num,
            shadow_tansho_hit, shadow_tansho_payout,
            shadow_fukusho_hit, shadow_fukusho_payout,
            shadow_umaren_hit, shadow_umaren_payout,
            shadow_wide_hit, shadow_wide_payout,
            shadow_sanrenp_hit, shadow_sanrenp_payout,
            was_recommended
        ) VALUES (
            :date, :race_id, :racecourse, :race_num, :race_class, :num_horses,
            :surface, :distance, :chaos_grade,
            :rl1_num, :rl1_name, :rl1_win_prob, :rl1_cal_prob,
            :rl2_num, :rl2_name, :rl3_num, :rl3_name,
            :winner_num, :winner_pop, :winner_odds, :second_num, :third_num,
            :shadow_tansho_hit, :shadow_tansho_payout,
            :shadow_fukusho_hit, :shadow_fukusho_payout,
            :shadow_umaren_hit, :shadow_umaren_payout,
            :shadow_wide_hit, :shadow_wide_payout,
            :shadow_sanrenp_hit, :shadow_sanrenp_payout,
            :was_recommended
        )''',
        rows,
    )
    conn.commit()
    conn.close()
    print(f'  ✅ shadow_bets: {len(rows)}レース記録完了')


def _build_shadow_row(result, bias_data, recommended_race_ids):
    """1レース分の shadow_bets 行データを構築する。"""
    race_id  = result.get('id', '')
    finishers = result.get('finishers', [])
    divs      = result.get('dividends', {})

    if not finishers or not race_id:
        return None

    # ── レース情報 ────────────────────────────────────────────────────
    race = {
        'id':           race_id,
        'date':         result.get('date', ''),
        'racecourse':   result.get('racecourse', ''),
        'race_num':     result.get('race_num', 0),
        'race_name':    result.get('race_name', ''),
        'race_class':   result.get('race_class', ''),
        'surface':      result.get('surface', '芝'),
        'distance':     result.get('distance', 1600),
        'track_condition': result.get('track_condition', '良'),
        'num_horses':   len(finishers),
        'horses': [
            {
                'num':           h.get('num', 0),
                'name':          h.get('name', ''),
                'running_style': h.get('running_style', '差し'),
                'weight_load':   h.get('weight_load', 56.0),
                'win_odds':      h.get('win_odds', 0),
                'bracket':       h.get('bracket', 0),
                'jockey':        h.get('jockey', ''),
                'trainer':       h.get('trainer', ''),
                'sex':           h.get('sex', '牡'),
                'age':           h.get('age', 4),
                'body_weight':   h.get('body_weight', 0),
            }
            for h in finishers
        ],
    }

    # ── AI予測（エンジンの現在キャッシュを使う） ───────────────────────
    try:
        scored = calc_all(race, bias_data)
    except Exception:
        scored = []

    chaos_grade = classify_race_chaos(scored) if scored else 'B'

    # RL順（rl_rank または win_prob降順）でソート
    by_rl = sorted(scored, key=lambda h: h.get('rl_rank', 99)) if scored else []
    rl1 = by_rl[0] if len(by_rl) > 0 else {}
    rl2 = by_rl[1] if len(by_rl) > 1 else {}
    rl3 = by_rl[2] if len(by_rl) > 2 else {}

    # ── 実結果 ───────────────────────────────────────────────────────
    by_place  = sorted(finishers, key=lambda h: h.get('place', 99))
    winner    = by_place[0] if by_place else {}
    second_h  = by_place[1] if len(by_place) > 1 else {}
    third_h   = by_place[2] if len(by_place) > 2 else {}

    winner_num = winner.get('num')
    second_num = second_h.get('num')
    third_num  = third_h.get('num')
    top3_nums  = {n for n in [winner_num, second_num, third_num] if n}

    # 人気順位（win_odds昇順でのランク）
    by_odds  = sorted(finishers, key=lambda h: h.get('win_odds') or 99)
    pop_rank = next((i + 1 for i, h in enumerate(by_odds) if h.get('num') == winner_num), None)

    # ── 払戻 ─────────────────────────────────────────────────────────
    fuku_map = {f['num']: f['payout'] for f in divs.get('fukusho', [])}
    wide_map = {
        tuple(sorted([w['nums'][0], w['nums'][1]])): w['payout']
        for w in divs.get('wide', [])
    }

    rl1_num = rl1.get('num') or rl1.get('horse_num')
    rl2_num = rl2.get('num') or rl2.get('horse_num')
    rl3_num = rl3.get('num') or rl3.get('horse_num')

    # 単勝
    tan_hit  = 1 if rl1_num and rl1_num == winner_num else 0
    tan_pay  = divs.get('tansho', {}).get('payout') if tan_hit else None

    # 複勝
    fuku_hit = 1 if rl1_num and rl1_num in top3_nums else 0
    fuku_pay = fuku_map.get(rl1_num) if fuku_hit else None

    # 馬連
    rl12 = {rl1_num, rl2_num} if rl1_num and rl2_num else set()
    top2 = {winner_num, second_num}
    uren_hit = 1 if rl12 and rl12 == top2 else 0
    uren_pay = divs.get('umaren', {}).get('payout') if uren_hit else None

    # ワイド
    wide_hit = 1 if rl12 and rl12 <= top3_nums else 0
    wide_pay = wide_map.get(tuple(sorted(rl12))) if wide_hit else None

    # 三連複
    rl123 = {rl1_num, rl2_num, rl3_num} if rl1_num and rl2_num and rl3_num else set()
    san_hit = 1 if rl123 and rl123 == top3_nums else 0
    san_pay = divs.get('sanrenpuku', {}).get('payout') if san_hit else None

    return {
        'date':             result.get('date', ''),
        'race_id':          race_id,
        'racecourse':       result.get('racecourse', ''),
        'race_num':         result.get('race_num', 0),
        'race_class':       result.get('race_class', ''),
        'num_horses':       len(finishers),
        'surface':          result.get('surface', '芝'),
        'distance':         result.get('distance', 1600),
        'chaos_grade':      chaos_grade,
        'rl1_num':          rl1_num,
        'rl1_name':         rl1.get('name', ''),
        'rl1_win_prob':     rl1.get('win_prob') or rl1.get('pn'),
        'rl1_cal_prob':     rl1.get('cal_prob'),
        'rl2_num':          rl2_num,
        'rl2_name':         rl2.get('name', ''),
        'rl3_num':          rl3_num,
        'rl3_name':         rl3.get('name', ''),
        'winner_num':       winner_num,
        'winner_pop':       pop_rank,
        'winner_odds':      winner.get('win_odds'),
        'second_num':       second_num,
        'third_num':        third_num,
        'shadow_tansho_hit':    tan_hit,
        'shadow_tansho_payout': tan_pay,
        'shadow_fukusho_hit':   fuku_hit,
        'shadow_fukusho_payout': fuku_pay,
        'shadow_umaren_hit':    uren_hit,
        'shadow_umaren_payout': uren_pay,
        'shadow_wide_hit':      wide_hit,
        'shadow_wide_payout':   wide_pay,
        'shadow_sanrenp_hit':   san_hit,
        'shadow_sanrenp_payout': san_pay,
        'was_recommended':  1 if race_id in recommended_race_ids else 0,
    }

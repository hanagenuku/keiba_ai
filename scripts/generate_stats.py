#!/usr/bin/env python3
"""keiba.db から stats.json を生成してアプリ向けに公開する"""
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.db import get_db_path


def generate_stats(base_dir=None):
    base_dir = base_dir or ROOT
    db_path = get_db_path(base_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {}

    # ── 実ベット: 週別ROI ────────────────────────────────────────────────
    bets = conn.execute(
        'SELECT date, bet_type, amount, is_hit, payout FROM bets WHERE is_hit >= 0 ORDER BY date'
    ).fetchall()

    weekly = {}
    for b in bets:
        week = b['date'][:7]  # YYYY-MM
        w = weekly.setdefault(week, {'invested': 0, 'recovered': 0, 'hit': 0, 'total': 0})
        w['invested']  += b['amount']
        w['recovered'] += b['payout']
        w['total']     += 1
        if b['is_hit'] == 1:
            w['hit'] += 1

    stats['weekly_roi'] = [
        {
            'month':    k,
            'invested': v['invested'],
            'recovered': v['recovered'],
            'roi':      round(v['recovered'] / v['invested'] * 100, 1) if v['invested'] else 0,
            'hit':      v['hit'],
            'total':    v['total'],
            'hit_rate': round(v['hit'] / v['total'] * 100, 1) if v['total'] else 0,
        }
        for k, v in sorted(weekly.items())
    ]

    # ── 実ベット: 券種別集計 ──────────────────────────────────────────────
    by_type = {}
    for b in bets:
        t = b['bet_type']
        d = by_type.setdefault(t, {'invested': 0, 'recovered': 0, 'hit': 0, 'total': 0})
        d['invested']  += b['amount']
        d['recovered'] += b['payout']
        d['total']     += 1
        if b['is_hit'] == 1:
            d['hit'] += 1

    stats['by_type'] = [
        {
            'type':     k,
            'invested': v['invested'],
            'recovered': v['recovered'],
            'roi':      round(v['recovered'] / v['invested'] * 100, 1) if v['invested'] else 0,
            'hit':      v['hit'],
            'total':    v['total'],
            'hit_rate': round(v['hit'] / v['total'] * 100, 1) if v['total'] else 0,
        }
        for k, v in sorted(by_type.items())
    ]

    # ── shadow_bets: RL精度集計 ───────────────────────────────────────────
    shadows = conn.execute(
        '''SELECT date, racecourse, race_num, race_class, num_horses, chaos_grade,
                  rl1_num, rl2_num, rl3_num,
                  winner_num, second_num, third_num,
                  shadow_tansho_hit,  shadow_tansho_payout,
                  shadow_fukusho_hit, shadow_fukusho_payout,
                  shadow_umaren_hit,  shadow_umaren_payout,
                  shadow_wide_hit,    shadow_wide_payout,
                  shadow_sanrenp_hit, shadow_sanrenp_payout,
                  was_recommended
           FROM shadow_bets ORDER BY date DESC LIMIT 1000'''
    ).fetchall()

    rl = {'rl1_win': 0, 'rl1_top3': 0, 'rl2_top3': 0, 'rl3_top3': 0, 'total': 0}
    chaos_rl = {}  # chaos_grade → rl stats
    for s in shadows:
        top3 = {s['winner_num'], s['second_num'], s['third_num']} - {None}
        rl['total'] += 1
        if s['rl1_num'] == s['winner_num']:
            rl['rl1_win'] += 1
        if s['rl1_num'] in top3:
            rl['rl1_top3'] += 1
        if s['rl2_num'] in top3:
            rl['rl2_top3'] += 1
        if s['rl3_num'] in top3:
            rl['rl3_top3'] += 1

        cg = s['chaos_grade'] or 'B'
        c = chaos_rl.setdefault(cg, {'rl1_win': 0, 'rl1_top3': 0, 'total': 0})
        c['total'] += 1
        if s['rl1_num'] == s['winner_num']:
            c['rl1_win'] += 1
        if s['rl1_num'] in top3:
            c['rl1_top3'] += 1

    n = rl['total']
    stats['rl_accuracy'] = {
        'total_races':   n,
        'rl1_win_rate':  round(rl['rl1_win']  / n * 100, 1) if n else 0,
        'rl1_top3_rate': round(rl['rl1_top3'] / n * 100, 1) if n else 0,
        'rl2_top3_rate': round(rl['rl2_top3'] / n * 100, 1) if n else 0,
        'rl3_top3_rate': round(rl['rl3_top3'] / n * 100, 1) if n else 0,
        'by_chaos': {
            grade: {
                'total':         d['total'],
                'rl1_win_rate':  round(d['rl1_win']  / d['total'] * 100, 1) if d['total'] else 0,
                'rl1_top3_rate': round(d['rl1_top3'] / d['total'] * 100, 1) if d['total'] else 0,
            }
            for grade, d in sorted(chaos_rl.items())
        },
    }

    # ── 直近レース一覧（最新50件） ────────────────────────────────────────
    stats['recent_races'] = [
        {
            'date':       s['date'],
            'racecourse': s['racecourse'],
            'race_num':   s['race_num'],
            'race_class': s['race_class'],
            'chaos':      s['chaos_grade'],
            'rl1':        s['rl1_num'],
            'rl2':        s['rl2_num'],
            'rl3':        s['rl3_num'],
            'win':        s['winner_num'],
            'sec':        s['second_num'],
            'thr':        s['third_num'],
            'tan_hit':    s['shadow_tansho_hit'],
            'fuku_hit':   s['shadow_fukusho_hit'],
            'fuku_pay':   s['shadow_fukusho_payout'],
            'wide_hit':   s['shadow_wide_hit'],
            'rec':        s['was_recommended'],
        }
        for s in list(shadows)[:50]
    ]

    stats['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    conn.close()

    out_path = os.path.join(base_dir, 'data', 'stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'✅ stats.json 生成: {out_path}  '
          f'({len(stats["recent_races"])}レース / {len(bets)}ベット)')


if __name__ == '__main__':
    generate_stats(ROOT)

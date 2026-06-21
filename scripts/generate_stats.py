#!/usr/bin/env python3
"""keiba.db / history.db から stats.json を生成してアプリ向けに公開する"""
import json
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.db import get_db_path, get_history_db_path


def _dist_cat(d):
    if not d: return '不明'
    if d <= 1400: return '短距離(〜1400)'
    if d <= 1800: return 'マイル(1401-1800)'
    if d <= 2200: return '中距離(1801-2200)'
    return '長距離(2201〜)'


def _head_cat(n):
    if not n: return '不明'
    if n <= 8:  return '少頭数(〜8頭)'
    if n <= 13: return '中頭数(9-13頭)'
    return '多頭数(14頭〜)'


def _class_cat(c):
    if not c: return 'その他'
    if any(x in c for x in ['G1', 'G２', 'G2', 'G３', 'G3', '重賞', 'グランプリ']):
        return '重賞'
    if any(x in c for x in ['OP', 'オープン', 'リステッド']):
        return 'OP/L'
    if '3勝' in c or '1600万' in c: return '3勝クラス'
    if '2勝' in c or '1000万' in c: return '2勝クラス'
    if '1勝' in c or '500万' in c:  return '1勝クラス'
    if '未勝利' in c: return '未勝利'
    if '新馬' in c:   return '新馬'
    return 'その他'


def _calc_upset_patterns(shadows):
    """AIの盲点パターンを shadow_bets から集計する。

    「upset」= AI上位3頭(rl1/rl2/rl3)に入っていない馬が実際の複勝内(1-3着)に来たケース。
    「full_miss」= AIの上位3頭が1頭も複勝内に入らなかったケース。
    様々な軸で集計し、盲点の大きい組み合わせをランキング表示する。
    """
    groups = {}

    def add(dim, key, upset, full_miss, longshot):
        g = groups.setdefault((dim, key), {'total': 0, 'upset': 0, 'full_miss': 0, 'longshot': 0})
        g['total']    += 1
        g['upset']    += int(upset)
        g['full_miss']+= int(full_miss)
        g['longshot'] += int(longshot)

    for s in shadows:
        ai3  = {s['rl1_num'], s['rl2_num'], s['rl3_num']} - {None}
        act3 = {s['winner_num'], s['second_num'], s['third_num']} - {None}
        if not ai3 or not act3:
            continue

        upset     = bool(act3 - ai3)           # AI外が複勝内に1頭以上
        full_miss = len(ai3 & act3) == 0       # AI上位3頭が全滅
        longshot  = bool(                       # 10倍超の馬が複勝内
            (s['winner_odds'] or 0) >= 10.0 and s['winner_num'] not in ai3
        )

        chaos = s['chaos_grade'] or 'B'
        heads = _head_cat(s['num_horses'])
        surf  = s['surface'] or '不明'
        dist  = _dist_cat(s['distance'])
        rc    = s['racecourse'] or '不明'
        cls   = _class_cat(s['race_class'])

        add('chaos',     chaos, upset, full_miss, longshot)
        add('heads',     heads, upset, full_miss, longshot)
        add('surface',   surf,  upset, full_miss, longshot)
        add('distance',  dist,  upset, full_miss, longshot)
        add('racecourse', rc,   upset, full_miss, longshot)
        add('class',     cls,   upset, full_miss, longshot)

        # 複合軸（盲点ランキング用）
        add('combo', f'{surf}・{dist}',                 upset, full_miss, longshot)
        add('combo', f'{chaos}グレード・{heads}',        upset, full_miss, longshot)
        add('combo', f'{rc}・{chaos}グレード',           upset, full_miss, longshot)
        add('combo', f'{surf}・{chaos}グレード・{heads}', upset, full_miss, longshot)

    def to_stat(v):
        n = v['total']
        return {
            'total':          n,
            'upset_rate':     round(v['upset']     / n * 100, 1) if n else 0,
            'full_miss_rate': round(v['full_miss'] / n * 100, 1) if n else 0,
            'longshot_rate':  round(v['longshot']  / n * 100, 1) if n else 0,
        }

    def dim_list(dim):
        return sorted(
            [{'label': k[1], **to_stat(v)}
             for k, v in groups.items() if k[0] == dim],
            key=lambda x: x['label']
        )

    # 盲点ランキング: 複合軸でデータ5件以上、upset_rate降順
    blind_spots = sorted(
        [{'label': k[1], **to_stat(v)}
         for k, v in groups.items()
         if k[0] == 'combo' and v['total'] >= 5],
        key=lambda x: -x['upset_rate']
    )[:10]

    total_g = {k: v for k, v in groups.items() if k[0] == 'chaos'}
    total   = sum(v['total']  for v in total_g.values())
    upset_t = sum(v['upset']  for v in total_g.values())

    return {
        'total_races':        total,
        'overall_upset_rate': round(upset_t / total * 100, 1) if total else 0,
        'by_chaos':     dim_list('chaos'),
        'by_heads':     dim_list('heads'),
        'by_surface':   dim_list('surface'),
        'by_distance':  dim_list('distance'),
        'by_racecourse': dim_list('racecourse'),
        'by_class':     dim_list('class'),
        'blind_spots':  blind_spots,
    }


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
                  surface, distance,
                  rl1_num, rl2_num, rl3_num,
                  winner_num, second_num, third_num,
                  winner_odds,
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

    # ── AIの盲点パターン ─────────────────────────────────────────────────
    stats['upset_patterns'] = _calc_upset_patterns(shadows)

    # ── 結果取得ステータス ────────────────────────────────────────────────
    hist_path = get_history_db_path(base_dir)
    rs: dict = {'last_date': None, 'races': 0, 'venues': [],
                'workflow_run_at': None, 'workflow_result': None}
    if os.path.exists(hist_path):
        try:
            hconn = sqlite3.connect(hist_path)
            hconn.row_factory = sqlite3.Row
            row = hconn.execute('SELECT MAX(date) as d FROM race_history').fetchone()
            last_date = row['d'] if row else None
            if last_date:
                rows = hconn.execute(
                    'SELECT racecourse, COUNT(*) as cnt FROM race_history '
                    'WHERE date=? GROUP BY racecourse ORDER BY cnt DESC',
                    (last_date,)
                ).fetchall()
                rs['last_date'] = last_date
                rs['races'] = sum(r['cnt'] for r in rows)
                rs['venues'] = [{'name': r['racecourse'], 'races': r['cnt']} for r in rows]
            hconn.close()
        except Exception:
            pass

    wf_path = os.path.join(base_dir, 'data', 'workflow_status.json')
    if os.path.exists(wf_path):
        try:
            with open(wf_path, encoding='utf-8') as f:
                wf = json.load(f)
            rs['workflow_run_at'] = wf.get('updated_at')
            rs['workflow_result'] = wf.get('status')
        except Exception:
            pass
    stats['results_status'] = rs

    stats['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    conn.close()

    out_path = os.path.join(base_dir, 'data', 'stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'✅ stats.json 生成: {out_path}  '
          f'({len(stats["recent_races"])}レース / {len(bets)}ベット)')


if __name__ == '__main__':
    generate_stats(ROOT)

#!/usr/bin/env python3
"""keiba.db / history.db から stats.json を生成してアプリ向けに公開する"""
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
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


def calc_divergence_analysis(conn):
    """AI予測 vs 市場オッズの乖離分析 + 結果との相関。

    馬ごとに「AI確率 / 市場確率」の比率を算出し、その比率帯別に
    実際の的中率（1着率・3着内率）を集計する。
    AIが市場より高く評価した馬が実際に走ったか？を定量化。

    Returns dict for stats.json, or None if data insufficient.
    """
    rows = conn.execute('''
        SELECT date, race_id, racecourse, race_num,
               horse_num, horse_name, popularity,
               tansho_odds, rl_rank, win_prob, cal_prob, fuku_prob,
               actual_place
        FROM race_predictions
        WHERE actual_place IS NOT NULL
          AND tansho_odds IS NOT NULL AND tansho_odds > 0
          AND win_prob IS NOT NULL AND win_prob > 0
    ''').fetchall()

    if len(rows) < 20:
        return None

    races = defaultdict(list)
    for r in rows:
        races[r['race_id']].append(dict(r))

    races = {k: v for k, v in races.items() if len(v) >= 4}
    if not races:
        return None

    records = []
    for race_id, horses in races.items():
        raw_mkt = [1.0 / h['tansho_odds'] for h in horses]
        mkt_total = sum(raw_mkt)
        if mkt_total <= 0:
            continue
        mkt_probs = [p / mkt_total for p in raw_mkt]

        for i, h in enumerate(horses):
            ai_p = h['win_prob']
            mkt_p = mkt_probs[i]
            ratio = ai_p / mkt_p if mkt_p > 0.001 else 0
            records.append({
                'date': h['date'],
                'race_id': race_id,
                'racecourse': h.get('racecourse', ''),
                'horse_num': h['horse_num'],
                'horse_name': h.get('horse_name', ''),
                'popularity': h.get('popularity', 99),
                'rl_rank': h.get('rl_rank', 99),
                'ai_prob': round(ai_p, 4),
                'mkt_prob': round(mkt_p, 4),
                'ratio': round(ratio, 3),
                'actual_place': h['actual_place'],
                'won': h['actual_place'] == 1,
                'top3': h['actual_place'] <= 3,
            })

    if not records:
        return None

    def _ratio_bucket(r):
        if r < 0.5:   return 'AI<<市場(<0.5x)'
        if r < 0.8:   return 'AI<市場(0.5-0.8x)'
        if r < 1.2:   return '一致(0.8-1.2x)'
        if r < 2.0:   return 'AI>市場(1.2-2.0x)'
        if r < 3.0:   return 'AI>>市場(2.0-3.0x)'
        return 'AI>>>市場(3.0x+)'

    bucket_order = [
        'AI<<市場(<0.5x)', 'AI<市場(0.5-0.8x)', '一致(0.8-1.2x)',
        'AI>市場(1.2-2.0x)', 'AI>>市場(2.0-3.0x)', 'AI>>>市場(3.0x+)',
    ]

    buckets = defaultdict(list)
    for rec in records:
        buckets[_ratio_bucket(rec['ratio'])].append(rec)

    bucket_stats = []
    for bname in bucket_order:
        br = buckets.get(bname, [])
        if not br:
            continue
        n = len(br)
        wins = sum(1 for r in br if r['won'])
        top3s = sum(1 for r in br if r['top3'])
        bucket_stats.append({
            'bucket': bname,
            'count': n,
            'win_rate': round(wins / n * 100, 1),
            'top3_rate': round(top3s / n * 100, 1),
            'avg_ai_prob': round(sum(r['ai_prob'] for r in br) / n, 4),
            'avg_mkt_prob': round(sum(r['mkt_prob'] for r in br) / n, 4),
        })

    ai_fav_wins = sum(1 for r in records if r['rl_rank'] == 1 and r['won'])
    ai_fav_total = sum(1 for r in records if r['rl_rank'] == 1)
    mkt_fav_wins = sum(1 for r in records if r['popularity'] == 1 and r['won'])
    mkt_fav_total = sum(1 for r in records if r['popularity'] == 1)

    agree = sum(1 for r in records
                if r['rl_rank'] == 1 and r['popularity'] == 1)
    disagree_total = ai_fav_total - agree
    disagree_ai_wins = sum(1 for r in records
                          if r['rl_rank'] == 1 and r['popularity'] != 1 and r['won'])
    disagree_mkt_wins = sum(1 for r in records
                           if r['popularity'] == 1 and r['rl_rank'] != 1 and r['won'])

    top_overvalued = sorted(
        [r for r in records if r['ratio'] >= 2.0],
        key=lambda x: -x['ratio'],
    )[:10]
    top_undervalued = sorted(
        [r for r in records if r['ratio'] <= 0.5],
        key=lambda x: x['ratio'],
    )[:10]

    def _simplify(r):
        return {
            'date': r['date'], 'racecourse': r['racecourse'],
            'horse': r['horse_name'], 'num': r['horse_num'],
            'ai': r['ai_prob'], 'mkt': r['mkt_prob'],
            'ratio': r['ratio'], 'place': r['actual_place'],
        }

    by_date = defaultdict(lambda: {'n': 0, 'ai_wins': 0, 'mkt_wins': 0,
                                    'agree': 0, 'disagree_ai': 0, 'disagree_mkt': 0})
    for r in records:
        d = r['date'][:10] if r['date'] else 'unknown'
        by_date[d]['n'] += 1
        if r['rl_rank'] == 1:
            if r['won']:
                by_date[d]['ai_wins'] += 1
            if r['popularity'] == 1:
                by_date[d]['agree'] += 1
            else:
                if r['won']:
                    by_date[d]['disagree_ai'] += 1
        if r['popularity'] == 1 and r['rl_rank'] != 1:
            if r['won']:
                by_date[d]['disagree_mkt'] += 1

    daily = [{'date': d, **v} for d, v in sorted(by_date.items())]

    return {
        'total_horses': len(records),
        'total_races': len(races),
        'bucket_stats': bucket_stats,
        'ai_fav_win_rate': round(ai_fav_wins / ai_fav_total * 100, 1) if ai_fav_total else 0,
        'mkt_fav_win_rate': round(mkt_fav_wins / mkt_fav_total * 100, 1) if mkt_fav_total else 0,
        'agree_count': agree,
        'disagree_count': disagree_total,
        'disagree_ai_wins': disagree_ai_wins,
        'disagree_mkt_wins': disagree_mkt_wins,
        'top_overvalued': [_simplify(r) for r in top_overvalued],
        'top_undervalued': [_simplify(r) for r in top_undervalued],
        'daily': daily,
    }


def calc_odds_movement_analysis(conn):
    """朝オッズ→直前オッズの変動と結果の相関分析。

    race_predictions.tansho_odds（朝オッズ）と odds_snapshots（直前オッズ）を突合し、
    変動幅×結果の関係を集計する。

    Returns dict or None if insufficient data.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if 'odds_snapshots' not in tables:
        return None

    rows = conn.execute('''
        SELECT rp.date, rp.race_id, rp.racecourse, rp.race_num,
               rp.horse_num, rp.horse_name, rp.popularity,
               rp.tansho_odds AS morning_odds, rp.rl_rank,
               rp.actual_place,
               os.tansho AS chokuzen_odds, os.captured_at
        FROM race_predictions rp
        INNER JOIN (
            SELECT race_id, horse_num, tansho, captured_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY race_id, horse_num
                       ORDER BY captured_at DESC
                   ) AS rn
            FROM odds_snapshots
            WHERE tansho IS NOT NULL AND tansho > 0
        ) os ON os.race_id = rp.race_id
              AND os.horse_num = rp.horse_num
              AND os.rn = 1
        WHERE rp.actual_place IS NOT NULL
          AND rp.tansho_odds IS NOT NULL AND rp.tansho_odds > 0
    ''').fetchall()

    if len(rows) < 10:
        return None

    records = []
    for r in [dict(row) for row in rows]:
        morning = r['morning_odds']
        chokuzen = r['chokuzen_odds']
        if morning <= 0 or chokuzen <= 0:
            continue
        change_pct = (chokuzen - morning) / morning * 100
        records.append({
            'date': r['date'],
            'race_id': r['race_id'],
            'racecourse': r.get('racecourse', ''),
            'horse_num': r['horse_num'],
            'horse_name': r.get('horse_name', ''),
            'popularity': r.get('popularity', 99),
            'rl_rank': r.get('rl_rank', 99),
            'morning_odds': round(morning, 1),
            'chokuzen_odds': round(chokuzen, 1),
            'change_pct': round(change_pct, 1),
            'actual_place': r['actual_place'],
            'won': r['actual_place'] == 1,
            'top3': r['actual_place'] <= 3,
        })

    if not records:
        return None

    def _move_bucket(pct):
        if pct <= -30:   return '急騰(↓30%+)'
        if pct <= -15:   return '上昇(↓15-30%)'
        if pct < 15:     return '横ばい(±15%)'
        if pct < 30:     return '下降(↑15-30%)'
        return '急落(↑30%+)'

    bucket_order = ['急騰(↓30%+)', '上昇(↓15-30%)', '横ばい(±15%)',
                    '下降(↑15-30%)', '急落(↑30%+)']

    buckets = defaultdict(list)
    for rec in records:
        buckets[_move_bucket(rec['change_pct'])].append(rec)

    bucket_stats = []
    for bname in bucket_order:
        br = buckets.get(bname, [])
        if not br:
            continue
        n = len(br)
        wins = sum(1 for r in br if r['won'])
        top3s = sum(1 for r in br if r['top3'])
        bucket_stats.append({
            'bucket': bname,
            'count': n,
            'win_rate': round(wins / n * 100, 1),
            'top3_rate': round(top3s / n * 100, 1),
            'avg_change_pct': round(sum(r['change_pct'] for r in br) / n, 1),
        })

    big_risers = sorted(
        [r for r in records if r['change_pct'] <= -20],
        key=lambda x: x['change_pct'],
    )[:10]
    big_fallers = sorted(
        [r for r in records if r['change_pct'] >= 20],
        key=lambda x: -x['change_pct'],
    )[:10]

    def _simplify(r):
        return {
            'date': r['date'], 'racecourse': r['racecourse'],
            'horse': r['horse_name'], 'num': r['horse_num'],
            'morning': r['morning_odds'], 'chokuzen': r['chokuzen_odds'],
            'change': r['change_pct'], 'place': r['actual_place'],
        }

    ai_agrees_market = []
    ai_disagrees_market = []
    for r in records:
        if r['change_pct'] <= -20 and r['rl_rank'] and r['rl_rank'] <= 3:
            ai_agrees_market.append(r)
        elif r['change_pct'] <= -20 and (not r['rl_rank'] or r['rl_rank'] > 5):
            ai_disagrees_market.append(r)

    return {
        'total_horses': len(records),
        'bucket_stats': bucket_stats,
        'big_risers': [_simplify(r) for r in big_risers],
        'big_fallers': [_simplify(r) for r in big_fallers],
        'ai_agrees_rising': {
            'count': len(ai_agrees_market),
            'top3_rate': round(
                sum(1 for r in ai_agrees_market if r['top3']) /
                len(ai_agrees_market) * 100, 1
            ) if ai_agrees_market else 0,
        },
        'ai_ignores_rising': {
            'count': len(ai_disagrees_market),
            'top3_rate': round(
                sum(1 for r in ai_disagrees_market if r['top3']) /
                len(ai_disagrees_market) * 100, 1
            ) if ai_disagrees_market else 0,
        },
    }


def _save_divergence_weekly(divergence, odds_movement, base_dir):
    """divergence_weekly.json に週次蓄積する。"""
    path = os.path.join(base_dir, 'data', 'divergence_weekly.json')
    existing = []
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []

    today = datetime.now().strftime('%Y-%m-%d')
    entry = {
        'date': today,
        'divergence': divergence,
        'odds_movement': odds_movement,
    }

    existing_dates = {e['date'] for e in existing}
    if today in existing_dates:
        for i, e in enumerate(existing):
            if e['date'] == today:
                existing[i] = entry
                break
    else:
        existing.append(entry)

    existing.sort(key=lambda x: x['date'])

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def calc_model_kpi(conn):
    """AI vs 市場の log-loss を race_predictions から算出する。

    レースごとに:
      - AI確率 = win_prob（softmax出力、合計1）
      - 市場確率 = (1/tansho_odds) を同一レース内で正規化（合計1）
      - 正解 = actual_place == 1
    log-loss = -mean( y*log(p) + (1-y)*log(1-p) )

    Returns dict with overall + weekly breakdown, or None if data insufficient.
    """
    EPS = 1e-7

    rows = conn.execute('''
        SELECT date, race_id, horse_num, win_prob, tansho_odds, actual_place
        FROM race_predictions
        WHERE actual_place IS NOT NULL
          AND win_prob IS NOT NULL AND win_prob > 0
          AND tansho_odds IS NOT NULL AND tansho_odds > 0
    ''').fetchall()

    if not rows:
        return None

    races = defaultdict(list)
    for r in rows:
        races[r['race_id']].append(r)

    # 不完全なレース（馬が1頭しかない等）を除外
    races = {k: v for k, v in races.items() if len(v) >= 2}
    if not races:
        return None

    ai_losses = []
    mkt_losses = []
    weekly_data = defaultdict(lambda: {'ai_sum': 0.0, 'mkt_sum': 0.0, 'n': 0,
                                       'date': None, 'races': 0})

    for race_id, horses in races.items():
        raw_mkt = [1.0 / h['tansho_odds'] for h in horses]
        mkt_total = sum(raw_mkt)
        if mkt_total <= 0:
            continue

        mkt_probs = [p / mkt_total for p in raw_mkt]
        date = horses[0]['date']
        week = date[:10] if date else 'unknown'

        for i, h in enumerate(horses):
            y = 1.0 if h['actual_place'] == 1 else 0.0
            ai_p = max(min(h['win_prob'], 1.0 - EPS), EPS)
            mkt_p = max(min(mkt_probs[i], 1.0 - EPS), EPS)

            ai_ll = -(y * math.log(ai_p) + (1 - y) * math.log(1 - ai_p))
            mkt_ll = -(y * math.log(mkt_p) + (1 - y) * math.log(1 - mkt_p))

            ai_losses.append(ai_ll)
            mkt_losses.append(mkt_ll)

            wd = weekly_data[week]
            wd['ai_sum'] += ai_ll
            wd['mkt_sum'] += mkt_ll
            wd['n'] += 1
            wd['date'] = date

        weekly_data[week]['races'] += 1

    n_total = len(ai_losses)
    if n_total == 0:
        return None

    ai_avg = sum(ai_losses) / n_total
    mkt_avg = sum(mkt_losses) / n_total
    delta = ai_avg - mkt_avg

    weekly_list = []
    for week in sorted(weekly_data.keys()):
        wd = weekly_data[week]
        if wd['n'] == 0:
            continue
        w_ai = wd['ai_sum'] / wd['n']
        w_mkt = wd['mkt_sum'] / wd['n']
        weekly_list.append({
            'date': week,
            'races': wd['races'],
            'horses': wd['n'],
            'ai_logloss': round(w_ai, 4),
            'mkt_logloss': round(w_mkt, 4),
            'delta': round(w_ai - w_mkt, 4),
        })

    return {
        'total_races': len(races),
        'total_horses': n_total,
        'ai_logloss': round(ai_avg, 4),
        'mkt_logloss': round(mkt_avg, 4),
        'delta': round(delta, 4),
        'verdict': 'AI優位' if delta < -0.001 else ('市場優位' if delta > 0.001 else '同等'),
        'weekly': weekly_list,
    }


def _save_kpi_weekly(kpi, base_dir):
    """kpi_weekly.json に累積追記する（週ごとの最新値を上書き）。"""
    path = os.path.join(base_dir, 'data', 'kpi_weekly.json')
    existing = []
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing_dates = {e['date'] for e in existing}
    for w in kpi.get('weekly', []):
        if w['date'] not in existing_dates:
            existing.append(w)
            existing_dates.add(w['date'])
        else:
            for i, e in enumerate(existing):
                if e['date'] == w['date']:
                    existing[i] = w
                    break

    existing.sort(key=lambda x: x['date'])

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


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

    # ── モデルKPI: AI vs 市場 log-loss ────────────────────────────────────
    kpi = calc_model_kpi(conn)
    if kpi:
        stats['model_kpi'] = kpi
        _save_kpi_weekly(kpi, base_dir)

    # ── AI vs 市場 乖離分析 ──────────────────────────────────────────────
    div = calc_divergence_analysis(conn)
    if div:
        stats['divergence_analysis'] = div

    # ── オッズ変動 × 結果 分析 ───────────────────────────────────────────
    odds_mv = calc_odds_movement_analysis(conn)
    if odds_mv:
        stats['odds_movement'] = odds_mv

    # 週次蓄積
    _save_divergence_weekly(div, odds_mv, base_dir)

    stats['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    conn.close()

    out_path = os.path.join(base_dir, 'data', 'stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'✅ stats.json 生成: {out_path}  '
          f'({len(stats["recent_races"])}レース / {len(bets)}ベット)')


if __name__ == '__main__':
    generate_stats(ROOT)

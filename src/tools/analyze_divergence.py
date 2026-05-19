"""
AI確率 vs 市場オッズの乖離分析スクリプト。

チューニング・キャリブレーション後の実際の精度を確認するための診断ツール。

実行方法（Google Colab）:
    import sys; sys.path.insert(0, f'{BASE_DIR}/src')
    from tools.analyze_divergence import run_analysis
    run_analysis(BASE_DIR)
"""
import json
import math
import os
import sqlite3


def run_analysis(base_dir, verbose=True):
    """乖離分析・キャリブレーション診断・ROI分析を実行する。

    Returns:
        dict: 各種指標
    """
    # keiba.db または history.db を検索
    db_path = None
    for name in ['keiba.db', 'history.db']:
        p = os.path.join(base_dir, 'data', name)
        if os.path.exists(p):
            db_path = p
            break
    if not db_path:
        print('⚠ DBが見つかりません。土日/金曜ノートブックで予測を実行した後に再実行してください。')
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    sim_rows = []

    if 'bet_simulation' in tables:
        try:
            sim_rows = conn.execute("""
                SELECT bs.ai_prob, bs.odds_est, bs.is_hit, bs.payout, bs.bet_type,
                       bs.num_horses, bs.ev
                FROM bet_simulation bs
                WHERE bs.ai_prob > 0 AND bs.odds_est > 1.0 AND bs.is_hit >= 0
            """).fetchall()
        except Exception:
            pass

    if not sim_rows and 'bets' in tables:
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(bets)").fetchall()}
            ev_expr = 'b.ev' if 'ev' in cols else '0.0'
            ai_expr = 'b.ai_prob' if 'ai_prob' in cols else ev_expr
            sim_rows = conn.execute(f"""
                SELECT {ai_expr} as ai_prob, b.odds_est, b.is_hit, b.payout,
                       b.bet_type, 0 as num_horses, {ev_expr} as ev
                FROM bets b
                WHERE b.is_hit >= 0
            """).fetchall()
        except Exception:
            pass

    conn.close()

    if not sim_rows:
        print('⚠ 分析対象データなし。土日/金曜ノートブックで予測・結果取得を実行した後に再実行してください。')
        return {}

    rows = [dict(r) for r in sim_rows]
    total = len(rows)

    # ── 1. 全体指標 ───────────────────────────────────────────────
    hits      = sum(1 for r in rows if r['is_hit'] == 1)
    paid      = sum(r['payout'] for r in rows if r['is_hit'] == 1)
    invested  = total * 100
    roi       = paid / invested if invested > 0 else 0.0
    hit_rate  = hits / total

    if verbose:
        print('=== 乖離分析レポート ===')
        print(f'\n【全体】 {total:,}件 | 的中率 {hit_rate:.3f} | ROI {roi:.3f}')

    # ── 2. AI確率バケット別の的中率・ROI ─────────────────────────
    def bucket_label(p):
        if p < 0.05:  return '< 5%'
        if p < 0.10:  return '5-10%'
        if p < 0.15:  return '10-15%'
        if p < 0.20:  return '15-20%'
        if p < 0.30:  return '20-30%'
        return '30%+'

    bucket_data = {}
    for r in rows:
        b = bucket_label(r.get('ai_prob', 0))
        bucket_data.setdefault(b, []).append(r)

    if verbose:
        print('\n【AI確率バケット別】')
        print(f'  {"バケット":>8s}  {"件数":>5s}  {"実的中率":>8s}  {"AI確率平均":>10s}  {"ROI":>7s}')
        bucket_order = ['< 5%','5-10%','10-15%','15-20%','20-30%','30%+']
        for b in bucket_order:
            if b not in bucket_data:
                continue
            br     = bucket_data[b]
            b_hit  = sum(1 for r in br if r['is_hit'] == 1) / len(br)
            b_ai   = sum(r.get('ai_prob', 0) for r in br) / len(br)
            b_roi  = sum(r['payout'] for r in br if r['is_hit'] == 1) / (len(br) * 100)
            calib  = '✅' if abs(b_ai - b_hit) < 0.03 else '⚠' if abs(b_ai - b_hit) < 0.07 else '❌'
            print(f'  {b:>8s}  {len(br):>5d}  {b_hit:>8.3f}  {b_ai:>10.4f}  {b_roi:>7.3f}  {calib}')

    # ── 3. AI vs 市場乖離バケット別 ──────────────────────────────
    mkt_rows = [r for r in rows if r.get('odds_est', 0) > 1.0]

    def ratio_bucket(p, o):
        if o <= 1.0:
            return None
        mkt_p = 1.0 / o
        ratio = p / mkt_p if mkt_p > 0 else 0
        if ratio < 0.5:   return '< 0.5x (市場>AI大)'
        if ratio < 0.8:   return '0.5-0.8x (市場>AI)'
        if ratio < 1.2:   return '0.8-1.2x (ほぼ一致)'
        if ratio < 2.0:   return '1.2-2.0x (AI>市場)'
        if ratio < 3.0:   return '2.0-3.0x (AI>>市場)'
        return '3.0x+ (AI>>>市場)'

    ratio_data = {}
    for r in mkt_rows:
        b = ratio_bucket(r.get('ai_prob', 0), r.get('odds_est', 10))
        if b:
            ratio_data.setdefault(b, []).append(r)

    if verbose and ratio_data:
        print('\n【AI vs 市場乖離バケット別】（妙味分析）')
        print(f'  {"乖離バケット":>26s}  {"件数":>5s}  {"的中率":>6s}  {"ROI":>7s}  判定')
        ratio_order = ['< 0.5x (市場>AI大)','0.5-0.8x (市場>AI)','0.8-1.2x (ほぼ一致)',
                       '1.2-2.0x (AI>市場)','2.0-3.0x (AI>>市場)','3.0x+ (AI>>>市場)']
        for b in ratio_order:
            if b not in ratio_data:
                continue
            br    = ratio_data[b]
            b_hit = sum(1 for r in br if r['is_hit'] == 1) / len(br)
            b_roi = sum(r['payout'] for r in br if r['is_hit'] == 1) / (len(br) * 100)
            mark  = '🎯' if b_roi > 1.05 else '➖' if b_roi > 0.9 else '💸'
            print(f'  {b:>26s}  {len(br):>5d}  {b_hit:>6.3f}  {b_roi:>7.3f}  {mark}')

    # ── 4. 券種別 ROI ─────────────────────────────────────────────
    type_data = {}
    for r in rows:
        t = r.get('bet_type', '不明')
        type_data.setdefault(t, []).append(r)

    if verbose:
        print('\n【券種別 ROI】')
        for t, tr in sorted(type_data.items(), key=lambda x: -len(x[1])):
            t_hit = sum(1 for r in tr if r['is_hit'] == 1) / len(tr)
            t_roi = sum(r['payout'] for r in tr if r['is_hit'] == 1) / (len(tr) * 100)
            bar   = '█' * int(t_roi * 20)
            print(f'  {t:>5s}: 的中率 {t_hit:.3f}, ROI {t_roi:.3f}  {bar}')

    # ── 5. EV バケット別実績 ──────────────────────────────────────
    ev_rows = [r for r in rows if r.get('ev', 0) > 0]
    if ev_rows and verbose:
        def ev_bucket(ev):
            if ev < 0.9:  return '< 0.90'
            if ev < 1.0:  return '0.90-1.00'
            if ev < 1.05: return '1.00-1.05'
            if ev < 1.10: return '1.05-1.10'
            if ev < 1.20: return '1.10-1.20'
            return '1.20+'

        ev_data = {}
        for r in ev_rows:
            b = ev_bucket(r.get('ev', 0))
            ev_data.setdefault(b, []).append(r)

        print('\n【EV バケット別実績】')
        print(f'  {"EVバケット":>12s}  {"件数":>5s}  {"的中率":>6s}  {"ROI":>7s}')
        for b in ['< 0.90','0.90-1.00','1.00-1.05','1.05-1.10','1.10-1.20','1.20+']:
            if b not in ev_data:
                continue
            br    = ev_data[b]
            b_hit = sum(1 for r in br if r['is_hit'] == 1) / len(br)
            b_roi = sum(r['payout'] for r in br if r['is_hit'] == 1) / (len(br) * 100)
            mark  = '✅' if b_roi > 1.0 else '❌'
            print(f'  {b:>12s}  {len(br):>5d}  {b_hit:>6.3f}  {b_roi:>7.3f}  {mark}')

    return {
        'total_bets': total,
        'hit_rate':   round(hit_rate, 4),
        'roi':        round(roi, 4),
        'bucket_stats': {b: {
            'n':       len(br),
            'hit':     round(sum(1 for r in br if r['is_hit']==1)/len(br), 4),
            'ai_avg':  round(sum(r.get('ai_prob',0) for r in br)/len(br), 4),
            'roi':     round(sum(r['payout'] for r in br if r['is_hit']==1)/(len(br)*100), 4),
        } for b, br in bucket_data.items()},
        'ratio_stats': {b: {
            'n':   len(br),
            'hit': round(sum(1 for r in br if r['is_hit']==1)/len(br), 4),
            'roi': round(sum(r['payout'] for r in br if r['is_hit']==1)/(len(br)*100), 4),
        } for b, br in ratio_data.items()},
    }

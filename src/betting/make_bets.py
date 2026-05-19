"""
EV計算・Kelly基準ベット額・券種選択・シミュレーション記録。
ノートブックのセル内関数を分離。
"""
import os
import pickle
import sqlite3

from src.features.engine import harville_pair_prob, harville_trio_prob
from src.utils.db import get_db_path

# ── デフォルト賭け金設定（init_betting() または直接書き換えで上書き可能）────
BANKROLL = 100_000
FUKU_AMT = 500
TAN_AMT  = 300
WIDE_AMT = 300
REN_AMT  = 300
TAN2_AMT = 300
SAN_AMT  = 300

# ── 券種選択モデル（オプション）────────────────────────────────────────────
_BET_SELECTOR    = None
_BET_SELECTOR_LE = None


def init_betting(base_dir,
                 bankroll=None, fuku_amt=None, tan_amt=None,
                 wide_amt=None, ren_amt=None, tan2_amt=None, san_amt=None):
    """買い目モジュールを初期化する。ノートブックのセル4直後に呼ぶ。

    Args:
        base_dir  : プロジェクトルート
        bankroll  : バンクロール（省略時 100,000）
        *_amt     : 各券種のデフォルト賭け金（省略時はモジュール定数を維持）
    """
    global BANKROLL, FUKU_AMT, TAN_AMT, WIDE_AMT, REN_AMT, TAN2_AMT, SAN_AMT
    global _BET_SELECTOR, _BET_SELECTOR_LE

    if bankroll  is not None: BANKROLL = bankroll
    if fuku_amt  is not None: FUKU_AMT = fuku_amt
    if tan_amt   is not None: TAN_AMT  = tan_amt
    if wide_amt  is not None: WIDE_AMT = wide_amt
    if ren_amt   is not None: REN_AMT  = ren_amt
    if tan2_amt  is not None: TAN2_AMT = tan2_amt
    if san_amt   is not None: SAN_AMT  = san_amt

    bs_path = os.path.join(base_dir, 'data', 'bet_selector_model.pkl')
    le_path = os.path.join(base_dir, 'data', 'bet_selector_le.pkl')
    if os.path.exists(bs_path) and os.path.exists(le_path):
        with open(bs_path, 'rb') as f: _BET_SELECTOR    = pickle.load(f)
        with open(le_path, 'rb') as f: _BET_SELECTOR_LE = pickle.load(f)
        print('✅ 券種選択モデル読み込み完了')
    else:
        print('⚠ 券種選択モデルなし → EVルールで代替')


def calc_ev(win_prob, odds):
    """期待値 = オッズ × 的中確率"""
    return round(odds * win_prob, 3)


def calc_kelly(win_prob, odds, bankroll=None, max_ratio=0.05):
    """1/4ケリー基準による最適ベット額（100円単位）"""
    if bankroll is None:
        bankroll = BANKROLL
    if odds <= 1.0 or win_prob <= 0:
        return 0
    b = odds - 1.0
    kelly_f = (win_prob * b - (1 - win_prob)) / b
    kelly_f = kelly_f * 0.25
    kelly_f = max(0, min(kelly_f, max_ratio))
    amount  = int(bankroll * kelly_f / 100) * 100
    return max(0, amount)


def make_bets(c):
    """EV × 適性係数スコアで最適券種を選択する（v5）。

    Args:
        c : ability_first_loose が返す候補辞書
            {'race': ..., 'scored': ..., 'top1': ..., 'score_gap': ..., 'chaos_score': ...}

    Returns:
        ベット辞書のリスト（最大2件）
    """
    race   = c['race']
    scored = c['scored']
    top1   = c['top1']
    nh  = race.get('num_horses', 16)
    sg  = c.get('score_gap', 0)
    ch  = c.get('chaos_score', 0)

    def go(i, k, d):
        return scored[i].get(k, d) if len(scored) > i else d

    o1 = go(0, 'win_odds', 10.0) or 10.0
    o2 = go(1, 'win_odds', 20.0) or 20.0
    o3 = go(2, 'win_odds', 30.0) or 30.0
    p1 = go(0, 'pn', 0.10)
    p2 = go(1, 'pn', 0.08)
    p3 = go(2, 'pn', 0.06)
    f1 = go(0, 'top3_prob', min(0.80, p1 * 3))

    itz = (o1 <= 2.5)
    i2k = (o2 / o1 <= 1.8 and o3 / o2 >= 1.8)
    ikn = (ch >= 0.55)
    ish = (nh <= 8)
    ita = (nh >= 14)
    imd = (3.0 <= o1 <= 12.0)
    icl = (sg >= 0.5 and o1 >= 3.0)

    # 各券種の確率・オッズ・EV・ベット額
    fp  = f1
    fo  = max(1.2, o1 * 0.30)
    fev = calc_ev(fp, fo)
    fa  = max(FUKU_AMT, min(calc_kelly(fp, fo), FUKU_AMT * 3))

    tev = calc_ev(p1, o1)
    ta  = max(TAN_AMT, min(calc_kelly(p1, o1), TAN_AMT * 3))

    pair12 = harville_pair_prob(p1, p2)
    wp  = pair12
    wo  = max(1.8, (o1 * o2) ** 0.5 * 0.65)
    wev = calc_ev(wp, wo)
    wa  = max(WIDE_AMT, min(calc_kelly(wp, wo), WIDE_AMT * 3))

    rp  = pair12
    ro  = max(3.0, (o1 * o2) ** 0.5 * 1.10)
    rev = calc_ev(rp, ro)
    ra  = max(REN_AMT, min(calc_kelly(rp, ro), REN_AMT * 3))

    t2p = min(p1 * p2 / max(1e-9, 1 - p1), 0.50)
    t2o = max(3.0, (o1 * o2) ** 0.5 * 1.3)
    t2ev = calc_ev(t2p, t2o)
    t2a  = max(TAN2_AMT, min(calc_kelly(t2p, t2o), TAN2_AMT * 3))

    sp  = harville_trio_prob(p1, p2, p3)
    so  = round(0.75 / sp, 1) if sp > 0 else 99.0
    sev = calc_ev(sp, so)
    sa  = max(SAN_AMT, min(calc_kelly(sp, so), SAN_AMT * 3))

    def ws(b, *rules):
        s = b
        for factor, cond in rules:
            s *= factor if cond else (2.0 - factor)
        return s

    fs  = ws(fev,  (1.1, ikn),  (0.95, ish))
    ts  = ws(tev,  (1.2, icl and imd), (0.7, itz), (0.8, o1 < 3.5))
    ws2 = ws(wev,  (1.2, itz or i2k),  (1.1, ita), (0.85, ikn))
    rs  = ws(rev,  (1.3, i2k and not itz), (1.1, ish), (0.6, itz), (0.8, ita))
    t2s = ws(t2ev, (1.2, icl and ish), (1.1, icl and imd), (0.6, itz), (0.75, ita))
    ss  = ws(sev,  (1.2, ish), (1.1, not ikn and not itz and not ita), (0.7, ikn), (0.8, ita))

    EV   = 1.00
    cds  = []

    # XGBoostで最適券種を予測（モデルがあれば）
    if _BET_SELECTOR is not None:
        import pandas as _pd
        X_pred = _pd.DataFrame([[
            nh, o1, o2, o3,
            o2 / o1 if o1 > 0 else 2.0,
            o3 / o2 if o2 > 0 else 2.0,
            ch,
            race.get('distance', 1600),
            1 if race.get('surface') == '芝' else 0,
            race.get('last_3f', 0) or 0,
        ]], columns=['n', 'o1', 'o2', 'o3', 'gap12', 'gap23',
                     'chaos', 'distance', 'surface', 'last3f'])
        _BET_SELECTOR_LE.classes_[_BET_SELECTOR.predict(X_pred)[0]]
        EV = 0.90

    s0n  = scored[0]['num']  if scored           else top1['num']
    s1n  = scored[1]['num']  if len(scored) > 1  else 0
    s2n  = scored[2]['num']  if len(scored) > 2  else 0
    s0nm = scored[0]['name'] if scored           else top1['name']
    s1nm = scored[1]['name'] if len(scored) > 1  else ''
    s2nm = scored[2]['name'] if len(scored) > 2  else ''

    if fev >= EV:
        cds.append(('複勝', fs, fev, {
            'type': '複勝', 'mark': '◎', 'nums': [top1['num']],
            'horse_name': top1['name'], 'odds': o1, 'odds_est': round(fo, 1),
            'amount': fa, 'ev': fev, 'prob': round(fp, 4),
            'pattern': f'複勝:EV{fev:.2f}xw{fs:.2f}'}))
    if tev >= EV and o1 >= 3.5:
        cds.append(('単勝', ts, tev, {
            'type': '単勝', 'mark': '◎', 'nums': [top1['num']],
            'horse_name': top1['name'], 'odds': o1, 'odds_est': o1,
            'amount': ta, 'ev': tev, 'prob': round(p1, 4),
            'pattern': f'単勝:EV{tev:.2f}xw{ts:.2f}'}))
    if wev >= EV and len(scored) >= 2:
        cds.append(('ワイド', ws2, wev, {
            'type': 'ワイド', 'mark': '◎○', 'nums': [s0n, s1n],
            'horse_name': f'{s0nm}-{s1nm}', 'odds': wo, 'odds_est': round(wo, 1),
            'amount': wa, 'ev': wev, 'prob': round(wp, 4),
            'pattern': f'ワイド:EV{wev:.2f}xw{ws2:.2f}'}))
    if rev >= EV and len(scored) >= 2:
        cds.append(('馬連', rs, rev, {
            'type': '馬連', 'mark': '◎○', 'nums': [s0n, s1n],
            'horse_name': f'{s0nm}-{s1nm}', 'odds': ro, 'odds_est': round(ro, 1),
            'amount': ra, 'ev': rev, 'prob': round(rp, 4),
            'pattern': f'馬連:EV{rev:.2f}xw{rs:.2f}'}))
    if t2ev >= EV and len(scored) >= 2:
        cds.append(('馬単', t2s, t2ev, {
            'type': '馬単', 'mark': '◎→○', 'nums': [s0n, s1n],
            'horse_name': f'{s0nm}→{s1nm}', 'odds': t2o, 'odds_est': round(t2o, 1),
            'amount': t2a, 'ev': t2ev, 'prob': round(t2p, 4),
            'pattern': f'馬単:EV{t2ev:.2f}xw{t2s:.2f}'}))
    if sev >= EV and len(scored) >= 3:
        from itertools import combinations as _comb
        gap12 = scored[0]['total'] - scored[1]['total'] if len(scored) > 1 else 0
        gap23 = scored[1]['total'] - scored[2]['total'] if len(scored) > 2 else 0
        n_av  = min(len(scored), 8)
        if gap12 >= 0.4 and gap23 >= 0.3:
            aite    = scored[2:5]
            tickets = [[s0n, s1n, h['num']] for h in aite]
            spat    = f'三連複2頭軸流し({len(tickets)}点)'
        elif gap12 >= 0.4:
            aite    = scored[1:5]
            tickets = [[s0n, h1['num'], h2['num']] for h1, h2 in _comb(aite, 2)]
            spat    = f'三連複1頭軸流し({len(tickets)}点)'
        else:
            n_box   = 5 if ikn else 4
            n_box   = min(n_box, n_av)
            tickets = [[h0['num'], h1['num'], h2['num']] for h0, h1, h2 in _comb(scored[:n_box], 3)]
            spat    = f'三連複{n_box}頭BOX({len(tickets)}点)'
        tickets   = tickets[:10]
        san_total = len(tickets) * 100
        san_nm    = (f'{s0nm}-{s1nm}-流し{len(tickets)}点' if '軸' in spat
                     else f'{s0nm}-BOX{len(tickets)}点')
        cds.append(('三連複', ss, sev, {
            'type': '三連複', 'mark': '◎',
            'nums': tickets[0], 'tickets': tickets,
            'horse_name': san_nm, 'odds': so, 'odds_est': so,
            'amount': san_total, 'unit_amount': 100,
            'ev': sev, 'prob': round(sp, 4),
            'pattern': f'{spat}:EV{sev:.2f}xw{ss:.2f}'}))

    if not cds:
        return [{'type': '複勝', 'mark': '◎', 'nums': [top1['num']],
                 'horse_name': top1['name'], 'odds': o1, 'odds_est': round(fo, 1),
                 'amount': FUKU_AMT, 'ev': fev, 'prob': round(fp, 4),
                 'pattern': f'複勝:EV低({fev:.2f})'}]

    cds.sort(key=lambda x: x[1], reverse=True)
    bets = [cds[0][3]]
    for ct, cs, cev, cb in cds[1:]:
        if cev >= 1.10 and ct != cds[0][0]:
            sb = dict(cb)
            sb['amount']  = max(300, cb['amount'] // 2)
            sb['pattern'] += ':サブ'
            bets.append(sb)
            break
    return bets


def log_bet_simulation(date_str, c, base_dir=None, db_path=None):
    """全券種を買った想定で記録する（データ蓄積後のROI分析用）。

    Args:
        date_str : 'YYYY-MM-DD'
        c        : ability_first_loose が返す候補辞書
        base_dir : プロジェクトルート（db_path 省略時に使用）
        db_path  : keiba.db のパス（直接指定する場合）
    """
    path   = db_path or get_db_path(base_dir)
    race   = c['race']
    scored = c['scored']
    rid    = race.get('id', '')
    rc     = race.get('racecourse', '')
    rnum   = race.get('race_num', 0)
    nh     = race.get('num_horses', 16)
    ch     = c.get('chaos_score', 0)
    sg     = c.get('score_gap', 0)

    def go(i, k, d):
        return scored[i].get(k, d) if len(scored) > i else d

    o1 = go(0, 'win_odds', 10.0) or 10.0
    o2 = go(1, 'win_odds', 20.0) or 20.0
    o3 = go(2, 'win_odds', 30.0) or 30.0
    p1 = go(0, 'pn', 0.10)
    p2 = go(1, 'pn', 0.08)
    p3 = go(2, 'pn', 0.06)

    itz = 1 if o1 <= 2.5 else 0
    i2k = 1 if (o2 / o1 <= 1.8 and o3 / o2 >= 1.8) else 0
    ikn = 1 if ch >= 0.55 else 0
    pr  = next((i + 1 for i, h in enumerate(
        sorted(scored, key=lambda h: h.get('win_odds') or 99))
        if h['name'] == scored[0]['name']), 99)

    s0n  = str(scored[0]['num'])
    s0nm = scored[0]['name']
    rows = [
        ('複勝', s0n, s0nm, max(1.1, o1 * 0.28), p1,
         calc_ev(min(0.95, p1 * 3), max(1.1, o1 * 0.28))),
        ('単勝', s0n, s0nm, o1, p1, calc_ev(p1, o1)),
    ]
    if len(scored) >= 2:
        s1n  = str(scored[1]['num'])
        s1nm = scored[1]['name']
        wo   = max(1.5, (o1 * o2) ** 0.5 * 0.45)
        ro   = max(2.0, (o1 * o2) ** 0.5 * 0.75)
        t2o  = max(3.0, (o1 * o2) ** 0.5 * 1.3)
        rows += [
            ('ワイド', f'{s0n}-{s1n}', f'{s0nm}-{s1nm}', wo,
             min(0.60, p1 * p2 * 6), calc_ev(min(0.60, p1 * p2 * 6), wo)),
            ('馬連',   f'{s0n}-{s1n}', f'{s0nm}-{s1nm}', ro,
             min(0.40, p1 * p2 * 2), calc_ev(min(0.40, p1 * p2 * 2), ro)),
            ('馬単',   f'{s0n}->{s1n}', f'{s0nm}->{s1nm}', t2o,
             min(0.30, p1 * p2 * 1.2), calc_ev(min(0.30, p1 * p2 * 1.2), t2o)),
        ]
    if len(scored) >= 3:
        s2n  = str(scored[2]['num'])
        s2nm = scored[2]['name']
        ps   = min(0.50, p1 * p2 * p3 * 6)
        soo  = round(0.75 / ps, 1) if ps > 0 else 99.0
        rows.append((
            '三連複',
            f'{s0n}-{scored[1]["num"]}-{s2n}',
            f'{s0nm}-{scored[1]["name"]}-{s2nm}',
            soo, ps, calc_ev(ps, soo),
        ))

    conn = sqlite3.connect(path)
    for bt, hn, hname, oe, ap, ev in rows:
        conn.execute(
            'INSERT INTO bet_simulation'
            '(date,race_id,racecourse,race_num,bet_type,horse_num,horse_name,'
            'odds_est,ai_prob,ev,num_horses,chaos,is_tanzen,is_2kyou,is_konsen,pop_rank,score_gap)'
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (date_str, rid, rc, rnum, bt, hn, hname,
             round(oe, 2), round(ap, 4), round(ev, 3),
             nh, round(ch, 3), itz, i2k, ikn, pr, round(sg, 3)),
        )
    conn.commit()
    conn.close()

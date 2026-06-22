"""
EV計算・Kelly基準ベット額・券種選択・シミュレーション記録。
ノートブックのセル内関数を分離。
"""
import os
import pickle
import sqlite3
from itertools import combinations

from src.features.engine import harville_pair_prob, harville_trio_prob
from src.utils.db import get_db_path, _connect

# ── デフォルト賭け金設定（init_betting() または直接書き換えで上書き可能）────
BANKROLL = 100_000
FUKU_AMT = 500
TAN_AMT  = 300
WIDE_AMT = 300
REN_AMT  = 300
TAN2_AMT = 300
SAN_AMT  = 300

# ── ルールベース券種選択の金額定数（フォワードテスト後に調整）────────────────
BET_AMOUNT_REN  = 500   # 馬連の基本金額（円）
BET_AMOUNT_WIDE = 300   # ワイドの基本金額（円）
BET_AMOUNT_FUKU = 500   # 複勝の基本金額（円）
BET_AMOUNT_SAN  = 200   # 三連複の基本金額（円）※少額固定

# ── EVフィルタ（市場オッズ取得時のみ適用）────────────────────────────────────
EV_THRESHOLD = 1.5  # 推定払戻 >= 投資額 × EV_THRESHOLD で買い

# ── 券種選択モデル（後方互換のため残存。ルールベースでは使用しない）──────────
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
        print('✅ 券種選択モデル読み込み完了（ルールベースのため予測には使用しない）')
    else:
        print('⚠ 券種選択モデルなし → ルールベースで動作')


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


def classify_chaos_grade(horses, chaos_score):
    """波乱度スコアとRL1位馬の人気から A/B/C を判定する。

    判定ルール（優先順）:
        chaos_score < 0.30 かつ rl_rank=1 の人気 <= 2  → 'A'（堅い）
        chaos_score > 0.55 または rl_rank=1 の人気 >= 6 → 'C'（大荒れ）
        それ以外 → 'B'（中荒れ）

    Args:
        horses      : 全馬の予測結果。必須キー: rl_rank, popularity
        chaos_score : engine.py の calc_chaos_score() の出力（0〜1）

    Returns:
        'A'（堅い）/ 'B'（中荒れ）/ 'C'（大荒れ）
    """
    top = next((h for h in horses if h.get('rl_rank') == 1), None)
    top_pop = top.get('popularity', 99) if top else 99

    if chaos_score < 0.30 and top_pop <= 2:
        return 'A'
    if chaos_score > 0.55 or top_pop >= 6:
        return 'C'
    return 'B'


def estimate_payout(bet_type, horse_odds_list):
    """券種ごとの推定払戻（100円あたり）を計算する。

    keiba_ai_app_v7.html の estimatePayout() と同じ式（JS側と同期すること）。
    horse_odds_list の各要素は detect_value_horses() が返す馬辞書を想定
    （'fukusho_odds' / 'tansho_odds' キーを持つ）。

    Returns:
        推定払戻額（円）。算出に必要なオッズが無い場合は None（推定不能）。
    """
    def tansho(h):
        return h.get('tansho_odds') or h.get('odds', 0)

    if bet_type == '複勝':
        fo = horse_odds_list[0].get('fukusho_odds')
        if not fo:
            return None
        return fo * 100

    if bet_type in ('馬連', 'ワイド'):
        if len(horse_odds_list) < 2:
            return None
        o1, o2 = tansho(horse_odds_list[0]), tansho(horse_odds_list[1])
        if not o1 or not o2:
            return None
        if bet_type == '馬連':
            return max(200, (o1 * o2) ** 0.5 * 75)
        return max(150, (o1 * o2) ** 0.5 * 45)

    if bet_type == '三連複':
        if len(horse_odds_list) < 3:
            return None
        odds = [tansho(h) for h in horse_odds_list[:3]]
        if not all(odds):
            return None
        return max(300, (odds[0] * odds[1] * odds[2]) ** 0.33 * 80)

    return None


def _apply_ev_filter(bets, value_horses):
    """市場オッズ取得済みの場合、推定払戻が投資額×EV_THRESHOLD未満の買い目を除外する。

    value_horses（detect_value_horses の戻り値）が市場オッズを一切含まない
    （= market_odds_map が空だった）場合はフィルタせずそのまま返す。
    """
    if not bets:
        return bets

    odds_by_num = {h.get('horse_num', h.get('num')): h for h in value_horses}
    has_odds = any(h.get('fukusho_odds') is not None or h.get('tansho_odds') is not None
                   for h in value_horses)
    if not has_odds:
        return bets

    filtered = []
    for b in bets:
        odds_list = [odds_by_num.get(n, {}) for n in b['nums'][:3]]
        est = estimate_payout(b['type'], odds_list)
        if est is None or est >= b['amount'] * EV_THRESHOLD:
            filtered.append(b)
    return filtered


def select_bet_type(horses, chaos_grade, value_horses, num_horses):
    """波乱度・頭数・バリュー馬を総合して最終買い目を生成する。

    100%ルールベース。XGBモデル（bet_selector_model.pkl）は使用しない。
    ★ 馬連・ワイドの対象馬は rl_rank（AI予測順位）上位を使用する。

    【多頭数(>=14頭)】
        A/B: 馬連（rl_rank 1×2）+ ワイド（rl_rank 1×2）
        C  : 馬連（rl_rank 1×2）+ 三連複（rl_rank 1〜3ボックス・少額）

    【少頭数(<=8頭)】
        → 馬連（rl_rank 1×2）のみ

    【中頭数(9-13頭)】
        A: 馬連（rl_rank 1×2）
        B + バリュー馬あり: 複勝（バリュー馬）+ ワイド（バリュー馬×rl_rank1位）
        B + バリュー馬なし: 複勝（rl_rank 1位）
        C + バリュー馬あり: 複勝（バリュー馬）少額
        C + バリュー馬なし: スキップ（空リスト）

    市場オッズ取得済み（value_horses に fukusho_odds/tansho_odds がある）場合は、
    上記で生成した候補をさらに _apply_ev_filter() でフィルタする
    （推定払戻 < 投資額×EV_THRESHOLD の券種は除外）。

    Args:
        horses       : 全馬の予測結果（rl_rank 付きを前提）
        chaos_grade  : 'A'/'B'/'C'
        value_horses : detect_value_horses() の戻り値（value_gap・fukusho_odds・
                       tansho_odds 付きの全馬リスト）
        num_horses   : 出走頭数

    Returns:
        買い目リスト。各要素: {type, nums, amount, reason}
        スキップ時は空リスト []
    """
    from src.betting.ev_filter import VALUE_GAP_THRESHOLD

    by_rl = sorted(horses, key=lambda h: h.get('rl_rank', 99))
    top1  = by_rl[0] if len(by_rl) > 0 else None
    top2  = by_rl[1] if len(by_rl) > 1 else None
    top3  = by_rl[2] if len(by_rl) > 2 else None
    if top1 is None:
        return []

    candidates = _select_bet_candidates(by_rl, top1, top2, top3,
                                         chaos_grade, value_horses, num_horses)
    return _apply_ev_filter(candidates, value_horses)


def _select_bet_candidates(by_rl, top1, top2, top3, chaos_grade, value_horses, num_horses):
    from src.betting.ev_filter import VALUE_GAP_THRESHOLD

    n1 = top1.get('horse_num', top1.get('num'))
    n2 = top2.get('horse_num', top2.get('num')) if top2 else None
    n3 = top3.get('horse_num', top3.get('num')) if top3 else None
    vh = [h for h in value_horses if h.get('value_gap', 0) >= VALUE_GAP_THRESHOLD]

    def ren(reason):
        return {'type': '馬連', 'nums': [n1, n2], 'amount': BET_AMOUNT_REN, 'reason': reason}

    def wide(a, b, reason):
        return {'type': 'ワイド', 'nums': [a, b], 'amount': BET_AMOUNT_WIDE, 'reason': reason}

    def fuku(num, reason):
        return {'type': '複勝', 'nums': [num], 'amount': BET_AMOUNT_FUKU, 'reason': reason}

    def san(nums, reason):
        return {'type': '三連複', 'nums': sorted(nums), 'amount': BET_AMOUNT_SAN, 'reason': reason}

    # ── 多頭数(>=14頭) ────────────────────────────────────────────────────────
    if num_horses >= 14:
        if chaos_grade in ('A', 'B'):
            if n2 is None:
                return [fuku(n1, f'多頭数({num_horses}頭)・馬連候補不足→複勝')]
            return [
                ren(f'多頭数({num_horses}頭)・馬連(RL1×2)'),
                wide(n1, n2, f'多頭数({num_horses}頭)・ワイド(RL1×2)'),
            ]
        else:  # C
            if n2 is None:
                return []
            bets = [ren(f'多頭数({num_horses}頭)+混戦・馬連(RL1×2)')]
            if n3 is not None:
                bets.append(san([n1, n2, n3],
                                f'多頭数({num_horses}頭)+混戦・三連複RL1〜3ボックス'))
            return bets

    # ── 少頭数(<=8頭) ─────────────────────────────────────────────────────────
    if num_horses <= 8:
        if n2 is None:
            return [fuku(n1, f'少頭数({num_horses}頭)・馬連候補不足→複勝')]
        return [ren(f'少頭数({num_horses}頭)・馬連(RL1×2)')]

    # ── 中頭数(9-13頭) ────────────────────────────────────────────────────────
    if chaos_grade == 'A':
        if n2 is None:
            return [fuku(n1, f'中頭数({num_horses}頭)+堅い・馬連候補不足→複勝')]
        return [ren(f'中頭数({num_horses}頭)+堅い・馬連(RL1×2)')]

    if chaos_grade == 'B':
        if vh:
            v_num = vh[0].get('horse_num', vh[0].get('num'))
            bets = [fuku(v_num, f'中頭数({num_horses}頭)+中荒れ・バリュー馬#{v_num}複勝')]
            if v_num != n1:
                bets.append(wide(v_num, n1,
                                 f'中頭数({num_horses}頭)+中荒れ・バリュー#{v_num}×RL1位#{n1}ワイド'))
            elif n2 is not None:
                bets.append(wide(v_num, n2,
                                 f'中頭数({num_horses}頭)+中荒れ・バリュー#{v_num}×RL2位#{n2}ワイド'))
            return bets
        return [fuku(n1, f'中頭数({num_horses}頭)+中荒れ・バリューなし→RL1位複勝')]

    # chaos_grade == 'C'
    if vh:
        v_num = vh[0].get('horse_num', vh[0].get('num'))
        return [fuku(v_num, f'中頭数({num_horses}頭)+大荒れ・バリュー馬#{v_num}複勝少額')]
    return []  # バリューなし → スキップ


def make_bets(c, market_odds_map=None):
    """最適券種を選択してベット辞書リストを返す。

    市場オッズ（market_odds_map）が渡された場合はルールベースで決定する。
    省略時は従来のEV×スコアリングロジックにフォールバック（後方互換）。

    Args:
        c               : ability_first_loose が返す候補辞書
                          {'race': ..., 'scored': ..., 'top1': ..., 'score_gap': ..., 'chaos_score': ...}
        market_odds_map : {horse_num: fukusho_odds}（省略時は従来ロジック）

    Returns:
        ベット辞書のリスト
    """
    race   = c['race']
    scored = c['scored']
    top1   = c['top1']
    nh  = race.get('num_horses', len(scored))
    sg  = c.get('score_gap', 0)
    ch  = c.get('chaos_score', 0)
    chaos_level = c.get('chaos_level', 'B')

    # ── ルールベース（market_odds_map が渡された場合）──────────────────────────
    if market_odds_map is not None:
        from src.betting.ev_filter import detect_value_horses
        from src.features.engine import calc_chaos_score
        for h in scored:
            if 'horse_num' not in h:
                h['horse_num'] = h.get('num')
            if 'cal_prob' not in h:
                h['cal_prob'] = h.get('pn', 0)
            if 'popularity' not in h:
                h['popularity'] = h.get('_pop', 99)
        chaos_score_val = c.get('chaos_score', calc_chaos_score(race, scored))
        grade     = classify_chaos_grade(scored, chaos_score_val)
        vh_all    = detect_value_horses(scored, market_odds_map)
        rule_bets = select_bet_type(scored, grade, vh_all, nh)
        if rule_bets:
            result = []
            for rb in rule_bets:
                nums      = rb['nums']
                first_num = nums[0]
                horse_obj = next((h for h in scored if h.get('num') == first_num
                                  or h.get('horse_num') == first_num), top1)
                result.append({
                    'type':        rb['type'],
                    'mark':        '◎',
                    'nums':        nums,
                    'horse_name':  horse_obj.get('name', ''),
                    'odds':        horse_obj.get('win_odds', 0) or 0,
                    'odds_est':    horse_obj.get('win_odds', 0) or 0,
                    'amount':      rb['amount'],
                    'ev':          0.0,
                    'prob':        round(horse_obj.get('pn', 0), 4),
                    'pattern':     rb['reason'],
                    'chaos_grade': grade,
                })
            return result

    # ── 従来の EV×スコアリングロジック（後方互換フォールバック）──────────────
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
    is_solid = (chaos_level == 'A')
    is_chaos = (chaos_level == 'C')

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

    fs  = ws(fev,  (1.1, ikn),  (0.95, ish),  (0.8, is_chaos))
    ts  = ws(tev,  (1.2, icl and imd), (0.7, itz), (0.8, o1 < 3.5),
                   (1.3, is_solid), (0.7, is_chaos))
    ws2 = ws(wev,  (1.2, itz or i2k),  (1.1, ita), (0.85, ikn), (0.7, is_solid))
    rs  = ws(rev,  (1.3, i2k and not itz), (1.1, ish), (0.6, itz), (0.8, ita),
                   (1.3, is_solid), (0.7, is_chaos))
    t2s = ws(t2ev, (1.2, icl and ish), (1.1, icl and imd), (0.6, itz), (0.75, ita))
    ss  = ws(sev,  (1.2, ish), (1.1, not ikn and not itz and not ita), (0.7, ikn), (0.8, ita))

    EV   = 1.00
    cds  = []

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


# ── 三連複フォーメーション設定 ────────────────────────────────────────────────
FORMATION_UNIT       = 100
FORMATION_MAX_POINTS = 15
FORMATION_AXIS_POP_LIMIT = 12
FORMATION_SYN_TARGET = 3.5   # 合成オッズの目標値


def _fnum(h):
    return h.get('horse_num', h.get('num'))


def _fpop(h):
    return h.get('popularity') or h.get('_pop') or 99


def _fodds(h):
    return h.get('win_odds', h.get('odds', 5)) or 5


def _pick_n(pool_rl, pool_pop, exclude, n):
    """AI順位優先・市場人気で補完しながら重複なくn頭選ぶ。"""
    seen = set(exclude)
    result = []
    for h in pool_rl:
        if _fnum(h) not in seen:
            result.append(_fnum(h))
            seen.add(_fnum(h))
        if len(result) >= n:
            break
    for h in pool_pop:
        if _fnum(h) not in seen:
            result.append(_fnum(h))
            seen.add(_fnum(h))
        if len(result) >= n:
            break
    return result[:n]


def _calc_syn_odds(tickets, h_map):
    """三連複フォーメーションの合成オッズ（的中時リターン/総投資額）を推定する。

    合成オッズ = 平均推定払戻 / 総投資額
    払戻推定: (o1*o2*o3)^0.5 * 100 （実績ベース近似）
    """
    if not tickets:
        return 0.0
    total_est = 0.0
    for nums in tickets:
        o = [_fodds(h_map.get(n, {})) for n in nums]
        est_pay = max(500, (o[0] * o[1] * o[2]) ** 0.5 * 100)
        total_est += est_pay
    avg_payout = total_est / len(tickets)
    return avg_payout / (len(tickets) * FORMATION_UNIT)


def _tickets_payout_range(tickets, h_map):
    """全買い目の想定配当から min/mid/max を返す（円）。"""
    payouts = []
    for nums in tickets:
        o = [_fodds(h_map.get(n, {})) for n in nums]
        payouts.append(max(300, int((o[0] * o[1] * o[2]) ** 0.33 * 80)))
    if not payouts:
        return {'min': 300, 'mid': 3000, 'max': 10000}
    ps = sorted(payouts)
    return {'min': ps[0], 'mid': ps[len(ps) // 2], 'max': ps[-1]}


def _make_pattern(name, tickets, axis_nums, second_nums, third_nums, h_map):
    syn = _calc_syn_odds(tickets, h_map)
    return {
        'name':        name,
        'tickets':     [sorted(t) for t in tickets],
        'axis_nums':   axis_nums,
        'second_nums': second_nums,
        'third_nums':  third_nums,
        'syn_odds':    round(syn, 2),
    }


def _trim_to_syn_target(tickets, h_map, target_syn=2.5, min_tickets=4):
    """推定払戻の低い組み合わせを除去して合成オッズを改善する。

    高払戻の組み合わせを残しながら、target_syn に達するか min_tickets になるまで削る。
    """
    if not tickets or len(tickets) <= min_tickets:
        return tickets

    def est_pay(nums):
        o = [_fodds(h_map.get(n, {})) for n in nums]
        return max(500, (o[0] * o[1] * o[2]) ** 0.5 * 100)

    current = sorted(tickets, key=est_pay, reverse=True)
    while len(current) > min_tickets:
        if _calc_syn_odds(current, h_map) >= target_syn:
            break
        current.pop()
    return current


def select_axis(by_rl, by_pop):
    """軸馬を選定する。RL1位基本。人気12超なら代替を探す。"""
    rl1 = by_rl[0]
    if _fpop(rl1) <= FORMATION_AXIS_POP_LIMIT:
        return rl1
    top3 = by_rl[:3]
    best = min(top3, key=lambda h: _fpop(h))
    if _fpop(best) <= 8:
        return best
    return min(by_pop[:3], key=lambda h: h.get('rl_rank', 99))


def _select_best_candidate(candidates, chaos_grade, gap12, rl2_pop):
    """chaos_grade / gap12 / rl2_pop のルールに基づいて候補パターンを選ぶ。"""
    def find(name):
        return next((p for p in candidates if p['name'] == name), None)

    if chaos_grade == 'C' and gap12 < 0.03:
        return find('D_4box') or find('E_1ax6box') or candidates[0]
    if chaos_grade == 'C':
        return find('E_1ax6box') or find('B_standard') or candidates[0]
    if chaos_grade == 'A' and gap12 >= 0.03 and rl2_pop <= 5:
        return find('C_2ax5flow') or find('A_1ax4box') or find('B_standard') or candidates[0]
    if chaos_grade == 'A' and gap12 >= 0.05:
        return find('A_1ax4box') or find('C_2ax5flow') or find('B_standard') or candidates[0]
    if chaos_grade == 'B' and gap12 >= 0.04 and rl2_pop <= 4:
        return find('C_2ax5flow') or find('B_standard') or candidates[0]
    return find('B_standard') or candidates[0]


def build_formation(horses, race, chaos_grade='B'):
    """三連複フォーメーションをパターン自動選択して生成する。

    chaos_grade と AI自信度（RL1-2位のpn差）から最適パターンを選択。

    パターン一覧:
      A_1ax4box  : 軸1頭×4頭BOX（6点）  ← chaos A + gap≥0.05
      B_standard : 軸1頭×2列2頭×3列4頭（〜10点）← 標準
      C_2ax5flow : 2頭軸×5頭流し（5点）  ← chaos A/B + RL2位も人気上位
      D_4box     : 4頭BOX（4点）          ← chaos C + 混戦（軸不明）
      E_1ax6box  : 軸1頭×6頭BOX（15点）  ← chaos C + 軸は1頭ある
    """
    if len(horses) < 4:
        return None

    by_rl  = sorted(horses, key=lambda h: h.get('rl_rank', 99))
    by_pop = sorted(horses, key=lambda h: _fpop(h))
    num_horses = race.get('num_horses', len(horses))
    h_map = {_fnum(h): h for h in horses}

    p1    = by_rl[0].get('pn', 0.05) if by_rl           else 0.05
    p2    = by_rl[1].get('pn', 0.03) if len(by_rl) > 1  else 0.03
    gap12 = p1 - p2

    axis = select_axis(by_rl, by_pop)
    if axis is None:
        return None
    ax1 = _fnum(axis)

    candidates = []

    # ── パターンC: 2頭軸×5頭流し（5点）──────────────────────────────────────
    rl2 = by_rl[1] if len(by_rl) > 1 else None
    if rl2 and chaos_grade in ('A', 'B') and gap12 >= 0.03 and _fpop(rl2) <= 5:
        ax2   = _fnum(rl2)
        mates = _pick_n(by_rl[2:], by_pop, {ax1, ax2}, 5)
        if len(mates) >= 2:
            tickets = [[ax1, ax2, m] for m in mates]
            candidates.append(_make_pattern('C_2ax5flow', tickets, [ax1, ax2], [], mates, h_map))

    # ── パターンA: 軸1頭×4頭BOX（6点）────────────────────────────────────────
    if chaos_grade == 'A' and gap12 >= 0.05 and num_horses <= 14:
        mates = _pick_n(by_rl[1:], by_pop, {ax1}, 4)
        if len(mates) >= 3:
            tickets = [[ax1] + list(c) for c in combinations(mates, 2)]
            candidates.append(_make_pattern('A_1ax4box', tickets, [ax1], mates[:2], mates[2:], h_map))

    # ── パターンB: 軸1頭×2列2頭×3列4頭（〜10点）─────────────────────────────
    if len(horses) >= 6:
        second = []
        for h in by_rl[1:6]:
            if _fnum(h) != ax1 and _fpop(h) <= 6:
                second.append(_fnum(h))
            if len(second) >= 2:
                break
        if len(second) < 2:
            for h in by_pop[:4]:
                if _fnum(h) != ax1 and _fnum(h) not in second:
                    second.append(_fnum(h))
                if len(second) >= 2:
                    break
        third = _pick_n(by_rl, by_pop, {ax1} | set(second), 4)
        all_mates = second + third
        if len(all_mates) >= 2:
            tickets = [[ax1] + list(c) for c in combinations(all_mates, 2)]
            candidates.append(_make_pattern('B_standard', tickets, [ax1], second, third, h_map))

    # ── パターンD: 4頭BOX（4点）──────────────────────────────────────────────
    if chaos_grade == 'C' and gap12 < 0.03 and len(horses) >= 4:
        box4    = [_fnum(h) for h in by_rl[:4]]
        tickets = [list(c) for c in combinations(box4, 3)]
        candidates.append(_make_pattern('D_4box', tickets, [], [], box4, h_map))

    # ── パターンE: 軸1頭×6頭BOX（15点）──────────────────────────────────────
    if chaos_grade == 'C' and len(horses) >= 8:
        mates6  = _pick_n(by_rl[1:], by_pop, {ax1}, 6)
        if len(mates6) >= 4:
            tickets = [[ax1] + list(c) for c in combinations(mates6, 2)][:FORMATION_MAX_POINTS]
            candidates.append(_make_pattern('E_1ax6box', tickets, [ax1], mates6[:2], mates6[2:], h_map))

    if not candidates:
        return None

    # ── ルールベースで最適パターンを選択 ─────────────────────────────────────
    rl2_pop = _fpop(by_rl[1]) if len(by_rl) > 1 else 99
    best = _select_best_candidate(candidates, chaos_grade, gap12, rl2_pop)

    if len(best['tickets']) > FORMATION_MAX_POINTS:
        best['tickets']  = best['tickets'][:FORMATION_MAX_POINTS]
        best['syn_odds'] = round(_calc_syn_odds(best['tickets'], h_map), 2)

    # 合成オッズが低すぎる場合は低払戻の組み合わせを除去して点数を絞る
    if best['syn_odds'] < 1.5:
        trimmed = _trim_to_syn_target(best['tickets'], h_map, target_syn=2.5, min_tickets=4)
        if len(trimmed) != len(best['tickets']):
            surviving = {n for t in trimmed for n in t}
            best = {
                'name':        best['name'],
                'tickets':     trimmed,
                'axis_nums':   best['axis_nums'],
                'second_nums': [n for n in best['second_nums'] if n in surviving],
                'third_nums':  [n for n in best['third_nums'] if n in surviving],
                'syn_odds':    round(_calc_syn_odds(trimmed, h_map), 2),
            }

    payout_range = _tickets_payout_range(best['tickets'], h_map)

    def _hinfo(n):
        h = h_map.get(n, {})
        return {'num': n, 'name': h.get('name', ''), 'pop': _fpop(h), 'odds': round(_fodds(h), 1)}

    second_set = set(best['second_nums'])
    axis_set   = set(best['axis_nums'])

    def _tier(nums):
        if len(best['axis_nums']) >= 2:
            return '軸流し'
        if not best['axis_nums']:
            return 'BOX'
        non_axis = [n for n in nums if n not in axis_set]
        in_s = sum(1 for n in non_axis if n in second_set)
        return '堅' if in_s == 2 else ('中穴' if in_s == 1 else '大穴')

    return {
        'pattern':      best['name'],
        'axis':         _hinfo(best['axis_nums'][0]) if best['axis_nums'] else None,
        'axis2':        _hinfo(best['axis_nums'][1]) if len(best['axis_nums']) > 1 else None,
        'second_row':   [_hinfo(n) for n in best['second_nums']],
        'third_row':    [_hinfo(n) for n in best['third_nums']],
        'bets':         [{'type': '三連複F', 'nums': t, 'amount': FORMATION_UNIT,
                          'tier': _tier(t)} for t in best['tickets']],
        'total_points': len(best['tickets']),
        'total_amount': len(best['tickets']) * FORMATION_UNIT,
        'payout_range': payout_range,
        'syn_odds':     best['syn_odds'],
    }


# 旧API後方互換（呼び出し元がまだ残っていれば）
def select_second_row(by_rl, by_pop, axis):
    ax = _fnum(axis)
    second = []
    for h in by_rl[:6]:
        if _fnum(h) != ax and _fpop(h) <= 8:
            second.append(h)
        if len(second) >= 3:
            break
    used = {ax} | {_fnum(h) for h in second}
    if len(second) < 2:
        for h in by_pop[:5]:
            if _fnum(h) not in used:
                second.append(h)
                used.add(_fnum(h))
            if len(second) >= 2:
                break
    return second[:3]


def select_third_row(by_rl, by_pop, axis, second_row):
    used = {_fnum(axis)} | {_fnum(h) for h in second_row}
    third = []
    for h in by_rl[:10]:
        if _fnum(h) not in used:
            third.append(h)
            used.add(_fnum(h))
        if len(third) >= 4:
            break
    for h in by_pop[:6]:
        if _fnum(h) not in used:
            third.append(h)
            used.add(_fnum(h))
        if len(third) >= 6:
            break
    return third[:6]


def generate_formation_bets(axis, second_row, third_row):
    axis_num    = _fnum(axis)
    second_nums = [_fnum(h) for h in second_row]
    third_nums  = [_fnum(h) for h in third_row]
    all_nums    = second_nums + third_nums
    bets = []
    for combo in combinations(all_nums, 2):
        if combo[0] == axis_num or combo[1] == axis_num:
            continue
        nums = sorted([axis_num, combo[0], combo[1]])
        in_second = sum(1 for n in combo if n in second_nums)
        tier = '堅' if in_second == 2 else ('中穴' if in_second == 1 else '大穴')
        bets.append({'type': '三連複F', 'nums': nums, 'amount': FORMATION_UNIT,
                     'tier': tier, 'reason': f'軸#{axis_num} ' + '×'.join(f'#{n}' for n in combo)})
    return bets


def estimate_payout_range(axis, second_row, third_row):
    a_o = _fodds(axis)
    if len(second_row) >= 2:
        s = [_fodds(h) for h in second_row[:2]]
        min_pay = max(300, int((a_o * s[0] * s[1]) ** 0.33 * 80))
    else:
        min_pay = 500
    if second_row and third_row:
        m1 = _fodds(second_row[0])
        m2 = _fodds(third_row[len(third_row) // 2])
        mid_pay = max(1000, int((a_o * m1 * m2) ** 0.33 * 80))
    else:
        mid_pay = 3000
    if len(third_row) >= 2:
        t = sorted([_fodds(h) for h in third_row], reverse=True)
        max_pay = max(5000, int((a_o * t[0] * t[1]) ** 0.33 * 80))
    else:
        max_pay = 10000
    return {'min': min_pay, 'mid': mid_pay, 'max': max_pay}


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

    conn = _connect(path)
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

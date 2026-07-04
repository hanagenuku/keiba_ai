"""
アプリ用 JSON 生成。
ノートブックの to_app_json を分離。
"""
from src.betting.ev_filter import (VENUE_ORDER,
                                    calc_market_probs, calc_value_score,
                                    detect_value_horses, is_maiden_race,
                                    classify_race_chaos)
from src.betting.make_bets import calc_ev, make_bets, build_formation
from src.features.engine import auto_comment, calc_all, calc_chaos_score


def _build_odds_cn(race):
    """直前オッズ取得用のCNAME情報をGAS向けにドット区切り文字列で返す。

    GAS側でsuffixをスキャン・キャッシュするため、
    {odds_base}..{race_num:02d}..{date_str} の形式で返す。
    race['_odds_cn'] が無い場合は None を返す。
    """
    cn_info = race.get('_odds_cn')
    if not cn_info:
        return None
    odds_base = cn_info['base'].replace('pw01dde01', 'pw151ouS3')
    return f"{odds_base}..{cn_info['race_num']:02d}..{cn_info['date_str']}"


def _assign_marks(scored, by_odds):
    """各馬に高/推/穴マークを付与する。1レースにつき穴は最大1頭、推は最大2頭。

    本命（AI確率1位）は必ず買い目に入るため、マーク付与とは独立して決まる。
    """
    market_probs = calc_market_probs(scored)

    # EV未計算なら計算する
    for i, h in enumerate(scored):
        if 'ev' not in h:
            ai_prob = h.get('pn', 0) or 0
            m_prob  = market_probs[i]
            odds    = h.get('win_odds') or 0
            vs      = calc_value_score(ai_prob, m_prob, odds)
            h['ev']       = vs['ev']
            h['prob_gap'] = vs['prob_gap']
            h['is_value'] = vs['is_value']

    # AI順位（win_prob降順）を付与
    ai_sorted = sorted(range(len(scored)), key=lambda i: scored[i].get('pn', 0), reverse=True)
    for rank, idx in enumerate(ai_sorted):
        scored[idx]['ai_rank'] = rank + 1

    # 人気順を付与
    for h in scored:
        h['_pop'] = next((i + 1 for i, x in enumerate(by_odds) if x['name'] == h['name']), 99)

    marks = {h['num']: '' for h in scored}

    # 高マーク: 3番人気以内 かつ AI上位3位 かつ EV>=1.0
    for h in scored:
        if h['_pop'] <= 3 and h['ai_rank'] <= 3 and h.get('ev', 0) >= 1.0:
            marks[h['num']] = '高'

    # 推マーク: AI順位が人気より3以上高い かつ AI上位8位 かつ EV>=1.2（最大2頭）
    osusume = [h for h in scored
               if h['ai_rank'] <= h['_pop'] - 3
               and h['ai_rank'] <= 8
               and h.get('ev', 0) >= 1.2
               and marks[h['num']] == '']
    osusume.sort(key=lambda h: h.get('ev', 0), reverse=True)
    for h in osusume[:2]:
        marks[h['num']] = '推'

    # 穴マーク: 6番人気以下 かつ AI上位5位 かつ EV>=1.5 かつ cal_prob>=0.15（最大1頭）
    ana_cands = [h for h in scored
                 if h['_pop'] >= 6
                 and h['ai_rank'] <= 5
                 and h.get('ev', 0) >= 1.5
                 and h.get('pn', 0) >= 0.15
                 and marks[h['num']] == '']
    if ana_cands:
        ana = max(ana_cands, key=lambda h: h.get('ev', 0))
        marks[ana['num']] = '穴'

    return marks


def _build_horses_list(scored, top1, by_odds, value_gap_map=None, odds_lookup=None):
    """アプリ表示用の馬リストを生成する（馬番順）。

    本命はAI確率1位の馬。マークは _assign_marks で決定。
    Val列はEV（pn × win_odds）を使用。オッズ欠損時はNone（"-"表示）。
    """
    marks = _assign_marks(scored, by_odds)
    odds_lookup = odds_lookup or {}

    horses = []
    for h in scored:
        pop = h.get('_pop') or next((i + 1 for i, x in enumerate(by_odds) if x['name'] == h['name']), 99)
        pn  = h.get('pn', 0)
        wo  = h.get('win_odds', 0) or 0

        ev_val = round(pn * wo, 3) if wo > 0 else None

        horse = {
            'n':        h['num'],
            'name':     h['name'],
            'odds':     wo,
            'score':    round(h['total'], 1),
            'pop':      pop,
            'ai_rank':  h.get('ai_rank', 99),
            'style':    h.get('running_style', '差し'),
            'tan_pct':  round(min(60,  pn * 100),       1),
            'ren_pct':  round(min(80,  pn * 2.0 * 100), 1),
            'fuku_pct': round((h.get('top3_prob') or pn) * 100, 1),
            'rl_rank':  h.get('rl_rank', 99),
            'cl_rank':  h.get('cl_rank', 99),
            'ev':       ev_val,
            'prob_gap': round(h.get('prob_gap', 0.0), 4),
            'cal_prob': round(h.get('cal_prob', pn), 4),
            'mark':     marks[h['num']],
        }
        od = odds_lookup.get(h['num'])
        if od:
            horse['tansho_odds']  = od.get('tansho_odds')
            horse['fukusho_odds'] = od.get('fukusho_odds')
        horses.append(horse)
    horses.sort(key=lambda x: x['n'])
    return horses


def _detect_trio_structure(trio):
    """三連複の買い目パターンを判定し、BOX/フォーメーション/リストに分類する。"""
    from math import comb

    keys = [tuple(sorted(b['key'])) for b in trio]
    pts = len(keys)
    all_nums = sorted({n for k in keys for n in k})

    if comb(len(all_nums), 3) == pts:
        return {'type': 'box', 'nums': all_nums}

    freq = {}
    for k in keys:
        for n in k:
            freq[n] = freq.get(n, 0) + 1

    axes = sorted(n for n, f in freq.items() if f == pts)
    others = sorted(n for n in all_nums if n not in axes)

    if len(axes) == 1 and comb(len(others), 2) == pts:
        return {
            'type': 'formation',
            'nums': all_nums,
            'legs': [[axes[0]], others, others],
        }

    if len(axes) == 2 and len(others) == pts:
        return {
            'type': 'formation',
            'nums': all_nums,
            'legs': [axes, axes, others],
        }

    return {'type': 'list', 'nums': all_nums}


def _format_gumbel_bets(gb, scored):
    """build_optimal_bets の出力をアプリ表示リストに変換する。"""
    if not gb:
        return []

    name_map = {}
    for h in scored:
        k = h.get('horse_num') or h.get('num')
        if k is not None:
            name_map[int(k)] = h.get('name', '')

    result = []

    for b in gb.get('win', []):
        num = b['key']
        result.append({
            'tag':   'tan',
            'label': '単勝',
            'horse': f'#{num} {name_map.get(num, "")}',
            'est':   f'{b["odds"]:.1f}倍',
            'ev':    round(b['ev'], 2),
            'prob':  round(b['prob'], 3),
            'amt':   f'¥{b["amount"]}',
        })

    for b in gb.get('place', []):
        num = b['key']
        result.append({
            'tag':   'fuku',
            'label': '複勝',
            'horse': f'#{num} {name_map.get(num, "")}',
            'est':   f'{b["odds"]:.1f}倍',
            'ev':    round(b['ev'], 2),
            'prob':  round(b['prob'], 3),
            'amt':   f'¥{b["amount"]}',
        })

    for b in gb.get('quinella', []):
        a, bb = b['key']
        result.append({
            'tag':   'umaren',
            'label': '馬連',
            'horse': f'#{a}-#{bb}',
            'est':   f'{b["odds"]:.1f}倍',
            'ev':    round(b['ev'], 2),
            'prob':  round(b['prob'], 3),
            'amt':   f'¥{b["amount"]}',
        })

    trio = gb.get('trio', [])
    if trio:
        pts  = len(trio)
        s    = gb.get('summary', {})
        pmin = s.get('payout_min', 0)
        pmax = s.get('payout_max', 0)
        syn  = s.get('syn_odds', 0)
        avg_ev = round(sum(b['ev'] for b in trio) / pts, 2)

        structure = _detect_trio_structure(trio)
        combos_short = ['-'.join(str(n) for n in b['key']) for b in trio]

        result.append({
            'tag':       'sanfuku',
            'label':     f'三連複({pts}点)',
            'trio_type': structure['type'],
            'legs':      structure.get('legs'),
            'nums':      structure['nums'],
            'combos':    combos_short,
            'est':       f'¥{pmin:,}〜¥{pmax:,}',
            'ev':        avg_ev,
            'syn_odds':  syn,
            'amt':       f'¥{pts * 100}',
        })

    return result


def _build_bet_list(bets):
    """ベット辞書リストをアプリ表示形式に変換する。三連複F はまとめて1行で表示。"""
    result = []
    san_f = [b for b in bets if b['type'] == '三連複F']

    for b in bets:
        if b['type'] == '三連複F':
            continue  # 後でまとめて追加
        if b['type'] == '三連複':
            # 従来EV路（レガシー）からの三連複
            tickets = b.get('tickets', [b['nums']])
            pts = len(tickets)
            result.append({
                'tag':   'san',
                'label': f'三連複({pts}点)',
                'horse': b.get('horse_name', '-'.join(f'#{n}' for n in b['nums'][:3])),
                'est':   f'{pts}点',
                'amt':   f'¥{b["amount"]}',
            })
            continue
        tag = ('fuku' if b['type'] == '複勝' else
               'tan'  if b['type'] == '単勝' else 'wide')
        odds_val = b.get('odds_est') or b.get('odds', 0) or 0
        est = f'{odds_val:.1f}倍' if odds_val > 0 else '-'
        result.append({
            'tag':   tag,
            'label': b['type'],
            'horse': f'#{b["nums"][0]} {b.get("horse_name", "")}',
            'est':   est,
            'amt':   f'¥{b["amount"]}',
        })

    if san_f:
        total_amt = sum(t['amount'] for t in san_f)
        pts = len(san_f)
        preview = ' / '.join('-'.join(f'#{n}' for n in t['nums']) for t in san_f[:2])
        if pts > 2:
            preview += f' 他{pts - 2}点'
        result.append({
            'tag':   'san',
            'label': f'三連複({pts}点)',
            'horse': preview,
            'est':   f'{pts}点',
            'amt':   f'¥{total_amt}',
        })

    return result


def to_app_json(selected, races_all, bias_data, jst_now, day_type='friday', market_odds_map=None, base_dir=None):
    """厳選レース＋全レース情報をアプリ用 JSON 形式で返す。

    Args:
        selected  : ability_first_loose が返す候補リスト
        races_all : その日の全レースリスト
        bias_data : 馬場バイアス辞書（省略可）
        jst_now   : datetime（JST）
        day_type  : 'friday' | 'saturday' | 'sunday'
        market_odds_map : {race_id: {horse_num: {'tansho': float, 'fukusho': float}}}
                          形式の市場オッズ（省略可）。レース単位で
                          detect_value_horses / make_bets に渡される。

    Returns:
        dict（JSON シリアライズ可能）
    """
    all_venues   = sorted({r['racecourse'] for r in races_all},
                          key=lambda v: VENUE_ORDER.get(v, 99))
    market_odds_map = market_odds_map or {}
    races_by_venue = {}
    selected_ids = {c['race']['id'] for c in selected}
    total_inv = 0

    # ── 厳選レース（rec=True） ─────────────────────────────────────
    for c in selected:
        race   = c['race']
        scored = c['scored']
        # 本命はAI確率（win_prob）1位の馬（EVではなく能力基準）
        top1   = max(scored, key=lambda h: h.get('pn', 0))
        c['top1'] = top1
        rc     = race['racecourse']
        if rc not in races_by_venue:
            races_by_venue[rc] = []

        _race_odds   = market_odds_map.get(race['id'], {})
        _num_horses  = race.get('num_horses', len(scored))
        by_odds = sorted(scored, key=lambda h: h.get('win_odds') or 99)

        # popularity を win_odds 順位で補完
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
            _h.setdefault('_pop', _rank)
            if 'horse_num' not in _h: _h['horse_num'] = _h.get('num')
            if 'cal_prob' not in _h:  _h['cal_prob']  = _h.get('pn', 0)

        _chaos_score_val = c.get('chaos_score', calc_chaos_score(race, scored))
        _grade    = classify_race_chaos(scored)  # ③ pnベース
        _vh_all   = detect_value_horses(scored, _race_odds)
        _vh_list  = [{'horse_num': _h.get('horse_num', _h.get('num')),
                      'horse_name': _h.get('name', ''),
                      'ev_direct':  round(_h.get('ev_direct', 0), 3),
                      'cal_prob':   round(_h.get('cal_prob', _h.get('pn', 0)), 4),
                      'market_prob': round(_h.get('market_prob', 0), 4)}
                     for _h in _vh_all if _h.get('is_value', False)]  # ⑤ EVベース
        _odds_lookup = {_h.get('horse_num', _h.get('num')):
                        {'tansho_odds': _h.get('tansho_odds'), 'fukusho_odds': _h.get('fukusho_odds')}
                        for _h in _vh_all}

        c.setdefault('chaos_score', _chaos_score_val)
        bets = make_bets(c, _race_odds or None)
        if not bets:
            bets = [{'type': '複勝', 'nums': [top1['num']],
                     'horse_name': top1.get('name', ''),
                     'odds': top1.get('win_odds', 0) or 0,
                     'odds_est': 0, 'amount': 500, 'ev': 0.0, 'prob': 0.0,
                     'pattern': 'fallback', 'chaos_grade': _grade}]

        # ④ フォーメーション生成（_build_horses_list が _pop をセット後に実行）
        _horses_list = _build_horses_list(scored, top1, by_odds, odds_lookup=_odds_lookup)
        for _h in scored:
            _h['popularity'] = _h.get('_pop', 99)
        _formation = build_formation(scored, race)
        if _formation and _formation.get('bets'):
            bets = bets + _formation['bets']

        total_inv += sum(b['amount'] for b in bets)

        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(50, int(60 + (pop_rank - 2) * 4 + c['score_gap'] * 20)))

        _types_str  = '+'.join(dict.fromkeys(b['type'] for b in bets))
        _bet_reason = f'★期待値あり: {_types_str}'

        try:
            from src.betting.bet_optimizer import make_bets_v2 as _mbv2
            _gb, _, _, _ = _mbv2(scored, race, base_dir,
                                  market_odds_map=market_odds_map, n_sims=3000)
            _gumbel_bets = _format_gumbel_bets(_gb, scored)
        except Exception:
            _gumbel_bets = []

        _entry = {
            'r':       race['race_num'],
            'race_id': race['id'],
            'odds_cn': _build_odds_cn(race),
            'name': race['race_name'],
            'dist': f'{race["distance"]}m{race["surface"]}',
            'rec':  True,
            'conf': conf,
            'honmei': {
                'n':     top1['num'],
                'name':  top1['name'],
                'odds':  top1.get('win_odds', 0) or 0,
                'score': top1['total'],
                'style': top1.get('running_style', '差し'),
            },
            'horses':       _horses_list,
            'bets':         _build_bet_list(bets),
            'gumbel_bets':  _gumbel_bets,
            'formation':    _formation,
            'chaos_level':  _grade,
            'chaos_grade':  _grade,
            'num_horses':   _num_horses,
            'value_horses': _vh_list,
            'bet_reason':   _bet_reason,
            'cmt': auto_comment(c, bias_data),
        }
        races_by_venue[rc].append(_entry)

    # ── 非厳選レース（rec=False・参考買い目つき）──────────────────
    for race in sorted(races_all,
                       key=lambda r: (VENUE_ORDER.get(r['racecourse'], 99),
                                      r['race_num'])):
        if race['id'] in selected_ids:
            continue
        rc = race['racecourse']
        if rc not in races_by_venue:
            races_by_venue[rc] = []

        scored = calc_all(race, bias_data)
        if not scored:
            continue
        top1  = scored[0]
        odds  = top1.get('win_odds') or 99
        gap   = top1['total'] - scored[1]['total'] if len(scored) > 1 else 0
        _num_horses2 = race.get('num_horses', len(scored))

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(1, int(60 + (pop_rank - 2) * 4 + gap * 20)))

        # popularity を win_odds 順位で補完
        for _rank, _h in enumerate(by_odds, 1):
            _h.setdefault('popularity', _rank)
            _h.setdefault('_pop', _rank)
            if 'horse_num' not in _h: _h['horse_num'] = _h.get('num')
            if 'cal_prob' not in _h:  _h['cal_prob']  = _h.get('pn', 0)

        win_prob    = top1.get('pn', 0)
        fuku_prob   = top1.get('top3_prob', min(0.80, win_prob * 3))
        ev_fuku     = calc_ev(fuku_prob, odds * 0.28)
        ev_tan      = calc_ev(win_prob, odds)
        # ③ 波乱度は pnベース
        _chaos_score2 = calc_chaos_score(race, scored)
        _grade2   = classify_race_chaos(scored)
        maiden_note = '⚠ 新馬戦：データ不足のため参考値' if is_maiden_race(race) else ''

        _race_odds2 = market_odds_map.get(race['id'], {})
        _vh_all2     = detect_value_horses(scored, _race_odds2)
        _vh_list2    = [{'horse_num': _h.get('horse_num', _h.get('num')),
                         'horse_name': _h.get('name', ''),
                         'ev_direct':  round(_h.get('ev_direct', 0), 3),
                         'cal_prob':   round(_h.get('cal_prob', _h.get('pn', 0)), 4),
                         'market_prob': round(_h.get('market_prob', 0), 4)}
                        for _h in _vh_all2 if _h.get('is_value', False)]  # ⑤ EVベース
        _odds_lookup2 = {_h.get('horse_num', _h.get('num')):
                         {'tansho_odds': _h.get('tansho_odds'), 'fukusho_odds': _h.get('fukusho_odds')}
                         for _h in _vh_all2}

        c_ref = {
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          max(ev_fuku, ev_tan),
            'chaos_score':     _chaos_score2,
            'chaos_level':     _grade2,
        }
        bets2 = make_bets(c_ref, _race_odds2 or None)
        if not bets2:
            bets2 = [{'type': '複勝', 'nums': [top1['num']],
                      'horse_name': top1.get('name', ''),
                      'odds': top1.get('win_odds', 0) or 0,
                      'odds_est': 0, 'amount': 500, 'ev': 0.0, 'prob': 0.0,
                      'pattern': 'fallback', 'chaos_grade': _grade2}]

        # ④ フォーメーション生成（_build_horses_list が _pop をセット後）
        _horses_list2 = _build_horses_list(scored, top1, by_odds, odds_lookup=_odds_lookup2)
        for _h in scored:
            _h['popularity'] = _h.get('_pop', 99)
        _formation2 = build_formation(scored, race)
        if _formation2 and _formation2.get('bets'):
            bets2 = bets2 + _formation2['bets']

        _types_str2  = '+'.join(dict.fromkeys(b['type'] for b in bets2))
        _bet_reason2 = f'参考: {_types_str2}'

        try:
            from src.betting.bet_optimizer import make_bets_v2 as _mbv2
            _gb2, _, _, _ = _mbv2(scored, race, base_dir,
                                   market_odds_map=market_odds_map, n_sims=3000)
            _gumbel_bets2 = _format_gumbel_bets(_gb2, scored)
        except Exception:
            _gumbel_bets2 = []

        _entry2 = {
            'r':           race['race_num'],
            'race_id':     race['id'],
            'odds_cn':     _build_odds_cn(race),
            'name':        race['race_name'],
            'dist':        f'{race["distance"]}m{race["surface"]}',
            'rec':         False,
            'conf':        conf,
            'chaos_level': _grade2,
            'chaos_grade': _grade2,
            'num_horses':  _num_horses2,
            'value_horses': _vh_list2,
            'bet_reason':  _bet_reason2,
            'honmei': {
                'n':     top1['num'],
                'name':  top1['name'],
                'odds':  top1.get('win_odds', 0) or 0,
                'score': top1['total'],
                'style': top1.get('running_style', '差し'),
            },
            'horses':      _horses_list2,
            'bets':        _build_bet_list(bets2),
            'gumbel_bets': _gumbel_bets2,
            'formation':   _formation2,
            'cmt':         auto_comment(c_ref, bias_data) + ('\n' + maiden_note if maiden_note else ''),
        }
        races_by_venue[rc].append(_entry2)

    bias_txt = '内外:フラット ペース:±0 時計:±0'
    bias_tag = 'フラット'
    if bias_data:
        bias_txt = bias_data.get('summary', bias_txt)
        spd      = bias_data.get('track_speed', 0)
        bias_tag = ('時計速め' if spd > 0.3 else '時計遅め' if spd < -0.3 else 'フラット')

    from datetime import timedelta
    # sunday予想は翌日の日付を表示（土曜夜に実行するため）
    display_dt = jst_now + timedelta(days=1) if day_type == 'sunday' else jst_now
    jday = ['月', '火', '水', '木', '金', '土', '日'][display_dt.weekday()]
    rec_count = len(selected)
    return {
        'generated_at':      jst_now.isoformat(),
        'date':              f'{display_dt.month}月{display_dt.day}日({jday})',
        'type':              day_type,
        'venues':            all_venues,
        'bias':              {'txt': bias_txt, 'tag': bias_tag},
        'recommended_count': rec_count,
        'message':           ('本日の推奨レースはありません。閾値を満たすレースがありませんでした。'
                              if rec_count == 0 else None),
        'stats':             {'invest': total_inv, 'rec': rec_count, 'roi': 150},
        'races':             races_by_venue,
    }

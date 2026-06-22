"""
アプリ用 JSON 生成。
ノートブックの to_app_json を分離。
"""
from src.betting.ev_filter import (VENUE_ORDER, VALUE_GAP_THRESHOLD,
                                    calc_market_probs, calc_value_score,
                                    classify_race_chaos, detect_value_horses,
                                    is_maiden_race)
from src.betting.make_bets import classify_chaos_grade, calc_ev, make_bets, build_formation
from src.features.engine import auto_comment, calc_all


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
    """
    marks = _assign_marks(scored, by_odds)
    value_gap_map = value_gap_map or {}
    odds_lookup = odds_lookup or {}

    horses = []
    for h in scored:
        pop = h.get('_pop') or next((i + 1 for i, x in enumerate(by_odds) if x['name'] == h['name']), 99)
        pn  = h.get('pn', 0)
        wo  = h.get('win_odds', 0) or 0

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
            'fuku_pct': round(h.get('top3_prob', pn) * 100, 1),
            'rl_rank':  h.get('rl_rank', 99),
            'cl_rank':  h.get('cl_rank', 99),
            'ev':       round(h.get('ev', 0.0), 3),
            'prob_gap': round(h.get('prob_gap', 0.0), 4),
            'cal_prob': round(h.get('cal_prob', pn), 4),
            'mark':     marks[h['num']],
        }
        od = odds_lookup.get(h['num'])
        if od:
            horse['tansho_odds']  = od.get('tansho_odds')
            horse['fukusho_odds'] = od.get('fukusho_odds')
        vg = value_gap_map.get(h['num'])
        if vg:
            horse['value_gap'] = round(vg, 3)
        horses.append(horse)
    horses.sort(key=lambda x: x['n'])
    return horses


def _build_bet_list(bets):
    """ベット辞書リストをアプリ表示形式に変換する。"""
    result = []
    for b in bets:
        tag = ('fuku' if b['type'] == '複勝' else
               'tan'  if b['type'] == '単勝' else 'wide')
        est = (f'推定{b["odds_est"]:.1f}倍' if b['type'] == '複勝'
               else f'{b["odds"]:.1f}倍')
        result.append({
            'tag':   tag,
            'label': b['type'],
            'horse': f'#{b["nums"][0]} {b["horse_name"]}',
            'est':   est,
            'amt':   f'¥{b["amount"]}',
        })
    return result


def to_app_json(selected, races_all, bias_data, jst_now, day_type='friday', market_odds_map=None):
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

    # ── 厳選レース ─────────────────────────────────────────────────
    for c in selected:
        race   = c['race']
        scored = c['scored']
        # 本命はAI確率（win_prob）1位の馬（EVではなく能力基準）
        top1   = max(scored, key=lambda h: h.get('pn', 0))
        c['top1'] = top1
        _race_odds = market_odds_map.get(race['id'], {})
        bets   = make_bets(c, _race_odds or None)
        if not bets:
            continue
        total_inv += sum(b['amount'] for b in bets)
        rc     = race['racecourse']
        if rc not in races_by_venue:
            races_by_venue[rc] = []

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(50, int(60 + (pop_rank - 2) * 4 + c['score_gap'] * 20)))

        # chaos_grade / value_horses / bet_reason の計算
        _chaos_score_val = c.get('chaos_score', 0)
        _num_horses      = race.get('num_horses', len(scored))
        for _h in scored:
            if 'horse_num' not in _h:
                _h['horse_num'] = _h.get('num')
            if 'cal_prob' not in _h:
                _h['cal_prob'] = _h.get('pn', 0)
            if 'popularity' not in _h:
                _h['popularity'] = _h.get('_pop', 99)
        _grade    = classify_chaos_grade(scored, _chaos_score_val)
        _vh_all   = detect_value_horses(scored, _race_odds)
        _vh_list  = [{'horse_num': _h.get('horse_num', _h.get('num')),
                      'horse_name': _h.get('name', ''),
                      'value_gap':  round(_h.get('value_gap', 0), 4),
                      'cal_prob':   round(_h.get('cal_prob', _h.get('pn', 0)), 4),
                      'market_prob': round(_h.get('market_prob', 0), 4)}
                     for _h in _vh_all if _h.get('value_gap', 0) >= VALUE_GAP_THRESHOLD]
        _vg_map   = {_h.get('horse_num', _h.get('num')): _h.get('value_gap', 0) for _h in _vh_all}
        _odds_lookup = {_h.get('horse_num', _h.get('num')):
                        {'tansho_odds': _h.get('tansho_odds'), 'fukusho_odds': _h.get('fukusho_odds')}
                        for _h in _vh_all}
        _vh_count = len(_vh_list)
        # bet_reason は頭数ベースのルールに合わせて生成
        if _num_horses >= 14:
            _bet_reason = (f'多頭数({_num_horses}頭)・馬連+' +
                           ('三連複' if _grade == 'C' else 'ワイド'))
        elif _num_horses <= 8:
            _bet_reason = f'少頭数({_num_horses}頭)・馬連'
        elif _grade == 'A':
            _bet_reason = f'中頭数({_num_horses}頭)+堅い・馬連'
        elif _grade == 'B':
            _bet_reason = (f'中頭数({_num_horses}頭)+中荒れ・' +
                           ('複勝+ワイド' if _vh_count > 0 else '複勝'))
        else:
            _bet_reason = (f'中頭数({_num_horses}頭)+大荒れ・' +
                           ('複勝少額' if _vh_count > 0 else 'スキップ'))

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
            'horses':      _build_horses_list(scored, top1, by_odds, _vg_map, _odds_lookup),
            'bets':        _build_bet_list(bets),
            'chaos_level': c.get('chaos_level', classify_race_chaos(scored)),
            'chaos_grade': _grade,
            'num_horses':  _num_horses,
            'value_horses': _vh_list,
            'bet_reason':  _bet_reason,
            'cmt': auto_comment(c, bias_data),
        }
        # _build_horses_list が scored に _pop をセットした後でフォーメーション生成
        for _h in scored:
            _h['popularity'] = _h.get('_pop', 99)
        _entry['formation'] = build_formation(scored, race)
        races_by_venue[rc].append(_entry)

    # ── 非厳選レース（全レース買い目つき）─────────────────────────
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

        by_odds  = sorted(scored, key=lambda h: h.get('win_odds') or 99)
        pop_rank = next((i + 1 for i, h in enumerate(by_odds)
                         if h['name'] == top1['name']), 99)
        conf = min(99, max(1, int(60 + (pop_rank - 2) * 4 + gap * 20)))

        win_prob    = top1.get('pn', 0)
        fuku_prob   = top1.get('top3_prob', min(0.80, win_prob * 3))
        ev_fuku     = calc_ev(fuku_prob, odds * 0.28)
        ev_tan      = calc_ev(win_prob, odds)
        chaos_lvl   = classify_race_chaos(scored)
        maiden_note = '⚠ 新馬戦：データ不足のため参考値' if is_maiden_race(race) else ''
        c_ref    = {
            'race':            race,
            'scored':          scored,
            'top1':            top1,
            'odds':            odds,
            'popularity_rank': pop_rank,
            'score_gap':       gap,
            'ev_fuku':         ev_fuku,
            'ev_tan':          ev_tan,
            'ev_max':          max(ev_fuku, ev_tan),
            'chaos_level':     chaos_lvl,
        }
        _race_odds2 = market_odds_map.get(race['id'], {})
        for _h in scored:
            if 'horse_num' not in _h:
                _h['horse_num'] = _h.get('num')
            if 'cal_prob' not in _h:
                _h['cal_prob'] = _h.get('pn', 0)
        _vh_all2     = detect_value_horses(scored, _race_odds2)
        _odds_lookup2 = {_h.get('horse_num', _h.get('num')):
                         {'tansho_odds': _h.get('tansho_odds'), 'fukusho_odds': _h.get('fukusho_odds')}
                         for _h in _vh_all2}
        _vg_map2 = {_h.get('horse_num', _h.get('num')): _h.get('value_gap', 0) for _h in _vh_all2}

        bets = make_bets(c_ref)

        races_by_venue[rc].append({
            'r':           race['race_num'],
            'race_id':     race['id'],
            'odds_cn':     _build_odds_cn(race),
            'name':        race['race_name'],
            'dist':        f'{race["distance"]}m{race["surface"]}',
            'rec':         False,
            'conf':        conf,
            'chaos_level': chaos_lvl,
            'honmei': {
                'n':     top1['num'],
                'name':  top1['name'],
                'odds':  top1.get('win_odds', 0) or 0,
                'score': top1['total'],
                'style': top1.get('running_style', '差し'),
            },
            'horses': _build_horses_list(scored, top1, by_odds, _vg_map2, _odds_lookup2),
            'bets':   _build_bet_list(bets),
            'cmt':    auto_comment(c_ref, bias_data) + ('\n' + maiden_note if maiden_note else ''),
        })

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

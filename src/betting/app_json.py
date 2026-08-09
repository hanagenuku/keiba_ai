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
        if h['_pop'] <= 3 and h['ai_rank'] <= 3 and (h.get('ev') or 0) >= 1.0:
            marks[h['num']] = '高'

    # 推マーク: AI順位が人気より3以上高い かつ AI上位8位 かつ EV>=1.2（最大2頭）
    osusume = [h for h in scored
               if h['ai_rank'] <= h['_pop'] - 3
               and h['ai_rank'] <= 8
               and (h.get('ev') or 0) >= 1.2
               and marks[h['num']] == '']
    osusume.sort(key=lambda h: h.get('ev') or 0, reverse=True)
    for h in osusume[:2]:
        marks[h['num']] = '推'

    # 穴マーク: 6番人気以下 かつ AI上位5位 かつ EV>=1.5 かつ cal_prob>=0.15（最大1頭）
    ana_cands = [h for h in scored
                 if h['_pop'] >= 6
                 and h['ai_rank'] <= 5
                 and (h.get('ev') or 0) >= 1.5
                 and h.get('pn', 0) >= 0.15
                 and marks[h['num']] == '']
    if ana_cands:
        ana = max(ana_cands, key=lambda h: h.get('ev') or 0)
        marks[ana['num']] = '穴'

    return marks


def _build_solo_ranks(scored):
    """市場ゼロAI順位（ability_margin 降順）を {horse_num: rank} で返す。

    `ability_margin` は残差学習モデルの `raw_margin - base_margin`、つまり
    市場アンカーを差し引いた「AI自身の評価」。表示専用の**併記**であり、
    買い目・軸・レース厳選・成績集計はすべて従来の `rl_rank` のままである。

    ⚠ この順位は当てる力では `rl_rank` に劣ることが実測済み。
    単勝AUCは市場ゼロAI 0.7778 / 市場ありAI 0.8252（2026-08-03）。
    「AIが市場より強気」の度合いで並べると回収率の傾きが**逆**になり、
    +6以上の帯は41.4%（過去人気の寄与も抜くと21.8%）。
    したがって馬券判断には使わず、「AI自身が何を見ているか」を
    人間が読むためだけの列とする。

    1頭でも `ability_margin` を持たない場合はレース全体で None を返す
    （非残差モデル運用時・SHAP/推論フォールバック時に、一部の馬だけ
    別基準の順位が混ざるのを防ぐ。index.html の再同期と同じ方針）。
    """
    if not scored or any(h.get('ability_margin') is None for h in scored):
        return {}
    # タイブレークは馬番昇順（意味を持たない中立な順序）。
    # ⚠ 着順や既存順位でタイブレークすると、そこから情報が漏れる
    #    （2026-08-08のPLレーティングのリークがこの形だった）。
    order = sorted(scored, key=lambda h: (-h['ability_margin'], h['num']))
    return {h['num']: i + 1 for i, h in enumerate(order)}


def _build_horses_list(scored, top1, by_odds, odds_lookup=None, base_dir=None):
    """アプリ表示用の馬リストを生成する（馬番順）。

    本命はAI確率1位の馬。マークは _assign_marks で決定。
    Val列はEV（pn × win_odds）を使用。オッズ欠損時はNone（"-"表示）。
    base_dir 指定時は人気×RL乖離マトリクスのセル判定（mx_flag）を馬ごとに付与
    （クライアント側の直前オッズ再計算でも同じフィルタを適用するため）。
    """
    marks = _assign_marks(scored, by_odds)
    odds_lookup = odds_lookup or {}
    solo_ranks = _build_solo_ranks(scored)

    _mx_classify = None
    if base_dir is not None:
        try:
            from src.betting.rank_matrix_filter import classify_horse as _mx_classify
        except Exception:
            _mx_classify = None

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
            # 市場ゼロAI順位（ability_margin降順）。表示専用の併記で、
            # 買い目・軸・集計は従来どおり rl_rank を使う。詳細は
            # _build_solo_ranks のdocstring（当てる力では rl_rank に劣る）。
            # 非残差モデル時はNone → アプリ側で列ごと非表示にする。
            'solo_rank': solo_ranks.get(h['num']),
            'cl_rank':  h.get('cl_rank', 99),
            'ev':       ev_val,
            'prob_gap': round(h.get('prob_gap', 0.0), 4),
            'cal_prob': round(h.get('cal_prob', pn), 4),
            'mark':     marks[h['num']],
            # 市場非依存のAI能力スコア（残差学習モデル時のみ非None）。
            # 直前オッズ取得時にクライアント側でbase_marginを引き直し、
            # 勝率・複勝率を最新オッズと同じ時点に再同期するために使う。
            'ability_margin': h.get('ability_margin'),
            # ability_marginのカテゴリ別SHAP寄与度（騎手/距離適性/コース適性等）。
            # 「AIが市場と違う評価をした理由」を人間可読に説明するための項目。
            # 非残差モデル時・SHAP計算失敗時はNone
            'ability_breakdown': h.get('ability_breakdown'),
            # 人気×RL乖離マトリクスのセル判定: 'boost'（実測回収率120%以上の
            # セル）/'suppress'（同50%未満）/None。クライアント側の
            # 直前オッズ再計算（recalcGumbelBets）でも同じフィルタを使う
            'mx_flag': _mx_classify(pop, h.get('rl_rank'), base_dir) if _mx_classify else None,
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



def _build_ai_pair_bets(scored):
    """AI強気ペア(A×D)の馬連・ワイド推奨。

    2026-07-31の大規模検証で比1.82/1.56（損益分岐1.29）。
    ⚠ 実配当が無く回収率は未検証。詳細は src/betting/ai_pair_bets.py の
    docstring を参照。AI_PAIR_BETS=0 で無効化できる。
    """
    try:
        from src.betting.ai_pair_bets import build_ai_pair_bets
        return build_ai_pair_bets(scored) or None
    except Exception:
        return None


def _format_axis_bets(ab, scored):
    """build_axis_bets の出力をアプリ表示リストに変換する。

    ⚠ 回収率は控除率を超えていない（詳細は axis_bets.py の冒頭）。
    表示側で「検証中」と明記すること。
    """
    if not ab:
        return []
    name_map = {}
    for h in scored:
        k = h.get('horse_num') or h.get('num')
        if k is not None:
            name_map[int(k)] = h.get('name', '')

    result = []
    s = ab.get('summary', {})
    axis = s.get('axis')

    for b in ab.get('win', []):
        n = b['key']
        result.append({
            'tag': 'tan', 'label': '単勝（軸）',
            'horse': f'#{n} {name_map.get(n, "")}',
            'est': f'{b["odds"]:.1f}倍',
            # ⚠ EVは出さない。表示EVはGumbelシミュレーション(T=2.5で平坦化)の
            # 確率から出しており、画面の勝率とも実際の回収率とも一致しない。
            # 実測（389レース・軸の単勝）:
            #   〜2.0倍  EV0.28 → 回収79.1%     5〜8倍 EV1.45 → 回収34.5%
            # EVが高い帯ほど回収率が低いという逆行が起きており、誤解を招く。
            'amt': f'¥{b["amount"]}',
        })

    # ── 馬連・ワイドは「軸 - 相手」でまとめる ────────────────────────────
    # 1点ずつ並べると全部に軸が入っていることが読み取れないため。
    def _grouped(items, tag, label, syn=None):
        if not items:
            return None
        mates, legs = [], []
        for b in items:
            x, y = b['key']
            m = y if x == axis else x
            mates.append(m)
            legs.append({'n': m, 'name': name_map.get(m, ''),
                         'odds': b['odds'], 'prob': b['prob']})
        lo = min(b['odds'] for b in items)
        hi = max(b['odds'] for b in items)
        e = {
            'tag': tag, 'label': f'{label}({len(items)}点)',
            'axis': axis,
            'horse': f'#{axis} - ' + '・'.join(f'#{m}' for m in mates),
            'mates': legs,
            'est': (f'{lo:.1f}倍' if lo == hi else f'{lo:.1f}〜{hi:.1f}倍'),
            'amt': f'¥{len(items) * 100}',
        }
        if syn:
            e['syn_odds'] = syn
        return e

    for e in (_grouped(ab.get('wide', []), 'wide', 'ワイド'),
              _grouped(ab.get('quinella', []), 'umaren', '馬連',
                       s.get('quinella_syn_odds'))):
        if e:
            result.append(e)

    trio = ab.get('trio', [])
    if trio:
        combos = ['-'.join(str(n) for n in b['key']) for b in trio]
        nums = sorted({n for b in trio for n in b['key']})
        result.append({
            'tag': 'sanfuku', 'label': f'三連複({len(trio)}点)',
            # legs を渡すと index.html がフォーメーション表示に切り替わる。
            # 渡さないと13点がバラバラに並び、A-BC-... の構造が読み取れない。
            'trio_type': 'formation',
            'legs': s.get('trio_legs'),
            'nums': nums, 'combos': combos,
            'est': f'¥{s.get("payout_min", 0):,}〜¥{s.get("payout_max", 0):,}',
            'syn_odds': s.get('trio_syn_odds'),
            'amt': f'¥{len(trio) * 100}',
        })
    return result


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
            tickets = b.get('tickets', [b['nums']])
            pts = len(tickets)
            structure = _detect_trio_structure([{'key': sorted(t)} for t in tickets])
            combos = ['-'.join(str(n) for n in sorted(t)) for t in tickets]
            result.append({
                'tag':       'sanfuku',
                'label':     f'三連複({pts}点)',
                'trio_type': structure['type'],
                'legs':      structure.get('legs'),
                'nums':      structure['nums'],
                'combos':    combos,
                'est':       f'{pts}点',
                'amt':       f'¥{b["amount"]}',
            })
            continue
        tag = ('fuku' if b['type'] == '複勝' else
               'tan'  if b['type'] == '単勝' else 'wide')
        odds_val = b.get('odds_est') or b.get('odds', 0) or 0
        est = f'{odds_val:.1f}倍' if odds_val >= 1.0 else '-'
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
        all_tickets = [t['nums'] for t in san_f]
        structure = _detect_trio_structure([{'key': sorted(t)} for t in all_tickets])
        combos = ['-'.join(str(n) for n in sorted(t)) for t in all_tickets]
        result.append({
            'tag':       'sanfuku',
            'label':     f'三連複({pts}点)',
            'trio_type': structure['type'],
            'legs':      structure.get('legs'),
            'nums':      structure['nums'],
            'combos':    combos,
            'est':       f'{pts}点',
            'amt':       f'¥{total_amt}',
        })

    return result


def to_app_json(selected, races_all, bias_data, jst_now, day_type='friday', market_odds_map=None, base_dir=None,
                odds_updated_count=None, parse_failures=None, same_day=False):
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
        odds_updated_count : apply_odds_to_races() が返したオッズ反映頭数（省略可）。
                          races_all の全出走頭数との比率を data_quality.odds_coverage
                          として出力する（オッズが実質未取得のまま予想が生成された
                          回を気づけるようにするため。2026-07-25調査で発見した
                          「オッズ0頭のまま予想が静かに生成される」問題への対応）。
        parse_failures  : fetch_races_on_date() が返す取得失敗レースのリスト（省略可）。
                          [{'racecourse': str, 'race_num': int, 'reason': str}, ...]
        same_day  : True の場合、表示日付を jst_now そのまま使う（当日実行の
                          refresh_today()用）。False（デフォルト）は従来通り
                          day_typeがsaturday/sundayなら+1日して表示する
                          （前夜に実行し翌日の予想を生成する既存フロー用）。

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
            _fb_odds = top1.get('win_odds', 0) or 0
            bets = [{'type': '複勝', 'nums': [top1['num']],
                     'horse_name': top1.get('name', ''),
                     'odds': _fb_odds,
                     'odds_est': 0, 'amount': 500,
                     'ev': None if _fb_odds == 0 else 0.0,
                     'prob': 0.0,
                     'pattern': 'fallback', 'chaos_grade': _grade}]

        # ④ フォーメーション生成（_build_horses_list が _pop をセット後に実行）
        _horses_list = _build_horses_list(scored, top1, by_odds, odds_lookup=_odds_lookup, base_dir=base_dir)
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
            # 軸1頭ベース（2026-08-08〜）。従来のEV方式は「AIと市場が最も
            # 食い違う組み合わせ」を選ぶ設計で、実測の的中率が2.7%と最下位
            # だったため置き換えた。詳細は src/betting/axis_bets.py 冒頭。
            from src.betting.axis_bets import make_axis_bets as _mab
            _ab, _, _ = _mab(scored, race, base_dir,
                             market_odds_map=market_odds_map, n_sims=3000)
            _gumbel_bets = _format_axis_bets(_ab, scored)
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
            'ai_pair_bets': _build_ai_pair_bets(scored),
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
            _fb_odds2 = top1.get('win_odds', 0) or 0
            bets2 = [{'type': '複勝', 'nums': [top1['num']],
                      'horse_name': top1.get('name', ''),
                      'odds': _fb_odds2,
                      'odds_est': 0, 'amount': 500,
                      'ev': None if _fb_odds2 == 0 else 0.0,
                      'prob': 0.0,
                      'pattern': 'fallback', 'chaos_grade': _grade2}]

        # ④ フォーメーション生成（_build_horses_list が _pop をセット後）
        _horses_list2 = _build_horses_list(scored, top1, by_odds, odds_lookup=_odds_lookup2, base_dir=base_dir)
        for _h in scored:
            _h['popularity'] = _h.get('_pop', 99)
        _formation2 = build_formation(scored, race)
        if _formation2 and _formation2.get('bets'):
            bets2 = bets2 + _formation2['bets']

        _types_str2  = '+'.join(dict.fromkeys(b['type'] for b in bets2))
        _bet_reason2 = f'参考: {_types_str2}'

        try:
            from src.betting.axis_bets import make_axis_bets as _mab
            _ab2, _, _ = _mab(scored, race, base_dir,
                              market_odds_map=market_odds_map, n_sims=3000)
            _gumbel_bets2 = _format_axis_bets(_ab2, scored)
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
            'ai_pair_bets': _build_ai_pair_bets(scored),
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
    # friday→saturday, sunday は翌日の日付を表示（前夜に実行するため）。
    # same_day=Trueの場合（refresh_today、当日実行）はjst_nowをそのまま使う。
    display_dt = (jst_now if same_day
                  else jst_now + timedelta(days=1) if day_type in ('saturday', 'sunday')
                  else jst_now)
    jday = ['月', '火', '水', '木', '金', '土', '日'][display_dt.weekday()]
    rec_count = len(selected)

    total_horses = sum(len(r.get('horses', [])) for r in races_all)
    odds_coverage = (round(odds_updated_count / total_horses, 3)
                     if (odds_updated_count is not None and total_horses > 0) else None)
    data_quality = {
        'odds_coverage':  odds_coverage,
        'parse_failures': parse_failures or [],
    }

    return {
        'generated_at':      jst_now.isoformat(),
        'date':              f'{display_dt.month}月{display_dt.day}日({jday})',
        'type':              day_type,
        'venues':            all_venues,
        'bias':              {'txt': bias_txt, 'tag': bias_tag},
        'recommended_count': rec_count,
        'message':           ('本日の推奨レースはありません。閾値を満たすレースがありませんでした。'
                              if rec_count == 0 else None),
        # ⚠ ここに回収率を入れないこと。
        # 以前は 'roi': 150 という**べた書きの固定値**が入っており、アプリ下部に
        # 「ROI予測 ~150%」と表示され続けていた（計算も予測もしていない数字）。
        # 実測は stats.json の weekly_roi で通算47.4%（2026-08-02時点）であり、
        # 見た人に真逆の印象を与える。実績はアプリ側が stats.json から読む。
        'stats':             {'invest': total_inv, 'rec': rec_count},
        'races':             races_by_venue,
        'data_quality':      data_quality,
    }

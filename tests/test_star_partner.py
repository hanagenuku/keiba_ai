"""☆を買い目に絡める（2026-11-18 ユーザー指示・tickets.include_star）。"""
from src.product import tickets as T


def _horses():
    # 本番の marked と同じ形（n・rl_rank・odds）。1番人気＝◎＝1番
    odds = {1: 2.0, 2: 4.0, 3: 5.5, 4: 8.0, 5: 12.0, 6: 20.0, 7: 25.0, 8: 60.0}
    return [{'n': n, 'rl_rank': n, 'odds': o, 'pop': n} for n, o in odds.items()]


def _race(star=None, conf=80):
    r = {'conf': conf, 'r': 11}
    if star is not None:
        r['star'] = {'n': star, 'name': 'X'}
    return r


def test_star_same_as_honmei_changes_no_ticket():
    base = T.build_tickets(_race(), _horses())
    same = T.build_tickets(_race(star=1), _horses())
    assert same['star_role'] == 'honmei'
    assert same['partners'] == base['partners'] and same['bets'] == base['bets']


def test_star_outside_partners_replaces_the_lowest_ranked_partner():
    base = T.build_tickets(_race(), _horses())
    assert 8 not in base['partners'] and len(base['partners']) == 5
    t = T.build_tickets(_race(star=8), _horses())
    assert t['star_role'] == 'partner'
    assert 8 in t['partners'] and len(t['partners']) == len(base['partners'])
    dropped = set(base['partners']) - set(t['partners'])
    assert len(dropped) == 1 and max(base['partners']) in dropped      # rl_rank が一番下の馬
    wide = next(b for b in t['bets'] if b['type'] == 'ワイド')
    assert [1, 8] in [sorted(c) for c in wide['combos']] and wide['points'] == 2


def test_star_already_a_partner_only_makes_sure_wide_has_it():
    t = T.build_tickets(_race(star=3), _horses())
    base = T.build_tickets(_race(), _horses())
    assert t['star_role'] == 'partner' and t['partners'] == base['partners']
    wide = next(b for b in t['bets'] if b['type'] == 'ワイド')
    assert [1, 3] in [sorted(c) for c in wide['combos']]


def test_star_without_odds_is_not_forced_in():
    hs = _horses()
    hs[7]['odds'] = None
    t = T.build_tickets(_race(star=8), hs)
    assert t['star_role'] is None and 8 not in t['partners']


def test_no_star_keeps_old_tickets():
    t = T.build_tickets(_race(), _horses())
    assert t['star_role'] is None

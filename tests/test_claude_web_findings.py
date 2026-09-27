"""Claude の web 見解層（claude_web）の回帰テスト。

🔴 このプロジェクトでは「予想の出力を後から書き換える層」が**5件すべて撤回**されている
（市場補正レイヤー / error_tags週次補正 / rank_matrix_filter / AI xx%バッジ /
ROI予測150%）。この層が6件目にならないための条件を、ここでコードとして固定する:

  1. 対象は 10R/11R だけ（web の記事が存在するのは特別・重賞だけ）
  2. 日付が違う見解は使わない（前の開催の印を今日の画面に出さない）
  3. 出典の無い見解は使わない
  4. このレースに居ない馬番は落とし、落とした理由を持って回る
  5. **確率・スコア・買い目・推奨フラグを1つも変えない**（← 5件との構造的な違い）
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.claude_web import (SCHEMA_VERSION, WEB_RACE_NUMS,
                                     build_claude_view, findings_path, load_findings)

JST = timezone(timedelta(hours=9))
RID = '20260927_06_11'


def _notes(**over):
    entry = {
        'order': [9, 13, 14],
        'marks': {'9': '◎', '13': '○'},
        'horses': {'13': {'note': '調教最高評価・馬場不問', 'direction': 'up',
                          'sources': ['https://example.com/a']}},
        'going': {'surface': '芝', 'state': '重', 'expected': 'やや重〜重'},
        'summary': 'まとめ',
        'sources': ['https://example.com/a'],
    }
    entry.update(over.pop('entry', {}))
    data = {'schema_version': SCHEMA_VERSION, 'date': '20260927',
            'collected_at': '2026-09-27T09:10:00+09:00', 'races': {RID: entry}}
    data.update(over)
    return data


HORSES = [{'num': n} for n in (9, 13, 14, 15)]


def test_target_races_are_10_and_11_only():
    assert WEB_RACE_NUMS == (10, 11)


def test_other_race_numbers_get_nothing():
    for rno in (1, 5, 9, 12):
        assert build_claude_view(RID, rno, HORSES, _notes(), '20260927') is None, rno


def test_view_is_built_for_target_race():
    v = build_claude_view(RID, 11, HORSES, _notes(), '20260927')
    assert v['order'] == [9, 13, 14]
    assert v['marks'] == {9: '◎', 13: '○'}
    assert v['web_ranks'] == {9: 1, 13: 2, 14: 3}
    assert v['horse_notes'][13]['direction'] == 'up'
    assert v['going']['state'] == '重'
    assert v['coverage'] == 0.75


def test_stale_notes_are_refused():
    """🔴 前の開催の印を今日の画面に出さない。"""
    assert build_claude_view(RID, 11, HORSES, _notes(), '20261004') is None


def test_notes_without_sources_are_refused():
    """出典が無い見解は通さない（source_registry と同じ思想）。"""
    n = _notes(entry={'sources': []})
    assert build_claude_view(RID, 11, HORSES, n, '20260927') is None


def test_unknown_horse_numbers_are_dropped_with_a_reason():
    n = _notes(entry={'order': [9, 99, 13, 13], 'marks': {'98': '◎', '9': '○'}})
    v = build_claude_view(RID, 11, HORSES, n, '20260927')
    assert v['order'] == [9, 13]
    assert 9 in v['marks'] and 98 not in v['marks']
    assert any('#99' in d for d in v['dropped'])
    assert any('#98' in d for d in v['dropped'])
    assert any('重複' in d for d in v['dropped'])


def test_undefined_mark_is_dropped():
    n = _notes(entry={'marks': {'9': '★'}})
    v = build_claude_view(RID, 11, HORSES, n, '20260927')
    assert 9 not in v['marks']
    assert any('未定義' in d for d in v['dropped'])


def test_empty_entry_yields_none():
    n = _notes(entry={'order': [], 'marks': {}, 'horses': {}})
    assert build_claude_view(RID, 11, HORSES, n, '20260927') is None


def test_load_findings_missing_file(tmp_path):
    assert load_findings(str(tmp_path)) is None


def test_load_findings_rejects_other_schema_version(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
    with open(findings_path(str(tmp_path)), 'w', encoding='utf-8') as f:
        json.dump({'schema_version': 999, 'races': {}}, f)
    assert load_findings(str(tmp_path)) is None


def test_load_findings_reads_valid_file(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
    with open(findings_path(str(tmp_path)), 'w', encoding='utf-8') as f:
        json.dump(_notes(), f, ensure_ascii=False)
    got = load_findings(str(tmp_path))
    assert got and RID in got['races']


# ── 🔴 いちばん重要な回帰テスト ────────────────────────────────────────
# 馬番は _notes() が参照する 9/13/14 を含める（含めないと全部「居ない馬」で落ちる）
SCORED_NUMS = (9, 13, 14, 15, 1, 2, 3, 4)


def _scored():
    """to_app_json が読む最低限の形の scored（calc_all の戻り値相当）。"""
    out = []
    n = len(SCORED_NUMS)
    for i, num in enumerate(SCORED_NUMS, 1):
        out.append({
            'num': num, 'name': f'馬{num}', 'total': 8.0 - i * 0.5,
            'win_prob': 0.30 - i * 0.02, 'pn': 0.30 - i * 0.02,
            'cal_prob': 0.50 - i * 0.03, 'top2_prob': 0.4, 'top3_prob': 0.55 - i * 0.03,
            'rl_rank': i, 'cl_rank': n + 1 - i, 'fit_rank': n + 1 - i,
            'ability_margin': 1.0 - i * 0.2, 'ability_breakdown': None,
            'win_odds': 2.0 + i, 'market_prob': 0.2, 'pop_gap': 0.0,
            'rating': 1.0 - i * 0.2, 'scores': {}, 'career': {},
            'running_style': '差し', 'history': [], '_pop': i,
        })
    return out


def _race():
    return {'racecourse': '中山', 'race_num': 11, 'id': RID, 'race_name': 'テストG1',
            'distance': 1200, 'surface': '芝',
            'horses': [{'num': n} for n in SCORED_NUMS]}


def _picks(gumbel_bets):
    """買い目から「選択」だけを取り出す（モンテカルロ由来の推定値を落とす）。"""
    # 組の**並び**もモンテカルロ確率の降順なので揺れる。集合で比べる。
    return [(b.get('tag'), b.get('label'), b.get('axis'), b.get('horse'),
             b.get('amt'), frozenset(b.get('combos') or ()),
             frozenset(b.get('nums') or ()),
             frozenset(m.get('n') for m in (b.get('mates') or ())))
            for b in (gumbel_bets or [])]


def _run(base_dir):
    from src.betting.app_json import to_app_json
    # 軸1頭ベースの買い目は np.random を使うモンテカルロ（3,000回）なので、
    # 比較の2回で乱数列を揃えないと「選択が変わった」の判定が揺れる。
    import numpy as np
    np.random.seed(20260927)
    with patch('src.betting.app_json.calc_all', return_value=_scored()):
        return to_app_json([], [_race()], None,
                           datetime(2026, 9, 27, 8, 0, tzinfo=JST),
                           day_type='sunday', base_dir=base_dir, same_day=True)


def test_notes_do_not_change_probabilities_or_bets(tmp_path):
    """🔴 web見解があってもAI側の数字・買い目が1つも変わらないこと。

    ここが崩れた瞬間、撤回された5つの後付け層と同じものになる。
    """
    plain = _run(str(tmp_path))
    os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
    with open(findings_path(str(tmp_path)), 'w', encoding='utf-8') as f:
        json.dump(_notes(), f, ensure_ascii=False)
    withweb = _run(str(tmp_path))

    a = plain['races']['中山'][0]
    b = withweb['races']['中山'][0]
    assert a.get('claude_view') is None
    assert b.get('claude_view') is not None, 'web見解が載っていない（配線漏れ）'

    for key in ('honmei', 'bets', 'formation', 'rec', 'conf',
                'chaos_grade', 'value_horses', 'bet_reason', 'cmt', 'ai_pair_bets'):
        assert a.get(key) == b.get(key), f'web見解が {key} を変えている'
    assert a['horses'] == b['horses'], 'web見解が馬ごとの数字を変えている'

    # gumbel_bets（軸1頭ベース）は make_axis_bets が 3,000回のモンテカルロを
    # 回すので、推定配当・推定確率は**この層と無関係に**実行ごとに揺れる。
    # 比較するのは「どの馬を何点いくらで買うか」＝決定的な部分だけにする。
    assert _picks(a['gumbel_bets']) == _picks(b['gumbel_bets']), \
        'web見解が買い目の選択を変えている'


def test_fit_rank_reaches_the_app_json(tmp_path):
    out = _run(str(tmp_path))
    horses = out['races']['中山'][0]['horses']
    assert all(h.get('fit_rank') is not None for h in horses), \
        'fit_rank が latest.json に出ていない（配線漏れ）'

"""Gumbel温度 T の既定値と、それが実際に効いていることを固定する。

2026-07-26 の2重sigmoid修正で rating のスケールが本来の幅に戻ったが、
T=2.5 は潰れていた頃のスケールで最適化された値のまま約1ヶ月放置されていた。
2026-08-20 に 206レース・6開催日の実着順で測り直し T=1.0 に変更した
（測定値は bet_optimizer.py のコメント参照）。
"""
import json

import pytest

from src.betting import axis_bets
from src.betting.bet_optimizer import (DEFAULT_GUMBEL_RATING_T,
                                       _GUMBEL_T_CACHE,
                                       _load_gumbel_rating_temperature)


def test_default_is_one():
    """既定値は 1.0（＝温度スケーリングをしない）。"""
    assert DEFAULT_GUMBEL_RATING_T == 1.0


def test_axis_bets_does_not_hardcode_its_own_default():
    """axis_bets 側に既定値を直書きしない（片方だけ古くなる事故を防ぐ）。

    2026-08-09③で監査した「対になっている処理」と同型。
    """
    src = open(axis_bets.__file__, encoding='utf-8').read()
    assert 'DEFAULT_GUMBEL_RATING_T' in src
    assert 'else 2.5' not in src


def test_json_overrides_default(tmp_path):
    """rating_temperature.json に値があればそちらが優先される。"""
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'rating_temperature.json').write_text(
        json.dumps({'calibration': {'gumbel_rating': {'T': 1.75}}}))
    _GUMBEL_T_CACHE.pop(str(tmp_path), None)
    assert _load_gumbel_rating_temperature(str(tmp_path)) == 1.75


def test_missing_json_falls_back_to_default(tmp_path):
    _GUMBEL_T_CACHE.pop(str(tmp_path), None)
    assert _load_gumbel_rating_temperature(str(tmp_path)) == DEFAULT_GUMBEL_RATING_T


def test_production_json_leaves_gumbel_unset():
    """本番の rating_temperature.json は gumbel_rating を持たない（既定値に落ちる）。

    ここに値を書くと bet_optimizer.py のコメントにある実測の根拠と食い違うため、
    変えるときは同じ規模で測り直すこと。
    """
    with open('data/rating_temperature.json', encoding='utf-8') as f:
        d = json.load(f)
    assert 'gumbel_rating' not in d.get('calibration', {})


@pytest.mark.parametrize('T,expect_sharper', [(1.0, True), (2.5, False)])
def test_temperature_actually_changes_win_probability(T, expect_sharper):
    """T が小さいほど最上位馬の勝率が高く出る（実際に効いていることの確認）。

    P(勝利) = softmax(rating / T) と数学的に等価なので、
    T=1.0 は T=2.5 より鋭くなる。
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
    ratings = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0]
    nums = list(range(1, len(ratings) + 1))
    orders = simulate_race([r / T for r in ratings], n_sims=20000)
    p_top = calc_ticket_probabilities(orders, nums)['win'][1]
    # T=1.0 なら 0.45 前後、T=2.5 なら 0.28 前後
    assert (p_top > 0.38) is expect_sharper, p_top

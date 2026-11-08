"""軸1頭ベース買い目が Plackett-Luce の厳密式で決定的になったことを固定する。

2026-11-08 まで `make_axis_bets` は `simulate_race`（seed なしのモンテカルロ）
で券種確率と推定配当を出していた。そのため **同じ入力でも実行ごとに答えが
変わっていた**（実測: 2026-11-08 の34レース・各5回で三連複の順序が5/5回
ばらつき、推定配当は三連複で中央値 10.4% / 最大 362% ずれ、合成オッズも
1.2倍/1.3倍と揺れた）。商品の画面に出る数字が再現しないのは精度の話ではなく
記録の話として許容できない（2026-10-07 の脚質 PYTHONHASHSEED と同じ扱い）。

加えて、直前オッズ取得後に index.html が同じ買い目を組み直せるためには
JS と Python が同じ数字を出さなければならない。乱数があると原理的に突合
できないので、厳密式に替えた。JS 側との突合は
`tests/test_axis_bets_resync.mjs`（node で実行）が担う。
"""
import itertools
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betting import axis_bets as AB                                # noqa: E402
from src.betting.axis_bets import make_axis_bets                       # noqa: E402
from src.betting.bet_optimizer import (DEFAULT_GUMBEL_RATING_T,        # noqa: E402
                                       _load_gumbel_rating_temperature)
from src.betting.pl_exact import (TAKEOUT, both_in_top3_prob,          # noqa: E402
                                  exact_payouts_from_win_odds,
                                  exact_ticket_probabilities,
                                  top2_set_prob, top3_set_prob,
                                  weights_from_ratings, win_prob)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, 'index.html')
FIXTURE = os.path.join(ROOT, 'tests', 'fixtures', 'axis_bets_crosscheck.json')


def _field(n=10, seed=20261108):
    """RL順位・単勝オッズ・cal_prob・rating の整合した出走表を作る。

    cal_prob は rating の単調増加関数にしておく（本番の cal_prob は
    isotonic(sigmoid(rating)) で、Isotonic回帰は単調非減少）。
    """
    rng = np.random.default_rng(seed)
    ratings = [float(x) for x in rng.normal(0.0, 1.2, n)]
    order = sorted(range(n), key=lambda i: -ratings[i])
    rl = {i: k + 1 for k, i in enumerate(order)}
    return [{'num': i + 1, 'name': f'H{i + 1}', 'rl_rank': rl[i],
             'win_odds': round(2.0 + 3.0 * (rl[i] - 1), 1),
             'cal_prob': 1.0 / (1.0 + math.exp(-ratings[i])),
             'rating': ratings[i]}
            for i in range(n)]


class TestExactFormulasAreSelfConsistent:
    """厳密式を、MCではなく**代数の恒等式**で検算する（別経路の突合）。"""

    def test_sums_match_known_constants(self):
        rng = np.random.default_rng(11)
        ratings = list(rng.normal(0, 1.3, 11))
        w = weights_from_ratings(ratings)
        s = sum(w)
        idx = range(len(w))
        assert sum(win_prob(w, s, i) for i in idx) == pytest.approx(1.0, abs=1e-12)
        assert sum(top2_set_prob(w, s, *c) for c in itertools.combinations(idx, 2)) \
            == pytest.approx(1.0, abs=1e-12)
        assert sum(top3_set_prob(w, s, *c) for c in itertools.combinations(idx, 3)) \
            == pytest.approx(1.0, abs=1e-12)
        # 1レースで「2頭ともが3着内」の組は必ず C(3,2)=3 組成立する
        assert sum(both_in_top3_prob(w, s, *c)
                   for c in itertools.combinations(idx, 2)) \
            == pytest.approx(3.0, abs=1e-12)

    def test_matches_monte_carlo_within_sampling_error(self):
        """計測器の感度検査: 本物（MC）と突き合わせて同じ値に寄るか。"""
        from src.betting.race_simulator import (calc_ticket_probabilities,
                                                simulate_race)
        rng = np.random.default_rng(7)
        ratings = list(rng.normal(0, 1.2, 9))
        nums = list(range(1, 10))
        np.random.seed(0)
        mc = calc_ticket_probabilities(simulate_race(ratings, n_sims=200000), nums)
        ex = exact_ticket_probabilities(ratings, nums)
        for bt in ('win', 'quinella', 'wide', 'trio'):
            for key, p in ex[bt].items():
                if p < 1e-4:
                    continue
                # 200,000回のMCの標準誤差は p<=0.5 で最大 0.0011。4SEで見る。
                assert abs(mc[bt].get(key, 0.0) - p) < 0.005, (bt, key, p)


class TestDeterministic:
    def test_same_input_gives_byte_identical_bets(self):
        field = _field()
        race = {'id': 'TEST', 'race_num': 1}
        outs = []
        for _ in range(4):
            ab, _, _ = make_axis_bets(field, race)
            outs.append(json.dumps(ab, sort_keys=True, default=str))
        assert len(set(outs)) == 1, 'make_axis_bets が実行ごとに違う買い目を出す'

    def test_no_monte_carlo_left_in_make_axis_bets(self):
        src = open(os.path.join(ROOT, 'src', 'betting', 'axis_bets.py')).read()
        body = src[src.index('def make_axis_bets('):]
        # docstring に経緯として MC の名前が出るので、コードだけを見る
        head, _, rest = body.partition('"""')
        _doc, _, code = rest.partition('"""')
        body = head + code
        for banned in ('simulate_race', 'estimate_payouts_from_win_odds',
                       'np.random', 'random.'):
            assert banned not in body, (
                f'make_axis_bets に {banned} が戻っている'
                '（乱数が入るとJSと突合できず、画面の数字が実行ごとに変わる）')


class TestGumbelTemperatureStillOne:
    """index.html は JSON を読めないので AX_GUMBEL_T=1.0 を直書きしている。

    誰かが data/rating_temperature.json に温度を設定したら、JS 側が黙って
    古い値で計算し続ける。ここで 1.0 を固定して、設定した瞬間に気づける
    ようにしてある（落ちたら index.html の AX_GUMBEL_T も直す）。
    """

    def test_default_is_one(self):
        assert DEFAULT_GUMBEL_RATING_T == 1.0

    def test_repo_config_does_not_override_it(self):
        assert _load_gumbel_rating_temperature(ROOT) == 1.0

    def test_js_hardcodes_the_same_value(self):
        m = re.search(r'const AX_GUMBEL_T = ([0-9.]+);', open(INDEX).read())
        assert m, 'index.html に AX_GUMBEL_T が無い'
        assert float(m.group(1)) == _load_gumbel_rating_temperature(ROOT)


class TestJsConstantsMirrorPython:
    """index.html の定数が axis_bets.py / pl_exact.py の写しであることを固定。

    片方だけ変えると「二重実装が黙って食い違う」（2026-08-09 の事故）に
    戻るので、ソースレベルで突き合わせる。
    """

    PAIRS = [
        ('AX_ANA_ODDS_MIN', lambda: AB.ANA_ODDS_MIN),
        ('AX_ANA_ODDS_MAX', lambda: AB.ANA_ODDS_MAX),
        ('AX_WIDE_MAX_POINTS', lambda: AB.WIDE_MAX_POINTS),
        ('AX_TRIO_MAX_POINTS', lambda: AB.TRIO_MAX_POINTS),
        ('AX_TRIO_SECOND_LEG', lambda: AB.TRIO_SECOND_LEG),
        ('AX_TRIO_THIRD_LEG_RL_MAX', lambda: AB.TRIO_THIRD_LEG_RL_MAX),
        ('AX_QUINELLA_MIN_SYN_ODDS', lambda: AB.QUINELLA_MIN_SYNTHETIC_ODDS),
        ('AX_UNIT', lambda: AB.UNIT),
    ]

    def test_constants_match(self):
        html = open(INDEX).read()
        for name, get in self.PAIRS:
            m = re.search(rf'\b{name}\s*=\s*([0-9.]+)', html)
            assert m, f'index.html に {name} が無い'
            assert float(m.group(1)) == float(get()), (
                f'{name}: index.html {m.group(1)} / Python {get()}')

    def test_takeout_matches(self):
        html = open(INDEX).read()
        m = re.search(r'const AX_TAKEOUT = \{([^}]*)\}', html)
        assert m, 'index.html に AX_TAKEOUT が無い'
        js = {k: float(v) for k, v in re.findall(r'(\w+):\s*([0-9.]+)',
                                                 m.group(1))}
        assert js == {k: float(v) for k, v in TAKEOUT.items()}

    def test_takeout_matches_payout_estimator(self):
        """pl_exact.TAKEOUT は payout_estimator._TAKEOUT の写し。"""
        from src.betting.payout_estimator import _TAKEOUT
        for k, v in TAKEOUT.items():
            assert _TAKEOUT[k] == v, k


class TestCrosscheckFixtureIsGenerated:
    """フィクスチャが生成スクリプトの出力と一致する（手書き改変の検出）。"""

    def test_regenerating_reproduces_the_file(self):
        before = open(FIXTURE).read()
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, 'tests',
                                         'make_axis_bets_crosscheck.py')],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        after = open(FIXTURE).read()
        assert before == after, (
            'tests/fixtures/axis_bets_crosscheck.json が生成結果と違う。'
            '手で書き換えずに make_axis_bets_crosscheck.py を実行すること。')

    def test_fixture_exercises_the_edges(self):
        cases = json.load(open(FIXTURE))['cases']
        # 打ち切り・床・列復元がどこかで実際に効いていること
        assert any(c['axis_bets']['summary'].get('quinella_syn_odds')
                   and len(c['axis_bets']['quinella']) < sum(
                       1 for h in c['horses']
                       if AB.ANA_ODDS_MIN <= h['odds'] < AB.ANA_ODDS_MAX)
                   for c in cases), '馬連の合成オッズ打ち切りが効くケースが無い'
        assert any(b['odds'] == 1.0
                   for c in cases for bt in ('wide', 'quinella', 'trio')
                   for b in c['axis_bets'][bt]), '推定配当の床1.0に当たるケースが無い'
        assert any(not c['axis_bets']['trio'] for c in cases), \
            '三連複が組めないケースが無い'
        assert any(not c['axis_bets']['wide'] for c in cases), \
            '中穴が居ないケースが無い'


class TestSelectionRulesUnchanged:
    """買い目の選び方・閾値を1つも動かしていないことを固定する。"""

    def test_thresholds(self):
        assert AB.ANA_ODDS_MIN == 7.0
        assert AB.ANA_ODDS_MAX == 30.0
        assert AB.WIDE_MAX_POINTS == 3
        assert AB.TRIO_MAX_POINTS == 24
        assert AB.TRIO_SECOND_LEG == 2
        assert AB.TRIO_THIRD_LEG_RL_MAX == 9
        assert AB.QUINELLA_MIN_SYNTHETIC_ODDS == 3.0
        assert AB.UNIT == 100

    def test_axis_is_still_rl1(self):
        field = _field()
        ab, _, _ = make_axis_bets(field, {'id': 'TEST'})
        rl1 = next(h['num'] for h in field if h['rl_rank'] == 1)
        assert ab['summary']['axis'] == rl1

    def test_payouts_never_below_the_stake(self):
        """JRAの払戻は元本100円を下回らない（床1.0）。"""
        out = exact_payouts_from_win_odds({1: 1.1, 2: 8.0, 3: 50.0, 4: 90.0})
        for bt, d in out.items():
            for key, o in d.items():
                assert o >= 1.0, (bt, key, o)

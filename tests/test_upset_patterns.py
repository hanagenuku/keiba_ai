"""_calc_upset_patterns（アプリ成績ページ「AI上位3頭の当たり方」）の回帰テスト。

2026-08-24 の修正の前は、この集計が
  ①「AI外れ率 94.4%」＝実際は「三連複AI上位3頭ボックス1点の的中率 5.6%」を
     裏返しただけの数字を、失敗率として大きく表示していた
  ② blind_spots を N>=5 かつ upset_rate 降順で並べていたため、N=5 のセルが
     75% の確率で「100%」になり、偶然が上位を独占していた
  ③ longshot_rate（穴馬率）が winner_odds 全件NULLのため永久に 0% を返していた
という3つの問題を抱えていた。ここではその3点を固定する。

North Star #6 に従い、手打ち dict ではなく本番と同じ sqlite3.Row で検証する。
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_stats import _BLIND_SPOT_MIN_N, _binom_sf, _calc_upset_patterns

_COLS = ('race_id racecourse race_class num_horses surface distance chaos_grade '
         'rl1_num rl2_num rl3_num winner_num winner_odds second_num third_num').split()


def _rows(specs):
    """specs: [(ai3, act3, extra), ...] を本番と同じ sqlite3.Row にして返す。"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE shadow_bets (%s)' % ', '.join(_COLS))
    for i, (ai3, act3, extra) in enumerate(specs):
        rec = dict(race_id=f'r{i}', racecourse='中京', race_class='3歳以上1勝クラス',
                   num_horses=16, surface='ダート', distance=1200,
                   chaos_grade='C', winner_odds=None)
        rec.update(extra)
        rec['rl1_num'], rec['rl2_num'], rec['rl3_num'] = ai3
        rec['winner_num'], rec['second_num'], rec['third_num'] = act3
        conn.execute('INSERT INTO shadow_bets VALUES (%s)' % ','.join('?' * len(_COLS)),
                     [rec[c] for c in _COLS])
    return conn.execute('SELECT * FROM shadow_bets').fetchall()


class TestHitRateIsReportedNotOnlyMissRate:
    def test_exact_match_counts_as_hit_not_upset(self):
        # AI上位3頭 == 実際の3着内（順番違い）は「的中」であって「外れ」ではない
        r = _calc_upset_patterns(_rows([((1, 2, 3), (3, 1, 2), {})]))
        assert r['overall_hit3_rate'] == 100.0
        assert r['overall_upset_rate'] == 0.0

    def test_hit3_and_upset_are_complementary(self):
        specs = [((1, 2, 3), (1, 2, 3), {})] + [((1, 2, 3), (4, 5, 6), {})] * 3
        r = _calc_upset_patterns(_rows(specs))
        assert r['overall_hit3_rate'] == 25.0
        assert round(r['overall_hit3_rate'] + r['overall_upset_rate'], 1) == 100.0

    def test_random_baseline_is_exposed(self):
        """無作為に3頭選んだ時の的中率が併記されること。

        これが無いと 5.6% が良いのか悪いのか読めない（16頭なら 1/C(16,3)=0.18%）。
        """
        r = _calc_upset_patterns(_rows([((1, 2, 3), (1, 2, 3), {'num_horses': 16})]))
        assert r['random_hit3_rate'] == pytest.approx(100 / 560, abs=0.01)

    def test_full_miss_is_distinct_from_upset(self):
        # 1頭だけ的中 → upset ではあるが全滅ではない
        r = _calc_upset_patterns(_rows([((1, 2, 3), (1, 8, 9), {})]))
        assert r['overall_upset_rate'] == 100.0
        assert r['overall_full_miss_rate'] == 0.0


class TestBlindSpotsAreNotSmallNNoise:
    def test_small_cells_are_excluded(self):
        """N が小さいセルは、全滅100%でも blind_spots に出さない。

        修正前は N>=5 だったため、全体全滅率が12%でも N=5 のセルは
        偶然だけで頻繁に上位に来ていた。
        """
        specs = [((1, 2, 3), (7, 8, 9), {'racecourse': '福島', 'chaos_grade': 'B'})] * 5
        specs += [((1, 2, 3), (1, 2, 3), {})] * 50
        r = _calc_upset_patterns(_rows(specs))
        labels = [b['label'] for b in r['blind_spots']]
        assert not any('福島' in x for x in labels)
        assert r['blind_spot_min_n'] == _BLIND_SPOT_MIN_N >= 30

    def test_ranked_by_chance_probability_not_by_rate(self):
        """並び順が「偶然そうなる確率」の昇順であること（率の降順ではない）。"""
        specs = [((1, 2, 3), (7, 8, 9), {'surface': '芝'})] * 40      # 全滅40/40
        specs += [((1, 2, 3), (7, 8, 9), {'surface': 'ダート'})] * 35  # 全滅35/60
        specs += [((1, 2, 3), (1, 2, 3), {'surface': 'ダート'})] * 25
        r = _calc_upset_patterns(_rows(specs))
        assert r['blind_spots'], '十分なNがあるのに blind_spots が空'
        ps = [b['p_chance'] for b in r['blind_spots']]
        assert ps == sorted(ps), f'p_chance 昇順で並んでいない: {ps}'
        for b in r['blind_spots']:
            assert 0.0 <= b['p_chance'] <= 1.0

    def test_binom_sf_matches_hand_calculation(self):
        # P(X>=2 | n=3, p=0.5) = 3*0.125 + 0.125 = 0.5
        assert _binom_sf(2, 3, 0.5) == pytest.approx(0.5)
        assert _binom_sf(0, 10, 0.3) == pytest.approx(1.0)


class TestLongshotRateRemoved:
    def test_no_fabricated_longshot_metric(self):
        """穴馬率は winner_odds が全件NULLで測れないので出力しないこと。

        測っていない値を 0% として出すのは 2026-08-07⑥（ROI 150%べた書き）・
        2026-08-18②（AI xx%バッジ）と同じ「作り話をしない」違反。
        """
        r = _calc_upset_patterns(_rows([((1, 2, 3), (4, 5, 6), {})] * 40))
        assert 'longshot_rate' not in r.get('by_surface', [{}])[0]
        for b in r['blind_spots']:
            assert 'longshot_rate' not in b


class TestProductionDataShape:
    def test_runs_on_real_shadow_bets_if_present(self):
        db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'keiba.db')
        if os.path.getsize(db) < 10000:      # LFSポインタのまま = 実データなし
            pytest.skip('keiba.db が LFS ポインタ（実データ未取得）')
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM shadow_bets').fetchall()
        if len(rows) < 50:
            pytest.skip('shadow_bets のデータが少なすぎる')
        r = _calc_upset_patterns(rows)
        assert r['total_races'] > 0
        assert 0 < r['overall_hit3_rate'] < 100
        # 無作為より明確に良いこと（悪化したら表示の前提が崩れる）
        assert r['overall_hit3_rate'] > r['random_hit3_rate'] * 2
        for b in r['blind_spots']:
            assert b['total'] >= _BLIND_SPOT_MIN_N

"""Plackett-Luce レーティングの検証。

このプロジェクトで最も繰り返された事故は「学習時は本物の値・推論時は
デフォルト値」という学習/推論パリティ違反（f_post / racecourse /
finish_time / agari_rank …）。PLレーティングは読み取りコードを1本に
することで構造的に防いでいるので、そこを最優先で固定する。
"""
import pytest

from src.features.pl_rating import (
    PLRatings, FEATURE_COLS, load_ratings, save_ratings, build_from_history,
)


class TestLeakPrevention:
    """リーク防止（North Star #8）。当日の結果を当日の特徴量に使わないこと。"""

    def test_unraced_horse_has_exactly_zero(self):
        r = PLRatings()
        f = r.field_features(['新馬A', '新馬B', '新馬C'])
        for x in f:
            assert x['f_pl_rating'] == 0.0
            assert x['f_pl_rating_n'] == 0

    def test_same_day_races_see_the_same_theta(self):
        """同じ日の1R目の結果が2R目の特徴量に漏れないこと。

        本番は朝に全レースを予想するので、当日の結果は使えない。
        """
        r = PLRatings()
        r.advance_day('2026-01-01')
        r.queue_result(['A', 'B', 'C'])          # 1R目の結果を積む
        before = r.field_features(['A'])[0]      # 2R目の特徴量
        assert before['f_pl_rating'] == 0.0, '当日の結果が同日の特徴量に漏れた'
        r.advance_day('2026-01-02')              # ここで初めて反映
        after = r.field_features(['A'])[0]
        assert after['f_pl_rating'] > 0.0

    def test_winner_gains_and_loser_loses(self):
        r = PLRatings()
        for d in range(1, 6):
            r.queue_result(['勝ち馬', '中位馬', '負け馬'])
            r.advance_day(f'2026-01-{d:02d}')
        th = {n: r.get(n)[0] for n in ('勝ち馬', '中位馬', '負け馬')}
        assert th['勝ち馬'] > th['中位馬'] > th['負け馬']

    def test_beating_a_strong_field_is_worth_more(self):
        """相手の強さを考慮する — これが既存 f_rl との本質的な差。

        f_rl / f_speed_fig_avg は過去走スコアの重み付き平均なので、
        「強い相手への1着」と「弱い相手への1着」を区別できない。
        ここが区別できなくなったらこの特徴量を入れる意味が無い。
        """
        # 相手のθだけを変えて、同じ「1着」の価値を比べる
        strong = PLRatings({'相手1': 0.8, '相手2': 0.8}, {'相手1': 20, '相手2': 20})
        weak = PLRatings({'相手1': -0.8, '相手2': -0.8}, {'相手1': 20, '相手2': 20})
        for r in (strong, weak):
            r.queue_result(['挑戦者', '相手1', '相手2'])
            r.advance_day('2026-02-01')
        assert strong.get('挑戦者')[0] > weak.get('挑戦者')[0], (
            '強い相手を破っても評価が上がっていない'
            '（相手の強さを見ていない＝既存特徴量と同じになってしまう）')

    def test_losing_to_a_strong_field_is_penalised_less(self):
        """強い相手への敗戦は、弱い相手への敗戦より罰が軽いこと。"""
        strong = PLRatings({'相手1': 0.8, '相手2': 0.8}, {'相手1': 20, '相手2': 20})
        weak = PLRatings({'相手1': -0.8, '相手2': -0.8}, {'相手1': 20, '相手2': 20})
        for r in (strong, weak):
            r.queue_result(['相手1', '相手2', '敗者'])
            r.advance_day('2026-02-01')
        assert strong.get('敗者')[0] > weak.get('敗者')[0]

    def test_small_field_is_not_used(self):
        r = PLRatings()
        r.queue_result(['A', 'B'])       # 2頭は MIN_FIELD 未満
        r.advance_day('2026-01-01')
        assert r.get('A')[0] == 0.0


class TestFieldFeatures:
    def test_all_declared_columns_are_produced(self):
        r = PLRatings({'A': 0.5, 'B': 0.1}, {'A': 9, 'B': 4})
        f = r.field_features(['A', 'B', 'C'])
        for x in f:
            assert set(x) == set(FEATURE_COLS)

    def test_rank_and_relative_values(self):
        r = PLRatings({'A': 0.6, 'B': 0.0, 'C': -0.6}, {'A': 5, 'B': 5, 'C': 5})
        f = r.field_features(['A', 'B', 'C'])
        assert [x['rl_f_pl_rating_rank'] for x in f] == [1, 2, 3]
        assert f[0]['rl_f_pl_rating'] > 0 > f[2]['rl_f_pl_rating']
        assert f[1]['rl_f_pl_rating'] == pytest.approx(0.0, abs=1e-9)

    def test_identical_field_has_no_spread(self):
        r = PLRatings({'A': 0.3, 'B': 0.3}, {'A': 5, 'B': 5})
        for x in r.field_features(['A', 'B']):
            assert x['rl_f_pl_rating'] == pytest.approx(0.0, abs=1e-9)
            assert x['rl_f_pl_rating_z'] == 0.0


class TestPersistenceParity:
    """保存→ロードで値が変わらないこと（学習/推論パリティの土台）。"""

    def test_roundtrip_is_exact(self, tmp_path):
        r = PLRatings()
        for d in range(1, 8):
            r.queue_result(['A', 'B', 'C', 'D'])
            r.advance_day(f'2026-03-{d:02d}')
        save_ratings(r, str(tmp_path))
        back = load_ratings(str(tmp_path))
        assert back.last_date == r.last_date
        for n in ('A', 'B', 'C', 'D'):
            assert back.get(n) == r.get(n)
        assert back.field_features(['A', 'B']) == r.field_features(['A', 'B'])

    def test_missing_file_returns_none(self, tmp_path):
        assert load_ratings(str(tmp_path)) is None

    def test_decays_toward_zero_over_time(self, tmp_path):
        r = PLRatings()
        for d in range(1, 8):
            r.queue_result(['A', 'B', 'C'])
            r.advance_day(f'2026-03-{d:02d}')
        save_ratings(r, str(tmp_path))
        near = load_ratings(str(tmp_path), today='2026-03-10')
        far = load_ratings(str(tmp_path), today='2027-03-10')
        assert abs(far.get('A')[0]) < abs(near.get('A')[0]), '長期休養で減衰していない'


class TestEngineWiring:
    """engine が学習・推論で同じ読み取りコードを通ること。"""

    def test_calc_features_reads_the_global(self):
        import src.features.engine as eng
        from src.features.engine import calc_features_for_xgb, add_relative_features
        prev = eng._PL_RATINGS
        try:
            eng._PL_RATINGS = PLRatings({'強い馬': 0.8}, {'強い馬': 20})
            race = {'distance': 1600, 'surface': '芝', 'racecourse': '東京',
                    'race_class': '1勝クラス', 'horses': []}
            hs = [{'name': '強い馬', 'num': 1, 'history': []},
                  {'name': '無名馬', 'num': 2, 'history': []}]
            xf = [calc_features_for_xgb(h, race) for h in hs]
            assert xf[0]['f_pl_rating'] == 0.8
            assert xf[0]['f_pl_rating_n'] == 20
            assert xf[1]['f_pl_rating'] == 0.0, '未知の馬は初出走扱い(θ=0)のはず'
            add_relative_features(xf)
            assert xf[0]['rl_f_pl_rating_rank'] == 1
            assert xf[0]['rl_f_pl_rating'] > xf[1]['rl_f_pl_rating']
        finally:
            eng._PL_RATINGS = prev

    def test_defaults_when_ratings_absent(self):
        """pkl未生成の環境でも例外を出さず、学習時と同じ既定値になること。"""
        import src.features.engine as eng
        from src.features.engine import calc_features_for_xgb
        prev = eng._PL_RATINGS
        try:
            eng._PL_RATINGS = None
            race = {'distance': 1600, 'surface': '芝', 'racecourse': '東京',
                    'race_class': '1勝クラス', 'horses': []}
            xf = calc_features_for_xgb({'name': 'X', 'num': 1, 'history': []}, race)
            assert xf['f_pl_rating'] == 0.0 and xf['f_pl_rating_n'] == 0
        finally:
            eng._PL_RATINGS = prev


class TestBuildFromHistory:
    """本番と同じ経路（save_history_db で書いたDB）で構築できること。"""

    def test_builds_and_orders_horses(self, tmp_path):
        from src.utils.db import save_history_db
        db = tmp_path / 'history.db'
        for d in range(1, 6):
            save_history_db([{
                'race_id': f'2026010{d}_01_01', 'date': f'2026-01-0{d}',
                'racecourse': '東京', 'distance': 1600, 'surface': '芝',
                'num_finishers': 3,
                'finishers': [
                    {'num': 1, 'name': '一位馬', 'place': 1, 'jockey': 'J'},
                    {'num': 2, 'name': '二位馬', 'place': 2, 'jockey': 'J'},
                    {'num': 3, 'name': '三位馬', 'place': 3, 'jockey': 'J'},
                ]}], db_path=str(db))
        r, n = build_from_history(str(db))
        assert n == 5
        assert r.get('一位馬')[0] > r.get('二位馬')[0] > r.get('三位馬')[0]
        assert r.get('一位馬')[1] == 5

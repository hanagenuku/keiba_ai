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


class TestTieBreakDoesNotLeakFinishOrder:
    """🔴 2026-08-08 に実際にやらかしたリークの回帰テスト。

    検証用スクリプトが θ を作る際、SQLが `ORDER BY date, race_id, place`
    （place=着順）で、順位を `(-th).argsort().argsort()` で付けていた。
    θが同値の馬（初出走馬など）ではタイの並び順がそのまま着順になり、
    `rl_f_pl_rating_rank` が実際の着順を漏らしていた。

        θが同値の行での「順位 vs 実着順」の相関
          検証スクリプト版        +0.8678  ← 着順そのもの
          本番パイプライン版       +0.1694  （正常）

    これで AUC が 0.8037 → 0.8404、回収率が 115% と偽の結果が出た。
    プラセボ対照は「馬固有の情報」を壊す操作なので、本物のシグナルも
    リークも同じように消し、**リーク検出には使えなかった**。
    """

    def test_rank_follows_input_order_not_any_result(self):
        """θが全部同値なら、順位は渡された順序だけで決まること。

        呼び出し側（build_training_data）は ORDER BY horse_num で渡すので、
        着順とは無関係になる。ここが結果由来の順序に依存し始めたらリーク。
        """
        r = PLRatings()
        names = ['A', 'B', 'C', 'D']
        f1 = r.field_features(names)
        assert [x['rl_f_pl_rating_rank'] for x in f1] == [1, 2, 3, 4]
        # 同じ馬集合を別の順で渡すと、順位も同じ「渡された順」になる
        f2 = r.field_features(['D', 'C', 'B', 'A'])
        assert [x['rl_f_pl_rating_rank'] for x in f2] == [1, 2, 3, 4]

    def test_training_data_passes_horses_in_horse_num_order(self):
        """学習データ生成が馬番順で渡していること（着順順だとリークする）。

        build_training_data の SQL が ORDER BY place に変わったら、
        タイブレークが着順を漏らす。ソースを直接固定する。
        """
        import inspect
        import src.tools.build_training_data as btd
        src = inspect.getsource(btd)
        i = src.index('SELECT horse_name, horse_num, place')
        stmt = src[i:i + 900]
        assert 'ORDER BY horse_num' in stmt, (
            '当該レースの出走馬取得が馬番順でなくなっている。'
            'ORDER BY place にするとθ同値時のタイブレークが着順を漏らす')

    def test_theta_ties_are_common_enough_to_matter(self):
        """タイは稀な例外ではない（初出走馬は全員θ=0）。

        「タイなんて滅多に起きない」と判断してガードを外さないための記録。
        本番データでは全159,703行のうち22,207行(14%)がθ同値だった。
        """
        r = PLRatings({'経験馬': 0.5}, {'経験馬': 10})
        f = r.field_features(['経験馬', '新馬1', '新馬2', '新馬3'])
        zeros = [x for x in f if x['f_pl_rating'] == 0.0]
        assert len(zeros) == 3, '初出走馬は全員θ=0で必ずタイになる'


class TestPLFeaturesReachTheModel:
    """PL特徴量が「計算されているのにモデルに繋がっていない」状態にならないこと。

    このプロジェクトは同型の事故を何度も踏んでいる:
      f_bias    週次で算出し保存しているのに calc_features_for_xgb に渡っていない
      corner_3  corner_all に移行したのに読む側が追従せず12特徴量が死んでいた
      f_post    3段フォールバックがあるのにXGB経路は最も粗い値しか見ていなかった

    train_xgb は除外リスト以外の数値列を全部拾うので、build_training_data を
    先に走らせれば PL 特徴量は**自動的に**モデルに入る（実測: 129→134特徴量）。
    その前提が壊れる唯一の経路は「誰かが除外リストに足す」ことなので、そこを固定する。

    ⚠ 効果の実測（本番CSV・5窓 walk-forward・3シード平均）:
        本番構成（市場アンカーあり）  平均 +0.0020  悪化した窓 0/5
        市場ゼロAI                  平均 +0.0046  悪化した窓 0/5
      小さいが10窓すべてでプラスで、符号が一度も反転しない。
    """

    def test_not_in_any_exclude_list(self):
        from src.tools.train_xgb import _EXCLUDE_COLS, _MARKET_FEAT_COLS
        from src.features.pl_rating import FEATURE_COLS
        for c in FEATURE_COLS:
            assert c not in _EXCLUDE_COLS, f'{c} が _EXCLUDE_COLS に入っている'
            assert c not in _MARKET_FEAT_COLS, (
                f'{c} が _MARKET_FEAT_COLS に入っている。'
                'PLは市場情報ではないので残差学習でも除外してはいけない')

    def test_all_columns_are_produced_by_the_feature_pipeline(self):
        """calc_features_for_xgb → add_relative_features で5列すべて揃うこと。

        学習データ生成も推論もこの2関数を通るので、ここが揃っていれば
        CSVにも列が載り、train_xgb が自動で拾う。
        """
        import src.features.engine as eng
        from src.features.engine import calc_features_for_xgb, add_relative_features
        from src.features.pl_rating import FEATURE_COLS
        prev = eng._PL_RATINGS
        try:
            eng._PL_RATINGS = PLRatings({'A': 0.4, 'B': -0.2}, {'A': 8, 'B': 3})
            race = {'distance': 1600, 'surface': '芝', 'racecourse': '東京',
                    'race_class': '1勝クラス', 'horses': []}
            xf = [calc_features_for_xgb({'name': n, 'num': i, 'history': []}, race)
                  for i, n in enumerate(['A', 'B', 'C'], 1)]
            add_relative_features(xf)
            for x in xf:
                for c in FEATURE_COLS:
                    assert c in x, f'{c} が特徴量に出ていない'
        finally:
            eng._PL_RATINGS = prev

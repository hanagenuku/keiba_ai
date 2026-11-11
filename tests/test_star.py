"""☆マーク（市場1番人気を着順で上回る可能性）の固定テスト。

🔴 何を守るか
  1. 学習/推論パリティ（North Star #3）: `to_matrix`（推論・dict）と
     `matrix_from_frame`（学習・DataFrame）が**同じ値**を返す。
     崩れると片方だけ静かに 5.0 で埋まる。
  2. 閾値 `STAR_TAU = 0.60` は事前登録値。動かさない。
  3. ☆は**表示専用**: 買い目（`axis_bets`）・`rl_rank`・推奨の範囲（`article_kind`）に
     入っていない。
  4. 記事に ☆ を出すときは併記文（`STAR_NOTE`）が**必ず**付く。
  5. 併記文に禁止語（妙味・狙い目・勝てる・儲か・必ず・確実）が無く、
     「回収率は改善していません」と「本数は週によって変わります」が入っている。
  6. P1（人気1位）が1頭に定まらないレースでは☆を出さない。

根拠: `hitrate/CRITERIA_C_RERUN.md` / `hitrate/CRITERIA_STAR.md` と
      `hitrate/RESULTS_C_RERUN.md` / `hitrate/RESULTS_STAR.md`
"""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src.features import star

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestParity:
    """学習側と推論側で同じ行列になること。"""

    def test_to_matrix_equals_matrix_from_frame(self):
        cols = ['a', 'b', 'c', 'd']
        recs = [
            {'a': 1.0, 'b': 2, 'c': None, 'd': float('nan')},
            {'a': -3.5, 'b': 0, 'c': 7.25, 'd': 1},
            {'a': None, 'c': 'xx', 'd': 2.5},          # b 欠落・c が非数値
        ]
        frame = pd.DataFrame(recs)
        a = star.to_matrix(recs, cols)
        b = star.matrix_from_frame(frame, cols)
        assert a.shape == b.shape == (3, 4)
        np.testing.assert_array_equal(a, b)

    def test_missing_filled_with_star_fill(self):
        m = star.to_matrix([{'a': None}], ['a', 'zzz'])
        assert m[0, 0] == star.STAR_FILL
        assert m[0, 1] == star.STAR_FILL

    def test_matrix_from_frame_handles_absent_column(self):
        f = pd.DataFrame([{'a': 1.0}])
        m = star.matrix_from_frame(f, ['a', 'nope'])
        assert m[0, 0] == 1.0
        assert m[0, 1] == star.STAR_FILL

    def test_build_delta_is_difference_from_p1(self):
        X = np.array([[1., 2.], [4., 6.], [0., 0.]], dtype=np.float32)
        d = star.build_delta(X, 1)
        np.testing.assert_array_equal(d, X - X[1])
        np.testing.assert_array_equal(d[1], np.zeros(2, dtype=np.float32))


class TestP1:
    def test_single_favourite(self):
        assert star.p1_index([3, 1, 2]) == 1

    def test_no_favourite_returns_none(self):
        assert star.p1_index([2, 3, 4]) is None

    def test_tied_favourite_returns_none(self):
        """人気1位が2頭いたら☆を出さない（オッズ全滅時の馬番順フォールバック対策）。"""
        assert star.p1_index([1, 1, 2]) is None

    def test_none_popularity_ignored(self):
        assert star.p1_index([None, 1, None]) == 1


class TestThreshold:
    def test_tau_is_pre_registered_value(self):
        """🔴 事前登録値。結果を見てから動かさない（CRITERIA_STAR.md §1）。"""
        assert star.STAR_TAU == 0.60

    def test_fill_is_five(self):
        assert star.STAR_FILL == 5.0

    def test_pick_star_respects_threshold(self):
        assert star.pick_star([1, 2, 3], [0.59, None, 0.10]) == (None, None)
        assert star.pick_star([1, 2, 3], [0.59, None, 0.61]) == (3, 0.61)

    def test_pick_star_takes_max(self):
        num, p = star.pick_star([5, 6, 7], [0.62, 0.71, None])
        assert (num, p) == (6, 0.71)

    def test_pick_star_tiebreak_is_lower_number(self):
        """同値は馬番の小さい方（意味を持たない中立な順序）。"""
        num, _ = star.pick_star([9, 4], [0.65, 0.65])
        assert num == 4

    def test_pick_star_all_none(self):
        assert star.pick_star([1, 2], [None, None]) == (None, None)


class TestNote:
    def test_note_mentions_roi_not_improved(self):
        assert '回収率は改善していません' in star.STAR_NOTE

    def test_note_mentions_count_varies(self):
        """基準④（頻度の安定）が落ちたので、本数が変わる旨の明示が条件。"""
        assert '本数は週によって変わります' in star.STAR_NOTE

    def test_note_says_not_a_bet_basis(self):
        assert '買い目の根拠ではありません' in star.STAR_NOTE

    @pytest.mark.parametrize('bad', ['妙味', '狙い目', '勝て', '儲か', '必ず', '確実',
                                     '回収率は'])
    def test_note_has_no_forbidden_wording(self, bad):
        if bad == '回収率は':
            # 「回収率は改善していません」だけは許す。それ以外の回収率の言及は不可
            hits = [m for m in re.findall(r'回収率[^。]*。', star.STAR_NOTE)
                    if '改善していません' not in m]
            assert not hits, hits
        else:
            assert bad not in star.STAR_NOTE


class TestDisplayOnly:
    """☆が買い目・順位・推奨の範囲に入っていないこと。"""

    def _src(self, rel):
        with open(os.path.join(BASE, rel), encoding='utf-8') as f:
            return f.read()

    @pytest.mark.parametrize('rel', ['src/betting/axis_bets.py',
                                     'src/product/scope.py',
                                     'src/betting/rank_matrix_filter.py'])
    def test_star_absent_from_betting_and_scope(self, rel):
        p = os.path.join(BASE, rel)
        if not os.path.exists(p):
            pytest.skip(f'{rel} が無い')
        src = self._src(rel)
        assert 'star_prob' not in src, f'{rel} が star_prob を読んでいる'
        assert 'STAR_TAU' not in src, f'{rel} が STAR_TAU を読んでいる'

    def test_article_emits_note_with_star(self):
        """記事で ☆ の見出しを出す箇所に併記文が必ず付く。"""
        src = self._src('src/product/article.py')
        i = src.find('本命は1番人気ではありません')
        assert i > 0, '☆の見出しが article.py に無い'
        block = src[i:i + 2200]
        assert "_star.get('note')" in block, '併記文を出していない'

    def test_app_json_exposes_star_without_touching_marks(self):
        import ast
        src = self._src('src/betting/app_json.py')
        assert '_build_star(scored)' in src
        # ☆が印（◎○▲△）の割り当てに混ざっていないことを AST で確かめる
        # （文字数で切ると隣の関数の docstring を拾ってしまう）
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == '_assign_marks'), None)
        if fn is None:
            pytest.skip('_assign_marks が無い')
        body = ast.unparse(fn)
        for bad in ('star_prob', 'STAR_TAU', 'pick_star', '_build_star'):
            assert bad not in body, f'_assign_marks が {bad} を読んでいる'

    def test_marks_module_does_not_read_star(self):
        """印（◎○▲△・市場の支持ほどAIは評価していない馬）は☆を読まない。"""
        src = self._src('src/product/marks.py')
        for bad in ('star_prob', 'STAR_TAU', 'pick_star'):
            assert bad not in src, f'marks.py が {bad} を読んでいる'

    def test_engine_sets_star_prob_none_without_model(self):
        """モデルが無い環境でも star_prob キーは付き、値は None。"""
        src = self._src('src/features/engine.py')
        assert "_o['star_prob'] = None" in src


class TestArtifacts:
    """成果物があるときは本番の形で読めること（North Star #6）。"""

    def test_model_files_consistent(self):
        mp = os.path.join(BASE, 'data', 'star_model.ubj')
        cp = os.path.join(BASE, 'data', 'star_calibrator.pkl')
        fp = os.path.join(BASE, 'data', 'star_feature_cols.json')
        if not all(os.path.exists(x) for x in (mp, cp, fp)):
            pytest.skip('☆モデルの成果物がまだ無い')
        with open(fp, encoding='utf-8') as f:
            meta = json.load(f)
        assert meta['star_tau'] == star.STAR_TAU
        assert meta['star_fill'] == star.STAR_FILL
        # 本番 residual モデルの列と一致していること
        rp = os.path.join(BASE, 'data', 'xgb_feature_cols_residual.json')
        if os.path.exists(rp):
            with open(rp, encoding='utf-8') as f:
                prod = list(json.load(f)['feature_cols'])
            assert list(meta['feature_cols']) == prod, '列順が本番モデルと違う'
        bst, iso, cols = star.load_star_model(BASE)
        assert bst is not None and iso is not None
        assert list(cols) == list(meta['feature_cols'])
        # 🔴 木の本数を版の既定に任せていない（engine と同じ方針）
        assert bst.num_boosted_rounds() > 0

    def test_probs_are_calibrated_range_and_p1_is_none(self):
        bst, iso, cols = star.load_star_model(BASE)
        if bst is None:
            pytest.skip('☆モデルの成果物がまだ無い')
        feats = [{c: float(i + 1) for c in cols} for i in range(5)]
        pops = [3, 1, 2, 4, 5]
        probs = star.star_probs(feats, pops, BASE)
        assert probs[1] is None, 'P1 自身に確率が付いている'
        vals = [p for p in probs if p is not None]
        assert len(vals) == 4
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_no_star_when_p1_undetermined(self):
        bst, _, cols = star.load_star_model(BASE)
        if bst is None:
            pytest.skip('☆モデルの成果物がまだ無い')
        feats = [{c: 1.0 for c in cols} for _ in range(3)]
        assert star.star_probs(feats, [2, 3, 4], BASE) == [None, None, None]


class TestEndToEndThroughCalcAll:
    """🔴 本番パイプライン（`calc_all`）で ☆ が出ることを確かめる。

    North Star #8「新特徴量は**本番パイプラインで**測る」／
    North Star #6「モデルは実際に save_model したファイルで」。
    """

    def _write_engine_files(self, tmp_path, cols):
        import pickle as pk

        import xgboost as xgb
        (tmp_path / 'data').mkdir(exist_ok=True)
        for name in ('horse_dist_dict.pkl', 'horse_course_dict.pkl',
                     'horse_venue_dist_dict.pkl', 'post_zone_bias.pkl'):
            with open(tmp_path / 'data' / name, 'wb') as f:
                pk.dump({}, f)
        rng = np.random.RandomState(7)
        X = rng.rand(40, len(cols))
        y = rng.randint(0, 2, size=40)
        d = xgb.DMatrix(X, label=y, feature_names=cols)
        b = xgb.train({'objective': 'binary:logistic', 'max_depth': 2}, d,
                      num_boost_round=3)
        b.save_model(str(tmp_path / 'data' / 'xgb_fukusho_model.pkl'))
        (tmp_path / 'data' / 'xgb_feature_cols.json').write_text(
            json.dumps({'feature_cols': cols, 'residual': True}))

    def _write_star_files(self, tmp_path, cols, always_high):
        """☆モデルを実際に save_model / pickle で書く。

        always_high=True なら較正器が必ず 0.9 を返すので ☆ が付く。
        """
        import pickle as pk

        import xgboost as xgb
        from sklearn.isotonic import IsotonicRegression
        rng = np.random.RandomState(11)
        X = rng.rand(60, len(cols))
        y = rng.randint(0, 2, size=60)
        d = xgb.DMatrix(X, label=y, feature_names=cols)
        b = xgb.train({'objective': 'binary:logistic', 'max_depth': 2}, d,
                      num_boost_round=4)
        b.save_model(str(tmp_path / 'data' / 'star_model.ubj'))
        raw = b.predict(d)
        tgt = np.ones_like(raw) if always_high else np.zeros_like(raw)
        iso = IsotonicRegression(out_of_bounds='clip').fit(raw, tgt)
        with open(tmp_path / 'data' / 'star_calibrator.pkl', 'wb') as f:
            pk.dump(iso, f)
        (tmp_path / 'data' / 'star_feature_cols.json').write_text(
            json.dumps({'feature_cols': cols, 'star_tau': star.STAR_TAU,
                        'star_fill': star.STAR_FILL}))

    def _race(self):
        return {'date': '2026-07-26', 'racecourse': '新潟', 'distance': 1600,
                'surface': '芝', 'race_class': '3勝クラス', 'track_condition': '良',
                'horses': [
                    {'num': 1, 'horse_num': 1, 'name': 'A', 'win_odds': 2.0,
                     'running_style': '差し', 'history': []},
                    {'num': 2, 'horse_num': 2, 'name': 'B', 'win_odds': 8.0,
                     'running_style': '先行', 'history': []},
                    {'num': 3, 'horse_num': 3, 'name': 'C', 'win_odds': 15.0,
                     'running_style': '逃げ', 'history': []},
                ]}

    def _run(self, tmp_path, with_star, always_high=True):
        cols = [f'f{i}' for i in range(5)]
        self._write_engine_files(tmp_path, cols)
        if with_star:
            self._write_star_files(tmp_path, cols, always_high)
        star._DIR = None          # load_star_model のキャッシュを捨てる
        from src.features import engine
        engine.init_engine(str(tmp_path))
        assert engine._XGB_RESIDUAL is True
        return engine.calc_all(self._race())

    def test_star_prob_set_for_non_favourite_and_none_for_p1(self, tmp_path):
        out = self._run(tmp_path, with_star=True)
        assert all('star_prob' in h for h in out)
        p1 = [h for h in out if (h.get('popularity') or 0) == 1]
        assert len(p1) == 1, '人気1位が1頭に定まっていない'
        assert p1[0]['star_prob'] is None, 'P1 自身に確率が付いている'
        others = [h['star_prob'] for h in out if h is not p1[0]]
        assert all(v is not None for v in others)
        assert all(0.0 <= v <= 1.0 for v in others)

    def test_build_star_picks_one_horse_with_note(self, tmp_path):
        from src.betting.app_json import _build_star
        out = self._run(tmp_path, with_star=True)
        st = _build_star(out)
        assert st is not None and st['n'] != 0
        assert st['prob'] >= star.STAR_TAU
        assert st['tau'] == star.STAR_TAU
        assert '回収率は改善していません' in st['note']
        assert '本数は週によって変わります' in st['note']

    def test_no_star_when_probabilities_below_tau(self, tmp_path):
        from src.betting.app_json import _build_star
        out = self._run(tmp_path, with_star=True, always_high=False)
        assert _build_star(out) is None, '閾値未満なのに☆が付いた'

    def test_star_prob_none_without_model(self, tmp_path):
        """☆モデルが無くても予想は通り、star_prob は None になる。"""
        from src.betting.app_json import _build_star
        out = self._run(tmp_path, with_star=False)
        assert all(h['star_prob'] is None for h in out)
        assert _build_star(out) is None

    def test_star_does_not_change_ranking(self, tmp_path):
        """☆を出しても rl_rank / ai_rank / total は変わらない（表示専用）。"""
        cols = [f'f{i}' for i in range(5)]
        self._write_engine_files(tmp_path, cols)
        star._DIR = None
        from src.features import engine
        engine.init_engine(str(tmp_path))
        before = engine.calc_all(self._race())
        self._write_star_files(tmp_path, cols, True)
        star._DIR = None
        engine.init_engine(str(tmp_path))
        after = engine.calc_all(self._race())
        key = lambda xs: [(h['num'], h.get('rl_rank'), h.get('ai_rank'),  # noqa: E731
                           round(h['total'], 6)) for h in xs]
        assert key(before) == key(after), '☆を入れたら順位・スコアが動いた'


class TestHonmeiMovesToStar:
    """🔴 2026-11-11 ユーザー決定「◎を動かす」の固定。

    守ること:
      1. ☆が出たレースでは ◎ がその馬に移る（○▲△ は rl_rank 順で詰まる）
      2. **買い目の軸も同じ馬に移る**（◎だけ動かす形は 2026-11-08 に実害を出した）
      3. `conf`（レース信頼度）は動かさない＝ RL1 の3着内確率のまま
      4. 有料／無料の振り分け（`article_kind`）は ☆ を読まない
      5. クライアント側（直前オッズ）は ☆ があるレースの ◎ と軸を動かさない
    """

    def _src(self, rel):
        with open(os.path.join(BASE, rel), encoding='utf-8') as f:
            return f.read()

    def test_mark_moves_to_star_and_rest_shift(self):
        from src.product import marks
        hs = [{'num': 1, 'rl_rank': 1}, {'num': 2, 'rl_rank': 2},
              {'num': 7, 'rl_rank': 4}, {'num': 9, 'rl_rank': 3}]
        got = {h['num']: h['product_mark']
               for h in marks.assign_marks(hs, star_num=7)}
        assert got[7] == '◎', '◎が☆の馬に移っていない'
        assert got[1] == '○' and got[2] == '▲' and got[9] == '△'

    def test_marks_unchanged_without_star(self):
        from src.product import marks
        hs = [{'num': 1, 'rl_rank': 1}, {'num': 2, 'rl_rank': 2},
              {'num': 9, 'rl_rank': 3}]
        got = {h['num']: h['product_mark'] for h in marks.assign_marks(hs)}
        assert got == {1: '◎', 2: '○', 9: '▲'}

    def test_axis_horse_follows_the_mark(self):
        from src.product import marks
        hs = [{'num': 1, 'rl_rank': 1}, {'num': 7, 'rl_rank': 3}]
        m = marks.assign_marks(hs, star_num=7)
        assert marks.axis_horse(m)['num'] == 7
        # 印を付けていないリストは従来どおり rl_rank 1位
        assert marks.axis_horse(hs)['num'] == 1

    def test_unknown_star_num_falls_back_to_rl1(self):
        from src.product import marks
        hs = [{'num': 1, 'rl_rank': 1}, {'num': 2, 'rl_rank': 2}]
        got = {h['num']: h['product_mark']
               for h in marks.assign_marks(hs, star_num=99)}
        assert got[1] == '◎', '☆の馬が居ないのに ◎ が消えた'

    def test_bet_axis_moves_with_the_mark(self):
        """🔴 買い目の軸も ☆ に動くこと（◎だけ動かすのは禁止）。"""
        from src.betting.axis_bets import select_axis
        scored = [{'num': 1, 'horse_num': 1, 'rl_rank': 1},
                  {'num': 7, 'horse_num': 7, 'rl_rank': 3}]
        assert select_axis(scored)['num'] == 1
        assert select_axis(scored, star_num=7)['num'] == 7
        # 居ない馬番なら RL1 に落ちる
        assert select_axis(scored, star_num=99)['num'] == 1

    def test_app_json_passes_star_to_the_axis(self):
        src = self._src('src/betting/app_json.py')
        assert src.count('star_num=(_star or {}).get(\'n\')') == 1
        assert src.count('star_num=(_star2 or {}).get(\'n\')') == 1

    def test_conf_is_not_recomputed_for_the_new_honmei(self):
        """🔴 conf は RL1 の3着内確率のまま（有料/無料の振り分けを動かさない）。"""
        import ast
        src = self._src('src/betting/app_json.py')
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == 'race_confidence')
        body = ast.unparse(fn)
        assert "rl_rank') == 1" in body, 'race_confidence が RL1 を見ていない'
        for bad in ('star', 'honmei_h'):
            assert bad not in body, f'race_confidence が {bad} を読んでいる'

    def test_article_kind_does_not_read_star(self):
        src = self._src('src/product/scope.py')
        for bad in ('star_prob', 'STAR_TAU', "get('star')", 'from_star'):
            assert bad not in src, f'scope.py が {bad} を読んでいる'

    def test_client_does_not_move_honmei_when_star_exists(self):
        html = self._src('index.html')
        assert 'if (race.honmei && !(race.star && race.star.n))' in html, \
            '☆があるレースで ◎ を RL1 に戻してしまう'
        assert 'buildAxisBetsJS(hs, probs, _axExactPayouts(winOdds),' in html, \
            'クライアントの買い目が ☆ を軸にしていない'

    def test_conf_badge_label_is_not_a_lie(self):
        """◎が☆に移るので「軸3着内」は嘘になる。ラベルを直してあること。"""
        html = self._src('index.html')
        assert '軸3着内 ' not in html
        assert '当てやすさ ' in html

    def test_article_states_that_honmei_is_the_star(self):
        src = self._src('src/product/article.py')
        assert '◎ と買い目の軸はこの馬にしています' in src

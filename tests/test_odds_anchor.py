"""D-2（base_margin を人気順位→実オッズ）の評価器の回帰テスト。

この評価は「再学習せずアンカーだけ差し替える」ことに全部乗っている。
前提が崩れたら数字が丸ごと嘘になるので、前提そのものを固定する。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_odds_anchor import (MAX_VALID_ODDS_BOOK_SUM, _apply_odds_drift,
                                      _odds_to_base_margin, require_drift)


class TestOddsToBaseMargin:
    """人気順位版と同じ「レース内で合計1に正規化した勝率のlogit」を返すこと。"""

    def test_normalizes_within_race(self):
        odds = [2.0, 4.0, 4.0]
        rid = ['r1'] * 3
        bm = _odds_to_base_margin(odds, rid)
        p = 1 / (1 + np.exp(-bm))
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        assert p[0] == pytest.approx(0.5)

    def test_takeout_is_removed(self):
        """控除率が違う盤でも同じ base_margin になること。

        Σ(1/オッズ) で割るので控除率は落ちる。ここが効いていないと
        控除率の違いがそのままアンカーの水準差になる。
        """
        a = _odds_to_base_margin([2.0, 4.0, 4.0], ['r'] * 3)     # Σ=1.00
        b = _odds_to_base_margin([2.5, 5.0, 5.0], ['r'] * 3)     # Σ=0.80
        assert np.allclose(a, b)

    def test_races_are_independent(self):
        """レースをまたいで正規化してはいけない。"""
        odds = [2.0, 4.0, 4.0, 1.5, 3.0, 6.0]
        rid = ['r1'] * 3 + ['r2'] * 3
        bm = _odds_to_base_margin(odds, rid)
        p = 1 / (1 + np.exp(-bm))
        assert p[:3].sum() == pytest.approx(1.0, abs=1e-9)
        assert p[3:].sum() == pytest.approx(1.0, abs=1e-9)

    def test_shorter_odds_get_a_higher_margin(self):
        bm = _odds_to_base_margin([1.5, 10.0, 30.0], ['r'] * 3)
        assert bm[0] > bm[1] > bm[2]


class TestBrokenOddsBookThreshold:
    """モデルを使わずに盤の壊れを見る閾値（2026-08-20 C-2 の実測に基づく）。"""

    def test_threshold_sits_in_the_measured_gap(self):
        # 正常な盤の最大 1.577 と 壊れた盤の最小 1.714 の間
        assert 1.577 < MAX_VALID_ODDS_BOOK_SUM < 1.714

    def test_a_normal_book_passes(self):
        odds = np.array([2.5, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0])
        assert (1.0 / odds).sum() <= MAX_VALID_ODDS_BOOK_SUM

    def test_the_2026_08_15_broken_book_is_caught(self):
        """全16頭が2.0〜2.2倍だった実際の壊れ方。1頭ずつ見ても異常に見えない。"""
        odds = np.full(16, 2.1)
        assert (1.0 / odds).sum() > MAX_VALID_ODDS_BOOK_SUM


class TestMustNotJudgeOnConfirmedOdds:
    """🔴 確定オッズだけで判定させない（2026-07-31・08-21 に2回踏んだ罠）。"""

    def test_missing_odds_drift_aborts(self):
        with pytest.raises(SystemExit):
            require_drift({'_all': np.zeros(10)}, None)

    def test_missing_popularity_drift_aborts(self):
        with pytest.raises(SystemExit):
            require_drift(None, {'_all': np.zeros(10)})

    def test_both_present_is_fine(self):
        require_drift({'_all': np.zeros(10)}, {'_all': np.zeros(10)})


class TestOddsDrift:
    def test_zero_drift_leaves_odds_unchanged(self):
        odds = np.array([2.0, 5.0, 11.0])
        out = _apply_odds_drift(odds, [1, 2, 3], {'_all': np.zeros(50)}, seed=0)
        assert np.allclose(out, odds)

    def test_drift_is_multiplicative_not_additive(self):
        """log比を注入するので、倍率で動く（人気薄ほど絶対値が大きく動く）。"""
        odds = np.array([2.0, 20.0])
        d = {'_all': np.full(50, np.log(2.0))}
        out = _apply_odds_drift(odds, [1, 2], d, seed=0)
        assert out[0] == pytest.approx(4.0)
        assert out[1] == pytest.approx(40.0)

    def test_same_seed_same_result(self):
        d = {'_all': np.linspace(-1, 1, 50)}
        a = _apply_odds_drift([2.0, 5.0], [1, 2], d, seed=7)
        b = _apply_odds_drift([2.0, 5.0], [1, 2], d, seed=7)
        assert np.allclose(a, b)


class TestAnchorSwapIsValidWithoutRetraining:
    """ability_margin が base_margin に依存しないという前提を本番モデルで固定する。

    これが成り立たなければ「再学習せずアンカーだけ差し替える」が丸ごと無効。
    North Star #8（別経路で作って突合する）。
    """

    def test_ability_margin_is_independent_of_the_anchor(self):
        import json
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mp = os.path.join(base, 'data', 'xgb_fukusho_model.pkl')
        fp = os.path.join(base, 'data', 'xgb_feature_cols.json')
        if not (os.path.exists(mp) and os.path.exists(fp)):
            pytest.skip('本番モデルが無い環境')
        with open(fp) as f:
            meta = json.load(f)
        if not meta.get('residual'):
            pytest.skip('残差学習モデルでない')

        import xgboost as xgb
        from src.tools.train_xgb import _load_xgb_model_any
        booster = _load_xgb_model_any(mp, is_residual=True)
        cols = meta['feature_cols']

        rng = np.random.default_rng(0)
        X = rng.normal(5, 2, (300, len(cols)))

        def margin(bm):
            d = xgb.DMatrix(X, feature_names=cols)
            d.set_base_margin(bm)
            return booster.predict(d, output_margin=True)

        ability = margin(np.zeros(300))
        bm = rng.uniform(-6, 6, 300)          # 実オッズ側は幅が広くなりうる
        assert np.abs((margin(bm) - bm) - ability).max() < 1e-4


class TestEndToEnd:
    """配線を通しで動かす。数字は合成なので無意味だが、**経路**を固定する。

    ランナー上でしか本番データが揃わないため、ここで落とせるものは
    ここで落とす（特徴量の再生成に20分かけてから落ちるのは高くつく）。
    """

    @staticmethod
    def _fixture(tmp, n_races=60, horses=10, with_odds=True, break_one_book=False,
                 missing_pop=False):
        import json
        import shutil
        import sqlite3

        import pandas as pd

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_meta = os.path.join(base, 'data', 'xgb_feature_cols.json')
        real_model = os.path.join(base, 'data', 'xgb_fukusho_model.pkl')
        if not (os.path.exists(real_meta) and os.path.exists(real_model)):
            pytest.skip('本番モデルが無い環境')

        d = tmp / 'data'
        (d / 'private').mkdir(parents=True)
        shutil.copy(real_meta, d / 'xgb_feature_cols.json')
        shutil.copy(real_model, d / 'xgb_fukusho_model.pkl')
        cols = json.load(open(real_meta))['feature_cols']

        rng = np.random.default_rng(0)
        rows = []
        for r in range(n_races):
            # race_id は必ず一意にする。重複すると盤が合体して
            # Σ(1/オッズ) が頭数ぶん膨らみ、盤の妥当性検査に弾かれる
            mo, dd = (r % 12) + 1, (r % 28) + 1
            rid = f'2025{mo:02d}{dd:02d}_0{r % 5 + 1}_{r:04d}'
            day = f'2025-{mo:02d}-{dd:02d}'
            # 人気1..N と、それに整合する実オッズ（Σ(1/odds)≒1.25）
            inv = np.sort(rng.dirichlet(np.ones(horses)))[::-1] * 1.25
            for i in range(horses):
                # 本番CSVでは popularity=99 のセンチネル等で約15%が NaN になる
                p = np.nan if (missing_pop and i == 0) else i + 1
                rows.append({'race_id': rid, 'date': day, 'horse_num': i + 1,
                             'f_popularity': p,
                             'is_fukusho': int(rng.random() < 0.3),
                             '_odds': 1.0 / inv[i],
                             **{c: rng.normal(5, 2) for c in cols}})
        df = pd.DataFrame(rows)
        csv = tmp / 'feat.csv'
        df.drop(columns=['_odds']).to_csv(csv, index=False)

        # netkeiba の実オッズ
        conn = sqlite3.connect(d / 'private' / 'netkeiba.db')
        conn.execute('CREATE TABLE netkeiba_odds (race_id TEXT, horse_num INT, '
                     'win_odds REAL)')
        conn.execute('CREATE TABLE fetch_log (race_id TEXT)')
        sub = df if with_odds else df[df['race_id'] != df['race_id'].iloc[0]]
        for _, x in sub.iterrows():
            o = x['_odds']
            if break_one_book and x['race_id'] == df['race_id'].iloc[0]:
                o = 2.1                      # 2026-08-15 の壊れ方（全馬同じ値）
            conn.execute('INSERT INTO netkeiba_odds VALUES (?,?,?)',
                         (x['race_id'], int(x['horse_num']), float(o)))
        conn.commit()
        conn.close()

        # ドリフト用の keiba.db（朝オッズと直前オッズの対を十分な数）
        conn = sqlite3.connect(d / 'keiba.db')
        conn.execute('CREATE TABLE race_predictions (race_id TEXT, horse_num INT, '
                     'tansho_odds REAL)')
        conn.execute('CREATE TABLE odds_snapshots (race_id TEXT, horse_num INT, '
                     'tansho REAL, captured_at TEXT)')
        for r in range(80):
            for i in range(10):
                late = float(2 + i * 3)
                morn = late * float(np.exp(rng.normal(0, 0.4)))
                conn.execute('INSERT INTO race_predictions VALUES (?,?,?)',
                             (f'd{r}', i + 1, morn))
                conn.execute('INSERT INTO odds_snapshots VALUES (?,?,?,?)',
                             (f'd{r}', i + 1, late, 'now'))
        conn.commit()
        conn.close()
        return str(csv)

    def _run(self, tmp, csv, monkeypatch, capsys):
        from scripts.eval_odds_anchor import main
        monkeypatch.setattr(sys, 'argv', [
            'eval', '--base-dir', str(tmp), '--features', csv,
            '--start', '2025-01-01', '--end', '2025-12-31', '--windows', '2'])
        main()
        return capsys.readouterr().out

    def test_runs_and_reports_all_three_anchors(self, tmp_path, monkeypatch, capsys):
        csv = self._fixture(tmp_path)
        out = self._run(tmp_path, csv, monkeypatch, capsys)
        for a in ('A  人気順位', 'A2 人気順位', 'B  実オッズ', 'C  実オッズ'):
            assert a in out, f'{a} が出ていない'
        # 2つの問いを混ぜずに出すこと（B-A は どちらの答えでもない）
        assert 'Q1 朝に生成する予想' in out and 'Q2 直前オッズを押した後' in out
        # 検算（前提が崩れていないこと）が必ず出ること
        assert 'ability_margin が base_margin に依存しない' in out

    def test_partial_book_races_are_dropped(self, tmp_path, monkeypatch, capsys):
        """一部の馬にしかオッズが無い盤は正規化が狂うので使わない。"""
        csv = self._fixture(tmp_path, with_odds=False)
        out = self._run(tmp_path, csv, monkeypatch, capsys)
        assert 'オッズ欠けの盤 1レース' in out

    def test_broken_book_is_dropped(self, tmp_path, monkeypatch, capsys):
        csv = self._fixture(tmp_path, break_one_book=True)
        out = self._run(tmp_path, csv, monkeypatch, capsys)
        assert 'Σ(1/オッズ)が異常な盤 1レース' in out

    def test_aborts_when_features_are_missing(self, tmp_path, monkeypatch, capsys):
        import pandas as pd
        csv = self._fixture(tmp_path)
        df = pd.read_csv(csv)
        import json
        cols = json.load(open(tmp_path / 'data' / 'xgb_feature_cols.json'))['feature_cols']
        df.drop(columns=[cols[0]]).to_csv(csv, index=False)
        with pytest.raises(SystemExit):
            self._run(tmp_path, csv, monkeypatch, capsys)

    def test_says_so_loudly_when_no_window_qualifies(self, tmp_path, monkeypatch,
                                                     capsys):
        """窓が1つも成立しない時に黙って終わらないこと。

        黙ると「測ったつもりで何も測っていない」状態になる。
        2026-08-25 に netkeiba 取得が「進んでいるように見えて0件」だったのと同型。
        """
        csv = self._fixture(tmp_path, n_races=20)      # 200行 → どの窓も500行未満
        out = self._run(tmp_path, csv, monkeypatch, capsys)
        assert '判定不能' in out
        assert '数字を採用しないこと' in out
        assert '① 全窓でプラス' not in out          # 空の判定を出さない

    def test_reports_criteria_when_windows_qualify(self, tmp_path, monkeypatch,
                                                   capsys):
        csv = self._fixture(tmp_path, n_races=300)     # 3,000行
        out = self._run(tmp_path, csv, monkeypatch, capsys)
        assert '① 全窓でプラス' in out and '② 平均 +0.01 以上' in out
        assert '判定不能' not in out
        # 上限(B)を根拠にしないという注意が必ず出ること
        assert 'Q1の根拠にしないこと' in out


class TestEvalWorkflowGuard:
    """評価ジョブの「リポジトリを汚していない」検査が誤爆しないこと。

    build_training_data は特徴量CSVだけでなく member_level_cache.pkl も毎回
    書き直す。単純に「git status が空」を求めると、特徴量の再生成に20分かけた
    あとで必ず落ちる。再生成される物だけを許す形になっていることを固定する。
    """

    @staticmethod
    def _guard():
        import yaml
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         '.github', 'workflows', 'eval-odds-anchor.yml')
        with open(p, encoding='utf-8') as f:
            steps = yaml.safe_load(f)['jobs']['eval']['steps']
        return next(s['run'] for s in steps if '想定外' in s.get('name', ''))

    def test_allows_what_build_training_data_regenerates(self):
        body = self._guard()
        for f in ('data/horse_features.csv', 'data/member_level_cache.pkl'):
            assert f in body, f'{f} は毎回書き直されるので許可が要る'

    def test_still_fails_on_anything_else(self):
        body = self._guard()
        assert 'exit 1' in body and '想定外のファイルが変更された' in body

    def test_build_training_data_writes_the_cache_we_allow(self):
        """許可リストの根拠が実装と一致していること（片方だけ変わるのを防ぐ）。"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(base, 'src', 'tools', 'build_training_data.py'),
                   encoding='utf-8').read()
        assert "'member_level_cache.pkl'" in src
        assert 'member_level_cache.pkl' in self._guard()


class TestMissingPopularity:
    """🔴 2026-08-25 の実行で落ちた件の回帰テスト。

    検定窓の約15%は f_popularity が NaN（popularity=99 のセンチネル等）。
    `_apply_popularity_drift` が `int(nan)` で ValueError を投げて評価が止まった。

    🔑 落とすのではなく**本番の学習と同じフィールド中央値で埋める**
       （train_xgb.py:268）。落とすと母集団が変わり、アンカーの比較が
       apples-to-apples でなくなる。
    """

    def test_fills_the_same_way_production_training_does(self):
        """補完方法が train_xgb と一致していること（片方だけ変わるのを防ぐ）。"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train = open(os.path.join(base, 'src', 'tools', 'train_xgb.py'),
                     encoding='utf-8').read()
        ev = open(os.path.join(base, 'scripts', 'eval_odds_anchor.py'),
                  encoding='utf-8').read()
        assert "fillna(train_df['_n_horses'] / 2)" in train
        assert "fillna(df['_n'] / 2)" in ev, '本番と違う埋め方をしている'

    def test_runs_with_missing_popularity(self, tmp_path, monkeypatch, capsys):
        csv = TestEndToEnd._fixture(tmp_path, n_races=300, missing_pop=True)
        out = TestEndToEnd()._run(tmp_path, csv, monkeypatch, capsys)
        assert '人気の欠損' in out
        assert '① 全窓でプラス' in out, '欠損があると評価が止まってしまう'


class TestVerificationCatchesNaN:
    """🔴 検算が NaN を素通りしていた（2026-08-25）。

    `|(raw-bm)-ability| 最大 nan` と表示されたのに検算は通っていた。
    `nan > 1e-4` は False なので、素の不等号では NaN を捕まえられない。
    「前提が壊れたら止める」ための検査が、壊れ方によっては止まらなかった。
    """

    def test_plain_comparison_would_let_nan_through(self):
        """なぜ書き方を変えたのかを固定する。"""
        r = np.array([np.nan, 1e-9])
        assert (r.max() > 1e-4) is np.False_ or not (r.max() > 1e-4)
        assert not (r.max() <= 1e-4)          # こちらは捕まえる

    def test_source_uses_the_nan_safe_form(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(base, 'scripts', 'eval_odds_anchor.py'),
                   encoding='utf-8').read()
        assert 'not (resid.max() <= 1e-4)' in src, \
            'NaN を素通りさせる書き方に戻っている'

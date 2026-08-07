"""過去走レコードの None フィールドで予想が落ちないことの回帰テスト。

## 背景（2026-08-07 の金曜予想が本番で落ちた事故）

2026-08-03② で「上がり順位が未記録なら None を明示する」設計に変更した
（学習側 `_get_history_before` / 推論側 `get_history_from_db` の両方）。
ところが消費側は `r.get('agari_rank', finishers)` と書かれており、
**`.get()` の既定値はキーが存在しない時にしか効かない**。
キーはあって値が None、という新しい状態が素通りして

    TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'

で `calc_all()` 全体が停止した。出走表もオッズも正常に取れていたのに、
その回の予想が丸ごと失われている。

North Star #6 に従い、手打ちdictではなく **本番と同じ経路**
（`save_history_db()` で書き込み → `get_history_from_db()` で読み出し）
で得たレコードを使って検証する。
"""
import pytest

from src.utils.db import save_history_db
from src.scraper.jra_scraper import get_history_from_db
from src.features.engine import (
    calc_race_content_score, f_rl, f_recent, f_maturity, calc_features_for_xgb,
)


def _write_history(db_path, *, agari_rank, place=3):
    """agari_rank が未記録の過去走を1件、本番と同じ書き込み経路で作る。"""
    save_history_db([{
        'race_id': '20260705_01_05', 'date': '2026-07-05', 'racecourse': '新潟',
        'distance': 1600, 'surface': '芝', 'track_condition': '良',
        'race_class': '1勝クラス', 'num_finishers': 14,
        'finishers': [{
            'num': 7, 'name': 'テスト馬', 'place': place,
            'agari_rank': agari_rank, 'agari3f': 35.2,
            'jockey': '武豊', 'trainer': '藤沢', 'sex': '牡', 'age': 4,
            'weight_load': 55.0, 'corner_all': '3-3-2-1',
            'finish_time': 95.1, 'popularity': 2, 'body_weight': 480,
        }],
    }], db_path=str(db_path))


@pytest.fixture
def hist_with_unknown_agari(tmp_path):
    """上がり順位が取れなかった過去走（本番で頻出する状態）。"""
    db = tmp_path / 'history.db'
    _write_history(db, agari_rank=None)
    hist = get_history_from_db('テスト馬', str(db))
    assert hist, '過去走が読めていない（テストの前提が壊れている）'
    assert hist[0]['agari_rank'] is None, (
        'このテストは agari_rank=None を前提にしている。'
        '取得側の仕様が変わったならテストも見直すこと')
    return hist


class TestNoneAgariRankDoesNotCrash:
    """2026-08-07 の本番停止そのものの再現。修正前は TypeError で失敗する。"""

    def test_calc_race_content_score(self, hist_with_unknown_agari):
        score = calc_race_content_score(hist_with_unknown_agari[0])
        assert 0 <= score <= 10

    def test_f_rl(self, hist_with_unknown_agari):
        h = {'history': hist_with_unknown_agari}
        assert 0 <= f_rl(h, {'distance': 1600, 'surface': '芝'}) <= 10

    def test_f_recent(self, hist_with_unknown_agari):
        h = {'history': hist_with_unknown_agari}
        assert 0 <= f_recent(h, {'distance': 1600, 'surface': '芝'}) <= 10

    def test_unknown_agari_scores_no_higher_than_worst_known(self, tmp_path):
        """不明は「最下位相当」に寄せる（元の既定値の意図）。

        None が最速(1位)として扱われると、データが無い馬ほど高評価という
        逆転が起きる。方向性まで固定しておく。
        """
        db_worst = tmp_path / 'w.db'
        _write_history(db_worst, agari_rank=14)      # 14頭中14位＝最下位
        worst = get_history_from_db('テスト馬', str(db_worst))[0]
        db_best = tmp_path / 'b.db'
        _write_history(db_best, agari_rank=1)        # 最速
        best = get_history_from_db('テスト馬', str(db_best))[0]
        db_none = tmp_path / 'n.db'
        _write_history(db_none, agari_rank=None)
        unknown = get_history_from_db('テスト馬', str(db_none))[0]

        s_worst = calc_race_content_score(worst)
        s_best = calc_race_content_score(best)
        s_unknown = calc_race_content_score(unknown)
        assert s_best > s_worst, '最速が最下位より高評価になっていない'
        assert s_unknown == pytest.approx(s_worst), (
            f'不明({s_unknown})が最下位({s_worst})相当になっていない')


class TestSentinelAgariRank:
    """save_history_db は取得失敗時に 99 を入れる。負の寄与にしないこと。

    2026-08-06 に horse_type.py で同型の「99 が素通りして -5.5 を返す」
    バグを修正した。engine.py 側にも同じ構造がある。
    """

    def test_sentinel_99_treated_as_unknown(self, tmp_path):
        db = tmp_path / 's.db'
        _write_history(db, agari_rank=99)
        rec = get_history_from_db('テスト馬', str(db))[0]
        score = calc_race_content_score(rec)
        assert 0 <= score <= 10, f'センチネル99で範囲外のスコア: {score}'


class TestNonePlaceDoesNotCrash:
    """place は推論側SQLで COALESCE されておらず None になりうる。"""

    def test_xgb_features_and_rule_scores(self, tmp_path):
        db = tmp_path / 'p.db'
        _write_history(db, agari_rank=5, place=None)
        hist = get_history_from_db('テスト馬', str(db))
        h = {'name': 'テスト馬', 'history': hist, 'num': 7,
             'jockey_rate': 0.15, 'trainer_rate': 0.12}
        race = {'distance': 1600, 'surface': '芝', 'racecourse': '新潟',
                'race_class': '1勝クラス', 'horses': [h]}
        assert 0 <= f_maturity(h, race) <= 10
        feats = calc_features_for_xgb(h, race)
        assert feats and 'f_recent' in feats

"""複勝の的中判定を「払戻表に行があるか」で行うことを固定する。

🔴 2026-10-20 の監査で見つかった実データの壊れ方:

    JRAの複勝は **7頭以下のレースでは2着までしか発売されない**。
    ところが決済側は `3着以内なら的中` としていたため、3着に入った馬を
    **「的中・払戻0円」**として記録していた。

      bets          : 5件（例 20260927_09_04 複勝#7）
      shadow_bets   : 4件（例 20260809_07_06 7頭立て・軸#1が3着）

    影響は2つ。①的中率が過大に出る ②回収率の分母に外れが混ざる。

🔑 頭数の規則をコードに書かない（「7頭以下なら2着まで」と書くと、同じ条件で
   発売されているワイドまで巻き込んで誤判定する）。**払戻表にその馬の行が
   あるかどうか**が唯一の正しい判定で、これなら頭数規則を一切書かずに済む。
   2026-10-02 に prediction_archive 側で確定させた規則と同じ。
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betting.shadow import record_all_shadow_bets  # noqa: E402
from src.utils.db import check_and_update_bets, init_db, save_race_predictions  # noqa: E402


def _seven_horse_race():
    """7頭立てで、軸が3着。複勝の払戻は1〜2着ぶんしか無い（実データと同じ形）。"""
    return {
        'id': '20261018_05_09', 'race_id': '20261018_05_09',
        'date': '2026-10-18', 'racecourse': '東京', 'race_num': 9,
        'race_name': 'テストS', 'surface': '芝', 'distance': 1600,
        'finishers': [
            {'num': 4, 'place': 1, 'name': 'ウマ4', 'win_odds': 2.0},
            {'num': 5, 'place': 2, 'name': 'ウマ5', 'win_odds': 3.0},
            {'num': 3, 'place': 3, 'name': 'ウマ3', 'win_odds': 4.0},
            {'num': 1, 'place': 4, 'name': 'ウマ1', 'win_odds': 9.0},
            {'num': 2, 'place': 5, 'name': 'ウマ2', 'win_odds': 11.0},
            {'num': 6, 'place': 6, 'name': 'ウマ6', 'win_odds': 21.0},
            {'num': 7, 'place': 7, 'name': 'ウマ7', 'win_odds': 31.0},
        ],
        'dividends': {
            'tansho':  {'num': 4, 'payout': 200},
            # 🔴 7頭立てなので複勝は2着まで。3着の#3に行が無い
            'fukusho': [{'num': 4, 'payout': 130}, {'num': 5, 'payout': 160}],
            'umaren':  {'nums': [4, 5], 'payout': 800},
            'wide':    [{'nums': [4, 5], 'payout': 300},
                        {'nums': [3, 4], 'payout': 250},
                        {'nums': [3, 5], 'payout': 280}],
            'sanrenpuku': {'nums': [3, 4, 5], 'payout': 1200},
        },
    }


class TestBetsSettlement:
    def _db(self, tmp_path):
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        return os.path.join(tmp_path, 'data', 'keiba.db')

    def _bet(self, path, bet_type, h1, h2=0, amount=500):
        conn = sqlite3.connect(path)
        conn.execute(
            'INSERT INTO bets (date, race_id, bet_type, horse_num, horse_num2, '
            'amount, is_hit, payout) VALUES (?,?,?,?,?,?,-1,0)',
            ('2026-10-18', '20261018_05_09', bet_type, h1, h2, amount))
        conn.commit()
        conn.close()

    def _row(self, path, bet_type):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        r = conn.execute('SELECT is_hit, payout FROM bets WHERE bet_type=?',
                         (bet_type,)).fetchone()
        conn.close()
        return r

    def test_third_place_in_a_seven_horse_race_is_a_miss(self, tmp_path):
        """3着でも複勝の払戻行が無ければ外れ（修正前は「的中・0円」）。"""
        path = self._db(tmp_path)
        self._bet(path, '複勝', 3)
        check_and_update_bets([_seven_horse_race()], db_path=path)
        r = self._row(path, '複勝')
        assert r['is_hit'] == 0 and r['payout'] == 0

    def test_second_place_still_pays(self, tmp_path):
        """同じレースでも2着なら従来どおり的中して払い戻される。"""
        path = self._db(tmp_path)
        self._bet(path, '複勝', 5, amount=500)
        check_and_update_bets([_seven_horse_race()], db_path=path)
        r = self._row(path, '複勝')
        assert r['is_hit'] == 1 and r['payout'] == 800   # 500円 × 160/100

    def test_wide_with_the_third_horse_still_hits(self, tmp_path):
        """ワイドは3着まで有効。複勝の規則を他券種に波及させない。"""
        path = self._db(tmp_path)
        self._bet(path, 'ワイド', 3, 4, amount=100)
        check_and_update_bets([_seven_horse_race()], db_path=path)
        r = self._row(path, 'ワイド')
        assert r['is_hit'] == 1 and r['payout'] == 250


class TestShadowSettlement:
    def test_shadow_fukusho_hit_follows_the_payout_table(self, tmp_path):
        """shadow_bets 側も同じ規則。軸が3着でも払戻行が無ければ不的中。"""
        os.makedirs(os.path.join(tmp_path, 'data'), exist_ok=True)
        init_db(str(tmp_path))
        race = _seven_horse_race()
        # 朝の予想スナップショット: 軸(RL1)=#3（実際は3着）
        save_race_predictions(
            {'id': race['id'], 'date': race['date'],
             'racecourse': race['racecourse'], 'race_num': race['race_num']},
            [{'num': n, 'name': f'ウマ{n}', 'rl_rank': i + 1,
              'win_prob': 0.3 - i * 0.02, 'cal_prob': 0.4, 'top3_prob': 0.5,
              'popularity': i + 1, 'win_odds': 2.0 + i}
             for i, n in enumerate([3, 4, 5, 1, 2, 6, 7])],
            base_dir=str(tmp_path))
        record_all_shadow_bets([race], str(tmp_path))

        conn = sqlite3.connect(os.path.join(tmp_path, 'data', 'keiba.db'))
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            'SELECT rl1_num, shadow_fukusho_hit, shadow_fukusho_payout, '
            'shadow_wide_hit FROM shadow_bets').fetchone()
        conn.close()
        assert r['rl1_num'] == 3
        assert r['shadow_fukusho_hit'] == 0, '3着を複勝的中にしない'
        assert not r['shadow_fukusho_payout']
        assert r['shadow_wide_hit'] == 1, 'ワイド(#3-#4)は当たっている'

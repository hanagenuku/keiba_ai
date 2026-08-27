"""f_recent 系が「直近の走り」を見ていることを固定する。

2026-08-27発見: hist は新しい順(DESC)なのに calc_features_for_xgb が
hist[-5:] で切っており、拾っていたのは「一番古い5走」だった。
直近1着の馬で f_last1_rank=16（学習・履歴10走）/ 5（推論・履歴5走）が返っていた。
重み .75**i も逆順で、最古走に最大重み1.0が乗っていた。

影響は本番モデルの重要度合計 7.34%（rl_f_recent_rank 7位・f_recent_fukusho 9位）。
修正後は f_last1_rank の重要度が 128位(0.39%) → 7〜9位(1.46%) に上がった。

⚠ ただし3窓のAUCは +0.0006 で事前登録の基準未達。
   これは「精度のための修正」ではなく「名前と中身を一致させる修正」である。
"""
import unittest

from src.features.engine import calc_features_for_xgb


def _run(place, date):
    return {'place': place, 'date': date, 'race_id': f'R{date}',
            'distance': 1600, 'surface': '芝', 'racecourse': '東京', 'class': '1勝',
            'condition': '良', 'track_condition': '良', 'agari3f': 35.0, 'last_3f': 35.0,
            'finishers': 16, 'num_finishers': 16, 'corner_all': '5-5',
            'popularity': 3, 'margin': 0.2, 'race_name': ''}


# 新しい順(DESC)。直近が1着、古いほど着順が悪い。
_HIST = [_run(1, '20260801'), _run(2, '20260701'), _run(3, '20260601'),
         _run(4, '20260501'), _run(5, '20260401'), _run(12, '20260301'),
         _run(13, '20260201'), _run(14, '20260101'), _run(15, '20251201'),
         _run(16, '20251101')]

_RACE = {'distance': 1600, 'surface': '芝', 'racecourse': '東京', 'race_class': '1勝',
         'track_condition': '良', 'race_name': 'テスト', 'first_3f': 35.0, 'horses': []}


def _feats(n_hist):
    horse = {'name': 'テスト馬', 'history': _HIST[:n_hist], 'horse_num': 5, 'age': 4,
             'sex': '牡', 'weight_load': 56.0, 'jockey': 'j', 'trainer': 't',
             'popularity': 3, 'win_odds': 5.0}
    return calc_features_for_xgb(horse, _RACE)


class TestRecentFormUsesMostRecentRuns(unittest.TestCase):
    def test_last_ranks_are_the_most_recent_runs(self):
        f = _feats(10)
        self.assertEqual(f['f_last1_rank'], 1.0, 'f_last1_rank は前走(直近)の着順であるべき')
        self.assertEqual(f['f_last2_rank'], 2.0)
        self.assertEqual(f['f_last3_rank'], 3.0)

    def test_training_and_inference_depths_agree(self):
        """学習は履歴10走・推論は5走を渡す。f_recent 系は直近5走なので一致するべき。"""
        a, b = _feats(10), _feats(5)
        for k in ('f_last1_rank', 'f_last2_rank', 'f_last3_rank',
                  'f_recent', 'f_recent_fukusho'):
            self.assertAlmostEqual(
                a[k], b[k], places=9,
                msg=f'{k} が履歴の深さで変わる（学習10走 vs 推論5走のパリティ違反）')

    def test_recent_weights_favour_the_latest_run(self):
        """直近が良い馬は、直近が悪い馬より f_recent が高くなるべき。"""
        good = _feats(5)['f_recent']                       # 1,2,3,4,5着
        horse = {'name': 'x', 'history': list(reversed(_HIST[:5])), 'horse_num': 5,
                 'age': 4, 'sex': '牡', 'weight_load': 56.0, 'jockey': 'j',
                 'trainer': 't', 'popularity': 3, 'win_odds': 5.0}
        bad = calc_features_for_xgb(horse, _RACE)['f_recent']   # 5,4,3,2,1着
        self.assertGreater(good, bad,
                           '直近1着の馬の f_recent が直近5着の馬より低い＝重みが逆')

    def test_no_history_falls_back(self):
        horse = {'name': 'x', 'history': [], 'horse_num': 5, 'age': 4, 'sex': '牡',
                 'weight_load': 56.0, 'jockey': 'j', 'trainer': 't',
                 'popularity': 3, 'win_odds': 5.0}
        f = calc_features_for_xgb(horse, _RACE)
        self.assertEqual(f['f_last1_rank'], 8.0)
        self.assertEqual(f['f_career_runs'], 0)


if __name__ == '__main__':
    unittest.main()

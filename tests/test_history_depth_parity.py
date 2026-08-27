"""学習と推論で過去走の深さが揃っていることを固定する。

🔴 2026-08-27まで 学習 limit=10 / 推論 limit=5 と食い違っていた。
本番の唯一の呼び出し get_history_from_db(name, path) は既定の5走のまま動いており、
同じ馬・同じ時点で **134特徴量のうち78個** が別の値になっていた
（f_career_runs は std比3.80、f_speed_trend 1.27、f_speed_last 0.95 …）。
6走以上ある馬＝直近窓の約52%が該当。

実測した影響（3窓×3シード・本番同型・残差・ドリフト注入ON）:
  AUC       +0.0002（実害は小さい）
  Brier     3窓とも10走側が良い
  LogLoss   3窓とも10走側が良い
つまり順位付けはほぼ変わらないが、cal_prob の較正がわずかに悪化していた。

「効果が小さいから直さない」ではなく、学習と推論が違う入力を見ている状態は
今後の全ての測定を歪めるため揃えた（実際 A1 で f_maturity を別実装と突合した際、
4.4%のズレの原因がこれだった）。
"""
import inspect
import re
import unittest

from src.scraper.jra_scraper import HISTORY_LIMIT, get_history_from_db
from src.tools.build_training_data import _get_history_before


class TestHistoryDepthParity(unittest.TestCase):
    def test_both_sides_default_to_the_shared_constant(self):
        inf = inspect.signature(get_history_from_db).parameters['limit'].default
        tr = inspect.signature(_get_history_before).parameters['limit'].default
        self.assertEqual(inf, HISTORY_LIMIT,
                         '推論側の既定が共有定数と違う')
        self.assertEqual(tr, HISTORY_LIMIT,
                         '学習側の既定が共有定数と違う')
        self.assertEqual(inf, tr,
                         f'学習と推論で過去走の深さが違う（学習{tr}走 / 推論{inf}走）。'
                         '片方だけ変えると78/134特徴量が別の値になる')

    def test_production_call_site_does_not_override_the_depth(self):
        """本番の呼び出しが limit を上書きしていないこと。

        既定を揃えても、呼び出し側で limit=5 と書かれていれば元の木阿弥になる。
        """
        src = open('src/scraper/jra_scraper.py', encoding='utf-8').read()
        calls = [m.group(1) for line in src.split('\n')
                 if not line.lstrip().startswith('def ')
                 for m in [re.search(r'get_history_from_db\(([^)]*)\)', line)] if m]
        self.assertTrue(calls, 'get_history_from_db の呼び出しが見つからない')
        for c in calls:
            self.assertNotIn('limit=', c,
                             f'本番の呼び出しが深さを上書きしている: get_history_from_db({c})')

    def test_training_call_site_uses_the_constant(self):
        src = open('src/tools/build_training_data.py', encoding='utf-8').read()
        m = re.search(r"_get_history_before\(conn, h\['name'\], date_str, limit=(\w+)\)", src)
        self.assertIsNotNone(m, '学習側の呼び出しが見つからない')
        self.assertEqual(m.group(1), '_HISTORY_LIMIT',
                         '学習側の呼び出しが定数ではなく直値になっている')


if __name__ == '__main__':
    unittest.main()

"""「作ったのに誰も使っていない」を機械的に検知する。

2026-08-26 の棚卸しで、この種の副産物が11件見つかった:
  - 計算しているのにモデルに届いていない特徴量
  - 溜めているのに一度も読まれないテーブル / JSON
  - 定義されているが呼ばれない関数（11個）
  - 集計しているのに画面に出ないセクション（2個）

毎回トライするたびに副産物が出る、という状態を止めるためのテスト。
⚠ 意図的に残すものは理由付きでホワイトリストに書く（黙って例外にしない）。
"""
import ast
import json
import os
import re
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _grep(pattern, *paths):
    r = subprocess.run(['grep', '-rn', '--include=*.py', '--include=*.html',
                        '--include=*.mjs', '--include=*.ipynb', pattern,
                        *[os.path.join(BASE, p) for p in paths]],
                       capture_output=True, text=True)
    return [l for l in r.stdout.split('\n') if l]


class TestNoUncalledFunctions:
    """定義したのに一度も呼ばれない公開関数を作らない。"""

    # 意図的に残すもの。**理由を必ず書く**
    ALLOW = {
        # ノートブック(Colab)から呼ばれる
        'ability_first_loose',
        # race_notes（不利メモ）機構の一部。UIから使う想定で蓄積待ち
        'recalc_all_handicaps',
        # 月次再学習で特徴量化する想定。まだ配線していない
        'calc_error_tag_features',
    }

    def test_no_new_dead_functions(self):
        dead = []
        for rel in ('src/features/engine.py', 'src/utils/db.py',
                    'src/betting/make_bets.py', 'src/betting/ev_filter.py',
                    'src/features/error_tags.py'):
            tree = ast.parse(open(os.path.join(BASE, rel), encoding='utf-8').read())
            for n in tree.body:
                if not isinstance(n, ast.FunctionDef) or n.name.startswith('_'):
                    continue
                if n.name in self.ALLOW:
                    continue
                hits = [l for l in _grep(f'{n.name}(', 'src', 'scripts', 'tests')
                        if f'def {n.name}(' not in l]
                if not hits:
                    dead.append(f'{rel}:{n.name}')
        assert not dead, (
            '呼ばれていない関数がある。使うか消すか、残すなら ALLOW に理由付きで:\n  '
            + '\n  '.join(dead))


class TestStatsSectionsAreDisplayed:
    """stats.json に出すなら画面にも出す。溜めるだけにしない。"""

    def test_every_section_reaches_the_app(self):
        stats = json.load(open(os.path.join(BASE, 'data', 'stats.json'), encoding='utf-8'))
        html = open(os.path.join(BASE, 'index.html'), encoding='utf-8').read()
        missing = [k for k in stats if k not in html]
        assert not missing, (
            f'集計しているのに画面に出ていない: {missing}\n'
            '  出すか、generate_stats.py 側で出力をやめること')


class TestNoDeadDataFiles:
    """data/ のファイルは必ず「読む」コードを持つ。書きっぱなしにしない。"""

    # 読み手が無くてよいもの（理由付き）
    ALLOW = {
        'latest.json',            # アプリがHTTPで取得（コード内参照は生成側のみ）
        'stats.json',             # 同上
        'workflow_status.json',   # 同上
        'horse_features.csv',     # 学習の中間生成物
        'horse_features_old.csv', # 上のバックアップ
        'kpi_weekly.json',        # 週次トレンドの記録。画面には model_kpi 側を出している
    }

    @staticmethod
    def _mentioned_in_code(fname, line):
        """コメント行の言及は「使っている」に数えない。

        correction_table.json は 2026-07-21 に呼び出しを消したあと、
        コメント1行だけが残って「参照あり」に見えていた。
        """
        body = line.split(':', 2)[-1]          # grep の "path:lineno:" を落とす
        if body.lstrip().startswith('#'):
            return False
        pos = body.find(fname)
        hash_pos = body.find('#')
        return pos >= 0 and (hash_pos < 0 or pos < hash_pos)

    def test_every_data_file_is_used_in_code(self):
        data = os.path.join(BASE, 'data')
        orphans = []
        for f in sorted(os.listdir(data)):
            if f in self.ALLOW or f.startswith('.') or os.path.isdir(os.path.join(data, f)):
                continue
            hits = _grep(f, 'src', 'scripts', 'index.html')
            if not any(self._mentioned_in_code(f, h) for h in hits):
                orphans.append(f)
        assert not orphans, (
            f'コード中で使われていない data/ ファイル（コメントのみの言及）: {orphans}\n'
            '  使うか消すか、残すなら ALLOW に理由付きで')

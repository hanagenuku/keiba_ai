"""sunday_results.py の対象日指定のテスト（2026-08-03導入）。

2026-08-02 の日曜結果取得で新潟・札幌のスキャンが全滅し24レースを落としたが、
翌日に再実行しても **当日(月曜)を見に行って6秒で空振り** に終わった。
target_date が当日固定で、過去日を取り直す手段が無かったため。
取りこぼしを後から回収できることは、データが永久に失われないための生命線。
"""
import os
import re
import sys
import textwrap
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SRC_PATH = os.path.join(os.path.dirname(__file__), '..',
                        'scripts', 'sunday_results.py')
JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 8, 3, 6, 54, tzinfo=JST)      # 再取得を試みた月曜朝


def _src():
    with open(SRC_PATH, encoding='utf-8') as f:
        return f.read()


def _resolve(env_val):
    """対象日決定ブロックだけを取り出して実行する。"""
    block = textwrap.dedent(
        _src().split('jst_now = datetime.now(JST)', 1)[1]
              .split('sess = create_session()', 1)[0])
    environ = {} if env_val is None else {'TARGET_DATE': env_val}
    ns = {'os': type('o', (), {'environ': environ})(),
          'jst_now': NOW, 'print': lambda *a, **k: None}
    exec(compile(block, 'target_date_block', 'exec'), ns)
    return ns['target_date']


class TestTargetDate:

    def test_defaults_to_today(self):
        assert _resolve(None) == '20260803'

    def test_empty_input_is_today(self):
        """workflow_dispatch の既定値は空文字で渡ってくる。"""
        assert _resolve('') == '20260803'

    def test_past_date_can_be_specified(self):
        """本件の核心: 取りこぼした開催を後から取り直せること。"""
        assert _resolve('20260802') == '20260802'

    def test_surrounding_spaces_are_tolerated(self):
        assert _resolve(' 20260802 ') == '20260802'

    @pytest.mark.parametrize('bad', ['2026-08-02', '202608', 'abc', '2026080x'])
    def test_invalid_date_is_rejected(self, bad):
        """静かに当日へフォールバックすると、また空振りに気づけない。"""
        with pytest.raises(SystemExit):
            _resolve(bad)

    def test_shap_and_error_tags_follow_target_date(self):
        """当日固定のままだと指定日と別の日を集計してしまう。"""
        src = _src()
        assert re.search(
            r"jst_date = f'\{target_date\[:4\]\}-\{target_date\[4:6\]\}-"
            r"\{target_date\[6:\]\}'", src)
        assert 'jst_date = jst_now' not in src


class TestWorkflowWiring:

    def test_workflow_passes_date_input(self):
        import yaml
        p = os.path.join(os.path.dirname(__file__), '..',
                         '.github', 'workflows', 'sunday-results.yml')
        with open(p, encoding='utf-8') as f:
            d = yaml.safe_load(f)
        trig = d[True] if True in d else d.get('on')
        assert 'date' in trig['workflow_dispatch']['inputs']
        step = [s for s in d['jobs']['results']['steps']
                if s.get('name') == 'Get Sunday results'][0]
        assert step['env']['TARGET_DATE'] == '${{ inputs.date }}'

"""カード途中のレースが黙って落ちる不具合の回帰テスト（2026-09-11）。

2026-09-12 の予想で中山R03が
    R03: suffix=B5 → パラメータエラー/ページなし
で落ちたのに `data_quality.parse_failures` が空のまま、アプリにも警告が出ず、
ユーザーが目視で気づくまで分からなかった。

原因は2つ:
  ① 近傍走査が ±60 しかなく、suffix が全域(0x00〜0xFF)のどこかにある場合に届かない
  ② 「ページなし」を一律 failures から除外していた
     （12レース未満の開催があるため、というのが当時の理由）

②は「後続レースが取れているか」で切り分けられる。Rn が取れず Rn+1 以降が
取れていれば、そのレースは実在するのに落ちたと断定できる。
"""
import pytest
from src.scraper import jra_scraper as js


def _setup(monkeypatch, ok_suffix_by_race, half_range=None, budget=None):
    """suffix が一致した時だけ soup を返すフェイク。要求した suffix を記録する。"""
    asked = []
    monkeypatch.setattr(js, 'get_kaisai_on_date',
                        lambda d, s=None: {'pw01dde0106X': '20260912'})
    monkeypatch.setattr(js, 'find_r01_shutuba', lambda b, d, s: 0x4B)
    monkeypatch.setattr(js, 'find_r01_odds', lambda b, d, s: 0xEE)
    monkeypatch.setattr(js, '_fill_pedigree', lambda *a, **k: None)
    monkeypatch.setattr(js.time, 'sleep', lambda *a: None)
    if half_range is not None:
        monkeypatch.setattr(js, 'SUFFIX_SCAN_HALF_RANGE', half_range)
    if budget is not None:
        monkeypatch.setattr(js, 'SUFFIX_SCAN_BUDGET_DEFAULT', budget)

    class _Tbl:
        def get_text(self, *a, **k): return 'ヘッダ'

    class _Soup:                       # 本番と同じく find_all('table') が使える形にする
        def find_all(self, _): return [_Tbl()]

    def fake_fetch(sess, base, r, date_str, sx):
        asked.append((r, sx))
        return (None, _Soup()) if ok_suffix_by_race.get(r) == sx else (None, None)

    monkeypatch.setattr(js, '_try_fetch_shutuba', fake_fetch)
    monkeypatch.setattr(js, '_parse_shutuba',
                        lambda soup, rc, r, d, pc, hp: {
                            'horses': [], 'race_name': f'R{r}', 'num_horses': 8,
                            'distance': 1600, 'surface': '芝'})
    return asked


class TestMidCardHoleIsRecorded:
    def test_hole_in_the_middle_is_a_failure(self, monkeypatch):
        """🔴 R03 だけ取れず R04 以降が取れる → 実在するのに落ちた。記録する。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13) if r != 3}
        _setup(monkeypatch, ok, half_range=2)   # 走査を狭めて必ず見つからなくする
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert len(races) == 11
        assert [f['race_num'] for f in failures] == [3]
        assert 'カード途中' in failures[0]['reason']

    def test_short_card_is_not_a_failure(self, monkeypatch):
        """末尾が無いだけ（12レース未満の開催）は失敗にしない。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 11)}
        _setup(monkeypatch, ok, half_range=2)
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert len(races) == 10
        assert failures == []

    def test_jump_race_skip_is_not_a_failure(self, monkeypatch):
        """障害レースのスキップは意図した挙動。記録しない（阪神R01の実例）。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        _setup(monkeypatch, ok)
        monkeypatch.setattr(js, '_parse_shutuba',
                            lambda soup, rc, r, d, pc, hp: None if r == 1 else {
                                'horses': [], 'race_name': f'R{r}', 'num_horses': 8,
                                'distance': 1600, 'surface': '芝'})
        monkeypatch.setattr(js, 'parse_header', lambda t: {'surface': '障害'})
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert len(races) == 11
        assert failures == []


class TestSuffixScanCoversFullRange:
    def test_finds_suffix_beyond_the_old_60_window(self, monkeypatch):
        """🔴 旧実装(±60)では見つからない距離にある suffix を拾えること。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        far = f'{(int(js.calc_suffix(0x4B, 3), 16) + 100) % 256:02X}'
        ok[3] = far                                  # 計算値から100離れた場所に置く
        _setup(monkeypatch, ok)
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert len(races) == 12, '全域走査で R03 を回収できていない'
        assert failures == []

    def test_scan_is_bounded_by_a_budget(self, monkeypatch):
        """⚠ North Star #4: 上限が無いとJRA不調時にタイムアウトする。"""
        asked = _setup(monkeypatch, {}, budget=50)   # 1レースも当たらない
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert races == []
        # 12レース × 256 = 3,072 まで行かず、予算50 + 各レースの初回試行で止まること
        assert len(asked) < 200, f'走査が予算で止まっていない: {len(asked)}件'

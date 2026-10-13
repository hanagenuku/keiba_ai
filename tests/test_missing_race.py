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


class TestMissingRaceReasonDoesNotClaimAGap:
    """🔴 2026-10-12 の欠陥①: 走査の空振りを「欠番」と誤って記録していた。

    同日の朝に `ページなし(カード途中の欠番)` と記録した 東京10R・京都10R は、
    11:08 の再実行と夜の結果取得では**両方取れている**（＝番組に実在する）。
    真の欠番（10/12 東京R4）と走査の空振りは1回の実行では区別できないので、
    理由には「観測した事実」だけを書き、断定しない。
    """

    def test_reason_never_says_a_programme_gap(self, monkeypatch):
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13) if r != 3}
        _setup(monkeypatch, ok, half_range=2)
        _, failures = js.fetch_races_on_date(None, '20260912', None)
        assert [f['race_num'] for f in failures] == [3]
        reason = failures[0]['reason']
        assert '欠番' not in reason, f'走査の空振りを欠番と断定している: {reason}'
        assert 'suffix' in reason, f'何が起きたのか分からない理由: {reason}'
        assert 'カード途中' in reason          # 位置の事実は残す

    def test_budget_exhaustion_is_told_apart_from_a_full_scan(self, monkeypatch):
        """予算で打ち切った（＝探しきっていない）ことが理由から分かる。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13) if r != 1}
        _setup(monkeypatch, ok, half_range=128, budget=4)
        _, failures = js.fetch_races_on_date(None, '20260912', None)
        assert [f['race_num'] for f in failures] == [1]
        assert '予算切れ' in failures[0]['reason'], failures[0]['reason']


class TestSingleRaceIsRetried:
    """🔴 会場単位の再試行（find_r01_*）はあったが、レース単位には無かった。

    2026-10-10 の夜の結果取得では 京都 R2/R4/R7/R12 の4レースが落ち、
    history.db にそのレースが永久に入らなかった。
    """

    def test_transient_miss_is_recovered_by_a_retry(self, monkeypatch):
        """走査中は全滅するが、少し待つと計算値で取れるレースを回収できる。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        asked = _setup(monkeypatch, ok, half_range=2)
        good_sx = js.calc_suffix(0x4B, 3)

        class _Tbl:
            def get_text(self, *a, **k): return 'ヘッダ'

        class _Soup:
            def find_all(self, _): return [_Tbl()]

        calls = {'r3': 0}

        def flaky(sess, base, r, date_str, sx):
            asked.append((r, sx))
            if r == 3:
                calls['r3'] += 1
                # 初回＋走査(±2=4件)の 5 件は一過性に失敗させる
                if calls['r3'] <= 5:
                    return (None, None)
                return (None, _Soup()) if sx == good_sx else (None, None)
            return (None, _Soup()) if ok.get(r) == sx else (None, None)

        monkeypatch.setattr(js, '_try_fetch_shutuba', flaky)
        races, failures = js.fetch_races_on_date(None, '20260912', None)
        assert len(races) == 12, 'レース単位の再試行で R03 を回収できていない'
        assert failures == []

    def test_retry_does_not_rescan_the_whole_range(self, monkeypatch):
        """⚠ North Star #4: 再試行は全域(256件)を叩き直さない。"""
        asked = _setup(monkeypatch, {}, half_range=2, budget=10_000)
        js.fetch_races_on_date(None, '20260912', None)
        per_race = {}
        for r, _sx in asked:
            per_race[r] = per_race.get(r, 0) + 1
        # 初回1 + 走査4 + (R10以降の単純式1) + 再試行は最大4件
        assert max(per_race.values()) <= 10, per_race


# ── 結果側（fetch_results）も同じ強さにする ──────────────────────────────
def _setup_results(monkeypatch, ok_suffix_by_race, half_range=None, budget=None,
                   flaky_race=None, flaky_times=0):
    """結果取得の近傍走査を本番と同じ形で回すフェイク。"""
    asked = []
    monkeypatch.setattr('src.scraper.calendar.get_kaisai_on_date',
                        lambda d, s=None: {'pw01dde0108X': d})
    monkeypatch.setattr(js, 'find_r01_result', lambda b, d, s: 0x4B)
    monkeypatch.setattr(js, '_fill_pedigree', lambda *a, **k: None)
    monkeypatch.setattr(js.time, 'sleep', lambda *a: None)
    if half_range is not None:
        monkeypatch.setattr(js, 'SUFFIX_SCAN_HALF_RANGE', half_range)
    if budget is not None:
        monkeypatch.setattr(js, 'SUFFIX_SCAN_BUDGET_DEFAULT', budget)

    class _Sess:                        # Step1 を落として出走表一覧へフォールバックさせる
        def post(self, *a, **k): raise RuntimeError('step1 unavailable')

    calls = {}

    def fake_fetch(sess, base, r, target_date, sx):
        asked.append((r, sx))
        if r == flaky_race:
            calls[r] = calls.get(r, 0) + 1
            if calls[r] <= flaky_times:
                return None
        return object() if ok_suffix_by_race.get(r) == sx else None

    monkeypatch.setattr(js, '_try_fetch_result', fake_fetch)
    monkeypatch.setattr(js, 'parse_result_soup',
                        lambda soup, rc, r, d, pc: {
                            'race_name': f'R{r}', 'finishers': [
                                {'place': 1, 'num': 1, 'name': 'ウマ', 'running_style': '逃'}]})
    return asked, _Sess()


class TestResultScanMatchesTheShutubaSide:
    """結果側の走査は ±60・予算なしで出走表側(±128・予算あり)より弱かった。"""

    def test_finds_suffix_beyond_the_old_60_window(self, monkeypatch):
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        ok[7] = f'{(int(js.calc_suffix(0x4B, 7), 16) + 100) % 256:02X}'
        _asked, sess = _setup_results(monkeypatch, ok)
        results = js.fetch_results(sess, '20261010')
        assert len(results) == 12, '全域走査で R07 を回収できていない（旧±60のまま）'

    def test_scan_is_bounded_by_a_budget(self, monkeypatch):
        """⚠ North Star #4: 走査を広げるなら同じ変更で予算を入れる。"""
        asked, sess = _setup_results(monkeypatch, {}, budget=50)
        results = js.fetch_results(sess, '20261010')
        assert results == []
        assert len(asked) < 200, f'走査が予算で止まっていない: {len(asked)}件'

    def test_transient_miss_is_recovered_by_a_retry(self, monkeypatch):
        """🔴 2026-10-10 に京都 R2/R4/R7/R12 が落ちて history.db に残らなかった件。"""
        ok = {r: js.calc_suffix(0x4B, r) for r in range(1, 13)}
        _asked, sess = _setup_results(monkeypatch, ok, half_range=2,
                                      flaky_race=2, flaky_times=5)
        results = js.fetch_results(sess, '20261010')
        assert len(results) == 12, 'レース単位の再試行で R02 を回収できていない'

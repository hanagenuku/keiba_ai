"""popularity 復旧スクリプトの検証。

North Star #6 に従い、手打ちdictではなく本番と同じ save_history_db() で
実際に sqlite3 のDBを作ってから検証する。
"""
import sqlite3


def _write(hist_path, date, n_horses, broken_count, race_id=None):
    """指定した開催日のレースを1本書き込む。先頭 broken_count 頭を
    popularity=99（取得失敗のセンチネル）にする。"""
    from src.utils.db import save_history_db
    rid = race_id or f'{date}_01_01'
    finishers = []
    for i in range(1, n_horses + 1):
        finishers.append({
            'num': i, 'name': f'馬{date}_{i}', 'place': i,
            'popularity': 99 if i <= broken_count else i,
            'trainer': '' if i <= broken_count else f'調教師{i}',
        })
    save_history_db([{
        'race_id': rid,
        'date': f'{date[:4]}-{date[4:6]}-{date[6:8]}',
        'racecourse': '福島', 'distance': 1200, 'surface': '芝',
        'finishers': finishers,
    }], db_path=str(hist_path))


class TestFindBrokenDates:
    def test_detects_fully_broken_day(self, tmp_path):
        from scripts.repair_popularity import find_broken_dates
        p = tmp_path / 'history.db'
        _write(p, '20260704', 10, 10)      # 全滅
        found = find_broken_dates(str(p))
        assert [d for d, _, _ in found] == ['20260704']
        assert found[0][1] == 10 and found[0][2] == 10

    def test_ignores_healthy_day(self, tmp_path):
        from scripts.repair_popularity import find_broken_dates
        p = tmp_path / 'history.db'
        _write(p, '20260628', 10, 0)       # 健全
        assert find_broken_dates(str(p)) == []

    def test_ignores_few_scratched_horses(self, tmp_path):
        """取消馬などで数頭だけ99になるのは正常なので拾わない。"""
        from scripts.repair_popularity import find_broken_dates
        p = tmp_path / 'history.db'
        _write(p, '20260628', 10, 2)       # 20%だけ不明
        assert find_broken_dates(str(p)) == []

    def test_returns_dates_sorted_oldest_first(self, tmp_path):
        from scripts.repair_popularity import find_broken_dates
        p = tmp_path / 'history.db'
        _write(p, '20260802', 10, 10)
        _write(p, '20260704', 10, 10)
        assert [d for d, _, _ in find_broken_dates(str(p))] == ['20260704', '20260802']


class TestRepairOrdering:
    def test_defaults_to_newest_first(self, tmp_path, capsys):
        """既定は新しい順。JRAの結果一覧は直近しか載っておらず、古い日から
        処理するとbudgetを取得不能な日で使い切ってしまうため。"""
        from scripts.repair_popularity import repair
        data = tmp_path / 'data'
        data.mkdir()
        p = data / 'history.db'
        _write(p, '20260802', 10, 10)
        _write(p, '20260704', 10, 10)
        repair(str(tmp_path), budget=1, dry_run=True)
        out = capsys.readouterr().out
        assert '20260802' in out, '新しい日から処理すべき'
        assert out.index('20260802') < out.index('…ほか') if '…ほか' in out else True

    def test_oldest_first_option(self, tmp_path, capsys):
        from scripts.repair_popularity import repair
        data = tmp_path / 'data'
        data.mkdir()
        p = data / 'history.db'
        _write(p, '20260802', 10, 10)
        _write(p, '20260704', 10, 10)
        repair(str(tmp_path), budget=1, dry_run=True, oldest_first=True)
        out = capsys.readouterr().out
        assert '20260704' in out


class TestCountBroken:
    def test_counts_only_target_date(self, tmp_path):
        from scripts.repair_popularity import _count_broken
        p = tmp_path / 'history.db'
        _write(p, '20260704', 10, 10)
        _write(p, '20260628', 10, 0)
        assert _count_broken(str(p), '20260704') == 10
        assert _count_broken(str(p), '20260628') == 0


class TestRepairIsIdempotentOnHealthyDb:
    def test_healthy_db_reports_nothing_to_do(self, tmp_path, capsys):
        from scripts.repair_popularity import repair
        data = tmp_path / 'data'
        data.mkdir()
        _write(data / 'history.db', '20260628', 10, 0)
        r = repair(str(tmp_path), budget=5, dry_run=True)
        assert r['processed'] == 0 and r['remaining'] == 0
        assert '復旧が必要な開催日はない' in capsys.readouterr().out

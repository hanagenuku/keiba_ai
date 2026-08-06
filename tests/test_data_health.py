"""列崩れ検知（check_data_health）の検証。

North Star #6 に従い、手打ちINSERTではなく本番と同じ save_history_db() で
実際に sqlite3 のDBを作ってから検証する。
"""
import pytest


def _write_month(hist_path, ym, n_races=3, n_horses=10, broken=None):
    """指定した年月にレースを書き込む。broken は壊す列の指定。

    broken='popularity' → popularity=99（取得失敗のセンチネル）
    broken='trainer'    → trainer=''
    """
    from src.utils.db import save_history_db
    broken = broken or set()
    results = []
    for r in range(1, n_races + 1):
        day = f'{ym}-0{min(r,9)}'
        finishers = []
        for i in range(1, n_horses + 1):
            finishers.append({
                'num': i, 'name': f'馬{ym}_{r}_{i}', 'place': i,
                'popularity': 99 if 'popularity' in broken else i,
                'trainer': '' if 'trainer' in broken else f'調教師{i}',
                'jockey': f'騎手{i}',
                'agari_rank': i, 'agari3f': 35.0,
                'finish_time': 90.0, 'weight_load': 55.0,
                'corner_all': '3-3-2-1', 'sex': '牡', 'age': 4,
                'body_weight': None if 'body_weight' in broken else 480,
            })
        results.append({
            'race_id': f'{ym.replace("-","")}0{r}_01_0{r}',
            'date': day, 'racecourse': '東京',
            'distance': 1600, 'surface': '芝', 'finishers': finishers,
        })
    save_history_db(results, db_path=str(hist_path))


class TestCheckDataHealth:
    def test_detects_sudden_popularity_breakage(self, tmp_path, capsys):
        """本番で実際に起きた形: 過去は健全 → 直近だけ全滅（列崩れ）。

        popularity=99 は NULL ではないので「IS NOT NULL」の充足率チェックを
        すり抜ける。値の範囲で見て初めて検知できる。
        """
        from scripts.check_data_health import check
        p = tmp_path / 'history.db'
        for ym in ['2026-01', '2026-02', '2026-03', '2026-04']:
            _write_month(p, ym)                       # 健全
        for ym in ['2026-05', '2026-06', '2026-07']:
            _write_month(p, ym, broken={'popularity'})  # 列崩れ
        rc = check(str(p), months=3, strict=True)
        out = capsys.readouterr().out
        assert rc == 1, 'strict指定時は異常があれば exit 1'
        # 全滅(100%)でも急落でも、異常として報告されればよい
        assert '🔴' in out and 'popularity' in out
        assert '1件の異常' in out
        assert 'base_margin' in out, '影響の説明が出るべき'

    def test_detects_partial_drop_not_only_total_loss(self, tmp_path, capsys):
        """完全に0%にならず「一部だけ壊れた」場合も急落として拾えること。
        （本番の2026-05は46%欠損という中間状態だった）"""
        from scripts.check_data_health import check
        p = tmp_path / 'history.db'
        for ym in ['2026-01', '2026-02', '2026-03', '2026-04']:
            _write_month(p, ym, n_races=4)
        # 直近は4レース中2レースだけ壊す（≒50%欠損）
        for ym in ['2026-05', '2026-06', '2026-07']:
            _write_month(p, ym, n_races=2)
            _write_month(p, ym, n_races=4, broken={'popularity'})
        rc = check(str(p), months=3, strict=True)
        out = capsys.readouterr().out
        assert rc == 1, '部分的な欠損も検知すべき'
        assert 'popularity' in out and '🔴' in out

    def test_healthy_db_passes(self, tmp_path, capsys):
        from scripts.check_data_health import check
        p = tmp_path / 'history.db'
        for ym in ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07']:
            _write_month(p, ym)
        rc = check(str(p), months=3, strict=True)
        assert rc == 0
        assert '列崩れの兆候なし' in capsys.readouterr().out

    def test_detects_trainer_breakage(self, tmp_path, capsys):
        """popularity と同時に壊れた trainer も検知できること。"""
        from scripts.check_data_health import check
        p = tmp_path / 'history.db'
        for ym in ['2026-01','2026-02','2026-03','2026-04']:
            _write_month(p, ym)
        for ym in ['2026-05','2026-06','2026-07']:
            _write_month(p, ym, broken={'trainer'})
        check(str(p), months=3)
        assert 'trainer' in capsys.readouterr().out

    def test_known_empty_columns_are_not_reported(self, tmp_path, capsys):
        """win_odds 等は元々ほぼ空と分かっているので警告しない
        （毎回鳴ると本物の異常が埋もれるため）。"""
        from scripts.check_data_health import check, KNOWN_EMPTY
        p = tmp_path / 'history.db'
        for ym in ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06','2026-07']:
            _write_month(p, ym)
        check(str(p), months=3)
        out = capsys.readouterr().out
        for col in ['win_odds', 'bracket', 'trainer_affiliation']:
            assert col in KNOWN_EMPTY
            assert f'\n{col:<14}' not in out

    def test_missing_db_raises(self, tmp_path):
        from scripts.check_data_health import check
        with pytest.raises(SystemExit):
            check(str(tmp_path / 'nope.db'))

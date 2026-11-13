"""本番の出馬表が持つキーで特徴量が正しく作られること（2026-11-13）。

学習（build_training_data）の馬は 'horse_num'・レースは 'race_id' を持つが、
本番の出馬表（parser.parse_horse / _parse_shutuba）は 'num'・'post_position'・'id' しか持たない。
'horse_num' / 'race_id' だけを読むコードは、テスト（学習側の形）では通り、本番でだけ
黙って既定値に落ちる。North Star #6 のとおり**本番と同じ形**で検査する。
"""
import os
import sqlite3
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import web_signals  # noqa: E402
from src.scraper.parser import parse_horse  # noqa: E402


def _parsed(num, sexage='牝4'):
    html = f'''<tr><td>{num}</td>
      <td><a href="/JRADB/accessU.html?CNAME=pw01dud002024103763/CB">テストウマ{num}</a></td>
      <td>{sexage}</td><td>55.0</td><td><a href="#">調教師名</a></td>
      <td><a href="#">騎手名</a></td><td>3.5</td></tr>'''
    return parse_horse(BeautifulSoup(html, 'lxml').find('tr').find_all('td'), '東京', '芝')


def test_parser_output_has_no_horse_num_key_so_readers_must_not_rely_on_it():
    h = _parsed(7)
    assert 'horse_num' not in h
    assert h['num'] == 7 and h['post_position'] == 7


def test_horse_number_reads_every_shape():
    assert web_signals.horse_number(_parsed(7)) == 7
    assert web_signals.horse_number({'horse_num': 3}) == 3
    assert web_signals.horse_number({'num': '12'}) == 12
    assert web_signals.horse_number({}) is None


def test_parser_keeps_sex():
    assert _parsed(1, '牝4')['sex'] == '牝'
    assert _parsed(1, '牡3')['sex'] == '牡'
    assert _parsed(1, 'セ5')['sex'] == 'セ'
    assert _parsed(1, '騸5')['sex'] == 'セ'


def test_f_post_varies_within_a_race_for_parser_shaped_horses():
    """本番の形の馬で f_post が馬番ごとに変わること（以前は全馬が馬番1扱いで一定だった）。"""
    from src.features import engine
    race = {'id': '20261108_05_11', 'racecourse': '東京', 'distance': 1600, 'surface': '芝',
            'race_class': '1勝', 'num_horses': 16, 'date': '20261108', 'horses': []}
    vals = []
    for n in (1, 8, 16):
        h = _parsed(n)
        h.update(history=[], running_style='差し', popularity=n)
        race['horses'] = [h]
        vals.append(engine.calc_features_for_xgb(h, race)['f_post'])
    # 東京芝1600 に枠の傾きが1つも設定されていなければ3つとも 5.0 になるので、その場合は検査しない
    if len({round(v, 6) for v in vals}) == 1 and abs(vals[0] - 5.0) < 1e-9:
        return
    assert len({round(v, 6) for v in vals}) == 3, vals


def test_web_signals_attach_and_features_for_production_shaped_race(tmp_path, monkeypatch):
    """レースが 'id'、馬が 'num' しか持たなくても、指数が貼られて特徴量になること。"""
    db = tmp_path / 'web_signals.db'
    c = sqlite3.connect(db)
    c.execute('CREATE TABLE web_signals (source TEXT, race_id TEXT, horse_num INTEGER, index_value INTEGER)')
    c.executemany('INSERT INTO web_signals VALUES (?,?,?,?)',
                  [('kayochinkeiba', '20261108_05_11', n, v) for n, v in ((1, 60), (7, 75), (12, 50))])
    c.commit(); c.close()
    race = {'id': '20261108_05_11', 'horses': [_parsed(1), _parsed(7), _parsed(12), _parsed(15)]}
    web_signals.attach_web_signals(race, str(db), sources=['kayochinkeiba'])
    assert race['web_signals']['kayochinkeiba'] == {1: 60.0, 7: 75.0, 12: 50.0}

    monkeypatch.setattr(web_signals, 'adopted_sources', lambda base_dir: [{'source': 'kayochinkeiba'}])
    rank_col, z_col = web_signals.feature_cols_for('kayochinkeiba')
    f7 = web_signals.calc_web_signal_features(race['horses'][1], race)
    assert f7[rank_col] == 1.0 and f7[z_col] > 0
    f15 = web_signals.calc_web_signal_features(race['horses'][3], race)   # 記事に居ない馬
    assert f15[rank_col] != f15[rank_col] and f15[z_col] != f15[z_col]    # NaN


def test_fill_missing_keeps_web_columns_nan_and_fills_the_rest():
    import numpy as np
    import pandas as pd
    rank_col, z_col = web_signals.feature_cols_for('kayochinkeiba')
    df = pd.DataFrame({'f_a': [1.0, np.nan], rank_col: [np.nan, 2.0], z_col: [np.nan, 0.5]})
    out = web_signals.fill_missing(df)
    assert out['f_a'].tolist() == [1.0, 5.0]
    assert np.isnan(out[rank_col][0]) and np.isnan(out[z_col][0])
    # web 由来の列が無ければ従来の fillna(5.0) と同じ
    plain = pd.DataFrame({'f_a': [np.nan], 'f_b': [2.0]})
    assert web_signals.fill_missing(plain).equals(plain.fillna(5.0))
    assert np.isnan(web_signals.default_for(rank_col)) and web_signals.default_for('f_a') == 5.0

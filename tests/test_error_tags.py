"""エラータグ自動分類・蓄積・補正のテスト"""
import json
import os
import tempfile

import pytest

from src.features.error_tags import (
    TAG_DEFINITIONS,
    _class_level,
    _condition_key,
    _dist_band,
    _calc_corrections,
    accumulate_tags,
    classify_race_tags,
    get_correction_factor,
    load_error_tags,
    save_error_tags,
    calc_error_tag_features,
)


# ── ヘルパー ─────────────────────────────────────────────────────────

def _make_horse(num, place, **kw):
    h = {'horse_num': num, 'place': place, 'horse_name': f'Horse{num}'}
    h.update(kw)
    return h


def _make_race(horses, **kw):
    race = {
        'race_id': 'R001',
        'date': '2026-07-13',
        'racecourse': '東京',
        'surface': '芝',
        'distance': 1600,
        'track_condition': '良',
        'race_class': '3勝クラス',
        'first_3f': None,
        'horses': horses,
    }
    race.update(kw)
    return race


def _make_predictions(horses_preds):
    """horses_preds: [(horse_num, rl_rank, win_prob), ...]"""
    return {
        num: {'rl_rank': rl, 'win_prob': wp}
        for num, rl, wp in horses_preds
    }


# ── 単体テスト ───────────────────────────────────────────────────────

class TestDistBand:
    def test_short(self):
        assert _dist_band(1200) == 'short'

    def test_mile(self):
        assert _dist_band(1600) == 'mile'

    def test_middle(self):
        assert _dist_band(2000) == 'middle'

    def test_long(self):
        assert _dist_band(2400) == 'long'

    def test_none(self):
        assert _dist_band(None) == 'unknown'


class TestConditionKey:
    def test_basic(self):
        key = _condition_key('東京', '芝', 1600, '良')
        assert key == '東京_芝_mile_良'

    def test_heavy(self):
        key = _condition_key('中山', 'ダート', 1200, '重')
        assert key == '中山_ダート_short_重'


class TestClassLevel:
    def test_levels(self):
        assert _class_level('新馬') == 1
        assert _class_level('未勝利') == 2
        assert _class_level('1勝クラス') == 3
        assert _class_level('2勝クラス') == 4
        assert _class_level('3勝クラス') == 5
        assert _class_level('OP') == 6
        assert _class_level('G3') == 7
        assert _class_level('G2') == 8
        assert _class_level('G1') == 9
        assert _class_level('') == 0


# ── 分類テスト ───────────────────────────────────────────────────────

class TestClassifyEscapeWin:
    def test_escape_win_detected(self):
        horses = [
            _make_horse(1, 1, running_style='逃げ'),
            _make_horse(2, 2, running_style='差し'),
            _make_horse(3, 3, running_style='先行'),
            *[_make_horse(i, i, running_style='差し') for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 6, 0.05), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'escape_win' in result['tags']

    def test_escape_win_not_detected_if_ai_top(self):
        horses = [
            _make_horse(1, 1, running_style='逃げ'),
            _make_horse(2, 2), _make_horse(3, 3),
        ]
        preds = _make_predictions([(1, 1, 0.30), (2, 2, 0.20), (3, 3, 0.10)])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is None or 'escape_win' not in result.get('tags', [])


class TestClassifyDistChange:
    def test_dist_short_win(self):
        horses = [
            _make_horse(1, 1, history=[{'distance': 2000, 'jockey': 'A'}]),
            _make_horse(2, 2, history=[{'distance': 1600, 'jockey': 'B'}]),
            _make_horse(3, 3, history=[{'distance': 1600, 'jockey': 'C'}]),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 5, 0.08), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses, distance=1600)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'dist_short_win' in result['tags']
        assert result['details']['dist_short_win']['diff'] == -400

    def test_dist_ext_win(self):
        horses = [
            _make_horse(1, 1, history=[{'distance': 1400, 'jockey': 'A'}]),
            _make_horse(2, 2), _make_horse(3, 3),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 5, 0.08), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses, distance=1800)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'dist_ext_win' in result['tags']


class TestClassifyHeavyUpset:
    def test_heavy_upset(self):
        horses = [
            _make_horse(1, 1, popularity=12),
            _make_horse(2, 2, popularity=1),
            _make_horse(3, 3, popularity=3),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 8, 0.03), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses, track_condition='重')
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'heavy_upset' in result['tags']

    def test_no_heavy_on_good_track(self):
        horses = [
            _make_horse(1, 1, popularity=12),
            _make_horse(2, 2), _make_horse(3, 3),
        ]
        preds = _make_predictions([(1, 8, 0.03), (2, 1, 0.20), (3, 2, 0.15)])
        race = _make_race(horses, track_condition='良')
        result = classify_race_tags(race, preds)
        assert result is None or 'heavy_upset' not in result.get('tags', [])


class TestClassifyMareUpset:
    def test_mare_upset(self):
        horses = [
            _make_horse(1, 1, sex='牝'),
            _make_horse(2, 2, sex='牡'),
            _make_horse(3, 3, sex='牡'),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 7, 0.04), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'mare_upset' in result['tags']


class TestClassifyYoungUpset:
    def test_young_upset(self):
        horses = [
            _make_horse(1, 1, age=3),
            _make_horse(2, 2, age=5),
            _make_horse(3, 3, age=4),
            *[_make_horse(i, i, age=4) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 8, 0.03), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'young_upset' in result['tags']


class TestClassifyJockeySwitch:
    def test_jockey_switch_win(self):
        horses = [
            _make_horse(1, 1, jockey='新騎手',
                        history=[{'distance': 1600, 'jockey': '旧騎手'}]),
            _make_horse(2, 2), _make_horse(3, 3),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 5, 0.08), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'jockey_switch_win' in result['tags']


class TestClassifyClassMiss:
    def test_class_miss_promotion(self):
        horses = [
            _make_horse(1, 1, history=[{
                'distance': 1600, 'jockey': 'A', 'race_class': '2勝クラス',
            }]),
            _make_horse(2, 2), _make_horse(3, 3),
            *[_make_horse(i, i) for i in range(4, 10)],
        ]
        preds = _make_predictions([
            (1, 7, 0.04), (2, 1, 0.20), (3, 2, 0.15),
            *[(i, i, 0.05) for i in range(4, 10)],
        ])
        race = _make_race(horses, race_class='3勝クラス')
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'class_miss' in result['tags']


class TestClassifyPositionBias:
    def test_inner_bias(self):
        horses = [
            _make_horse(1, 1), _make_horse(2, 2), _make_horse(3, 3),
            *[_make_horse(i, i) for i in range(4, 13)],
        ]
        preds = _make_predictions(
            [(i, i, 0.08) for i in range(1, 13)]
        )
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is not None
        assert 'position_bias' in result['tags']
        assert result['details']['position_bias']['bias'] == 'inner'


# ── 蓄積・補正テスト ────────────────────────────────────────────────

class TestAccumulateAndCorrect:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            data = {
                'updated_at': '2026-07-13',
                'entries': [{'race_id': 'R1', 'tags': ['escape_win']}],
                'corrections': {},
            }
            save_error_tags(data, tmpdir)
            loaded = load_error_tags(tmpdir)
            assert len(loaded['entries']) == 1
            assert loaded['entries'][0]['tags'] == ['escape_win']

    def test_accumulate_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            entry = {
                'race_id': 'R1', 'date': '2026-07-13',
                'venue': '東京', 'surface': '芝', 'distance': 1600,
                'track_condition': '良', 'condition_key': '東京_芝_mile_良',
                'tags': ['escape_win'], 'details': {},
            }
            accumulate_tags(tmpdir, [entry])
            accumulate_tags(tmpdir, [entry])  # duplicate
            loaded = load_error_tags(tmpdir)
            assert len(loaded['entries']) == 1

    def test_corrections_need_min_samples(self):
        entries = [
            {
                'race_id': f'R{i}', 'condition_key': '東京_芝_mile_良',
                'tags': ['escape_win'], 'details': {},
            }
            for i in range(10)  # < MIN_SAMPLES_FOR_CORRECTION (20)
        ]
        corrections = _calc_corrections(entries)
        assert '東京_芝_mile_良' not in corrections

    def test_corrections_with_enough_samples(self):
        entries = [
            {
                'race_id': f'R{i}', 'condition_key': '東京_芝_mile_良',
                'tags': ['escape_win'], 'details': {},
            }
            for i in range(25)
        ]
        corrections = _calc_corrections(entries)
        assert '東京_芝_mile_良' in corrections
        entry = corrections['東京_芝_mile_良']
        assert entry['n'] == 25
        assert entry['factor'] >= 1.0


class TestGetCorrectionFactor:
    def test_no_data_returns_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            factor = get_correction_factor(tmpdir, '東京', '芝', 1600, '良')
            assert factor == 1.0

    def test_horse_specific_bonus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            data = {
                'updated_at': '2026-07-13',
                'entries': [],
                'corrections': {
                    '東京_芝_mile_良': {
                        'factor': 1.1,
                        'n': 30,
                        'top_tags': [{'tag': 'escape_win', 'count': 15}],
                        'active_adjustments': [
                            {'tag': 'escape_win', 'local_rate': 0.6,
                             'base_rate': 0.2},
                        ],
                    }
                },
            }
            save_error_tags(data, tmpdir)

            # 逃げ馬 → ボーナスあり
            horse_escape = {'running_style': '逃げ'}
            factor = get_correction_factor(
                tmpdir, '東京', '芝', 1600, '良', horse=horse_escape)
            assert factor > 1.1

            # 差し馬 → ボーナスなし
            horse_sashi = {'running_style': '差し'}
            factor2 = get_correction_factor(
                tmpdir, '東京', '芝', 1600, '良', horse=horse_sashi)
            assert factor2 == 1.1


class TestErrorTagFeatures:
    def test_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            feats = calc_error_tag_features(
                {}, tmpdir, '東京', '芝', 1600, '良')
            assert feats['f_et_correction'] == 1.0
            assert all(v == 0.0 for k, v in feats.items()
                       if k.startswith('f_et_') and k != 'f_et_correction')

    def test_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data')
            os.makedirs(data_dir)
            data = {
                'updated_at': '2026-07-13',
                'entries': [],
                'corrections': {
                    '東京_芝_mile_良': {
                        'factor': 1.15,
                        'n': 30,
                        'top_tags': [
                            {'tag': 'escape_win', 'count': 10},
                            {'tag': 'dist_short_win', 'count': 5},
                        ],
                        'active_adjustments': [],
                    }
                },
            }
            save_error_tags(data, tmpdir)
            feats = calc_error_tag_features(
                {}, tmpdir, '東京', '芝', 1600, '良')
            assert feats['f_et_correction'] == 1.15
            assert abs(feats['f_et_escape_win_rate'] - 10 / 30) < 0.001
            assert abs(feats['f_et_dist_short_win_rate'] - 5 / 30) < 0.001


class TestNoTagsReturnsNone:
    def test_all_predicted_correctly(self):
        horses = [
            _make_horse(1, 1), _make_horse(2, 2), _make_horse(3, 3),
        ]
        preds = _make_predictions([(1, 1, 0.30), (2, 2, 0.20), (3, 3, 0.15)])
        race = _make_race(horses)
        result = classify_race_tags(race, preds)
        assert result is None

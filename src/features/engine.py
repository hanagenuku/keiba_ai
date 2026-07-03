"""
特徴量エンジン v5 クリーン版
ノートブックのセル12から抽出。グローバル変数はinit_engine()で注入する。
"""
import math
import os
import pickle

from src.utils.config import POST_BIAS, POST_BIAS_BY_ZONE
from src.models.predict import softmax_probs, calibrate_and_renormalize
from src.models.calibration_xgb import load_xgb_calibrator

# ── グローバルコンテキスト（init_engine()で設定） ─────────────────
_XGB_FUKUSHO_MODEL  = None
_XGB_FEATURE_COLS   = None
_CALIBRATOR         = None
_XGB_CALIBRATOR     = None
_PACE_MODEL         = None
_SPEED_INDEX_CALC   = None  # SpeedIndexCalculator
_MEMBER_LEVEL_CACHE = {}    # race_id → float (前走メンバーレベルキャッシュ)
_KEIBA_DB_PATH      = None  # race_predictions 参照用
_W                 = {
    'rl':       0.35,   # Phase 2: スピード指数ベースRLスコア
    'distance': 0.20,
    'pace':     0.15,
    'maturity': 0.10,   # Phase 2: G1/重賞経験スコア
    'trainer':  0.08,
    'jockey':   0.04,
    'blood':    0.03,
    'post':     0.03,
    'bias':     0.02,
    'recent':   0.00,   # 後方互換（旧optimal_weights.jsonに存在する場合）
    'weight':   0.00,
    'rotation': 0.05,   # Phase 3: 前走メンバーレベル・ローテーション
}
_horse_venue_dist_dict = {}  # (name, venue, dist_zone, surface) → {出走, 勝率, 複勝率}
_post_zone_bias        = {}  # (venue, dist_zone) → float  正=内枠有利, 負=外枠有利
_jockey_dict           = {}  # (騎手名, 競馬場, surface) → 勝率
_trainer_dict          = {}  # 調教師名 → 勝率
_hist_db_path          = None  # Phase 3: DBパス（calc_prev_member_levelで使用）
_BASE_DIR              = None  # init_engine で設定。course_profiles等の遅延ロードに使用
_COURSE_PROFILES       = None  # data/course_profiles.json の内容（コース適性特徴量で使用）


def init_engine(base_dir,
                xgb_model=None, xgb_feature_cols=None,
                calibrator=None, pace_model=None,
                weights=None,
                horse_dist_dict=None, horse_course_dict=None,
                horse_venue_dist_dict=None):
    """エンジンのグローバル変数を設定する。ノートブックのセル4で呼ぶ"""
    global _XGB_FUKUSHO_MODEL, _XGB_FEATURE_COLS, _CALIBRATOR, _XGB_CALIBRATOR, _PACE_MODEL
    global _W, _horse_dist_dict, _horse_course_dict, _horse_venue_dist_dict, _post_zone_bias
    global _jockey_dict, _trainer_dict, _hist_db_path, _SPEED_INDEX_CALC, _MEMBER_LEVEL_CACHE
    global _KEIBA_DB_PATH, _BASE_DIR
    _BASE_DIR      = base_dir
    _hist_db_path  = os.path.join(base_dir, 'data', 'history.db')
    _KEIBA_DB_PATH = os.path.join(base_dir, 'data', 'keiba.db')

    # スピード指数キャッシュをロード（なければ history.db から自動構築）
    try:
        from src.features.speed_index import load_speed_index_calculator
        _SPEED_INDEX_CALC = load_speed_index_calculator(base_dir)
    except Exception:
        _SPEED_INDEX_CALC = None

    # 前走メンバーレベルキャッシュをロード（なければ history.db から自動構築）
    try:
        ml_path = os.path.join(base_dir, 'data', 'member_level_cache.pkl')
        if os.path.exists(ml_path):
            with open(ml_path, 'rb') as _f:
                _MEMBER_LEVEL_CACHE = pickle.load(_f)
            print(f'  📊 メンバーレベルキャッシュ: {len(_MEMBER_LEVEL_CACHE):,}件')
        elif os.path.exists(_hist_db_path):
            print('  🔨 メンバーレベルキャッシュ構築中...')
            _MEMBER_LEVEL_CACHE = build_member_level_cache(base_dir)
            print(f'  ✅ メンバーレベルキャッシュ完了: {len(_MEMBER_LEVEL_CACHE):,}件')
    except Exception as _e:
        _MEMBER_LEVEL_CACHE = {}
        print(f'  ⚠ メンバーレベルキャッシュ失敗: {_e}')

    if xgb_model is not None:
        _XGB_FUKUSHO_MODEL = xgb_model
    elif os.path.exists(f'{base_dir}/data/xgb_fukusho_model.pkl'):
        with open(f'{base_dir}/data/xgb_fukusho_model.pkl', 'rb') as f:
            _XGB_FUKUSHO_MODEL = pickle.load(f)

    if xgb_feature_cols is not None:
        _XGB_FEATURE_COLS = xgb_feature_cols
    elif os.path.exists(f'{base_dir}/data/xgb_feature_cols.json'):
        import json
        with open(f'{base_dir}/data/xgb_feature_cols.json', encoding='utf-8') as f:
            info = json.load(f)
            _XGB_FEATURE_COLS = info['feature_cols']

    if calibrator is not None:
        _CALIBRATOR = calibrator
    elif os.path.exists(f'{base_dir}/data/calibrator.pkl'):
        with open(f'{base_dir}/data/calibrator.pkl', 'rb') as f:
            _CALIBRATOR = pickle.load(f)

    # XGB専用キャリブレーター（calibrate_xgb.py で学習・保存）
    _XGB_CALIBRATOR = load_xgb_calibrator(base_dir)

    if pace_model is not None:
        _PACE_MODEL = pace_model
    elif os.path.exists(f'{base_dir}/data/pace_model.pkl'):
        with open(f'{base_dir}/data/pace_model.pkl', 'rb') as f:
            _PACE_MODEL = pickle.load(f)

    if weights is not None:
        _W = weights
    elif os.path.exists(f'{base_dir}/data/optimal_weights.json'):
        import json
        with open(f'{base_dir}/data/optimal_weights.json') as f:
            opt = json.load(f)
    if weights is not None:
        _W = weights
    elif os.path.exists(f'{base_dir}/data/optimal_weights.json'):
        import json
        with open(f'{base_dir}/data/optimal_weights.json') as f:
            opt = json.load(f)
        w_keys = ['pace', 'recent', 'rl', 'maturity', 'jockey', 'trainer',
                  'blood', 'distance', 'post', 'bias', 'weight', 'rotation']
        # Phase 2 のデフォルト値（optimal_weights.json に未登録のキーに適用）
        ph2_defaults = {
            'rl': 0.35, 'maturity': 0.10, 'distance': 0.20, 'pace': 0.15,
            'trainer': 0.08, 'jockey': 0.04, 'blood': 0.03, 'post': 0.03,
            'bias': 0.02, 'recent': 0.00, 'weight': 0.00, 'rotation': 0.05,
        }
        new_w = {k: opt.get(k, ph2_defaults.get(k, 0)) for k in w_keys}
        ws = sum(new_w.values())
        if ws > 0.05:
            new_w = {k: round(v / ws, 4) for k, v in new_w.items()}
            _W = new_w


    if horse_dist_dict is not None:
        _horse_dist_dict = horse_dist_dict
    elif os.path.exists(f'{base_dir}/data/horse_dist_dict.pkl'):
        with open(f'{base_dir}/data/horse_dist_dict.pkl', 'rb') as f:
            _horse_dist_dict = pickle.load(f)

    if horse_course_dict is not None:
        _horse_course_dict = horse_course_dict
    elif os.path.exists(f'{base_dir}/data/horse_course_dict.pkl'):
        with open(f'{base_dir}/data/horse_course_dict.pkl', 'rb') as f:
            _horse_course_dict = pickle.load(f)

    if horse_venue_dist_dict is not None:
        _horse_venue_dist_dict = horse_venue_dist_dict
    elif os.path.exists(f'{base_dir}/data/horse_venue_dist_dict.pkl'):
        with open(f'{base_dir}/data/horse_venue_dist_dict.pkl', 'rb') as f:
            _horse_venue_dist_dict = pickle.load(f)

    if os.path.exists(f'{base_dir}/data/post_zone_bias.pkl'):
        with open(f'{base_dir}/data/post_zone_bias.pkl', 'rb') as f:
            _post_zone_bias = pickle.load(f)

    import csv as _csv

    def _norm(s):
        return (s or '').replace(' ', '').replace('　', '')

    jockey_csv = os.path.join(base_dir, 'data', 'jockey_db.csv')
    if os.path.exists(jockey_csv):
        try:
            with open(jockey_csv, encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    key = (_norm(row.get('騎手', '')), row.get('競馬場', ''), row.get('surface', ''))
                    try:
                        _jockey_dict[key] = float(row['勝率'])
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass

    trainer_csv = os.path.join(base_dir, 'data', 'trainer_db.csv')
    if os.path.exists(trainer_csv):
        try:
            with open(trainer_csv, encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    try:
                        _trainer_dict[_norm(row['調教師'])] = float(row['勝率'])
                    except (KeyError, ValueError):
                        pass
        except Exception:
            pass

    # pkl未作成 or スペース正規化前の旧pkl → DBから再構築
    jk_pkl = os.path.join(base_dir, 'data', 'jockey_stats_dict.pkl')
    if os.path.exists(jk_pkl):
        try:
            with open(jk_pkl, 'rb') as _f:
                _jk_test = pickle.load(_f)
            # キーにスペースが含まれていれば旧データ（正規化前）→ 再構築
            if any(' ' in k[0] or '　' in k[0]
                   for k in _jk_test if isinstance(k, tuple) and k and k[0]):
                os.remove(jk_pkl)
        except Exception:
            pass
    if (not _horse_dist_dict or not _horse_venue_dist_dict or not _post_zone_bias
            or not os.path.exists(jk_pkl)):
        _build_horse_dicts(base_dir)
    else:
        # pklが存在する場合は読み込んでCSV未登録分を補完
        tr_pkl = os.path.join(base_dir, 'data', 'trainer_stats_dict.pkl')
        try:
            with open(jk_pkl, 'rb') as f:
                jk_from_pkl = pickle.load(f)
            for k, v in jk_from_pkl.items():
                if k not in _jockey_dict:
                    _jockey_dict[k] = v
        except Exception:
            pass
        try:
            with open(tr_pkl, 'rb') as f:
                tr_from_pkl = pickle.load(f)
            for k, v in tr_from_pkl.items():
                if k not in _trainer_dict:
                    _trainer_dict[k] = v
        except Exception:
            pass

    xgb_ok = _XGB_FUKUSHO_MODEL is not None
    print(f'✅ 特徴量エンジン初期化完了 (XGB:{xgb_ok}, Cal:{_CALIBRATOR is not None}, '
          f'XGB_Cal:{_XGB_CALIBRATOR is not None}, '
          f'馬別統計:{len(_horse_dist_dict)}件, 枠バイアス:{len(_post_zone_bias)}件, '
          f'騎手:{len(_jockey_dict)}件, 調教師:{len(_trainer_dict)}件)')


def _build_horse_dicts(base_dir):
    """history.db / keiba.db から馬別統計・枠順バイアスを構築してpklに保存する。"""
    import sqlite3
    from collections import defaultdict

    global _horse_dist_dict, _horse_course_dict, _horse_venue_dist_dict, _post_zone_bias

    db_path = None
    for name in ['history.db', 'keiba.db']:
        p = os.path.join(base_dir, 'data', name)
        if os.path.exists(p):
            db_path = p
            break
    if not db_path:
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if 'horse_history' not in tables:
        conn.close()
        return

    rows = conn.execute("""
        SELECT hh.horse_name, hh.distance, hh.surface, hh.racecourse,
               hh.place, hh.horse_num,
               (SELECT COUNT(*) FROM horse_history hh2
                WHERE hh2.race_id = hh.race_id) AS finishers
        FROM horse_history hh
        WHERE hh.place IS NOT NULL AND hh.distance IS NOT NULL
    """).fetchall()

    rows_jk = conn.execute("""
        SELECT jockey, trainer, racecourse, surface, place
        FROM horse_history
        WHERE place IS NOT NULL
    """).fetchall()
    conn.close()

    dist_stat   = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    course_stat  = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    vd_stat      = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    post_stat    = defaultdict(lambda: {'i_w': 0, 'i_n': 0, 'o_w': 0, 'o_n': 0})
    jockey_stat  = defaultdict(lambda: {'runs': 0, 'wins': 0})
    trainer_stat = defaultdict(lambda: {'runs': 0, 'wins': 0})

    for r in rows:
        try:
            name     = r['horse_name'] or ''
            dist     = int(r['distance'] or 1600)
            surf     = r['surface'] or '芝'
            rc       = r['racecourse'] or ''
            place    = int(r['place'] or 99)
            hnum     = int(r['horse_num'] or 8)
            zone     = dist_zone_label(dist)
            is_win   = 1 if place == 1 else 0
            is_fuku  = 1 if place <= 3 else 0

            dist_stat[(name, zone)]['出走']  += 1
            dist_stat[(name, zone)]['1着']   += is_win
            dist_stat[(name, zone)]['複勝']  += is_fuku

            course_stat[(name, rc, surf)]['出走'] += 1
            course_stat[(name, rc, surf)]['1着']  += is_win
            course_stat[(name, rc, surf)]['複勝'] += is_fuku

            vd_stat[(name, rc, zone, surf)]['出走'] += 1
            vd_stat[(name, rc, zone, surf)]['1着']  += is_win
            vd_stat[(name, rc, zone, surf)]['複勝'] += is_fuku

            pk = (rc, zone)
            if hnum <= 4:
                post_stat[pk]['i_n'] += 1
                post_stat[pk]['i_w'] += is_win
            else:
                post_stat[pk]['o_n'] += 1
                post_stat[pk]['o_w'] += is_win
        except Exception:
            pass

    for r in rows_jk:
        try:
            # スペース除去で正規化（出馬表と結果ページでスペース有無が異なるため）
            jn   = (r['jockey']  or '').replace(' ', '').replace('　', '')
            tr   = (r['trainer'] or '').replace(' ', '').replace('　', '')
            rc   = r['racecourse'] or ''
            surf = r['surface'] or '芝'
            place = int(r['place'] or 99)
            is_win = 1 if place == 1 else 0
            if jn:
                jockey_stat[(jn, rc, surf)]['runs'] += 1
                jockey_stat[(jn, rc, surf)]['wins'] += is_win
                jockey_stat[(jn, '', '')]['runs']   += 1
                jockey_stat[(jn, '', '')]['wins']   += is_win
            if tr:
                trainer_stat[tr]['runs'] += 1
                trainer_stat[tr]['wins'] += is_win
        except Exception:
            pass

    def _make_dict(stat):
        out = {}
        for k, v in stat.items():
            if v['出走'] > 0:
                out[k] = {
                    '出走':  v['出走'],
                    '勝率':  round(v['1着'] / v['出走'], 4),
                    '複勝率': round(v['複勝'] / v['出走'], 4),
                }
        return out

    new_jockey_dict  = {k: round(v['wins'] / v['runs'], 4)
                        for k, v in jockey_stat.items() if v['runs'] >= 10}
    new_trainer_dict = {k: round(v['wins'] / v['runs'], 4)
                        for k, v in trainer_stat.items() if v['runs'] >= 10}

    new_dist   = _make_dict(dist_stat)
    new_course = _make_dict(course_stat)
    new_vd     = _make_dict(vd_stat)

    new_post_bias = {}
    for (rc, zone), v in post_stat.items():
        if v['i_n'] >= 10 and v['o_n'] >= 10:
            i_rate = v['i_w'] / v['i_n']
            o_rate = v['o_w'] / v['o_n']
            new_post_bias[(rc, zone)] = round((i_rate - o_rate) * 20, 3)

    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    for fname, obj in [
        ('horse_dist_dict.pkl',       new_dist),
        ('horse_course_dict.pkl',     new_course),
        ('horse_venue_dist_dict.pkl', new_vd),
        ('post_zone_bias.pkl',        new_post_bias),
        ('jockey_stats_dict.pkl',     new_jockey_dict),
        ('trainer_stats_dict.pkl',    new_trainer_dict),
    ]:
        with open(os.path.join(data_dir, fname), 'wb') as f:
            pickle.dump(obj, f)

    _horse_dist_dict       = new_dist
    _horse_course_dict     = new_course
    _horse_venue_dist_dict = new_vd
    _post_zone_bias        = new_post_bias
    # CSV未カバー騎手・調教師を補完（CSV登録済みはそのまま優先）
    global _jockey_dict, _trainer_dict, _hist_db_path
    for k, v in new_jockey_dict.items():
        if k not in _jockey_dict:
            _jockey_dict[k] = v
    for k, v in new_trainer_dict.items():
        if k not in _trainer_dict:
            _trainer_dict[k] = v
    print(f'  [統計構築] 距離:{len(new_dist)}件 コース:{len(new_course)}件 '
          f'距離×会場:{len(new_vd)}件 枠バイアス:{len(new_post_bias)}件 '
          f'騎手:{len(new_jockey_dict)}件 調教師:{len(new_trainer_dict)}件')


# ── 血統DB ────────────────────────────────────────────────────
SIRE_DB = {
    # ── 芝・短距離〜マイル系 ──────────────────────────────────
    'ロードカナロア':    {'peak': 2.5, 'type': 'early',    'dist': {'sp': 1.0, 'mi': 0.9, 'md': 0.6, 'lo': 0.3},  'surf': {'芝': 1.0, 'ダート': 0.6}},
    'ダイワメジャー':    {'peak': 2.5, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.7, 'lo': 0.4},  'surf': {'芝': 1.0, 'ダート': 0.7}},
    'モーリス':          {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.7, 'mi': 1.0, 'md': 0.85,'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'アドマイヤマーズ':  {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.8, 'mi': 1.0, 'md': 0.8, 'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'シルバーステート':  {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.75,'lo': 0.4},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'イスラボニータ':    {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.85,'mi': 1.0, 'md': 0.8, 'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ミッキーアイル':    {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.7, 'lo': 0.3},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'サリオス':          {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.85,'mi': 1.0, 'md': 0.8, 'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'グランアレグリア':  {'peak': 3.0, 'type': 'early',    'dist': {'sp': 1.0, 'mi': 0.9, 'md': 0.6, 'lo': 0.3},  'surf': {'芝': 1.0, 'ダート': 0.3}},
    'アグネスタキオン':  {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.8, 'mi': 1.0, 'md': 0.8, 'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.6}},
    'フジキセキ':        {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.75,'lo': 0.4},  'surf': {'芝': 1.0, 'ダート': 0.6}},
    'マルターズアポジー':{'peak': 3.5, 'type': 'early',    'dist': {'sp': 0.7, 'mi': 1.0, 'md': 0.8, 'lo': 0.5},  'surf': {'芝': 1.0, 'ダート': 0.6}},
    # ── 芝・マイル〜中距離系 ─────────────────────────────────
    'キズナ':            {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.4}},
    'ディープインパクト':{'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 1.0, 'ダート': 0.3}},
    'キングカメハメハ':  {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 1.0, 'ダート': 0.75}},
    'ドゥラメンテ':      {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.85,'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.5}},
    'サートゥルナーリア':{'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'リアルスティール':  {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 1.0, 'md': 0.9, 'lo': 0.7},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'コントレイル':      {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.85,'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.4}},
    'レイデオロ':        {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ジャスタウェイ':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 1.0, 'md': 0.9, 'lo': 0.7},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ルーラーシップ':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 1.0, 'ダート': 0.6}},
    'ヴィクトワールピサ':{'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 0.8, 'ダート': 0.9}},
    'スワーヴリチャード':{'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.4, 'mi': 0.7, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'エイシンフラッシュ':{'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ネオユニヴァース':  {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 0.9, 'ダート': 0.7}},
    'シンボリクリスエス':{'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.85}, 'surf': {'芝': 1.0, 'ダート': 0.6}},
    'タニノギムレット':  {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 1.0, 'md': 0.9, 'lo': 0.6},  'surf': {'芝': 0.9, 'ダート': 0.7}},
    'スクリーンヒーロー':{'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.8, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'デクラレーションオブウォー': {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 1.0, 'md': 0.9, 'lo': 0.6}, 'surf': {'芝': 1.0, 'ダート': 0.6}},
    # ── 芝・中距離〜長距離系 ─────────────────────────────────
    'エピファネイア':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.4, 'mi': 0.7, 'md': 1.0, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ハーツクライ':      {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.95,'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ハービンジャー':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'ブラックタイド':    {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.95,'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'キタサンブラック':  {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.4, 'mi': 0.7, 'md': 1.0, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ゴールドシップ':    {'peak': 5.0, 'type': 'late',     'dist': {'sp': 0.2, 'mi': 0.5, 'md': 0.8, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'オルフェーヴル':    {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.4, 'mi': 0.7, 'md': 0.95,'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    'ステイゴールド':    {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'マンハッタンカフェ':{'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'スペシャルウィーク':{'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'サトノダイヤモンド':{'peak': 4.0, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 1.0, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'タイトルホルダー':  {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'フェノーメノ':      {'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.95,'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'トーセンジョーダン':{'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.3, 'mi': 0.6, 'md': 0.9, 'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.4}},
    'ノヴェリスト':      {'peak': 5.0, 'type': 'late',     'dist': {'sp': 0.2, 'mi': 0.5, 'md': 0.85,'lo': 1.0},  'surf': {'芝': 1.0, 'ダート': 0.3}},
    'バゴ':              {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.4, 'mi': 0.7, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 1.0, 'ダート': 0.5}},
    # ── ダート主体系 ──────────────────────────────────────────
    'ヘニーヒューズ':    {'peak': 3.0, 'type': 'early',    'dist': {'sp': 1.0, 'mi': 0.9, 'md': 0.6, 'lo': 0.3},  'surf': {'芝': 0.4, 'ダート': 1.0}},
    'ルヴァンスレーヴ':  {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 0.9, 'md': 1.0, 'lo': 0.7},  'surf': {'芝': 0.3, 'ダート': 1.0}},
    'ゴールドアリュール':{'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 0.3, 'ダート': 1.0}},
    'パイロ':            {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.7, 'lo': 0.3},  'surf': {'芝': 0.4, 'ダート': 1.0}},
    'シニスターミニスター':{'peak': 3.5, 'type': 'standard','dist': {'sp': 0.8, 'mi': 1.0, 'md': 0.8, 'lo': 0.5}, 'surf': {'芝': 0.3, 'ダート': 1.0}},
    'サウスヴィグラス':  {'peak': 3.5, 'type': 'early',    'dist': {'sp': 1.0, 'mi': 0.8, 'md': 0.5, 'lo': 0.2},  'surf': {'芝': 0.5, 'ダート': 1.0}},
    'スマートファルコン':{'peak': 4.5, 'type': 'late',     'dist': {'sp': 0.5, 'mi': 0.9, 'md': 1.0, 'lo': 0.9},  'surf': {'芝': 0.2, 'ダート': 1.0}},
    'エスポワールシチー':{'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.7, 'mi': 1.0, 'md': 0.9, 'lo': 0.7},  'surf': {'芝': 0.3, 'ダート': 1.0}},
    'カジノドライブ':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.5, 'mi': 0.9, 'md': 1.0, 'lo': 0.8},  'surf': {'芝': 0.3, 'ダート': 1.0}},
    'コパノリッキー':    {'peak': 4.0, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 1.0, 'md': 0.9, 'lo': 0.7},  'surf': {'芝': 0.2, 'ダート': 1.0}},
    'クロフネ':          {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.8, 'mi': 1.0, 'md': 0.85,'lo': 0.6},  'surf': {'芝': 0.7, 'ダート': 1.0}},
    'ナダル':            {'peak': 3.0, 'type': 'early',    'dist': {'sp': 0.9, 'mi': 1.0, 'md': 0.75,'lo': 0.4},  'surf': {'芝': 0.6, 'ダート': 1.0}},
}
DEF_SIRE = {'peak': 3.5, 'type': 'standard', 'dist': {'sp': 0.6, 'mi': 0.8, 'md': 1.0, 'lo': 0.8}, 'surf': {'芝': 0.8, 'ダート': 0.6}}

CLASS_RANK = {
    '新馬': 0, '未勝利': 1, '1勝クラス': 2, '2勝クラス': 3, '3勝クラス': 4,
    'オープン': 5, '重賞': 6, 'G3': 6, 'G2': 7, 'G1': 8,
}
CLASS_RANK_NAME = {
    0: '新馬', 1: '未勝利', 2: '1勝クラス', 3: '2勝クラス', 4: '3勝クラス',
    5: 'オープン', 6: '重賞/G3', 7: 'G2', 8: 'G1',
}


def dz(d):
    return 'sp' if int(d) <= 1400 else 'mi' if int(d) <= 1800 else 'md' if int(d) <= 2200 else 'lo'


def dist_zone_label(d):
    if d is None:
        return None
    d = int(d)
    if d <= 1400: return '短距離'
    if d <= 1800: return 'マイル'
    if d <= 2200: return '中距離'
    return '長距離'


def get_race_class_rank(text):
    for k, v in [('G1', 8), ('G2', 7), ('G3', 6), ('重賞', 6), ('オープン', 5),
                 ('3勝', 4), ('2勝', 3), ('1勝', 2), ('未勝利', 1), ('新馬', 0)]:
        if k in text:
            return v
    return 3


# ── ペース調整済みパフォーマンス指数 ─────────────────────────
_PI_SURF_BASE = {
    '芝':    {'良': 34.0, '稍重': 34.3, '重': 34.7, '不良': 35.2},
    'ダート': {'良': 35.5, '稍重': 35.3, '重': 35.1, '不良': 35.0},
}
_PI_FIRST_BASE = {
    1200: 34.2, 1400: 34.5, 1600: 34.8, 1800: 35.2,
    2000: 35.5, 2200: 35.8, 2400: 36.0, 2500: 36.2,
}


def calc_performance_index(last_3f, first_3f=0, corner_pos=8,
                            finishers=16, distance=1600,
                            surface='芝', condition='良'):
    """ペース調整済みパフォーマンス指数（0〜100）"""
    if not last_3f or float(last_3f) <= 0:
        return 50.0
    last_3f = float(last_3f)
    dist = int(distance)
    surf_base = _PI_SURF_BASE.get(surface, _PI_SURF_BASE['芝'])
    base_last = surf_base.get(condition, surf_base['良'])
    dist_adj = (dist - 1600) / 100 * 0.15
    base_last += dist_adj
    pace_burden = 0.0
    if first_3f and float(first_3f) > 0:
        first_3f = float(first_3f)
        dist_keys = sorted(_PI_FIRST_BASE.keys())
        base_first = _PI_FIRST_BASE.get(min(dist_keys, key=lambda k: abs(k - dist)), 34.8)
        pace_burden = (base_first - first_3f) * 0.4
    position_adj = 0.0
    if corner_pos and int(corner_pos) > 0:
        cp = int(corner_pos)
        n = max(finishers, 8)
        mid = n / 2
        position_adj = (mid - cp) / n * 0.5
    adjusted = last_3f - pace_burden - position_adj
    index = 50.0 + (base_last - adjusted) * 10
    return round(max(0.0, min(100.0, index)), 1)


# ── Phase 2: クラス別基準上がりタイム ────────────────────────────────
CLASS_BASE_AGARI = {
    'G1':    33.5,
    'G2':    33.8,
    'G3':    34.0,
    'L':     34.2,
    'OP':    34.3,
    'オープン': 34.3,
    '3勝':   34.5,
    '2勝':   34.8,
    '1勝':   35.2,
    '未勝利': 35.8,
    '新馬':  36.0,
}
TRACK_CONDITION_ADJUST = {'良': 0.0, '稍重': +0.3, '重': +0.6, '不良': +1.0}



def calc_competitiveness(history):
    """
    過去走の着差÷着順の平均。値が小さいほど「着順の割に接戦」。

    例:
      8着で0.5秒差 → 0.5/8 = 0.0625（実力は上位に近い）
      2着で1.0秒差 → 1.0/2 = 0.50（着順ほど強くない）

    Parameters
    ----------
    history : list of dict
        過去走データ。各要素に 'place', 'time_diff_sec' を含む

    Returns
    -------
    f_competitiveness : float  着差/着順の平均（直近5走）
    f_competitive_best : float  着差/着順の最小値（最も接戦だった走）
    """
    vals = []
    for r in history[:5]:  # 直近5走
        place = r.get('place', 99)
        diff = r.get('time_diff_sec')
        if place is None or place <= 0 or place >= 99:
            continue
        if diff is None or diff < 0:
            continue
        if place == 1:
            vals.append(0.0)  # 1着は着差0
        else:
            vals.append(diff / place)

    if not vals:
        return 0.5, 0.5  # デフォルト（中間値）

    return round(sum(vals) / len(vals), 4), round(min(vals), 4)


def calc_race_content_score(r):
    """1走分の内容スコア（着順ではなく中身を評価）"""
    place      = r.get('place', 10)
    finishers  = max(r.get('finishers', 16), 2)
    margin     = r.get('margin', 0.0)
    agari_rank = r.get('agari_rank', finishers)
    race_class = r.get('class', r.get('race_class', '1勝'))

    # 1. 相対着順スコア
    pos_score = max(0, 10 * (1 - (place - 1) / max(finishers - 1, 1)))

    # 2. 接戦ボーナス（0.3秒以内の負けは評価下げない）
    if place > 1 and 0 < margin <= 0.3:
        pos_score = min(10, pos_score + 1.5)

    # 3. 上がり順位ボーナス（最重要）
    agari_pct   = (finishers - agari_rank) / max(finishers - 1, 1)
    agari_bonus = agari_pct * 3.0  # 最速で+3.0

    # 4. クラス格係数
    class_mult = {
        'G1': 2.0, 'G2': 1.6, 'G3': 1.4, 'L': 1.2, 'OP': 1.1, 'オープン': 1.1,
        '3勝': 1.0, '2勝': 0.85, '1勝': 0.7, '未勝利': 0.5, '新馬': 0.5,
    }.get(race_class, 1.0)

    return min(10, (pos_score + agari_bonus) * class_mult * 0.45 + 2.0)


def f_rl(h, race):
    """スピード指数・上がり実績ベースのRLスコア (0-10)"""
    hist = h.get('history', [])
    if not hist:
        odds = h.get('win_odds')
        if odds and 1.0 <= odds < 50.0:
            return max(1.0, min(8.0, 10.0 - math.log(odds, 2)))
        return 5.0

    scores = []
    for i, r in enumerate(hist[:5]):
        last_3f    = r.get('last_3f') or r.get('agari3f') or 0
        race_class = r.get('class', r.get('race_class', '1勝クラス'))
        agari_rank = r.get('agari_rank', 9)
        num_fin    = max(r.get('finishers', 16), 8)
        track_cond = r.get('condition', r.get('track_condition', '良'))

        base = CLASS_BASE_AGARI.get(race_class, 35.0)
        base += TRACK_CONDITION_ADJUST.get(track_cond, 0)

        if last_3f > 0:
            speed_idx = (base - last_3f) * 10 + 50
        else:
            speed_idx = 50

        agari_pct  = (num_fin - agari_rank) / max(num_fin - 1, 1)
        agari_bonus = agari_pct * 15

        class_mult = {'G1': 1.5, 'G2': 1.3, 'G3': 1.2, 'L': 1.1, 'OP': 1.05,
                      'オープン': 1.05}.get(race_class, 1.0)

        raw    = (speed_idx + agari_bonus) * class_mult
        weight = 0.75 ** i
        scores.append((raw, weight))

    total_w      = sum(w for _, w in scores)
    weighted_avg = sum(s * w for s, w in scores) / total_w

    return max(0, min(10, (weighted_avg - 40) / 5))


def f_maturity(h, race):
    """G1・重賞・OP経験による完成度スコア (0-10)"""
    hist = h.get('history', [])

    class_points = {'G1': 5, 'G2': 3, 'G3': 2, 'L': 1.5, 'OP': 1, 'オープン': 1}
    place_mult   = {1: 2.0, 2: 1.5, 3: 1.2}

    total = 0
    for r in hist:
        rc = r.get('class', r.get('race_class', ''))
        pt = class_points.get(rc, 0)
        if pt > 0:
            p     = r.get('place', 9)
            total += pt * place_mult.get(p, 1.0 if p <= 5 else 0.5)

    return min(10, total / 1.5)



# ── Phase 3: ローテーション・メンバーレベル ───────────────────────────
import sqlite3 as _sqlite3

PREP_RACE_PROFILES = {
    'オークス': {
        '桜花賞':        {'level': 5, 'dist_match': 0.3},
        'フローラS':     {'level': 4, 'dist_match': 0.9},
        'フローラステークス': {'level': 4, 'dist_match': 0.9},
        '忘れな草賞':    {'level': 3, 'dist_match': 0.95},
        'スイートピーS': {'level': 3, 'dist_match': 0.8},
        'スイートピーステークス': {'level': 3, 'dist_match': 0.8},
    },
    '日本ダービー': {
        '皐月賞':    {'level': 5, 'dist_match': 0.6},
        '青葉賞':    {'level': 4, 'dist_match': 1.0},
        '京都新聞杯':{'level': 3, 'dist_match': 0.9},
    },
    '天皇賞（春）': {
        '阪神大賞典':    {'level': 4, 'dist_match': 0.85},
        '日経賞':        {'level': 4, 'dist_match': 0.80},
        '天皇賞（秋）':  {'level': 5, 'dist_match': 0.50},
    },
    '有馬記念': {
        'ジャパンカップ': {'level': 5, 'dist_match': 0.80},
        '天皇賞（秋）':   {'level': 5, 'dist_match': 0.75},
        '菊花賞':         {'level': 4, 'dist_match': 0.70},
    },
}


def build_member_level_cache(base_dir, cutoff_date=None):
    """対戦相手のその後の成績から前走メンバーレベルキャッシュを構築して保存する。

    各レースのスコア = (対戦相手の その後の勝利数×3 + 複勝数) / その後のレース数
    → 0-10スケールに変換（×2.5）し、高い = 強い相手と戦った実績

    計算はすべてメモリ上で行いDBクエリを最小化する。

    Parameters
    ----------
    base_dir    : プロジェクトルート
    cutoff_date : str or None
        データリーク防止用カットオフ日付（'YYYY-MM-DD' 形式）。
        指定すると、各レースのメンバーレベル計算に使う「その後の成績」を
        この日付より前に限定する（学習データ構築時に使用）。
        None（デフォルト）のとき全データを使用（推論時・運用時）。
    """
    import sqlite3 as _sq
    from collections import defaultdict

    db_path    = os.path.join(base_dir, 'data', 'history.db')
    cache_path = os.path.join(base_dir, 'data', 'member_level_cache.pkl')

    conn = _sq.connect(db_path)
    rows = conn.execute(
        'SELECT race_id, date, horse_name, place FROM horse_history '
        'WHERE place IS NOT NULL AND place < 99 ORDER BY date'
    ).fetchall()
    conn.close()

    # cutoff_date を文字列に正規化（比較はISO文字列で行う）
    cutoff_str = None
    if cutoff_date is not None:
        cutoff_str = str(cutoff_date).replace('-', '')[:8]
        # 'YYYYMMDD' → 'YYYY-MM-DD' 形式に正規化
        if len(cutoff_str) == 8:
            cutoff_str = f'{cutoff_str[:4]}-{cutoff_str[4:6]}-{cutoff_str[6:8]}'

    # horse → [(date, race_id, place)] の時系列辞書
    horse_timeline = defaultdict(list)
    for race_id, date, horse_name, place in rows:
        if horse_name:
            horse_timeline[horse_name].append((str(date), race_id, int(place)))
    # 日付昇順に整列（DBからORDER BY dateで来るが念のため）
    for name in horse_timeline:
        horse_timeline[name].sort(key=lambda x: x[0])

    # race_id → [(horse_name, date, place)]
    race_horses = defaultdict(list)
    for race_id, date, horse_name, place in rows:
        if horse_name:
            race_horses[race_id].append((horse_name, str(date), int(place)))

    cache = {}
    for race_id, horses_in_race in race_horses.items():
        if len(horses_in_race) < 5:
            cache[race_id] = 5.0
            continue

        race_date = horses_in_race[0][1]
        opp_scores = []

        for horse_name, _, _ in horses_in_race:
            timeline = horse_timeline.get(horse_name, [])
            # この日付より後のレースを最大5件（cutoff_date が指定された場合はその日付未満に限定）
            if cutoff_str is not None:
                subsequent = [(p, rid) for d, rid, p in timeline
                              if d > race_date and d < cutoff_str][:5]
            else:
                subsequent = [(p, rid) for d, rid, p in timeline if d > race_date][:5]
            if not subsequent:
                continue
            wins = sum(1 for p, _ in subsequent if p == 1)
            top3 = sum(1 for p, _ in subsequent if p <= 3)
            raw_score = (wins * 3 + top3) / len(subsequent)  # 0-4 range
            opp_scores.append(raw_score)

        if opp_scores:
            avg_raw = sum(opp_scores) / len(opp_scores)
            cache[race_id] = round(min(10.0, avg_raw * 2.5), 3)
        else:
            cache[race_id] = 5.0

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    return cache


def calc_prev_member_level(prev_race_id):
    """前走のメンバーレベルを算出（同レース出走馬の上位3頭上がり平均）"""
    if not prev_race_id or not _hist_db_path:
        return 5.0
    try:
        conn = _sqlite3.connect(_hist_db_path)
        rows = conn.execute(
            'SELECT place, agari3f FROM horse_history WHERE race_id=?',
            (prev_race_id,)
        ).fetchall()
        conn.close()
        if len(rows) < 5:
            return 5.0
        top3_agari = [r[1] for r in sorted(rows, key=lambda x: x[0])[:3] if r[1] and r[1] > 0]
        if not top3_agari:
            return 5.0
        avg_top3 = sum(top3_agari) / len(top3_agari)
        # 33秒台=高レベル(10)、36秒台=低レベル(3)
        return max(3.0, min(10.0, (36.5 - avg_top3) * 2.5 + 3.0))
    except Exception:
        return 5.0


def f_rotation(h, race):
    """前走メンバーレベル・ローテーション適性スコア (0-10)"""
    hist = h.get('history', [])
    if not hist:
        return 5.0

    prev_race_id   = hist[0].get('race_id')
    member_level   = calc_prev_member_level(prev_race_id)

    # ローテーションボーナス（PREP_RACE_PROFILESに合致する場合）
    curr_race_name = race.get('race_name', '')
    prev_race_name = hist[0].get('race_name', '')
    rot_bonus = 0.0
    if curr_race_name and prev_race_name:
        profile = PREP_RACE_PROFILES.get(curr_race_name, {})
        match = profile.get(prev_race_name)
        if match:
            rot_bonus = (match['level'] / 5.0 * 4.0) * match.get('dist_match', 0.8)

    return min(10.0, member_level * 0.7 + rot_bonus * 0.3 + 1.5)

def f_recent(h, race):
    hist = h.get('history', [])
    if not hist:
        odds = h.get('win_odds')
        if odds and 1.0 <= odds < 50.0:
            return max(1.0, min(8.0, 10.0 - math.log(odds, 2)))
        age = h.get('age', 4)
        wl  = h.get('weight_load', 56.0) or 56.0
        return max(4.0, min(6.5, 5.5 - (age - 4) * 0.1 + (56.0 - wl) * 0.05))
    ws = [.75 ** i for i in range(len(hist))]
    tw = sum(ws)
    sc = 0.0
    for i, r in enumerate(hist):
        sc += (ws[i] / tw) * calc_race_content_score(r)
    return max(0, min(10, sc))



# 距離帯別ペース×脚質スコア（短距離は逃げ優位が小さく、長距離は大きい）
PACE_STYLE_SCORE = {
    '短距離': {
        'high': {'逃げ': -1, '先行':  0, '差し': +2, '追込': +3},
        'mid':  {'逃げ': +2, '先行': +3, '差し': +1, '追込':  0},
        'slow': {'逃げ': +4, '先行': +3, '差し': -1, '追込': -2},
    },
    'マイル': {
        'high': {'逃げ': -2, '先行': -1, '差し': +3, '追込': +4},
        'mid':  {'逃げ': +1, '先行': +2, '差し': +1, '追込':  0},
        'slow': {'逃げ': +4, '先行': +3, '差し': -1, '追込': -2},
    },
    '中距離': {
        'high': {'逃げ': -3, '先行': -2, '差し': +3, '追込': +4},
        'mid':  {'逃げ': +1, '先行': +2, '差し': +1, '追込':  0},
        'slow': {'逃げ': +3, '先行': +3, '差し':  0, '追込': -1},
    },
    '長距離': {
        'high': {'逃げ': -4, '先行': -3, '差し': +3, '追込': +5},
        'mid':  {'逃げ':  0, '先行': +2, '差し': +2, '追込': +1},
        'slow': {'逃げ':  0, '先行': +3, '差し': +1, '追込':  0},
    },
}

# コース別ペース傾向（正=先行有利傾向=スローペースになりやすい、負=差し有利傾向）
VENUE_PACE_TENDENCY = {
    '中山': {'短距離': +0.05, 'マイル': +0.10, '中距離': +0.15, '長距離': +0.20},
    '小倉': {'短距離': +0.08, 'マイル': +0.10, '中距離': +0.12, '長距離': +0.08},
    '阪神': {'短距離': +0.05, 'マイル': +0.05, '中距離': +0.08, '長距離': +0.03},
    '福島': {'短距離': +0.05, 'マイル': +0.08, '中距離': +0.10, '長距離': +0.05},
    '函館': {'短距離': +0.05, 'マイル': +0.08, '中距離': +0.08, '長距離': +0.05},
    '札幌': {'短距離': +0.03, 'マイル': +0.05, '中距離': +0.05, '長距離': +0.03},
    '東京': {'短距離': -0.03, 'マイル': -0.08, '中距離': -0.10, '長距離': -0.05},
    '新潟': {'短距離': -0.05, 'マイル': -0.10, '中距離': -0.08, '長距離': -0.05},
    '京都': {'短距離': +0.00, 'マイル': +0.00, '中距離': +0.00, '長距離': +0.05},
    '中京': {'短距離': +0.00, 'マイル': -0.03, '中距離': -0.05, '長距離': -0.05},
}


def calc_pace_distribution(race):
    horses = race.get('horses', [])
    n = max(len(horses), 1)
    esc   = race.get('escape_count', 0)
    front = race.get('front_count', 0)
    if _PACE_MODEL is not None:
        front_density = (esc + front) / n
        agari_vals = [h.get('agari3f') for h in horses if h.get('agari3f')]
        avg_agari = sum(agari_vals) / len(agari_vals) if agari_vals else 36.0
        std_agari = (sum((x - avg_agari) ** 2 for x in agari_vals) / len(agari_vals)) ** 0.5 if len(agari_vals) > 1 else 1.5
        surface_num = 1 if race.get('surface') == '芝' else 0
        distance = int(race.get('distance', 1600))
        import pandas as _pd
        X_df = _pd.DataFrame([[
            esc, front, front_density,
            avg_agari, std_agari, n,
            distance, surface_num,
        ]], columns=[
            'front_count', 'senkou_count', 'front_density',
            'avg_agari3f', 'std_agari3f', 'runner_count',
            'distance', 'surface_num',
        ])
        proba = _PACE_MODEL.predict_proba(X_df)[0]
        return {
            'slow': round(float(proba[0]), 3),
            'mid':  round(float(proba[1]), 3),
            'high': round(float(proba[2]), 3),
        }
    fp = (esc * 1.0 + front * 0.5) / n
    inner_front = sum(
        1 for h in horses
        if h.get('running_style') in ('逃げ', '先行') and h.get('post_position', 8) <= 4
    )
    fp += inner_front * 0.05
    if fp >= 0.30:
        p_high = min(0.75, fp * 2.2)
        p_slow = max(0.05, 0.40 - fp * 1.2)
    elif fp <= 0.08:
        p_slow = min(0.70, (0.15 - fp) * 5)
        p_high = max(0.05, fp * 1.5)
    else:
        p_high = max(0.10, min(0.60, fp * 1.5))
        p_slow = max(0.10, min(0.60, (0.20 - fp) * 2))
    p_mid = max(0.05, 1.0 - p_high - p_slow)

    # コース別ペース傾向を加味（正=スローペース傾向）
    venue    = race.get('racecourse', '')
    zone     = dist_zone_label(race.get('distance', 1600))
    tendency = VENUE_PACE_TENDENCY.get(venue, {}).get(zone, 0.0)
    p_slow = max(0.05, p_slow + tendency)
    p_high = max(0.05, p_high - tendency)
    p_mid  = max(0.05, 1.0 - p_high - p_slow)

    total = p_high + p_mid + p_slow
    return {
        'high': round(p_high / total, 3),
        'mid':  round(p_mid / total, 3),
        'slow': round(p_slow / total, 3),
    }


def f_pace(h, race):
    style = h.get('running_style', '差し')
    dist  = race.get('distance', 1600)
    zone  = dist_zone_label(dist)
    # race['pace_dist'] が既に計算済みなら再利用
    pace_dist = race.get('pace_dist') or calc_pace_distribution(race)
    zone_scores = PACE_STYLE_SCORE.get(zone, PACE_STYLE_SCORE['マイル'])
    ev = (pace_dist['high'] * zone_scores['high'].get(style, 0) +
          pace_dist['mid']  * zone_scores['mid'].get(style, 0) +
          pace_dist['slow'] * zone_scores['slow'].get(style, 0))

    # 先行・逃げの外枠ペナルティ（頭数比例）
    n   = race.get('num_horses', 16)
    pos = h.get('post_position', 8)
    if style in ('逃げ', '先行') and n >= 14 and pos >= n * 0.65:
        ev -= 0.8
    elif style in ('逃げ', '先行') and n >= 10 and pos >= n * 0.75:
        ev -= 0.4

    # 逃げ多頭ペナルティ（競り合い激化）
    esc_count = race.get('escape_count', 0)
    if style == '逃げ' and esc_count >= 2:
        ev -= (esc_count - 1) * 0.5
    elif style == '先行' and esc_count >= 3:
        ev -= (esc_count - 2) * 0.2

    score = (ev + 4) / 8 * 10
    return max(0, min(10, score))


def f_dist_v2(h, race):
    name = h.get('name', '')
    dist = race.get('distance', 2000)
    rc   = race.get('racecourse', '')
    surf = race.get('surface', '芝')
    zone = dist_zone_label(dist)
    scores = []

    # 1. 距離帯別成績
    dist_rec = _horse_dist_dict.get((name, zone))
    if dist_rec and dist_rec['出走'] >= 3:
        conf = min(1.0, dist_rec['出走'] / 10)
        scores.append(min(10, max(0, dist_rec['複勝率'] / 0.35 * 6)) * conf + 5.0 * (1 - conf))
    else:
        b = h.get('best_distance')  # 出馬表から取得できる場合のみ使う
        if b is not None:
            d = abs(int(dist) - int(b))
            scores.append(10 if d == 0 else 8.5 if d <= 200 else 7 if d <= 400 else 5 if d <= 600 else 3)
        else:
            scores.append(5.0)  # 距離適性不明：中立値（全馬10.0に固定されるのを防ぐ）

    # 2. 競馬場×芝ダート別成績
    course_rec = _horse_course_dict.get((name, rc, surf))
    if course_rec and course_rec['出走'] >= 3:
        conf = min(1.0, course_rec['出走'] / 10)
        scores.append(min(10, max(0, course_rec['複勝率'] / 0.35 * 6)) * conf + 5.0 * (1 - conf))
    else:
        scores.append(5.0)

    # 3. 競馬場×距離帯×芝ダート別成績（組み合わせ適性）
    vd_rec = _horse_venue_dist_dict.get((name, rc, zone, surf))
    if vd_rec and vd_rec['出走'] >= 3:
        conf = min(1.0, vd_rec['出走'] / 8)
        scores.append(min(10, max(0, vd_rec['複勝率'] / 0.35 * 6)) * conf + 5.0 * (1 - conf))

    base_score = round(sum(scores) / len(scores), 2)

    # 長距離初挑戦ペナルティ（2000m以上で過去出走ゼロの場合 -1.0）
    if zone == '長距離':
        long_rec  = _horse_dist_dict.get((name, '長距離'))
        long_runs = long_rec['出走'] if long_rec else 0
        if long_runs == 0:
            hist      = h.get('history', [])
            long_hist = sum(1 for r in hist if (r.get('distance') or 0) >= 2000)
            if long_hist == 0:
                base_score = max(0, base_score - 1.0)

    return base_score


def f_blood(h, race):
    sd   = SIRE_DB.get(h.get('sire', ''), DEF_SIRE)
    dd   = SIRE_DB.get(h.get('dam_sire', ''), DEF_SIRE)
    age  = h.get('age', 4)
    surf = race.get('surface', '芝')
    z    = dz(race.get('distance', 2000))
    ds   = sd['dist'].get(z, .7) * .7 + dd['dist'].get(z, .7) * .3
    ss   = sd['surf'].get(surf, .7) * .7 + dd['surf'].get(surf, .7) * .3
    peak = sd['peak']
    gt   = sd['type']
    diff = age - peak
    if abs(diff) <= .5:    gw = 10.0
    elif diff < 0:
        gw = 8.5 if gt == 'late' and diff >= -2 else 9.0 if gt == 'early' and diff >= -0.5 else max(4, 10 + diff * 1.5)
    else:
        gw = max(3, 10 - diff * 2.5) if gt == 'early' and diff >= 1.5 else max(5, 10 - diff)
    pt = min(10, (peak - age) * 3) if gt == 'late' and age < peak - .5 else 6.5 if gt == 'standard' and age <= 3 else 5.0
    return min(10, max(0, ds * 10 * .4 + ss * 10 * .3 + gw * .2 + pt * .1))


def f_jockey(h, race):
    rate = h.get('jockey_rate', 0.15)
    base = rate / 0.30 * 10

    # 市場乖離補正: 騎手実力がオッズに織り込まれていない部分だけを評価する
    # 有名騎手が低オッズ馬に乗る場合 → 市場が既に評価済み → 補正を下げる
    # 実力騎手が高オッズ馬に乗る場合 → 市場が未評価   → 補正を上げる
    win_odds = h.get('win_odds') or 10.0
    market_prob = 1.0 / max(win_odds, 1.1)
    gap = rate - market_prob          # 正=市場未評価の実力、負=市場が過大評価
    adjustment = gap * 8.0            # 0.10差 → ±0.8点

    return min(10, max(0, base + adjustment))


def f_trainer(h):
    return min(10, max(0, h.get('trainer_rate', 0.12) / 0.20 * 10))


def calc_unlucky_features(horse_name, race_date, db_path=None):
    """手動入力した不利メモ（race_notes）を特徴量化する。

    過去走の total_handicap（不利・出遅れ・展開ロスの補正値合計）を集計する。
    補正値はスキーマ駆動で保存時にキャッシュ済みなので、ここでは集計するだけ。
    データが無い馬・未入力の馬はすべて 0 を返すため、メモが空でも安全に動作する。

    Returns
    -------
    dict:
        f_unlucky_recent : 直近3走の補正値合計の平均
        f_unlucky_last   : 前走の補正値
        f_unlucky_max    : 過去5走で最大の補正値
        f_note_coverage  : メモが入力されている走数（信頼度の目安）
    """
    default = {'f_unlucky_recent': 0.0, 'f_unlucky_last': 0.0,
               'f_unlucky_max': 0.0, 'f_note_coverage': 0}
    path = db_path or _KEIBA_DB_PATH
    if not horse_name or not path or not os.path.exists(path):
        return default
    import sqlite3 as _sq
    try:
        conn = _sq.connect(path)
        rows = conn.execute(
            "SELECT total_handicap FROM race_notes "
            "WHERE horse_name = ? AND date < ? "
            "ORDER BY date DESC LIMIT 5",
            (horse_name, race_date),
        ).fetchall()
        conn.close()
    except _sq.OperationalError:
        return default

    handicaps = [r[0] for r in rows if r[0] is not None]
    if not handicaps:
        return default
    recent3 = handicaps[:3]
    return {
        'f_unlucky_recent': round(sum(recent3) / len(recent3), 2),
        'f_unlucky_last': handicaps[0],
        'f_unlucky_max': max(handicaps),
        'f_note_coverage': len(handicaps),
    }


def f_weight(h):
    return max(0, min(10, (58.0 - h.get('weight_load', 56.0)) * 2))


def f_post(h, race):
    p    = h.get('post_position', 8)
    n    = race.get('num_horses', 16)
    rc   = race.get('racecourse', '')
    dist = race.get('distance', 1600)
    zone = dist_zone_label(dist)
    pos_ratio = (p - 1) / max(n - 1, 1)  # 0=内枠, 1=外枠

    # 優先度: データ実績 > 距離帯別設定値 > 競馬場単体設定値
    # 正=内枠有利, 負=外枠有利
    bv = _post_zone_bias.get((rc, zone))
    if bv is None:
        bv = POST_BIAS_BY_ZONE.get(rc, {}).get(zone)
    if bv is None:
        bv = -POST_BIAS.get(rc, 0)  # 旧設定は外枠正値なので符号反転

    return max(0, min(10, 5.0 - bv * (pos_ratio - 0.5) * 4))


def f_bias(h, race, bias_data):
    if not bias_data:
        return 5.0
    rc = race.get('racecourse', '')
    bd = bias_data.get('by_course', {}).get(rc, bias_data)
    p  = h.get('post_position', 8)
    st = h.get('running_style', '差し')
    b  = 5.0 + bd.get('inner_outer', 0) * (.5 if p <= 4 else -.5 if p >= 12 else 0)
    b += bd.get('pace_bias', 0) * (.6 if st in ('逃げ', '先行') else -.3)
    b += bd.get('track_speed', 0) * .3
    return max(0, min(10, b))


def analyze_career(h, race):
    import re
    hist = h.get('history', [])
    flags = {}
    comments = []
    if not hist:
        return {'flags': flags, 'comments': comments}
    prev = hist[0]
    curr_dist = race.get('distance', 2000)
    curr_surf = race.get('surface', '芝')
    prev_dist = prev.get('distance', curr_dist)
    prev_surf = prev.get('surface', curr_surf)
    curr_class = get_race_class_rank(race.get('class', race.get('race_name', '')))
    prev_class = get_race_class_rank(prev.get('class') or '')
    skip_class = (prev.get('class') is None) or bool(re.match(r'^R\d{2}$', prev.get('class') or ''))
    curr_name = CLASS_RANK_NAME.get(curr_class, '?')
    prev_name = CLASS_RANK_NAME.get(prev_class, '?')
    class_diff = curr_class - prev_class
    if not skip_class:
        if class_diff >= 2:
            flags['class_up_big'] = True
            comments.append('⚠ 大幅クラス上昇(' + prev_name + '→' + curr_name + ')')
        elif class_diff == 1:
            flags['class_up'] = True
            comments.append('📈 クラス上昇(' + prev_name + '→' + curr_name + ')')
        elif class_diff <= -1:
            flags['class_down'] = True
            comments.append('📉 クラス降格(' + prev_name + '→' + curr_name + ')')
    dist_diff = curr_dist - prev_dist
    zone = dist_zone_label(curr_dist)
    has_dist_rec = _horse_dist_dict.get((h.get('name', ''), zone))
    if dist_diff <= -400:
        flags['dist_shorten_big'] = True
        comments.append('↙ 大幅距離短縮(' + str(prev_dist) + 'm→' + str(curr_dist) + 'm)' +
                        (' ★初挑戦・注目' if not has_dist_rec else ' ※実績あり'))
    elif dist_diff <= -200:
        flags['dist_shorten'] = True
        comments.append('↙ 距離短縮(' + str(prev_dist) + 'm→' + str(curr_dist) + 'm)' +
                        (' ★この距離帯初挑戦・注目' if not has_dist_rec else ''))
    if prev_surf != curr_surf:
        flags['surface_change'] = True
        sire = h.get('sire', '')
        surf_fit = SIRE_DB.get(sire, DEF_SIRE)['surf'].get(curr_surf, 0.7)
        direction = '芝→ダート' if curr_surf == 'ダート' else 'ダート→芝'
        if surf_fit >= 0.8:
            key = 'surface_to_dirt_suitable' if curr_surf == 'ダート' else 'surface_to_turf_suitable'
            flags[key] = True
            comments.append('🔄 ' + direction + '転向 ★血統的に向き(' + sire + ') 注目')
        else:
            comments.append('🔄 ' + direction + '転向（血統適性やや低め）')

    # 前走からの間隔分析
    prev_date = prev.get('date', '')
    curr_date = race.get('date', '')
    if prev_date and curr_date:
        try:
            from datetime import datetime
            d0 = datetime.strptime(str(prev_date)[:10], '%Y-%m-%d')
            d1 = datetime.strptime(str(curr_date)[:10], '%Y-%m-%d')
            interval = (d1 - d0).days
            if interval <= 14:
                flags['short_interval'] = True
                comments.append(f'⚡ 連闘・中2週以内（{interval}日）')
            elif interval >= 90:
                flags['long_layoff'] = True
                comments.append(f'💤 長期休養明け（{interval}日）')
            elif 21 <= interval <= 35:
                flags['good_interval'] = True
        except Exception:
            pass

    return {'flags': flags, 'comments': comments}


def apply_career_flags(total, career):
    flags = career['flags']
    adj = 0.0
    if flags.get('class_up_big'):    adj -= 1.0
    elif flags.get('class_up'):      adj -= 0.5
    if flags.get('class_down'):      adj += 0.3
    if flags.get('surface_to_dirt_suitable') or flags.get('surface_to_turf_suitable'):
        adj += 0.5
    elif flags.get('surface_change'):
        adj -= 0.3
    if flags.get('short_interval'):  adj -= 0.3
    if flags.get('long_layoff'):     adj -= 0.4
    if flags.get('good_interval'):   adj += 0.2
    return round(total + adj, 2)


def calc_prediction_gap_features(horse_name, race_date):
    """過去レースでのAI予測(rl_rank)と実着順の乖離を特徴量として返す。

    データが溜まるまでデフォルト値(0.0)を返すので既存モデルに影響しない。
    race_date: 'YYYY-MM-DD' または 'YYYYMMDD'
    """
    if not _KEIBA_DB_PATH:
        return 0.0, 0.0, 0.0
    try:
        import sqlite3 as _sq
        # 日付を YYYY-MM-DD に正規化
        rd = str(race_date).replace('-', '')[:8]
        date_norm = f'{rd[:4]}-{rd[4:6]}-{rd[6:8]}' if len(rd) == 8 else str(race_date)
        conn = _sq.connect(_KEIBA_DB_PATH)
        rows = conn.execute("""
            SELECT prediction_gap FROM race_predictions
            WHERE horse_name = ? AND date < ? AND actual_place IS NOT NULL
            ORDER BY date DESC LIMIT 3
        """, (horse_name, date_norm)).fetchall()
        conn.close()
        gaps = [r[0] for r in rows if r[0] is not None]
        if not gaps:
            return 0.0, 0.0, 0.0
        avg  = sum(gaps) / len(gaps)
        worst = max(gaps, key=abs)
        std  = (sum((g - avg) ** 2 for g in gaps) / len(gaps)) ** 0.5 if len(gaps) > 1 else 0.0
        return round(avg, 2), round(worst, 2), round(std, 2)
    except Exception:
        return 0.0, 0.0, 0.0


# ── コース適性特徴量（course_profiles.json 駆動）─────────────────────────
def load_course_profiles(base_dir=None):
    """data/course_profiles.json を読み込む（モジュールロード時に1回キャッシュ）。

    base_dir 省略時は init_engine() で設定した _BASE_DIR を使う。
    """
    global _COURSE_PROFILES
    if _COURSE_PROFILES is None:
        bd = base_dir or _BASE_DIR
        if bd is None:
            return None
        path = os.path.join(bd, 'data', 'course_profiles.json')
        if not os.path.exists(path):
            return None
        import json as _json
        with open(path, encoding='utf-8') as f:
            _COURSE_PROFILES = _json.load(f)
    return _COURSE_PROFILES


def get_course_profile(racecourse, surface, base_dir=None):
    """競馬場+コース（芝/ダート）のプロファイルを取得。未定義なら None。"""
    profiles = load_course_profiles(base_dir)
    if not profiles:
        return None
    key = f'{racecourse}_{surface}'
    return profiles.get('courses', {}).get(key)


def _default_course_features():
    """データなし・コース未定義時のデフォルト特徴量。"""
    return {
        'f_same_course_rate': 0.0,
        'f_same_turn_rate':   0.0,
        'f_straight_match':   0.0,
        'f_uphill_match':     0.0,
        'f_agari_at_similar': 99.0,
        'f_course_coverage':  0,
    }


def calc_course_aptitude_features(horse_name, today_racecourse, today_surface,
                                  history, base_dir=None):
    """馬の過去走をコース形状で分類し、今日のコースへの適性を計算する。

    history : list of dict（racecourse, surface, place, agari3f/last_3f を含む）
    base_dir 省略時は _BASE_DIR を使用。

    Returns dict:
        f_same_course_rate : この競馬場+コースでの過去好走率（3着内率）
        f_same_turn_rate   : 同じ回り（右/左）での好走率
        f_straight_match   : 直線長クラスが合う過去走での好走率
        f_uphill_match     : 坂の有無が合う過去走での好走率
        f_agari_at_similar : 似た直線のコースでの最速上がり（小さいほど良い）
        f_course_coverage  : 同一コースでの経験走数（信頼度）
    """
    today = get_course_profile(today_racecourse, today_surface, base_dir)
    if today is None:
        return _default_course_features()

    same_course    = []  # 同一競馬場+コース
    same_turn      = []  # 同じ回り
    straight_match = []  # 直線長クラスが同じ
    uphill_match   = []  # 坂の有無が同じ
    agari_list     = []  # 直線長が近いコースでの上がり

    for hrec in history:
        rc    = hrec.get('racecourse', '')
        sf    = hrec.get('surface', '')
        place = hrec.get('place')
        agari = hrec.get('agari3f') or hrec.get('last_3f')

        if place is None or place <= 0 or place >= 99:
            continue

        prof = get_course_profile(rc, sf, base_dir)
        if prof is None:
            continue

        is_top3 = 1 if place <= 3 else 0

        if rc == today_racecourse and sf == today_surface:
            same_course.append(is_top3)
        if prof.get('turn') == today.get('turn'):
            same_turn.append(is_top3)
        if prof.get('straight_class') == today.get('straight_class'):
            straight_match.append(is_top3)
            try:
                a = float(agari) if agari else 0.0
            except (TypeError, ValueError):
                a = 0.0
            if a > 0:
                agari_list.append(a)
        if prof.get('has_uphill') == today.get('has_uphill'):
            uphill_match.append(is_top3)

    def _rate(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        'f_same_course_rate': _rate(same_course),
        'f_same_turn_rate':   _rate(same_turn),
        'f_straight_match':   _rate(straight_match),
        'f_uphill_match':     _rate(uphill_match),
        'f_agari_at_similar': round(min(agari_list), 1) if agari_list else 99.0,
        'f_course_coverage':  len(same_course),
    }


# ── 脚質×コース×展開適性 ──────────────────────────────────────────────────

# running_style の表記ゆれを正規化（DB=日本語、テスト=英語の両方を受け入れる）
_STYLE_NORM = {
    '逃げ': 'escape',  'escape': 'escape',
    '先行': 'front',   'front':  'front',
    '差し': 'stalk',   'stalk':  'stalk',
    '追込': 'closer',  'closer': 'closer',
}

# ペース判定の展開別・脚質別適合度マトリクス
_PACE_FIT_MATRIX = {
    'high':   {'escape': 0.20, 'front': 0.35, 'stalk': 0.70, 'closer': 0.60},
    'middle': {'escape': 0.50, 'front': 0.60, 'stalk': 0.45, 'closer': 0.35},
    'slow':   {'escape': 0.75, 'front': 0.65, 'stalk': 0.30, 'closer': 0.20},
}


def estimate_horse_style(horse):
    """過去走の running_style から馬の主な脚質を推定する（英語キーで返す）。

    直近5走の最頻値を採用。running_style が無い場合は corner_3 で補完。
    戻り値: 'escape' / 'front' / 'stalk' / 'closer' / None
    """
    from collections import Counter
    styles = []
    for rec in horse.get('history', [])[:5]:
        rs = rec.get('running_style')
        normalized = _STYLE_NORM.get(rs)
        if normalized:
            styles.append(normalized)
        elif rec.get('corner_3') is not None:
            try:
                pos = float(rec['corner_3'])
                if pos <= 2:
                    styles.append('escape')
                elif pos <= 5:
                    styles.append('front')
                elif pos <= 10:
                    styles.append('stalk')
                else:
                    styles.append('closer')
            except (TypeError, ValueError):
                pass

    if not styles:
        return None
    return Counter(styles).most_common(1)[0][0]


def predict_race_pace(horses):
    """出走馬の脚質構成からレースのペースを予測する。

    逃げ馬が多い → ハイペース（差し有利）
    逃げ馬が少ない → スローペース（先行有利）

    Returns dict: pace / n_escape / n_front / n_stalk / n_closer /
                  front_ratio / favored_style
    """
    n_escape = n_front = n_stalk = n_closer = 0
    for h in horses:
        style = estimate_horse_style(h)
        if style == 'escape':
            n_escape += 1
        elif style == 'front':
            n_front += 1
        elif style == 'stalk':
            n_stalk += 1
        elif style == 'closer':
            n_closer += 1

    n_horses = len(horses) if horses else 1
    front_ratio = (n_escape + n_front) / n_horses

    if n_escape >= 3 or front_ratio >= 0.5:
        pace = 'high'
        favored = 'stalk'
    elif n_escape <= 1 and front_ratio <= 0.3:
        pace = 'slow'
        favored = 'front'
    else:
        pace = 'middle'
        favored = 'front'

    return {
        'pace':         pace,
        'n_escape':     n_escape,
        'n_front':      n_front,
        'n_stalk':      n_stalk,
        'n_closer':     n_closer,
        'front_ratio':  round(front_ratio, 2),
        'favored_style': favored,
    }


def calc_style_course_fit(horse, race, base_dir=None):
    """馬の脚質がこのコースの有利脚質とどれだけ合うか（0〜1）。

    course_profiles.json の style_advantage から取得。
    コース未定義または脚質不明時は 0.25（4脚質均等）を返す。
    """
    profile = get_course_profile(
        race.get('racecourse', ''), race.get('surface', '芝'), base_dir
    )
    if not profile or 'style_advantage' not in profile:
        return 0.25

    horse_style = estimate_horse_style(horse)
    if horse_style is None:
        return 0.25

    return float(profile['style_advantage'].get(horse_style, 0.25))


def calc_pace_fit(horse, pace_info):
    """馬の脚質が予測されるペースに合うか（0〜1）。

    pace_info は predict_race_pace() の戻り値。
    脚質不明時は 0.5 を返す。
    """
    horse_style = estimate_horse_style(horse)
    if horse_style is None:
        return 0.5

    pace = pace_info.get('pace', 'middle')
    return float(_PACE_FIT_MATRIX.get(pace, _PACE_FIT_MATRIX['middle']).get(horse_style, 0.5))


def calc_features_for_xgb(h, race):
    """XGBoost複勝予測モデルへの入力特徴量を生成"""
    import numpy as _np
    feats = {}
    hist  = h.get('history', [])
    rc    = race.get('racecourse', '')
    surf  = race.get('surface', '芝')
    dist  = int(race.get('distance', 1600) or 1600)
    n     = max(len(race.get('horses', [])), 8)

    # 初出走フラグ（新馬・初出走は過去走なし → fillna(5.0) と区別するため）
    feats['f_is_debut'] = float(1 if not hist else 0)

    c3_list = [r.get('corner_3') for r in hist
               if r.get('corner_3') is not None and r.get('corner_3') == r.get('corner_3')]
    c3_arr  = [float(v) for v in c3_list if v is not None]

    if c3_arr:
        feats['f_pos_avg_3']   = float(sum(c3_arr[-3:]) / len(c3_arr[-3:]))
        feats['f_pos_std_3']   = float(_np.std(c3_arr[-3:])) if len(c3_arr[-3:]) >= 2 else 3.0
        feats['f_p_front']     = float(sum(1 for v in c3_arr if v <= 3) / len(c3_arr))
        feats['f_p_mid']       = float(sum(1 for v in c3_arr if 3 < v <= 8) / len(c3_arr))
        feats['f_p_back']      = float(sum(1 for v in c3_arr if v > 8) / len(c3_arr))
        feats['f_last1_pos3c'] = float(c3_arr[-1]) if len(c3_arr) >= 1 else 8.0
        feats['f_last2_pos3c'] = float(c3_arr[-2]) if len(c3_arr) >= 2 else 8.0
        feats['f_last3_pos3c'] = float(c3_arr[-3]) if len(c3_arr) >= 3 else 8.0
    else:
        style = h.get('running_style', '差し')
        sp = {'逃げ': 1.5, '先行': 3.5, '差し': 7.0, '追込': 11.0}.get(style, 7.0)
        feats.update({
            'f_pos_avg_3': sp, 'f_pos_std_3': 2.0,
            'f_p_front': 0.8 if style == '逃げ' else 0.5 if style == '先行' else 0.1,
            'f_p_mid':   0.1 if style in ('逃げ', '先行') else 0.5,
            'f_p_back':  0.1 if style in ('逃げ', '先行') else 0.4,
            'f_last1_pos3c': sp, 'f_last2_pos3c': sp, 'f_last3_pos3c': sp,
        })

    ag_list = [float(r.get('last_3f') or r.get('agari3f') or 0) for r in hist
               if (r.get('last_3f') or r.get('agari3f'))]
    if ag_list:
        feats['f_late_speed'] = float(sum(ag_list[-3:]) / len(ag_list[-3:]))
        feats['f_last1_3f']   = ag_list[-1] if len(ag_list) >= 1 else 37.0
        feats['f_last2_3f']   = ag_list[-2] if len(ag_list) >= 2 else 37.0
        feats['f_last3_3f']   = ag_list[-3] if len(ag_list) >= 3 else 37.0
    else:
        feats.update({'f_late_speed': 37.0, 'f_last1_3f': 37.0, 'f_last2_3f': 37.0, 'f_last3_3f': 37.0})

    feats['f_early_speed'] = float(race.get('first_3f', 0) or 36.0)

    if hist:
        places = [r.get('place', 10) for r in hist[-5:]]
        ws_ = [.75 ** i for i in range(len(places) - 1, -1, -1)]
        tw_ = sum(ws_)
        ps_ = [max(0, 10 - (p - 1) * 10 / 15) for p in places]
        feats['f_recent']         = float(sum(s * w for s, w in zip(ps_, ws_)) / tw_) if tw_ > 0 else 5.0
        feats['f_recent_fukusho'] = float(sum(1 for p in places if p <= 3) / len(places))
        feats['f_career_runs']    = min(20, len(hist))
        feats['f_last1_rank']     = float(places[-1]) if len(places) >= 1 else 8.0
        feats['f_last2_rank']     = float(places[-2]) if len(places) >= 2 else 8.0
        feats['f_last3_rank']     = float(places[-3]) if len(places) >= 3 else 8.0
    else:
        feats.update({
            'f_recent': 5.0, 'f_recent_fukusho': 0.33, 'f_career_runs': 0,
            'f_last1_rank': 8.0, 'f_last2_rank': 8.0, 'f_last3_rank': 8.0,
        })

    def _dz(d):
        if d <= 1400: return 'sp'
        if d <= 1800: return 'mi'
        if d <= 2200: return 'md'
        return 'lo'

    dz_ = _dz(dist)
    if hist:
        same_zone   = [r for r in hist if _dz(int(r.get('distance', 1600) or 1600)) == dz_]
        same_course = [r for r in hist if r.get('racecourse', '') == rc and r.get('surface', '') == surf]
        feats['f_dist_fukusho']   = float(sum(1 for r in same_zone if r.get('place', 10) <= 3) / len(same_zone)) if same_zone else 0.33
        feats['f_course_fukusho'] = float(sum(1 for r in same_course if r.get('place', 10) <= 3) / len(same_course)) if same_course else 0.33
    else:
        feats['f_dist_fukusho']   = 0.33
        feats['f_course_fukusho'] = 0.33

    if c3_arr and len(c3_arr) >= 2:
        high_front = [r for r, c in zip(hist[-len(c3_arr):], c3_arr) if c <= 4]
        slow_back  = [r for r, c in zip(hist[-len(c3_arr):], c3_arr) if c >= 8]
        feats['f_perf_highpace'] = float(sum(1 for r in high_front if r.get('place', 10) <= 3) / len(high_front)) if high_front else 0.3
        feats['f_perf_slowpace'] = float(sum(1 for r in slow_back if r.get('place', 10) <= 3) / len(slow_back)) if slow_back else 0.3
    else:
        feats['f_perf_highpace'] = 0.3
        feats['f_perf_slowpace'] = 0.3

    feats['f_jockey']      = min(10, max(0, h.get('jockey_rate', 0.15) / 0.30 * 10))
    feats['f_jockey_rate'] = float(h.get('jockey_rate', 0.15))
    feats['f_trainer']     = min(10, max(0, h.get('trainer_rate', 0.12) / 0.20 * 10))

    bv  = POST_BIAS.get(rc, 0)
    num = int(h.get('horse_num', 1) or 1)
    feats['f_post'] = max(0, min(10, 5.0 + bv * ((num - 1) / max(n - 1, 1) - 0.5) * 4))

    if hist and hist[-1].get('date'):
        try:
            import datetime as _dt
            last_d = _dt.datetime.strptime(hist[-1]['date'], '%Y-%m-%d')
            curr_d = _dt.datetime.strptime(race.get('date', hist[-1]['date']), '%Y-%m-%d')
            days = (curr_d - last_d).days
            feats['f_days_since_last'] = float(min(days, 365))
            feats['f_interval_score'] = (
                -1.5 if days <= 7 else 0.5 if days <= 14 else
                1.0 if days <= 28 else 0.0 if days <= 56 else -0.5
            )
        except Exception:
            feats['f_days_since_last'] = 30.0
            feats['f_interval_score']  = 0.0
    else:
        feats['f_days_since_last'] = 30.0
        feats['f_interval_score']  = 0.0

    feats['f_blood']       = f_blood(h, race)
    feats['f_weight_load'] = max(0, min(10, (58.0 - float(h.get('weight_load', 56.0) or 56.0)) * 2))
    feats['f_pace']        = f_pace(h, race)

    si_list_ = []
    for r_ in hist[-5:]:
        lf_ = r_.get('last_3f') or r_.get('agari3f') or 0
        if lf_ and float(lf_) > 0:
            try:
                si_ = calc_performance_index(
                    float(lf_),
                    float(r_.get('first_3f') or 0),
                    float(r_.get('corner_3') or 8) if r_.get('corner_3') == r_.get('corner_3') else 8,
                    max(int(r_.get('finishers', 16) or 16), 8),
                    int(r_.get('distance', 1600) or 1600),
                    r_.get('surface', '芝'), '良',
                )
                si_list_.append(si_)
            except Exception:
                pass
    if si_list_:
        feats['f_speed_avg']   = float(sum(si_list_) / len(si_list_))
        feats['f_speed_max']   = float(max(si_list_))
        feats['f_speed_last']  = float(si_list_[-1])
        feats['f_speed_trend'] = float(si_list_[-1] - si_list_[0]) if len(si_list_) >= 2 else 0.0
    else:
        feats.update({'f_speed_avg': 50.0, 'f_speed_max': 50.0, 'f_speed_last': 50.0, 'f_speed_trend': 0.0})

    # ── Stage 3 新特徴量 ────────────────────────────────────────
    # 性別（牡=0, 牝=1, セ=2）
    _sex_map = {'牡': 0, '牝': 1, 'セ': 2, '騸': 2}
    feats['f_sex']    = float(_sex_map.get(h.get('sex', '牡') or '牡', 0))
    # 年齢
    feats['f_age']    = float(int(h.get('age', 4) or 4))

    # 馬場状態（当該レース）
    _tc_map = {'良': 0.0, '稍重': 1.0, '重': 2.0, '不良': 3.0}
    feats['f_track_cond'] = float(_tc_map.get(race.get('track_condition', '良') or '良', 0.0))

    # 重馬場適性（過去走で稍重以上での複勝率）
    heavy_runs = [r for r in hist if r.get('track_condition', '良') in ('稍重', '重', '不良')]
    feats['f_heavy_track_rate'] = (
        float(sum(1 for r in heavy_runs if r.get('place', 10) <= 3) / len(heavy_runs))
        if heavy_runs else 0.33
    )

    # クラスレベル（当該レースの格）
    _cls_map = {'新馬': 1, '未勝利': 2, '1勝': 3, '1勝クラス': 3, '2勝': 4, '2勝クラス': 4,
                '3勝': 5, '3勝クラス': 5, 'OP': 6, 'オープン': 6, 'L': 7, 'G3': 8, 'G2': 9, 'G1': 10}
    feats['f_class_level'] = float(_cls_map.get(race.get('race_class', '') or '', 3))

    # クラスジャンプ（前走クラスとの差）
    if hist:
        prev_cls = _cls_map.get(hist[-1].get('race_class', '') or hist[-1].get('class', '') or '', 3)
        feats['f_class_jump'] = float(feats['f_class_level'] - prev_cls)
    else:
        feats['f_class_jump'] = 0.0

    # 走破タイム平均・勝ち馬差平均（±200m以内の過去走のみ使用）
    # 距離が近いレースに絞ることで異距離混在による誤評価を防ぐ
    spd_list = []
    td_list  = []
    for r in hist:
        rd = int(r.get('distance') or 0)
        if rd == 0 or abs(rd - dist) > 200:
            continue
        ft = r.get('finish_time')
        if ft:
            spd_list.append(rd / float(ft))
        td = r.get('time_diff_sec')
        if td is not None and td != 0:
            td_list.append(float(td) / (rd / 1000.0))
    # 今走距離 / 平均速度 で秒換算に戻す（XGBモデルの入力スケールを維持）
    if spd_list:
        avg_spd = sum(spd_list) / len(spd_list)
        feats['f_finish_time_avg'] = float(dist / avg_spd) if avg_spd else float('nan')
    else:
        feats['f_finish_time_avg'] = float('nan')
    feats['f_time_diff_avg'] = float(sum(td_list) / len(td_list)) if td_list else float('nan')

    # ── 前走メンバーレベル（対戦相手のその後の成績で強度を評価）────────
    ml_levels = []
    for past_race in hist[:5]:
        rid = past_race.get('race_id', '')
        if rid and _MEMBER_LEVEL_CACHE:
            ml_levels.append(_MEMBER_LEVEL_CACHE.get(rid, 5.0))
    if ml_levels:
        feats['f_member_level_avg']  = float(sum(ml_levels) / len(ml_levels))
        feats['f_member_level_max']  = float(max(ml_levels))
        feats['f_member_level_last'] = float(ml_levels[0])  # 直近走
    else:
        feats['f_member_level_avg']  = 5.0
        feats['f_member_level_max']  = 5.0
        feats['f_member_level_last'] = 5.0

    # 競争力指数（着差÷着順：小さいほど着順より実力が上位に近い）
    f_comp_avg, f_comp_best = calc_competitiveness(hist)
    feats['f_competitiveness']      = float(f_comp_avg)
    feats['f_competitive_best']     = float(f_comp_best)

    # ── スピード指数（±200m以内の過去走のみ使用）────────────────────────
    # 今走距離と近い条件の過去走に絞ることで距離跨ぎの誤評価を防ぐ
    if _SPEED_INDEX_CALC is not None:
        figs = []
        for r in hist:
            rd = int(r.get('distance') or 0)
            if rd == 0 or abs(rd - dist) > 200:
                continue
            fig = _SPEED_INDEX_CALC.calc_speed_figure(
                finish_time    = r.get('finish_time'),
                distance       = rd,
                surface        = r.get('surface') or surf,
                track_condition= r.get('track_condition') or '良',
                date           = r.get('date') or '',
                racecourse     = r.get('racecourse') or rc,
            )
            if fig is not None:
                figs.append(fig)
        if figs:
            feats['f_speed_fig_last'] = float(figs[0])
            feats['f_speed_fig_avg']  = float(sum(figs) / len(figs))
            feats['f_speed_fig_max']  = float(max(figs))
        else:
            feats['f_speed_fig_last'] = float('nan')
            feats['f_speed_fig_avg']  = float('nan')
            feats['f_speed_fig_max']  = float('nan')
    else:
        feats['f_speed_fig_last'] = float('nan')
        feats['f_speed_fig_avg']  = float('nan')
        feats['f_speed_fig_max']  = float('nan')

    # ── 予測乖離特徴量（race_predictions テーブルから取得）──────────────
    horse_name = h.get('name', '')
    race_date  = race.get('date', '')
    gap_avg, gap_worst, gap_std = calc_prediction_gap_features(horse_name, race_date)
    feats['f_pred_gap_avg']         = float(gap_avg)
    feats['f_pred_gap_worst']       = float(gap_worst)
    feats['f_pred_gap_consistency'] = float(gap_std)

    # ── コース適性特徴量（course_profiles.json 駆動）──────────────────
    # 過去走をコース形状（直線長・回り・坂）で分類し、今走コースへの適性を算出。
    # course_profiles.json が無い／コース未定義ならデフォルト0で安全にフォールバック。
    course_feats = calc_course_aptitude_features(
        h.get('name', ''), rc, surf, hist,
    )
    feats.update(course_feats)

    # ── 脚質×コース・展開適性（3特徴量） ──────────────────────────────────
    # pace_info はレース単位で1回だけ計算し race['_pace_info_cache'] に保存して使い回す。
    if '_pace_info_cache' not in race:
        race['_pace_info_cache'] = predict_race_pace(race.get('horses', []))
    pace_info = race['_pace_info_cache']
    feats['f_style_course_fit'] = calc_style_course_fit(h, race)
    feats['f_pace_fit']         = calc_pace_fit(h, pace_info)
    feats['f_style_total_fit']  = round(
        (feats['f_style_course_fit'] + feats['f_pace_fit']) / 2, 3
    )

    return feats


def add_relative_features(all_xfeats):
    """22個の相対特徴量をフィールド全馬のxfeatsに一括追加する（in-place）。

    XGB予測の前に全馬の絶対特徴量が揃った状態で呼ぶこと。
    rank: 1=ベスト。vs_field: 正=平均より良い。
    """
    n = len(all_xfeats)
    if n == 0:
        return

    def _assign(src_key, default, prefix, reverse=True):
        import math
        vals = [xf.get(src_key, default) for xf in all_xfeats]
        # NaN を除いた有効値で mean を計算
        valid_vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        mean_v = sum(valid_vals) / len(valid_vals) if valid_vals else default
        # NaN は sort で末尾に置く（reverse=True なら最低ランク扱い）
        def _sort_key(kv):
            v = kv[1]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return float('-inf') if reverse else float('inf')
            return v
        indexed = sorted(enumerate(vals), key=_sort_key, reverse=reverse)
        ranks = [0] * n
        for rank_i, (idx, _) in enumerate(indexed, start=1):
            ranks[idx] = rank_i
        sign = 1 if reverse else -1  # lower_is_better → vs_field = mean - val
        for i, xf in enumerate(all_xfeats):
            xf[prefix + '_rank']     = float(ranks[i])
            xf[prefix + '_vs_field'] = round(sign * (vals[i] - mean_v), 4)

    # RL系: スピード指数・近走成績（高いほど良い）
    _assign('f_speed_avg',      50.0,  'rl_f_speed_avg')
    _assign('f_speed_last',     50.0,  'rl_f_speed_last')
    _assign('f_recent',          5.0,  'rl_f_recent')
    _assign('f_recent_fukusho', 0.33,  'rl_f_recent_fukusho')
    # 上がりタイム: 低いほど速い → reverse=False、vs_field = mean - val（正=速い）
    _assign('f_late_speed',     37.0,  'rl_f_late_speed',  reverse=False)

    # CL系: 騎手・調教師・距離・コース・血統適性（高いほど良い）
    _assign('f_jockey',          5.0,  'cl_f_jockey')
    _assign('f_trainer',         5.0,  'cl_f_trainer')
    _assign('f_dist_fukusho',   0.33,  'cl_f_dist_fukusho')
    _assign('f_course_fukusho', 0.33,  'cl_f_course_fukusho')
    _assign('f_blood',           5.0,  'cl_f_blood')
    # Stage 3 新特徴量の相対化
    _assign('f_heavy_track_rate', 0.33, 'cl_f_heavy_track')
    _assign('f_weight_load',      5.0,  'cl_f_weight_load')
    # finish_time_avg: 低いほど速い（データなし馬は float('nan') → ワーストランク扱い）
    _assign('f_finish_time_avg',  float('nan'),  'rl_f_finish_time',  reverse=False)
    _assign('f_time_diff_avg',    float('nan'),  'rl_f_time_diff',    reverse=False)
    # スピード指数（高いほど速い = 強い）
    _assign('f_speed_fig_last', float('nan'), 'rl_f_speed_fig_last')
    _assign('f_speed_fig_avg',  float('nan'), 'rl_f_speed_fig_avg')
    _assign('f_speed_fig_max',  float('nan'), 'rl_f_speed_fig_max')
    # 前走メンバーレベル（高いほど強い相手と戦った実績）
    _assign('f_member_level_avg',  5.0, 'rl_f_member_level_avg')
    _assign('f_member_level_last', 5.0, 'rl_f_member_level_last')
    # 予測乖離（正=AI過小評価、負=AI過大評価。乖離なし馬は 0）
    _assign('f_pred_gap_avg',   0.0, 'rl_f_pred_gap_avg',   reverse=False)
    _assign('f_pred_gap_worst', 0.0, 'rl_f_pred_gap_worst', reverse=False)
    # 脚質×コース・展開適性（高いほど有利）
    _assign('f_style_course_fit', 0.25, 'cl_f_style_course_fit')
    _assign('f_pace_fit',         0.50, 'cl_f_pace_fit')
    _assign('f_style_total_fit',  0.375,'cl_f_style_total_fit')

    # f_rl_rank / f_cl_rank: 複合スコアで順位付け
    for xf in all_xfeats:
        xf['_rl_c'] = (
            xf.get('f_speed_avg',       50.0) * 0.40 +
            xf.get('f_recent',           5.0) * 2.00 +
            (37.0 - xf.get('f_late_speed', 37.0)) * 5.00 +
            xf.get('f_recent_fukusho',  0.33) * 5.00
        )
        xf['_cl_c'] = (
            xf.get('f_jockey',          5.0) * 0.30 +
            xf.get('f_dist_fukusho',   0.33) * 10.0 +
            xf.get('f_course_fukusho', 0.33) * 10.0 +
            xf.get('f_blood',           5.0) * 0.10 +
            xf.get('f_trainer',         5.0) * 0.20
        )

    rl_s = sorted(enumerate([x['_rl_c'] for x in all_xfeats]), key=lambda kv: kv[1], reverse=True)
    cl_s = sorted(enumerate([x['_cl_c'] for x in all_xfeats]), key=lambda kv: kv[1], reverse=True)
    rl_r, cl_r = [0] * n, [0] * n
    for ri, (idx, _) in enumerate(rl_s, 1):
        rl_r[idx] = ri
    for ri, (idx, _) in enumerate(cl_s, 1):
        cl_r[idx] = ri
    for i, xf in enumerate(all_xfeats):
        xf['f_rl_rank'] = float(rl_r[i])
        xf['f_cl_rank'] = float(cl_r[i])
        xf.pop('_rl_c', None)
        xf.pop('_cl_c', None)


def calc_fukusho_prob(win_prob, num_horses):
    """後方互換用。calc_harville_probs 移行後はこちらは使わない。"""
    p = max(0.0, min(1.0, win_prob))
    return round(min(0.80, p * 3.0), 4)


def calc_harville_probs(win_probs):
    """Harville公式でtop2/top3確率を計算する。

    win_probs: ソフトマックスで正規化済みの確率リスト（和が1）
    Returns  : [(top2_prob, top3_prob), ...] — 各馬のtop2/top3確率

    計算量: O(n²) for top2, O(n³) for top3。n=18でも数百マイクロ秒。
    """
    n = len(win_probs)
    ps = [max(1e-9, float(p)) for p in win_probs]

    # P(i in top2) = P(i 1st) + sum_{j≠i} p_j * p_i / (1-p_j)
    top2 = []
    for i in range(n):
        v = ps[i]
        for j in range(n):
            if j == i:
                continue
            dj = 1.0 - ps[j]
            if dj > 1e-9:
                v += ps[j] * ps[i] / dj
        top2.append(min(1.0, v))

    if n < 3:
        return list(zip(top2, top2))

    # P(i in top3) = P(i in top2)
    #   + sum_{j≠i} sum_{k≠i,k≠j} p_j*(p_k/(1-p_j))*(p_i/(1-p_j-p_k))
    top3 = []
    for i in range(n):
        v = top2[i]
        for j in range(n):
            if j == i:
                continue
            dj = 1.0 - ps[j]
            if dj < 1e-9:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                djk = 1.0 - ps[j] - ps[k]
                if djk < 1e-9:
                    continue
                v += ps[j] * (ps[k] / dj) * (ps[i] / djk)
        top3.append(min(1.0, v))

    return list(zip(top2, top3))


def harville_pair_prob(p_i, p_j):
    """P(i と j が両方top2に入る確率) — ワイド・馬連のEV計算用。

    = p_i * p_j/(1-p_i) + p_j * p_i/(1-p_j)
    """
    di = max(1e-9, 1.0 - p_i)
    dj = max(1e-9, 1.0 - p_j)
    return round(min(1.0, p_i * p_j / di + p_j * p_i / dj), 6)


def harville_trio_prob(p_i, p_j, p_k):
    """P(i, j, k がすべてtop3に入る確率) — 三連複のEV計算用。

    全6通りの着順を Harville公式で合算。
    """
    ps = [p_i, p_j, p_k]
    # 3!=6 permutations of (0,1,2)
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    total = 0.0
    for a, b, c in perms:
        pa, pb, pc = ps[a], ps[b], ps[c]
        da  = max(1e-9, 1.0 - pa)
        dab = max(1e-9, 1.0 - pa - pb)
        total += pa * (pb / da) * (pc / dab)
    return round(min(1.0, total), 6)


def calc_chaos_score(race, scored):
    """レースの混戦度を0〜1で返す"""
    if not scored:
        return 0.5
    scores = [h['total'] for h in scored]
    top = scores[0] if scores[0] > 0 else 1.0
    if len(scores) >= 3:
        gap = (scores[0] - scores[2]) / top
    elif len(scores) >= 2:
        gap = (scores[0] - scores[1]) / top
    else:
        gap = 0.5
    return round(max(0.0, min(1.0, 1.0 - gap * 2)), 3)


def auto_comment(cand, bias_data):
    """厳選レースのAIコメントを自動生成"""
    race   = cand.get('race', {})
    scored = cand.get('scored', [])
    if not scored:
        return 'データ不足のためコメント生成できません。'
    top1  = scored[0]
    name  = top1.get('name', '不明')
    style = top1.get('running_style', '差し')
    score = top1.get('total', 0)
    odds  = top1.get('win_odds', 0) or 0
    chaos = cand.get('chaos_score', 0.5)
    pace_dist = race.get('pace_dist') or {}
    if pace_dist:
        pace_key = max(pace_dist, key=pace_dist.get)
        pace_str = {'high': 'ハイペース', 'mid': 'ミドル', 'slow': 'スロー'}.get(pace_key, 'ミドル')
        pace_pct = int(pace_dist[pace_key] * 100)
    else:
        pace_str = 'ミドル'
        pace_pct = 50
    bias_str  = f'馬場は{bias_data.get("summary", "フラット")}。' if bias_data else ''
    chaos_str = '混戦模様で波乱含み。' if chaos >= 0.7 else 'やや拮抗したメンバー構成。' if chaos >= 0.4 else '能力差明確な一戦。'
    odds_str  = (f'断然の1番人気({odds:.1f}倍)だが信頼度高い。' if odds <= 3.0 else
                 f'中穴({odds:.1f}倍)で期待値十分。' if odds <= 7.0 else
                 f'穴馬({odds:.1f}倍)で高配当狙い。')
    return (f'{bias_str}{pace_str}想定({pace_pct}%)で{style}有利な展開。'
            f'{chaos_str}◎{name}(スコア{score:.1f})が最有力。{odds_str}')


def diagnose_race(race, bias_data=None):
    """全馬同確率バグ診断用：レース内の馬データ品質をチェックし問題箇所を列挙する。

    ノートブックから print(diagnose_race(race)) で使用。
    """
    horses = race.get('horses', [])
    n = len(horses)
    if n == 0:
        return '⚠ horsesが空'

    hist_ok    = sum(1 for h in horses if h.get('history'))
    odds_ok    = sum(1 for h in horses if h.get('win_odds') and 1.0 <= h['win_odds'] < 50.0)
    jockey_ok  = sum(1 for h in horses if h.get('jockey'))
    trainer_ok = sum(1 for h in horses if h.get('trainer'))
    sire_ok    = sum(1 for h in horses if h.get('sire'))
    age_var    = len({h.get('age', 4) for h in horses})
    wl_var     = len({h.get('weight_load', 56.0) for h in horses})
    style_var  = len({h.get('running_style', '差し') for h in horses})

    in_dist_dict   = sum(1 for h in horses
                         if _horse_dist_dict.get((h.get('name', ''),
                                                  dist_zone_label(race.get('distance', 1600)))))
    in_jockey_dict = sum(1 for h in horses
                         if _jockey_dict.get((h.get('jockey', ''), '', ''))
                         or _jockey_dict.get((h.get('jockey', ''),
                                              race.get('racecourse', ''),
                                              race.get('surface', ''))))
    in_trainer_dict = sum(1 for h in horses if _trainer_dict.get(h.get('trainer', '')))

    lines = [
        f'━━━ {race.get("race_name", "?")} ({n}頭) 診断 ━━━',
        f'history取得:       {hist_ok}/{n} 頭  {"❌" if hist_ok < n*0.5 else "✅"}',
        f'有効オッズ:        {odds_ok}/{n} 頭  {"⚠" if odds_ok == 0 else "✅"}',
        f'騎手名パース:      {jockey_ok}/{n} 頭  {"❌" if jockey_ok < n*0.8 else "✅"}',
        f'調教師名パース:    {trainer_ok}/{n} 頭  {"❌" if trainer_ok < n*0.8 else "✅"}',
        f'父名パース:        {sire_ok}/{n} 頭',
        f'年齢のユニーク数:  {age_var}  {"❌全馬同じ" if age_var == 1 else "✅"}',
        f'斤量のユニーク数:  {wl_var}  {"❌全馬同じ" if wl_var == 1 else "✅"}',
        f'脚質のユニーク数:  {style_var}  {"❌全馬同じ" if style_var == 1 else "✅"}',
        f'_horse_dist_dictヒット:  {in_dist_dict}/{n} 頭',
        f'_jockey_dictヒット:      {in_jockey_dict}/{n} 頭',
        f'_trainer_dictヒット:     {in_trainer_dict}/{n} 頭',
        f'horse_dist_dict総数:  {len(_horse_dist_dict)}件',
        f'jockey_dict総数:       {len(_jockey_dict)}件',
        f'trainer_dict総数:      {len(_trainer_dict)}件',
        f'bias_data:             {"あり" if bias_data else "なし"}',
    ]
    # 先頭3頭のサンプル
    lines.append('━ 先頭3頭サンプル ━')
    for h in horses[:3]:
        lines.append(
            f'  #{h.get("num", "?")} {h.get("name", "?")} '
            f'age={h.get("age", "?")} wl={h.get("weight_load", "?")} '
            f'jockey={h.get("jockey", "")!r} trainer={h.get("trainer", "")!r} '
            f'sire={h.get("sire", "")!r} odds={h.get("win_odds")} '
            f'style={h.get("running_style", "?")} hist={len(h.get("history", []))}'
        )
    return '\n'.join(lines)



def calc_rl_cl_ranks(scored_horses):
    """RL/CL の生スコアを計算し、ランクを付与する"""
    RL_FEATURES = ['rl', 'recent', 'pace', 'maturity', 'rotation']
    CL_FEATURES = ['distance', 'post', 'bias', 'jockey', 'blood']

    for h in scored_horses:
        sc = h['scores']
        rl_w = {k: _W.get(k, 0) for k in RL_FEATURES}
        cl_w = {k: _W.get(k, 0) for k in CL_FEATURES}
        rl_sum = sum(rl_w.values()) or 1
        cl_sum = sum(cl_w.values()) or 1
        h['rl_raw'] = sum(sc.get(k, 5.0) * rl_w[k] for k in RL_FEATURES) / rl_sum
        h['cl_raw'] = sum(sc.get(k, 5.0) * cl_w[k] for k in CL_FEATURES) / cl_sum

    sorted_rl = sorted(scored_horses, key=lambda h: h['rl_raw'], reverse=True)
    sorted_cl = sorted(scored_horses, key=lambda h: h['cl_raw'], reverse=True)
    for i, h in enumerate(sorted_rl):
        h['rl_rank'] = i + 1
    for i, h in enumerate(sorted_cl):
        h['cl_rank'] = i + 1
    return scored_horses


def get_xgb_rating(xfeats_list, model=None, feature_cols=None):
    """XGBの生マージン（log-odds, output_margin=True）を能力値として返す。

    Parameters
    ----------
    xfeats_list : list of dict
        各馬の特徴量辞書（calc_features_for_xgb の出力）
    model       : XGBClassifier（省略時は init_engine でロード済みのモデルを使用）
    feature_cols: 特徴量名リスト（省略時は _XGB_FEATURE_COLS を使用）

    Returns
    -------
    list of float — 各馬の能力値（インデックスは xfeats_list と対応）
    """
    import pandas as _pd
    import xgboost as _xgb
    m  = model or _XGB_FUKUSHO_MODEL
    fc = feature_cols or _XGB_FEATURE_COLS
    if m is None or not fc:
        return [0.0] * len(xfeats_list)
    rows = [{c: xf.get(c, 5.0) for c in fc} for xf in xfeats_list]
    X    = _pd.DataFrame(rows)[fc].fillna(5.0)
    dmat = _xgb.DMatrix(X, feature_names=list(fc))
    return [float(v) for v in m.get_booster().predict(dmat, output_margin=True)]


def calc_all(race, bias_data=None):
    """全馬スコア計算（XGBoost or 重み合算フォールバック）"""
    out = []
    pace_dist = calc_pace_distribution(race)
    race['pace_dist'] = pace_dist
    p_high = pace_dist.get('high', 0.2)
    p_slow = pace_dist.get('slow', 0.3)

    use_xgb = (_XGB_FUKUSHO_MODEL is not None and _XGB_FEATURE_COLS is not None)

    rc   = race.get('racecourse', '')
    surf = race.get('surface', '芝')

    # ── Pass 1: 全馬の絶対特徴量を収集 ───────────────────────────────────
    horse_data = []  # (h, sc, career, xfeats)
    for h in race['horses']:
        # 騎手・調教師名から勝率を引く。スペース除去で正規化してlookup
        if 'jockey_rate' not in h:
            jn = h.get('jockey', '').replace(' ', '').replace('　', '')
            h['jockey_rate'] = (_jockey_dict.get((jn, rc, surf))
                                or _jockey_dict.get((jn, '', ''))
                                or 0.15)
        if 'trainer_rate' not in h:
            tn = h.get('trainer', '').replace(' ', '').replace('　', '')
            h['trainer_rate'] = _trainer_dict.get(tn, 0.12)

        sc = {
            'pace':     f_pace(h, race),
            'recent':   f_recent(h, race),
            'rl':       f_rl(h, race),
            'maturity': f_maturity(h, race),
            'jockey':   f_jockey(h, race),
            'trainer':  f_trainer(h),
            'blood':    f_blood(h, race),
            'distance': f_dist_v2(h, race),
            'post':     f_post(h, race),
            'bias':     f_bias(h, race, bias_data),
            'weight':   f_weight(h),
            'rotation': f_rotation(h, race),
        }
        career = analyze_career(h, race)
        xfeats = calc_features_for_xgb(h, race) if use_xgb else {}
        horse_data.append((h, sc, career, xfeats))

    # ── 相対特徴量をフィールド全体で一括計算 ────────────────────────────
    if use_xgb:
        add_relative_features([xf for _, _, _, xf in horse_data])

    # ── Pass 2: XGB予測（相対特徴量込み） ───────────────────────────────
    for h, sc, career, xfeats in horse_data:
        if use_xgb:
            try:
                import pandas as _pd_xgb
                import xgboost as _xgb_lib
                xrow   = {c: xfeats.get(c, 5.0) for c in _XGB_FEATURE_COLS}
                X_pred = _pd_xgb.DataFrame([xrow])[_XGB_FEATURE_COLS].fillna(5.0)
                prob   = float(_XGB_FUKUSHO_MODEL.predict_proba(X_pred)[0][1])
                raw_prob = prob
                # 能力値（Phase1）: XGB生マージン（sigmoid前のlog-odds）
                _dmat  = _xgb_lib.DMatrix(X_pred, feature_names=list(_XGB_FEATURE_COLS))
                rating = float(_XGB_FUKUSHO_MODEL.get_booster().predict(_dmat, output_margin=True)[0])
                # XGB専用キャリブレーターを適用（複勝確率表示用）
                if _XGB_CALIBRATOR is not None:
                    import numpy as _np_cal
                    cal_prob = float(_np_cal.clip(_XGB_CALIBRATOR.transform([prob])[0], 0.01, 0.99))
                else:
                    cal_prob = prob
                # softmaxにはキャリブ前のraw_probを使う（calibratorのフロアで潰れるのを防ぐ）
                # round()は精度損失で同一確率が発生するため使わない
                total  = raw_prob * 10
                prob   = cal_prob  # 表示用複勝確率はcal_probを保持
            except Exception:
                total = sum(sc.get(k, 5.0) * _W.get(k, 0) for k in _W if _W.get(k, 0) > 0)
                total = apply_career_flags(total, career)
                prob  = 1 / (1 + math.exp(-(total - 5.5) * .8))
                rating = total - 5.0  # ルールベーススコアを0中心にシフト
        else:
            total = sum(sc.get(k, 5.0) * _W.get(k, 0) for k in _W if _W.get(k, 0) > 0)
            total = apply_career_flags(total, career)
            prob  = 1 / (1 + math.exp(-(total - 5.5) * .8))
            rating = total - 5.0
            if _CALIBRATOR is not None:
                prob = float(_CALIBRATOR.transform([prob])[0])

        win_odds    = h.get('win_odds') or 10.0
        market_prob = round(1 / win_odds, 4)
        out.append({
            **h,
            'scores':      sc,
            'total':       round(total, 2),
            'win_prob':    prob,
            'cal_prob':    prob,   # キャリブレーション済み複勝確率（win_probはsoftmaxで後ほど上書きされるため別キーで保持）
            'career':      career,
            'market_prob': market_prob,
            'pop_gap':     round(prob - market_prob, 4),
            'rating':      round(rating, 4),  # 能力値（XGB生マージン or ルールベースtotal-5）
        })

    if not out:
        return out

    n          = len(out)
    scores     = [h['total'] for h in out]
    max_score  = max(scores)
    min_score  = min(scores)
    mean_score = sum(scores) / n
    std_score  = (sum((s - mean_score) ** 2 for s in scores) / n) ** 0.5 or 0.1

    late_vals = []
    for h in out:
        hist = h.get('history', [])
        ag = hist[0].get('agari3f', 37.0) if hist and hist[0].get('agari3f') else 37.0
        late_vals.append(float(ag))
    mean_late = sum(late_vals) / len(late_vals) if late_vals else 37.0

    early_vals = []
    for h in out:
        style = h.get('running_style', '差し')
        early_vals.append({'逃げ': 1, '先行': 2, '差し': 3, '追込': 4}.get(style, 3))

    sorted_scores = sorted(scores, reverse=True)
    sorted_late   = sorted(late_vals)

    for i, h in enumerate(out):
        h['f_score_rank']     = sorted_scores.index(h['total']) + 1
        h['f_score_gap']      = round(max_score - h['total'], 2)
        h['f_score_gap_norm'] = round((max_score - h['total']) / std_score, 2)
        own_front = h.get('scores', {}).get('pace', 5.0)
        race_front_mean = sum(x.get('scores', {}).get('pace', 5.0) for x in out) / n
        h['f_pos_vs_field']   = round(own_front - race_front_mean, 2)
        h['f_late_vs_field']  = round(mean_late - late_vals[i], 2)
        h['f_early_rank']     = early_vals[i]
        h['f_late_rank']      = sorted_late.index(late_vals[i]) + 1
        h['f_front_adv']      = round(h['f_pos_vs_field'] * p_slow, 3)
        h['f_back_adv']       = round(-h['f_pos_vs_field'] * p_high, 3)
        score_range = max(max_score - min_score, 0.1)
        h['f_relative_score'] = round((h['total'] - min_score) / score_range * 10, 2)

    RELATIVE_WEIGHT = 0.10
    for h in out:
        pace_bonus = (h['f_front_adv'] + h['f_back_adv']) * 0.5
        rel = h['f_relative_score']
        h['total'] = round(
            h['total'] * (1 - RELATIVE_WEIGHT) + rel * RELATIVE_WEIGHT + pace_bonus, 2
        )

    all_totals = [h['total'] for h in out]
    win_probs = softmax_probs(all_totals, temperature=2.0)
    # 旧IsotonicCalibratorは同一確率フロアを生じさせるためスキップ
    # （XGB raw_probを入力しているため再calibrationは不要）
    for h, p in zip(out, win_probs):
        h['win_prob'] = round(p, 6)

    # Harville: top2/top3 per-horse probabilities from win_prob
    win_ps = [h['win_prob'] for h in out]
    harville = calc_harville_probs(win_ps)
    for h, (t2, t3) in zip(out, harville):
        h['top2_prob'] = round(t2, 6)
        h['top3_prob'] = round(t3, 6)

    for x in out:
        x['pn']      = x['win_prob']
        x['pop_gap'] = round(x['win_prob'] - x['market_prob'], 4)

    out = sorted(out, key=lambda x: x['total'], reverse=True)
    calc_rl_cl_ranks(out)

    # RL順位をwin_prob（AI確率）の順位に統一（rl_rawは旧加重スコアで逆転が起きるため）
    for i, h in enumerate(sorted(out, key=lambda h: h['win_prob'], reverse=True)):
        h['rl_rank'] = i + 1

    return out

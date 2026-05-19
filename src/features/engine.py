"""
特徴量エンジン v5 クリーン版
ノートブックのセル12から抽出。グローバル変数はinit_engine()で注入する。
"""
import math
import os
import pickle

from src.utils.config import POST_BIAS, POST_BIAS_BY_ZONE

# ── グローバルコンテキスト（init_engine()で設定） ─────────────────
_XGB_FUKUSHO_MODEL = None
_XGB_FEATURE_COLS  = None
_CALIBRATOR        = None
_PACE_MODEL        = None
_W                 = {
    'pace': 0.25, 'recent': 0.20, 'jockey': 0.15, 'trainer': 0.10,
    'blood': 0.10, 'distance': 0.08, 'post': 0.06, 'bias': 0.04, 'weight': 0.02,
}
_horse_dist_dict   = {}
_horse_course_dict = {}
_horse_venue_dist_dict = {}  # (name, venue, dist_zone, surface) → {出走, 勝率, 複勝率}
_post_zone_bias        = {}  # (venue, dist_zone) → float  正=内枠有利, 負=外枠有利


def init_engine(base_dir,
                xgb_model=None, xgb_feature_cols=None,
                calibrator=None, pace_model=None,
                weights=None,
                horse_dist_dict=None, horse_course_dict=None,
                horse_venue_dist_dict=None):
    """エンジンのグローバル変数を設定する。ノートブックのセル4で呼ぶ"""
    global _XGB_FUKUSHO_MODEL, _XGB_FEATURE_COLS, _CALIBRATOR, _PACE_MODEL
    global _W, _horse_dist_dict, _horse_course_dict, _horse_venue_dist_dict, _post_zone_bias

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
        w_keys = ['pace', 'recent', 'jockey', 'trainer', 'blood', 'distance', 'post', 'bias', 'weight']
        new_w = {k: opt[k] for k in w_keys if k in opt}
        if new_w and abs(sum(new_w.values()) - 1.0) < 0.05:
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

    # pkl未作成の場合はDBから自動構築
    if not _horse_dist_dict or not _horse_venue_dist_dict or not _post_zone_bias:
        _build_horse_dicts(base_dir)

    xgb_ok = _XGB_FUKUSHO_MODEL is not None
    print(f'✅ 特徴量エンジン初期化完了 (XGB:{xgb_ok}, Cal:{_CALIBRATOR is not None}, '
          f'馬別統計:{len(_horse_dist_dict)}件, 枠バイアス:{len(_post_zone_bias)}件)')


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
    conn.close()

    dist_stat  = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    course_stat = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    vd_stat    = defaultdict(lambda: {'出走': 0, '1着': 0, '複勝': 0})
    post_stat  = defaultdict(lambda: {'i_w': 0, 'i_n': 0, 'o_w': 0, 'o_n': 0})

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
    ]:
        with open(os.path.join(data_dir, fname), 'wb') as f:
            pickle.dump(obj, f)

    _horse_dist_dict       = new_dist
    _horse_course_dict     = new_course
    _horse_venue_dist_dict = new_vd
    _post_zone_bias        = new_post_bias
    print(f'  [統計構築] 距離:{len(new_dist)}件 コース:{len(new_course)}件 '
          f'距離×会場:{len(new_vd)}件 枠バイアス:{len(new_post_bias)}件')


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


def f_recent(h, race):
    hist = h.get('history', [])
    if not hist:
        odds = h.get('win_odds') or 10.0
        return max(1.0, min(8.0, 10.0 - math.log(odds, 2)))
    ws = [.75 ** i for i in range(len(hist))]
    tw = sum(ws)
    sc = 0.0
    for i, r in enumerate(hist):
        place     = r.get('place', 10)
        finishers = max(r.get('finishers', 16), 2)
        margin    = r.get('margin', 0.0)
        if place == 1:   ps = 10.0
        elif place == 2: ps = 7.0
        elif place == 3: ps = 5.0
        else: ps = max(0.0, 4.0 * (1.0 - (place - 3) / max(finishers - 3, 1)))
        if place == 1:       ma = 0.0
        elif margin <= 0.0:  ma = 0.0
        elif margin <= 0.2:  ma = +0.5
        elif margin <= 0.5:  ma = 0.0
        elif margin <= 1.0:  ma = -0.5
        else: ma = min(-2.0, -margin * 0.5)
        last_3f  = r.get('last_3f') or 0
        first_3f = r.get('first_3f') or 0
        corner_3 = r.get('corner_3') or 0
        surf     = r.get('surface', '芝')
        dist     = r.get('distance', 1600)
        cond     = r.get('track_condition', '良')
        fins     = max(r.get('finishers', 16), 8)
        if last_3f > 0:
            pi = calc_performance_index(last_3f, first_3f, corner_3, fins, dist, surf, cond)
            speed_bonus = max(-2.0, min(2.0, (pi - 50) / 10))
        else:
            speed_bonus = 0.0
        sc += (ws[i] / tw) * (ps + ma + speed_bonus)
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
        'slow': {'逃げ': +2, '先行': +3, '差し': +1, '追込':  0},
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
        c = int(dist)
        b = h.get('best_distance', c)
        d = abs(c - b)
        scores.append(10 if d == 0 else 8.5 if d <= 200 else 7 if d <= 400 else 5 if d <= 600 else 3)

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

    return round(sum(scores) / len(scores), 2)


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


def calc_features_for_xgb(h, race):
    """XGBoost複勝予測モデルへの入力特徴量を生成"""
    import numpy as _np
    feats = {}
    hist  = h.get('history', [])
    rc    = race.get('racecourse', '')
    surf  = race.get('surface', '芝')
    dist  = int(race.get('distance', 1600) or 1600)
    n     = max(len(race.get('horses', [])), 8)

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

    return feats


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

    for h in race['horses']:
        # 騎手・調教師名から勝率を引く（出馬表から名前が取れた場合）
        if 'jockey_rate' not in h:
            jn = h.get('jockey', '')
            h['jockey_rate'] = (_jockey_dict.get((jn, rc, surf))
                                or _jockey_dict.get((jn, '', ''))
                                or 0.15)
        if 'trainer_rate' not in h:
            tn = h.get('trainer', '')
            h['trainer_rate'] = _trainer_dict.get(tn, 0.12)

        sc = {
            'pace':     f_pace(h, race),
            'recent':   f_recent(h, race),
            'jockey':   f_jockey(h, race),
            'trainer':  f_trainer(h),
            'blood':    f_blood(h, race),
            'distance': f_dist_v2(h, race),
            'post':     f_post(h, race),
            'bias':     f_bias(h, race, bias_data),
            'weight':   f_weight(h),
        }
        career = analyze_career(h, race)

        if use_xgb:
            try:
                import pandas as _pd_xgb
                xfeats = calc_features_for_xgb(h, race)
                xrow   = {c: xfeats.get(c, 5.0) for c in _XGB_FEATURE_COLS}
                X_pred = _pd_xgb.DataFrame([xrow])[_XGB_FEATURE_COLS].fillna(5.0)
                prob   = float(_XGB_FUKUSHO_MODEL.predict_proba(X_pred)[0][1])
                if _CALIBRATOR is not None:
                    import numpy as _np_cal
                    prob = float(_np_cal.clip(_CALIBRATOR.transform([prob])[0], 0.01, 0.99))
                total  = round(prob * 10, 2)
            except Exception:
                total = sum(sc[k] * _W[k] for k in _W)
                total = apply_career_flags(total, career)
                prob  = 1 / (1 + math.exp(-(total - 5.5) * .8))
        else:
            total = sum(sc[k] * _W[k] for k in _W)
            total = apply_career_flags(total, career)
            prob  = 1 / (1 + math.exp(-(total - 5.5) * .8))
            if _CALIBRATOR is not None:
                prob = float(_CALIBRATOR.transform([prob])[0])

        win_odds    = h.get('win_odds') or 10.0
        market_prob = round(1 / win_odds, 4)
        out.append({
            **h,
            'scores':      sc,
            'total':       round(total, 2),
            'win_prob':    prob,
            'career':      career,
            'market_prob': market_prob,
            'pop_gap':     round(prob - market_prob, 4),
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

    # Softmax: race-relative win probabilities that sum to 1 across the field.
    # Replaces per-horse sigmoid which gave ~0.5 to every horse regardless of field size.
    all_totals = [h['total'] for h in out]
    max_t = max(all_totals)
    exp_scores = [math.exp((t - max_t) * 0.8) for t in all_totals]
    sum_exp = sum(exp_scores)
    for h, e in zip(out, exp_scores):
        h['win_prob'] = round(e / sum_exp, 6)

    if _CALIBRATOR is not None:
        # Calibrator shifts individual probabilities; renormalize to keep sum=1
        probs_cal = [float(_CALIBRATOR.transform([h['win_prob']])[0]) for h in out]
        s_cal = sum(probs_cal) or 1.0
        for h, p in zip(out, probs_cal):
            h['win_prob'] = round(p / s_cal, 6)

    # Harville: top2/top3 per-horse probabilities from win_prob
    win_ps = [h['win_prob'] for h in out]
    harville = calc_harville_probs(win_ps)
    for h, (t2, t3) in zip(out, harville):
        h['top2_prob'] = round(t2, 6)
        h['top3_prob'] = round(t3, 6)

    for x in out:
        x['pn']      = x['win_prob']
        x['pop_gap'] = round(x['win_prob'] - x['market_prob'], 4)

    return sorted(out, key=lambda x: x['total'], reverse=True)

"""
過去レース実績から特徴量重みをSciPyで最適化するスクリプト。
history.db（horse_history + race_history）を使用。

実行方法（Google Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.tune_weights import run_tuning
    run_tuning(BASE_DIR)
"""
import argparse
import json
import math
import os
import pickle
import sqlite3
import sys
from collections import defaultdict

WEIGHT_KEYS = ['pace', 'recent', 'jockey', 'trainer', 'blood', 'distance', 'post', 'bias', 'weight']
DEFAULT_W   = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02]


# ── DBロード ─────────────────────────────────────────────────────

def _find_db(base_dir):
    """history.db または keiba.db を探して返す"""
    for name in ['history.db', 'keiba.db']:
        p = os.path.join(base_dir, 'data', name)
        if os.path.exists(p):
            return p, name
    return None, None


def _diagnose_db(base_dir):
    db_path, db_name = _find_db(base_dir)
    if not db_path:
        print('  [DB診断] DBファイルが見つかりません')
        return
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f'\n  [DB診断] {db_name}')
    print(f'    テーブル: {tables}')
    for t in tables:
        n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        print(f'    {t}: {n:,}件')
    conn.close()


def _load_jockey_dict(base_dir):
    path = os.path.join(base_dir, 'data', 'jockey_db.csv')
    if not os.path.exists(path):
        return {}
    try:
        import csv
        d = {}
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row.get('騎手', ''), row.get('競馬場', ''), row.get('surface', ''))
                try:
                    d[key] = float(row['勝率'])
                except (KeyError, ValueError):
                    pass
        return d
    except Exception:
        return {}


def _load_trainer_dict(base_dir):
    path = os.path.join(base_dir, 'data', 'trainer_db.csv')
    if not os.path.exists(path):
        return {}
    try:
        import csv
        d = {}
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    d[row['調教師']] = float(row['勝率'])
                except (KeyError, ValueError):
                    pass
        return d
    except Exception:
        return {}


def load_training_data(base_dir):
    """history.db から訓練データを構築する。

    horse_history の各レースについて、
    「そのレース以前の各馬の実績」を history キーに詰めて
    エンジンのスコア関数に渡せる形にする。

    Returns:
        samples : [(scores_matrix, winner_idx), ...]
        meta    : {races_loaded, races_skipped, horses_per_race_avg}
    """
    db_path, db_name = _find_db(base_dir)
    if not db_path:
        raise FileNotFoundError(f'history.db / keiba.db が見つかりません: {base_dir}/data/')

    jdict = _load_jockey_dict(base_dir)
    tdict = _load_trainer_dict(base_dir)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # テーブル名を確認して分岐
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if 'horse_history' in tables:
        # ── history.db スキーマ ──────────────────────────────────
        # race_history から全レースIDを取得
        race_rows = conn.execute("""
            SELECT rh.race_id, rh.date, rh.racecourse, rh.distance, rh.surface
            FROM race_history rh
            WHERE EXISTS (
                SELECT 1 FROM horse_history hh
                WHERE hh.race_id = rh.race_id AND hh.place = 1
            )
            ORDER BY rh.date
        """).fetchall()

        # horse_history を全件ロード（馬名→時系列リスト）
        all_horse_rows = conn.execute("""
            SELECT hh.race_id, hh.date, hh.racecourse, hh.horse_name,
                   hh.horse_num, hh.place, hh.running_style,
                   hh.agari3f, hh.jockey, hh.trainer,
                   hh.corner_3, hh.distance, hh.surface,
                   rh.first_3f,
                   (SELECT COUNT(*) FROM horse_history hh2
                    WHERE hh2.race_id = hh.race_id) AS finishers
            FROM horse_history hh
            LEFT JOIN race_history rh ON hh.race_id = rh.race_id
            ORDER BY hh.date
        """).fetchall()

        # 馬名 → 出走履歴リスト（日付順）
        horse_hist_map = defaultdict(list)
        # race_id → {horse_num: row}
        race_horse_map = defaultdict(dict)
        for row in all_horse_rows:
            d = dict(row)
            horse_hist_map[d['horse_name']].append(d)
            race_horse_map[d['race_id']][d['horse_num']] = d

        # 騎手・調教師勝率をDBから補完
        if not jdict:
            rows = conn.execute("""
                SELECT jockey, COUNT(*) AS runs,
                       SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
                FROM horse_history
                WHERE jockey IS NOT NULL AND jockey != ''
                GROUP BY jockey HAVING COUNT(*) >= 10
            """).fetchall()
            jdict = {(r['jockey'], '', ''): r['wins'] / r['runs'] for r in rows}

        if not tdict:
            rows = conn.execute("""
                SELECT trainer, COUNT(*) AS runs,
                       SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
                FROM horse_history
                WHERE trainer IS NOT NULL AND trainer != ''
                GROUP BY trainer HAVING COUNT(*) >= 10
            """).fetchall()
            tdict = {r['trainer']: r['wins'] / r['runs'] for r in rows}

        print(f'  騎手DB: {len(jdict)}件 / 調教師DB: {len(tdict)}件')

        winner_map = {r['race_id']: r['horse_num'] for r in conn.execute(
            "SELECT race_id, horse_num FROM horse_history WHERE place = 1")}

    elif 'races' in tables:
        # ── keiba.db スキーマ（フォールバック） ─────────────────
        race_rows = conn.execute("""
            SELECT r.id AS race_id, r.date, r.racecourse, r.distance, r.surface
            FROM races r
            WHERE r.raw_json IS NOT NULL
              AND EXISTS (SELECT 1 FROM results res WHERE res.race_id = r.id AND res.place = 1)
            ORDER BY r.date
        """).fetchall()
        winner_map = {r[0]: r[1] for r in conn.execute(
            "SELECT race_id, horse_num FROM results WHERE place = 1")}
        horse_hist_map = None
        race_horse_map = None

        if not jdict:
            rows = conn.execute("""
                SELECT jockey, COUNT(*) AS runs,
                       SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
                FROM results WHERE jockey IS NOT NULL AND jockey != ''
                GROUP BY jockey HAVING COUNT(*) >= 5
            """).fetchall()
            jdict = {(r[0], '', ''): r[2] / r[1] for r in rows}
        if not tdict:
            rows = conn.execute("""
                SELECT trainer, COUNT(*) AS runs,
                       SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
                FROM results WHERE trainer IS NOT NULL AND trainer != ''
                GROUP BY trainer HAVING COUNT(*) >= 5
            """).fetchall()
            tdict = {r[0]: r[2] / r[1] for r in rows}
        print(f'  騎手DB: {len(jdict)}件 / 調教師DB: {len(tdict)}件')
    else:
        conn.close()
        raise RuntimeError(f'未知のDBスキーマ。テーブル一覧: {tables}')

    conn.close()

    # エンジンのスコア関数をインポート
    from src.features.engine import (
        f_pace, f_recent, f_jockey, f_trainer,
        f_blood, f_dist_v2, f_post, f_weight,
        analyze_career, apply_career_flags,
        calc_pace_distribution,
    )

    samples = []
    skipped = 0
    total_h = 0

    for race_row in race_rows:
        race_id  = race_row['race_id'] if hasattr(race_row, 'keys') else race_row[0]
        date_str = race_row['date']    if hasattr(race_row, 'keys') else race_row[1]
        rc       = race_row['racecourse'] if hasattr(race_row, 'keys') else race_row[2]
        dist     = race_row['distance']   if hasattr(race_row, 'keys') else race_row[3]
        surf     = race_row['surface']    if hasattr(race_row, 'keys') else race_row[4]

        winner_num = winner_map.get(race_id)
        if winner_num is None:
            skipped += 1
            continue

        if horse_hist_map is not None:
            # history.db: race_horse_map から当該レースの全馬を取得
            runners = race_horse_map.get(race_id, {})
            if len(runners) < 3:
                skipped += 1
                continue

            race = {
                'racecourse': rc, 'surface': surf or '芝',
                'distance': int(dist or 1600),
                'num_horses': len(runners),
                'escape_count': 0, 'front_count': 0,
                'date': date_str,
            }
            try:
                race['pace_dist'] = calc_pace_distribution(race)
            except Exception:
                race['pace_dist'] = {'high': 0.3, 'mid': 0.4, 'slow': 0.3}

            scores_matrix = []
            winner_idx    = -1

            for idx, (hnum, hrow) in enumerate(sorted(runners.items())):
                hname = hrow['horse_name']
                jockey  = hrow.get('jockey', '') or ''
                trainer = hrow.get('trainer', '') or ''

                # 当該レースより前の実績を取得
                prior = [
                    r for r in horse_hist_map[hname]
                    if r['date'] < date_str
                ]
                # 直近10件に絞る
                prior = prior[-10:]

                # history形式に変換
                history = []
                for pr in prior:
                    try:
                        ag = float(pr['agari3f']) if pr['agari3f'] else 0
                        c3 = int(float(pr['corner_3'])) if pr['corner_3'] is not None else 0
                        history.append({
                            'place':           int(pr['place'] or 10),
                            'finishers':       int(pr.get('finishers') or 16),
                            'distance':        int(pr['distance'] or 1600),
                            'surface':         pr['surface'] or '芝',
                            'class':           '',
                            'margin':          0.0,
                            'agari3f':         ag,
                            'last_3f':         ag,
                            'first_3f':        float(pr.get('first_3f') or 0),
                            'corner_3':        c3,
                            'date':            pr['date'],
                            'racecourse':      pr.get('racecourse', ''),
                            'track_condition': '良',
                        })
                    except Exception:
                        pass

                jockey_r  = (jdict.get((jockey, rc, surf))
                             or jdict.get((jockey, '', ''))
                             or 0.15)
                trainer_r = tdict.get(trainer, 0.12)

                h = {
                    'name':         hname,
                    'num':          hnum,
                    'horse_num':    hnum,
                    'post_position': hnum,
                    'win_odds':     10.0,
                    'jockey':       jockey,
                    'trainer':      trainer,
                    'jockey_rate':  jockey_r,
                    'trainer_rate': trainer_r,
                    'running_style': hrow.get('running_style', '差し') or '差し',
                    'weight_load':  56.0,
                    'age':          4,
                    'sire':         '',
                    'dam_sire':     '',
                    'history':      history,
                }

                try:
                    sc = {
                        'pace':     f_pace(h, race),
                        'recent':   f_recent(h, race),
                        'jockey':   f_jockey(h, race),
                        'trainer':  f_trainer(h),
                        'blood':    f_blood(h, race),
                        'distance': f_dist_v2(h, race),
                        'post':     f_post(h, race),
                        'bias':     5.0,
                        'weight':   f_weight(h),
                    }
                    career_adj = apply_career_flags(0.0, analyze_career(h, race))
                except Exception:
                    sc         = {k: 5.0 for k in WEIGHT_KEYS}
                    career_adj = 0.0

                scores_matrix.append({'sc': sc, 'adj': career_adj})
                if hnum == winner_num:
                    winner_idx = idx

        else:
            # keiba.db: raw_json から構築
            conn2 = sqlite3.connect(db_path)
            raw = conn2.execute(
                "SELECT raw_json FROM races WHERE id=?", (race_id,)).fetchone()
            conn2.close()
            if not raw or not raw[0]:
                skipped += 1
                continue
            try:
                race = json.loads(raw[0])
            except Exception:
                skipped += 1
                continue
            try:
                race['pace_dist'] = calc_pace_distribution(race)
            except Exception:
                race['pace_dist'] = {'high': 0.3, 'mid': 0.4, 'slow': 0.3}

            horses = race.get('horses', [])
            if len(horses) < 3:
                skipped += 1
                continue

            scores_matrix = []
            winner_idx    = -1
            for idx, h in enumerate(horses):
                jockey  = h.get('jockey', '')
                trainer = h.get('trainer', '')
                h['jockey_rate']  = (jdict.get((jockey, rc, surf))
                                     or jdict.get((jockey, '', ''))
                                     or h.get('jockey_rate', 0.15))
                h['trainer_rate'] = tdict.get(trainer) or h.get('trainer_rate', 0.12)
                try:
                    sc = {
                        'pace':     f_pace(h, race),
                        'recent':   f_recent(h, race),
                        'jockey':   f_jockey(h, race),
                        'trainer':  f_trainer(h),
                        'blood':    f_blood(h, race),
                        'distance': f_dist_v2(h, race),
                        'post':     f_post(h, race),
                        'bias':     5.0,
                        'weight':   f_weight(h),
                    }
                    career_adj = apply_career_flags(0.0, analyze_career(h, race))
                except Exception:
                    sc         = {k: 5.0 for k in WEIGHT_KEYS}
                    career_adj = 0.0
                scores_matrix.append({'sc': sc, 'adj': career_adj})
                hnum = h.get('num') or h.get('horse_num')
                if hnum == winner_num:
                    winner_idx = idx

        if winner_idx < 0 or len(scores_matrix) < 3:
            skipped += 1
            continue

        samples.append((scores_matrix, winner_idx))
        total_h += len(scores_matrix)

    meta = {
        'races_loaded':        len(samples),
        'races_skipped':       skipped,
        'horses_per_race_avg': round(total_h / max(len(samples), 1), 1),
    }
    return samples, meta


# ── 最適化 ───────────────────────────────────────────────────────

def softmax_probs(scores_matrix, weights, temperature=0.8):
    totals = [
        sum(h['sc'][k] * weights[k] for k in WEIGHT_KEYS) + h['adj']
        for h in scores_matrix
    ]
    max_t = max(totals)
    exp_s = [math.exp((t - max_t) * temperature) for t in totals]
    s     = sum(exp_s) or 1.0
    return [e / s for e in exp_s]


def neg_log_likelihood(w_raw, samples):
    w_pos = [max(1e-4, x) for x in w_raw]
    s     = sum(w_pos)
    w     = {k: w_pos[i] / s for i, k in enumerate(WEIGHT_KEYS)}
    total = 0.0
    for scores_matrix, winner_idx in samples:
        probs = softmax_probs(scores_matrix, w)
        total += math.log(max(1e-9, probs[winner_idx]))
    return -total / len(samples)


def accuracy_at_1(w_arr, samples):
    w_pos = [max(1e-4, x) for x in w_arr]
    s     = sum(w_pos)
    w     = {k: w_pos[i] / s for i, k in enumerate(WEIGHT_KEYS)}
    hits  = 0
    for scores_matrix, winner_idx in samples:
        probs = softmax_probs(scores_matrix, w)
        if probs.index(max(probs)) == winner_idx:
            hits += 1
    return hits / len(samples)


def run_tuning(base_dir, n_restarts=5, verbose=True):
    """重みチューニングを実行して optimal_weights.json に保存する。"""
    try:
        from scipy.optimize import minimize
    except ImportError:
        print('❌ scipy が必要です: !pip install scipy')
        return None

    import numpy as np
    import random

    if verbose:
        print('=== 重みチューニング開始 ===')
        _diagnose_db(base_dir)
        print('  データロード中...')

    samples, meta = load_training_data(base_dir)
    if verbose:
        print(f'  レース: {meta["races_loaded"]:,}件 (スキップ: {meta["races_skipped"]}件)')
        print(f'  平均出走頭数: {meta["horses_per_race_avg"]}頭')

    if len(samples) < 20:
        print(f'❌ データ不足（{len(samples)}件）。最低20レース必要です。')
        return None
    if len(samples) < 100:
        print(f'⚠ データ少なめ（{len(samples)}件）。結果の精度は限定的ですが続行します。')

    base_loss = neg_log_likelihood(DEFAULT_W, samples)
    base_acc  = accuracy_at_1(DEFAULT_W, samples)
    if verbose:
        print(f'\n  [ベースライン] NLL={base_loss:.4f}, Acc@1={base_acc:.3f}')

    constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0},)
    bounds      = [(0.01, 0.50)] * len(WEIGHT_KEYS)

    best_result = None
    best_loss   = float('inf')

    for trial in range(n_restarts):
        if trial == 0:
            w0 = DEFAULT_W[:]
        else:
            r  = [random.random() for _ in WEIGHT_KEYS]
            w0 = [x / sum(r) for x in r]

        res = minimize(
            neg_log_likelihood,
            w0,
            args=(samples,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8},
        )
        if res.fun < best_loss:
            best_loss   = res.fun
            best_result = res

        if verbose:
            print(f'  [試行 {trial+1}/{n_restarts}] NLL={res.fun:.4f}  {"✅" if res.success else "⚠"}')

    opt_w_arr = np.clip(best_result.x, 0, None)
    opt_w_arr /= opt_w_arr.sum()
    opt_w     = {k: round(float(opt_w_arr[i]), 4) for i, k in enumerate(WEIGHT_KEYS)}

    opt_loss = neg_log_likelihood(list(opt_w_arr), samples)
    opt_acc  = accuracy_at_1(list(opt_w_arr), samples)

    if verbose:
        print(f'\n  [最適化結果] NLL={opt_loss:.4f} (改善: {base_loss-opt_loss:+.4f}), '
              f'Acc@1={opt_acc:.3f} (改善: {opt_acc-base_acc:+.3f})')
        print('\n  最適重み:')
        for k, v in sorted(opt_w.items(), key=lambda x: -x[1]):
            d   = round(v - dict(zip(WEIGHT_KEYS, DEFAULT_W))[k], 4)
            bar = '█' * int(v * 40)
            print(f'    {k:10s}: {v:.4f} ({d:+.4f})  {bar}')

    out_path = os.path.join(base_dir, 'data', 'optimal_weights.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = {
        **opt_w,
        '_meta': {
            'races':        meta['races_loaded'],
            'nll':          round(opt_loss, 6),
            'accuracy_at1': round(opt_acc, 4),
            'baseline_nll': round(base_loss, 6),
            'baseline_acc': round(base_acc, 4),
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存完了: {out_path}')
    return opt_w


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--restarts', type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, args.base_dir)
    from src.features.engine import init_engine
    init_engine(args.base_dir)
    run_tuning(args.base_dir, n_restarts=args.restarts)

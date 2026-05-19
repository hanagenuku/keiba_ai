"""
過去レース実績から特徴量重みをSciPyで最適化するスクリプト。

実行方法（Google Colab）:
    import sys; sys.path.insert(0, f'{BASE_DIR}/src')
    from tools.tune_weights import run_tuning
    run_tuning(BASE_DIR)

コマンドライン:
    python -m src.tools.tune_weights --base-dir /path/to/keiba_ai
"""
import argparse
import json
import math
import os
import pickle
import sqlite3
import sys

WEIGHT_KEYS = ['pace', 'recent', 'jockey', 'trainer', 'blood', 'distance', 'post', 'bias', 'weight']
DEFAULT_W   = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02]


# ── データロード ─────────────────────────────────────────────────

def _load_jockey_dict(base_dir):
    """jockey_db.csv → {(騎手名, 競馬場, surface): 勝率}"""
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
    except Exception as e:
        print(f'  ⚠ jockey_db 読み込みエラー: {e}')
        return {}


def _load_trainer_dict(base_dir):
    """trainer_db.csv → {調教師名: 勝率}"""
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
    except Exception as e:
        print(f'  ⚠ trainer_db 読み込みエラー: {e}')
        return {}


def _load_jockey_from_results(db_path):
    """results テーブルから騎手勝率を計算（jockey_db がない場合のフォールバック）"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT jockey, COUNT(*) AS runs,
               SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
        FROM results
        WHERE jockey IS NOT NULL AND jockey != ''
        GROUP BY jockey HAVING COUNT(*) >= 10
    """).fetchall()
    conn.close()
    return {r[0]: r[2] / r[1] for r in rows}


def _load_trainer_from_results(db_path):
    """results テーブルから調教師勝率を計算"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT trainer, COUNT(*) AS runs,
               SUM(CASE WHEN place=1 THEN 1 ELSE 0 END) AS wins
        FROM results
        WHERE trainer IS NOT NULL AND trainer != ''
        GROUP BY trainer HAVING COUNT(*) >= 10
    """).fetchall()
    conn.close()
    return {r[0]: r[2] / r[1] for r in rows}


def _diagnose_db(base_dir):
    """DBの中身を簡単に確認して原因を報告する"""
    db_path = os.path.join(base_dir, 'data', 'keiba.db')
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    try:
        n_races   = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        n_raw     = conn.execute("SELECT COUNT(*) FROM races WHERE raw_json IS NOT NULL").fetchone()[0]
        n_results = conn.execute("SELECT COUNT(DISTINCT race_id) FROM results").fetchone()[0]
        n_place1  = conn.execute("SELECT COUNT(DISTINCT race_id) FROM results WHERE place=1").fetchone()[0]
        n_matched = conn.execute("""
            SELECT COUNT(*) FROM races r
            WHERE r.raw_json IS NOT NULL
              AND EXISTS (SELECT 1 FROM results res WHERE res.race_id=r.id AND res.place=1)
        """).fetchone()[0]
        print(f'\n  [DB診断]')
        print(f'    races テーブル総数  : {n_races}件')
        print(f'    raw_json あり       : {n_raw}件')
        print(f'    results に紐づく    : {n_results}件')
        print(f'    place=1 あり        : {n_place1}件')
        print(f'    チューニング対象    : {n_matched}件')
        if n_races > n_matched:
            print(f'    ⚠ {n_races - n_matched}件が除外 '
                  '（raw_jsonなし or resultsに1着記録なし）')
    except Exception as e:
        print(f'    診断エラー: {e}')
    finally:
        conn.close()


def load_training_data(base_dir):
    """DBとCSVからトレーニングデータを構築する。

    Returns:
        samples: [(scores_matrix, winner_idx), ...]
                 scores_matrix: [{factor: score, ...}, ...] 馬ごとの9因子スコア
                 winner_idx   : 1着馬のインデックス（scores_matrix内）
        meta   : {races_loaded, horses_per_race_avg, ...}
    """
    db_path = os.path.join(base_dir, 'data', 'keiba.db')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'DB not found: {db_path}')

    # 騎手・調教師勝率の取得
    jdict  = _load_jockey_dict(base_dir)
    tdict  = _load_trainer_dict(base_dir)
    if not jdict:
        jdict_fallback = _load_jockey_from_results(db_path)
        # fallback: keyed by name only (no racecourse/surface breakdown)
        jdict = {(k, '', ''): v for k, v in jdict_fallback.items()}
    if not tdict:
        tdict = _load_trainer_from_results(db_path)
    print(f'  騎手DB: {len(jdict)}件 / 調教師DB: {len(tdict)}件')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    race_rows = conn.execute("""
        SELECT r.id, r.raw_json
        FROM races r
        WHERE r.raw_json IS NOT NULL
          AND EXISTS (SELECT 1 FROM results res WHERE res.race_id = r.id AND res.place = 1)
        ORDER BY r.date
    """).fetchall()

    # 各レースの1着馬番号をキャッシュ
    winners = {}
    for row in conn.execute("SELECT race_id, horse_num FROM results WHERE place = 1"):
        winners[row[0]] = row[1]

    conn.close()

    # エンジンのスコア関数をインポート
    from src.features.engine import (
        f_pace, f_recent, f_jockey, f_trainer,
        f_blood, f_dist_v2, f_post, f_weight,
        analyze_career, apply_career_flags,
        calc_pace_distribution, _horse_dist_dict, _horse_course_dict,
    )

    samples   = []
    skipped   = 0
    total_h   = 0

    for race_row in race_rows:
        try:
            race = json.loads(race_row['raw_json'])
        except Exception:
            skipped += 1
            continue

        winner_num = winners.get(race_row['id'])
        if winner_num is None:
            skipped += 1
            continue

        # ペース配分（固定コンテキスト）
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
            # 騎手勝率のエンリッチ
            jockey   = h.get('jockey', '')
            rc       = race.get('racecourse', '')
            surf     = race.get('surface', '芝')
            jockey_r = (jdict.get((jockey, rc, surf))
                        or jdict.get((jockey, '', ''))
                        or h.get('jockey_rate', 0.15))
            h['jockey_rate'] = jockey_r

            trainer   = h.get('trainer', '')
            trainer_r = tdict.get(trainer) or h.get('trainer_rate', 0.12)
            h['trainer_rate'] = trainer_r

            try:
                sc = {
                    'pace':     f_pace(h, race),
                    'recent':   f_recent(h, race),
                    'jockey':   f_jockey(h, race),
                    'trainer':  f_trainer(h),
                    'blood':    f_blood(h, race),
                    'distance': f_dist_v2(h, race),
                    'post':     f_post(h, race),
                    'bias':     5.0,   # バイアスは過去データで未記録→中立
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
    """重みベクトルを受け取りソフトマックス確率を返す"""
    totals = [
        sum(h['sc'][k] * weights[k] for k in WEIGHT_KEYS) + h['adj']
        for h in scores_matrix
    ]
    max_t = max(totals)
    exp_s = [math.exp((t - max_t) * temperature) for t in totals]
    s     = sum(exp_s) or 1.0
    return [e / s for e in exp_s]


def neg_log_likelihood(w_raw, samples):
    """目的関数: 実際の1着馬の平均負対数尤度"""
    # 制約: weights >=0, sum=1 は scipy が保証するが念のためクリップ
    w_pos = [max(1e-4, x) for x in w_raw]
    s     = sum(w_pos)
    w     = {k: w_pos[i] / s for i, k in enumerate(WEIGHT_KEYS)}

    total = 0.0
    for scores_matrix, winner_idx in samples:
        probs = softmax_probs(scores_matrix, w)
        total += math.log(max(1e-9, probs[winner_idx]))

    return -total / len(samples)


def accuracy_at_1(w_arr, samples):
    """診断指標: AI1位が実際に1着だった割合"""
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
    """重みチューニングを実行して optimal_weights.json に保存する。

    Args:
        base_dir   : Google Drive上のプロジェクトルート
        n_restarts : 異なる初期値から最適化を繰り返す回数（局所解回避）
        verbose    : 途中経過を表示するか

    Returns:
        dict: 最適化された重み
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print('❌ scipy が必要です: pip install scipy')
        return None

    import numpy as np
    import random

    if verbose:
        print('=== 重みチューニング開始 ===')
        print('  データロード中...')

    samples, meta = load_training_data(base_dir)
    if verbose:
        print(f'  レース: {meta["races_loaded"]:,}件 (スキップ: {meta["races_skipped"]}件)')
        print(f'  平均出走頭数: {meta["horses_per_race_avg"]}頭')
        # DBの状況を診断
        _diagnose_db(base_dir)

    if len(samples) < 20:
        print(f'❌ データ不足（{len(samples)}件）。最低20レース必要です。')
        return None
    if len(samples) < 50:
        print(f'⚠ データ少なめ（{len(samples)}件）。結果の精度は限定的ですが続行します。')

    # 初期重みでのベースライン
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
            # ランダム初期値（合計1に正規化）
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

    opt_w_arr = best_result.x
    opt_w_arr = np.clip(opt_w_arr, 0, None)
    opt_w_arr /= opt_w_arr.sum()
    opt_w     = {k: round(float(opt_w_arr[i]), 4) for i, k in enumerate(WEIGHT_KEYS)}

    opt_loss = neg_log_likelihood(list(opt_w_arr), samples)
    opt_acc  = accuracy_at_1(list(opt_w_arr), samples)

    if verbose:
        print(f'\n  [最適化結果] NLL={opt_loss:.4f} (改善: {base_loss-opt_loss:+.4f}), '
              f'Acc@1={opt_acc:.3f} (改善: {opt_acc-base_acc:+.3f})')
        print('\n  最適重み:')
        for k, v in sorted(opt_w.items(), key=lambda x: -x[1]):
            d = round(v - dict(zip(WEIGHT_KEYS, DEFAULT_W))[k], 4)
            bar = '█' * int(v * 40)
            print(f'    {k:10s}: {v:.4f} ({d:+.4f})  {bar}')

    # 保存
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
    parser = argparse.ArgumentParser(description='特徴量重みチューニング')
    parser.add_argument('--base-dir', required=True, help='keiba_aiプロジェクトルートパス')
    parser.add_argument('--restarts', type=int, default=5, help='最適化の試行回数')
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.base_dir, 'src'))
    from src.features.engine import init_engine
    init_engine(args.base_dir)

    run_tuning(args.base_dir, n_restarts=args.restarts)

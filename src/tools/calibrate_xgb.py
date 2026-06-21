"""
XGB raw_prob を実測複勝率にマッピングする isotonic calibrator を学習する。

【背景】
既存 src/tools/calibrate.py は「重み付きスコアモデル」の softmax後確率 (≒0.01〜0.20)
を入力として学習されている。一方、XGB raw_prob は 0.6〜0.8 の範囲で出力されるため、
calibrate.py で学習した isotonic は XGB に適用すると学習範囲外で端点圧縮が発生し、
全入力が同一値（≒0.27）にマップされる症状を起こす。

本スクリプトは XGB raw_prob を直接の入力として、実測複勝率 (3着以内に入る確率) に
マップする専用のキャリブレーターを学習・保存する。

【出力】
data/xgb_calibrator.pkl  ← 既存 calibrator.pkl とは別ファイルで管理

【実行方法（Google Colab）】
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.calibrate_xgb import run_xgb_calibration
    run_xgb_calibration(BASE_DIR)

【コマンドライン】
    python -m src.tools.calibrate_xgb --base-dir /path/to/keiba_ai

【設計判断】
- 時系列分割: 最新N日を test に分離して汎化性能を測定 (デフォルト 30日)
- ビン数: デフォルト 20 (細かく刻んだ方がXGBの順位情報を活用できる)
- 単調化: PAV (Pool Adjacent Violators) で強制単調増加
"""
import argparse
import os
import sys
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd

from src.models.calibration import IsotonicCalibrator
from src.models.calibration_xgb import save_xgb_calibrator


# ── ペア構築 ────────────────────────────────────────────────────

def _precompute_horse_past_races(conn):
    """全馬の過去レース履歴を事前に dict に格納する。

    {horse_name: [(date_str, race_id), ...]} の形で、date 昇順に並べた辞書を返す。
    f_member_level の計算に必要な race_id を正確に渡すために使用する。
    """
    rows = conn.execute(
        "SELECT horse_name, date, race_id FROM horse_history ORDER BY date ASC, id ASC"
    ).fetchall()
    from collections import defaultdict
    d = defaultdict(list)
    for r in rows:
        d[r['horse_name']].append((str(r['date'] or ''), r['race_id']))
    return d


def _build_xgb_pairs(base_dir, verbose=True):
    """history.db から (raw_prob, is_fukusho, date, race_id) のペアを構築。

    Returns:
        pd.DataFrame with columns: raw_prob, is_fukusho, date, race_id, num_horses
    """
    from src.features.engine import (init_engine, calc_features_for_xgb,
                                      _XGB_FUKUSHO_MODEL, _XGB_FEATURE_COLS)

    if _XGB_FUKUSHO_MODEL is None:
        if verbose:
            print('  エンジン未初期化 → init_engine 実行')
        init_engine(base_dir)
        from src.features.engine import _XGB_FUKUSHO_MODEL as _M, _XGB_FEATURE_COLS as _C
        _XGB_FUKUSHO_MODEL = _M
        _XGB_FEATURE_COLS = _C

    if _XGB_FUKUSHO_MODEL is None:
        raise RuntimeError('XGBモデルがロードできません')

    hist_path = os.path.join(base_dir, 'data', 'history.db')
    conn = sqlite3.connect(hist_path)
    conn.row_factory = sqlite3.Row

    # 全レース取得（horse_history と JOIN して実頭数で絞る）
    races_rows = conn.execute("""
        SELECT r.race_id, r.date, r.racecourse, r.race_num, r.distance, r.surface,
               r.race_class, r.race_name, r.track_condition, r.pace_label,
               COUNT(h.id) AS actual_n_horses
        FROM race_history r
        INNER JOIN horse_history h ON r.race_id = h.race_id
        GROUP BY r.race_id
        HAVING actual_n_horses >= 3
        ORDER BY r.date ASC
    """).fetchall()

    if verbose:
        print(f'  対象レース: {len(races_rows):,}件')

    # f_member_level 計算のため馬別過去レース履歴を事前構築
    horse_past = _precompute_horse_past_races(conn)

    pairs = []
    skipped = 0

    for ri, row in enumerate(races_rows):
        if verbose and ri % 500 == 0 and ri > 0:
            print(f'  処理中... {ri}/{len(races_rows)}  (ペア: {len(pairs):,})')

        race_id  = row['race_id']
        date_str = str(row['date'] or '')
        horses_rows = conn.execute(
            'SELECT * FROM horse_history WHERE race_id=? ORDER BY place ASC',
            (race_id,)
        ).fetchall()

        if len(horses_rows) < 3:
            skipped += 1
            continue

        race_dict = {
            'id': race_id, 'date': date_str, 'racecourse': row['racecourse'],
            'race_num': row['race_num'], 'distance': row['distance'],
            'surface': row['surface'], 'race_class': row['race_class'] or '',
            'num_horses': len(horses_rows), 'race_name': row['race_name'] or '',
            'horses': [], 'pace_dist': {'high': 0.3, 'mid': 0.4, 'slow': 0.3},
        }

        for hrow in horses_rows:
            name = hrow['horse_name'] or ''
            # このレース日より前の過去レースID（最大5件）を history に渡す
            past = [{'race_id': rid} for d, rid in horse_past.get(name, [])
                    if d < date_str][-5:]
            past.reverse()  # 直近順

            h = {
                'num': hrow['horse_num'] or 0,
                'name': name,
                'jockey': hrow['jockey'] or '',
                'trainer': hrow['trainer'] or '',
                'age': hrow['age'] or 3,
                'weight_load': hrow['weight_load'] or 55.0,
                'history': past,
                'place': hrow['place'] or 99,
                'agari3f': hrow['agari3f'],
                'corner_3': hrow['corner_3'],
                'running_style': hrow['running_style'] or '',
            }
            race_dict['horses'].append(h)

        try:
            xfeats = [calc_features_for_xgb(h, race_dict) for h in race_dict['horses']]
            X = pd.DataFrame([{c: xf.get(c, 5.0) for c in _XGB_FEATURE_COLS}
                              for xf in xfeats])[_XGB_FEATURE_COLS].fillna(5.0)
            raw_probs = _XGB_FUKUSHO_MODEL.predict_proba(X)[:, 1].tolist()

            for h, raw_prob in zip(race_dict['horses'], raw_probs):
                is_fukusho = 1 if (1 <= h['place'] <= 3) else 0
                pairs.append({
                    'raw_prob': raw_prob,
                    'is_fukusho': is_fukusho,
                    'date': row['date'],
                    'race_id': race_id,
                    'num_horses': len(race_dict['horses']),
                })
        except Exception:
            skipped += 1
            continue

    conn.close()

    if verbose:
        print(f'  完了: {len(pairs):,}ペア, スキップ {skipped}')

    return pd.DataFrame(pairs)


# ── 時系列分割 ─────────────────────────────────────────────────

def _split_train_test(df_pairs, test_days=30, verbose=True):
    """時系列分割: 最新 test_days 日を test に。

    日付フォーマット混在対応: 'YYYY-MM-DD' と 'YYYYMMDD' が混在しているため
    ハイフンを除去して YYYYMMDD 形式に統一してからパースする。
    """
    df = df_pairs.copy()
    # 両フォーマットに対応: ハイフンを除去 → YYYYMMDD として解釈
    date_str = df['date'].astype(str).str.replace('-', '', regex=False).str.slice(0, 8)
    df['date_obj'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['date_obj']).copy()
    if verbose and len(df) < n_before:
        print(f'  ⚠ 日付パース失敗: {n_before - len(df)}件を除外')

    max_date = df['date_obj'].max()
    cutoff = max_date - timedelta(days=test_days)

    df_train = df[df['date_obj'] < cutoff].copy()
    df_test = df[df['date_obj'] >= cutoff].copy()

    return df_train, df_test


# ── 評価指標 ──────────────────────────────────────────────────

def _evaluate(p, y, label=''):
    """Brier, Log loss, ECE を計算して表示。"""
    p = np.clip(p, 0.001, 0.999)
    brier = ((p - y) ** 2).mean()
    log_loss_val = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    df = pd.DataFrame({'p': p, 'y': y}).sort_values('p').reset_index(drop=True)
    n_bins = 10
    bin_size = max(1, len(df) // n_bins)
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(df)
        chunk = df.iloc[start:end]
        if len(chunk) > 0:
            ece += abs(chunk['p'].mean() - chunk['y'].mean()) * len(chunk) / len(df)

    print(f'  [{label:10s}] Brier={brier:.4f}  LogLoss={log_loss_val:.4f}  ECE={ece:.4f}')
    return {'brier': brier, 'log_loss': log_loss_val, 'ece': ece}


# ── isotonic 学習 ─────────────────────────────────────────────

def _fit_isotonic(pairs_df, n_bins=20):
    """ビン統計から IsotonicCalibrator を構築（sklearn PAV使用）。"""
    from sklearn.isotonic import IsotonicRegression as _IR
    df = pairs_df.copy().sort_values('raw_prob').reset_index(drop=True)
    bin_size = max(1, len(df) // n_bins)

    bins_x = []
    bins_y = []
    for i in range(0, len(df), bin_size):
        chunk = df.iloc[i:i + bin_size]
        if len(chunk) < 5:
            continue
        bins_x.append(float(chunk['raw_prob'].mean()))
        bins_y.append(float(chunk['is_fukusho'].mean()))

    if len(bins_x) < 3:
        return None

    # sklearn の IsotonicRegression で単調増加を強制（C実装で高速）
    ir = _IR(increasing=True, out_of_bounds='clip')
    ir.fit(bins_x, bins_y)
    y_mono = ir.predict(bins_x).tolist()

    return IsotonicCalibrator(bins_x, y_mono)


# ── メインエントリ ────────────────────────────────────────────

def run_xgb_calibration(base_dir, test_days=30, n_bins=20, verbose=True):
    """XGB専用キャリブレーションを実行して xgb_calibrator.pkl に保存。

    Args:
        base_dir : プロジェクトルート
        test_days: test期間の日数（時系列分割）
        n_bins   : isotonic学習時のビン数
        verbose  : 診断情報を表示するか

    Returns:
        IsotonicCalibrator オブジェクト (失敗時は None)
    """
    if verbose:
        print('=== XGB専用キャリブレーション開始 ===\n')

    # ① ペア構築
    print('[1/4] history.db から (raw_prob, is_fukusho) ペアを構築...')
    df_pairs = _build_xgb_pairs(base_dir, verbose=verbose)

    if len(df_pairs) < 500:
        print(f'❌ データ不足（{len(df_pairs)}ペア）。最低500ペア必要。')
        return None

    if verbose:
        print(f'\n  全ペア: {len(df_pairs):,}')
        print(f'  日付範囲: {df_pairs["date"].min()} 〜 {df_pairs["date"].max()}')
        print(f'  実測複勝率: {df_pairs["is_fukusho"].mean():.4f}')
        print(f'  raw_prob 平均: {df_pairs["raw_prob"].mean():.4f}')

    # ② 時系列分割
    print(f'\n[2/4] 時系列分割（test = 直近{test_days}日）...')
    df_train, df_test = _split_train_test(df_pairs, test_days=test_days)
    print(f'  train: {len(df_train):,}ペア')
    if len(df_train) > 0:
        print(f'         ({df_train["date"].min()} 〜 {df_train["date"].max()})')
    print(f'  test : {len(df_test):,}ペア')
    if len(df_test) > 0:
        print(f'         ({df_test["date"].min()} 〜 {df_test["date"].max()})')

    if len(df_train) < 500:
        print('❌ train データ不足。test_days を減らす必要あり。')
        return None

    # ③ 学習
    print(f'\n[3/4] Isotonic Calibrator 学習（{n_bins}ビン PAV）...')
    cal = _fit_isotonic(df_train, n_bins=n_bins)
    if cal is None:
        print('❌ キャリブレーション失敗（データ不足の可能性）')
        return None
    print(f'  ✅ 学習完了: {cal}')

    # ④ 評価
    print(f'\n[4/4] 性能評価:')

    # Train データでの評価
    y_train = df_train['is_fukusho'].values
    p_train_raw = df_train['raw_prob'].values
    p_train_cal = np.array(cal.transform(p_train_raw.tolist()))
    p_train_base = np.full(len(df_train), y_train.mean())

    print('\n  [Train データ]')
    _evaluate(p_train_raw,  y_train, label='Raw XGB')
    _evaluate(p_train_base, y_train, label='Baseline')
    _evaluate(p_train_cal,  y_train, label='Calibrated')

    # Test データでの評価（汎化性能の本命）
    if len(df_test) >= 100:
        y_test = df_test['is_fukusho'].values
        p_test_raw = df_test['raw_prob'].values
        p_test_cal = np.array(cal.transform(p_test_raw.tolist()))
        p_test_base = np.full(len(df_test), y_test.mean())

        print('\n  [Test データ（未来予測の汎化性能 ★最重要★）]')
        _evaluate(p_test_raw,  y_test, label='Raw XGB')
        _evaluate(p_test_base, y_test, label='Baseline')
        _evaluate(p_test_cal,  y_test, label='Calibrated')

        # レース別 cal_prob 合計 (理論3.0)
        df_test_with_cal = df_test.copy()
        df_test_with_cal['cal_prob'] = p_test_cal
        race_sums = df_test_with_cal.groupby('race_id')['cal_prob'].sum()
        print(f'\n  [Test レース別 cal_prob 合計（理論 ≒ 3.0）]')
        print(f'    平均  : {race_sums.mean():.3f}')
        print(f'    中央値: {race_sums.median():.3f}')
        print(f'    範囲  : [{race_sums.min():.3f}, {race_sums.max():.3f}]')
    else:
        print(f'\n  ⚠ Test データ不足 ({len(df_test)}ペア) → 汎化性能の評価は省略')
        print(f'     test_days を増やすか、データ蓄積を待つ。')

    # ⑤ 保存
    out_path = save_xgb_calibrator(cal, base_dir)
    print(f'\n✅ 保存完了: {out_path}')
    print(f'   ファイル名: xgb_calibrator.pkl')
    print(f'   既存 calibrator.pkl (重み付きスコア用) とは別管理')

    return cal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGB専用キャリブレーション')
    parser.add_argument('--base-dir', required=True,
                        help='keiba_aiプロジェクトルートパス')
    parser.add_argument('--test-days', type=int, default=30,
                        help='テスト期間の日数（時系列分割）')
    parser.add_argument('--n-bins', type=int, default=20,
                        help='isotonic学習時のビン数')
    args = parser.parse_args()

    sys.path.insert(0, args.base_dir)
    run_xgb_calibration(args.base_dir, test_days=args.test_days, n_bins=args.n_bins)

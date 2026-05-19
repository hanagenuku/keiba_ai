"""
過去レース実績からAI確率のキャリブレーション（Platt scaling）を行うスクリプト。

optimal_weights.json が存在すればそれを使い、なければデフォルト重みで実行する。
history.db (horse_history / race_history) および keiba.db の両方に対応。

実行方法（Google Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.calibrate import run_calibration
    run_calibration(BASE_DIR)

コマンドライン:
    python -m src.tools.calibrate --base-dir /path/to/keiba_ai
"""
import argparse
import json
import math
import os
import pickle
import sys

from src.models.calibration import (  # noqa: F401  後方互換のため再エクスポート
    IsotonicCalibrator,
    PlattCalibrator,
    save_calibrator,
)

WEIGHT_KEYS = ['pace', 'recent', 'jockey', 'trainer', 'blood', 'distance', 'post', 'bias', 'weight']


def _load_weights(base_dir):
    path = os.path.join(base_dir, 'data', 'optimal_weights.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        w = {k: data[k] for k in WEIGHT_KEYS if k in data}
        if len(w) == len(WEIGHT_KEYS):
            print(f'  最適重みを使用: {path}')
            return w
    defaults = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.06, 0.04, 0.02]
    print('  optimal_weights.json なし → デフォルト重みを使用')
    return {k: defaults[i] for i, k in enumerate(WEIGHT_KEYS)}


def _build_prob_outcome_pairs(base_dir, weights):
    """全過去レースの (ai_win_prob, is_win) ペアを返す。

    tune_weights.load_training_data を流用して history.db / keiba.db 両対応。
    """
    from src.tools.tune_weights import load_training_data

    samples, meta = load_training_data(base_dir)
    print(f'  レース: {meta["races_loaded"]:,}件 (スキップ: {meta["races_skipped"]}件)')

    pairs_win  = []
    pairs_top3 = []

    for scores_matrix, winner_idx in samples:
        totals = [
            sum(h['sc'][k] * weights[k] for k in WEIGHT_KEYS) + h['adj']
            for h in scores_matrix
        ]
        max_t = max(totals)
        exp_s = [math.exp((t - max_t) * 0.8) for t in totals]
        sum_e = sum(exp_s) or 1.0
        probs = [e / sum_e for e in exp_s]

        n = len(probs)
        for i, p in enumerate(probs):
            pairs_win.append((p, 1 if i == winner_idx else 0))

            # Harville top3 近似確率
            v = p
            for j in range(n):
                if j == i:
                    continue
                dj = max(1e-9, 1.0 - probs[j])
                v += probs[j] * p / dj
            pairs_top3.append((min(1.0, v), 0))  # is_top3 は不明なので0（win calibにのみ使用）

    return pairs_win, pairs_top3


# ── キャリブレーション ────────────────────────────────────────────

def _bin_calibration_stats(pairs, n_bins=10):
    """ペアをビン分割して各ビンのAI確率平均と実際の的中率を返す。"""
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    bin_size     = max(1, len(pairs_sorted) // n_bins)
    bins_x, bins_y = [], []
    for i in range(0, len(pairs_sorted), bin_size):
        chunk = pairs_sorted[i:i + bin_size]
        if len(chunk) < 5:
            continue
        bins_x.append(sum(p for p, _ in chunk) / len(chunk))
        bins_y.append(sum(y for _, y in chunk) / len(chunk))
    return bins_x, bins_y


def _fit_platt(pairs):
    """Platt scaling: scipy で A,B を最適化。"""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    def neg_ll(ab):
        A, B = ab
        total = 0.0
        for p, y in pairs:
            logit = A * p + B
            # numerically stable
            if logit >= 0:
                log_p_pos = -logit - math.log(1 + math.exp(-logit))
                log_p_neg = -math.log(1 + math.exp(-logit))
            else:
                log_p_pos = -math.log(1 + math.exp(logit))
                log_p_neg = logit - math.log(1 + math.exp(logit))
            total += y * log_p_pos + (1 - y) * log_p_neg
        return -total

    res = minimize(neg_ll, [-1.0, 0.0], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    return PlattCalibrator(A=res.x[0], B=res.x[1])


def _fit_isotonic(pairs, n_bins=20):
    """ビン統計からIsotonicCalibratorを構築する。"""
    bins_x, bins_y = _bin_calibration_stats(pairs, n_bins=n_bins)
    if len(bins_x) < 3:
        return None
    # 単調増加を強制（PAVアルゴリズム簡易版）
    y_mono = list(bins_y)
    changed = True
    while changed:
        changed = False
        for i in range(len(y_mono) - 1):
            if y_mono[i] > y_mono[i + 1]:
                avg = (y_mono[i] + y_mono[i + 1]) / 2
                y_mono[i] = y_mono[i + 1] = avg
                changed = True
    return IsotonicCalibrator(bins_x, y_mono)


def _brier_score(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _expected_calibration_error(pairs, n_bins=10):
    """ECE: ビンごとの |AI確率 - 実的中率| の加重平均"""
    if not pairs:
        return 0.0
    bins_x, bins_y = _bin_calibration_stats(pairs, n_bins=n_bins)
    total_n = len(pairs)
    bin_size = max(1, total_n // n_bins)
    ece = 0.0
    for x, y in zip(bins_x, bins_y):
        frac = min(bin_size, total_n) / total_n
        ece += frac * abs(x - y)
    return round(ece, 4)


def run_calibration(base_dir, method='isotonic', verbose=True):
    """キャリブレーションを実行して calibrator.pkl に保存する。

    Args:
        base_dir : プロジェクトルート
        method   : 'isotonic'（推奨）または 'platt'
        verbose  : 診断情報を表示するか

    Returns:
        calibrator オブジェクト
    """
    if verbose:
        print('=== キャリブレーション開始 ===')

    weights = _load_weights(base_dir)
    if verbose:
        print('  確率ペアを構築中...')

    pairs_win, pairs_top3 = _build_prob_outcome_pairs(base_dir, weights)

    if len(pairs_win) < 100:
        print(f'❌ データ不足（{len(pairs_win)}件）。最低100馬必要です。')
        return None

    if verbose:
        actual_win_rate = sum(y for _, y in pairs_win) / len(pairs_win)
        ai_win_avg      = sum(p for p, _ in pairs_win) / len(pairs_win)
        print(f'  総馬数: {len(pairs_win):,}頭 (うち1着: {sum(y for _,y in pairs_win):,}頭)')
        print(f'  AI平均単勝確率: {ai_win_avg:.4f}  実際の勝率: {actual_win_rate:.4f}')
        print(f'  Brier score (before): {_brier_score(pairs_win):.4f}')
        print(f'  ECE (before): {_expected_calibration_error(pairs_win):.4f}')

    # キャリブレーション適合
    if method == 'platt':
        cal = _fit_platt(pairs_win)
        if cal is None:
            print('  ⚠ scipy なし → isotonic にフォールバック')
            cal = _fit_isotonic(pairs_win)
    else:
        cal = _fit_isotonic(pairs_win)

    if cal is None:
        print('❌ キャリブレーションの適合に失敗しました。')
        return None

    # 適合後の診断
    if verbose:
        cal_pairs = [(cal.transform([p])[0], y) for p, y in pairs_win]
        print(f'\n  キャリブレーター: {cal}')
        print(f'  Brier score (after): {_brier_score(cal_pairs):.4f}')
        print(f'  ECE (after): {_expected_calibration_error(cal_pairs):.4f}')

        # ビン別の診断表
        bins_before = _bin_calibration_stats(pairs_win, n_bins=10)
        bins_after  = _bin_calibration_stats(cal_pairs, n_bins=10)
        print('\n  ビン別キャリブレーション診断:')
        print(f'  {"AIスコア":>10s}  {"実的中率":>8s}  {"補正後":>8s}  {"件数":>6s}')
        total_n  = len(pairs_win)
        bin_size = max(1, total_n // 10)
        for i, (bx, by) in enumerate(zip(*bins_before)):
            cal_x = cal.transform([bx])[0] if i < len(bins_after[0]) else bx
            n     = min(bin_size, total_n - i * bin_size)
            bar   = '█' * int(by * 30)
            print(f'  {bx:>10.4f}  {by:>8.4f}  {cal_x:>8.4f}  {n:>6d}  {bar}')

    # 保存
    out_path = save_calibrator(cal, base_dir)
    print(f'\n✅ 保存完了: {out_path}')

    return cal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI確率キャリブレーション')
    parser.add_argument('--base-dir', required=True, help='keiba_aiプロジェクトルートパス')
    parser.add_argument('--method', default='isotonic', choices=['isotonic', 'platt'],
                        help='キャリブレーション手法')
    args = parser.parse_args()

    sys.path.insert(0, args.base_dir)
    from src.features.engine import init_engine
    init_engine(args.base_dir)

    run_calibration(args.base_dir, method=args.method)

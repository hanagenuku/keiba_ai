"""
スコア → 確率変換ユーティリティ。
engine.calc_all が使うソフトマックス・キャリブレーション適用処理を独立させる。
"""
import math


def softmax_probs(totals, temperature=0.8):
    """スコアリストをソフトマックスで勝率確率に変換する。和が 1 になる。

    Args:
        totals      : 各馬の合計スコアリスト
        temperature : 温度パラメータ。大きいほど確率が均等になる

    Returns:
        確率リスト（float）
    """
    max_t = max(totals)
    exps = [math.exp((t - max_t) * temperature) for t in totals]
    s = sum(exps)
    return [e / s for e in exps]


def calibrate_and_renormalize(win_probs, calibrator):
    """キャリブレーションを適用して再正規化する。

    calibrator が None の場合はそのまま返す。
    適用後に再正規化して和が 1 になるよう保証する。

    Args:
        win_probs  : softmax_probs の出力など、和が 1 の確率リスト
        calibrator : PlattCalibrator / IsotonicCalibrator、または None

    Returns:
        再正規化済み確率リスト（float）
    """
    if calibrator is None:
        return win_probs
    cal_probs = [float(calibrator.transform([p])[0]) for p in win_probs]
    s = sum(cal_probs) or 1.0
    return [p / s for p in cal_probs]


def predict_race(race, bias_data=None):
    """レース予想を実行する。engine.calc_all のラッパー。

    ノートブックからの呼び出しを predict モジュールに統一することで、
    将来的なモデル差し替えをこのファイルだけで完結できる。

    Args:
        race      : レース辞書（horses リストを含む）
        bias_data : 馬場バイアス辞書（省略可）

    Returns:
        calc_all と同じ形式のリスト（スコア降順）
    """
    from src.features.engine import calc_all
    return calc_all(race, bias_data)

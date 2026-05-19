"""
キャリブレーターの定義と I/O ユーティリティ。
src/tools/calibrate.py の実行ロジックから分離。
"""
import math
import os
import pickle


class PlattCalibrator:
    """P_cal = sigmoid(A * p + B) — A<0 のとき単調増加になる。"""

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def transform(self, probs):
        return [1.0 / (1.0 + math.exp(self.A * p + self.B)) for p in probs]

    def __repr__(self):
        return f'PlattCalibrator(A={self.A:.4f}, B={self.B:.4f})'


class IsotonicCalibrator:
    """単調増加を保証するノンパラメトリックキャリブレーター。"""

    def __init__(self, x_bins, y_vals):
        self.x_bins = x_bins
        self.y_vals = y_vals

    def transform(self, probs):
        out = []
        for p in probs:
            if p <= self.x_bins[0]:
                out.append(self.y_vals[0])
            elif p >= self.x_bins[-1]:
                out.append(self.y_vals[-1])
            else:
                for i in range(len(self.x_bins) - 1):
                    if self.x_bins[i] <= p < self.x_bins[i + 1]:
                        t = (p - self.x_bins[i]) / (self.x_bins[i + 1] - self.x_bins[i])
                        out.append(self.y_vals[i] * (1 - t) + self.y_vals[i + 1] * t)
                        break
                else:
                    out.append(self.y_vals[-1])
        return out

    def __repr__(self):
        return f'IsotonicCalibrator({len(self.x_bins)} breakpoints)'


def load_calibrator(base_dir):
    """calibrator.pkl を読み込む。ファイルがなければ None を返す。"""
    path = os.path.join(base_dir, 'data', 'calibrator.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_calibrator(cal, base_dir):
    """calibrator.pkl に保存する。"""
    path = os.path.join(base_dir, 'data', 'calibrator.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cal, f)
    return path

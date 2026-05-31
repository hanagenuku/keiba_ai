"""
XGB専用キャリブレーターの I/O ユーティリティ。

既存 src/models/calibration.py は触らず、新規ファイルとして追加することで
既存システムへの影響を回避する。
IsotonicCalibrator クラス本体は既存ファイルから再利用する。

保存先: data/xgb_calibrator.pkl
（既存 calibrator.pkl は重み付きスコア用なので別ファイルで管理）
"""
import os
import pickle


def load_xgb_calibrator(base_dir):
    """xgb_calibrator.pkl を読み込む。ファイルがなければ None を返す。"""
    path = os.path.join(base_dir, 'data', 'xgb_calibrator.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_xgb_calibrator(cal, base_dir):
    """xgb_calibrator.pkl に保存する。"""
    path = os.path.join(base_dir, 'data', 'xgb_calibrator.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(cal, f)
    return path

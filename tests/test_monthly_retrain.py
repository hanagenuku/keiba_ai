"""monthly_retrain.py の回帰テスト。

2026-07-01の月次自動再学習が、存在しない calibrate_xgb 関数を import しようと
して0秒で即座に失敗していたことが判明した（calibrate_xgb.py には
run_xgb_calibration という別名の関数しか存在しない）。この種の「import自体が
壊れている」バグはモジュールを読み込むだけで検知できるため、軽量な回帰テストと
して固定する。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_monthly_retrain_module_imports_without_error():
    """scripts.monthly_retrain が例外なくimportできる（2026-07-01の実障害の再発防止）。"""
    module = importlib.import_module('scripts.monthly_retrain')
    assert hasattr(module, 'main')
    assert hasattr(module, 'AUC_THRESHOLD')


def test_monthly_retrain_calls_train_xgb_with_residual_true():
    """本番は残差学習モデルで稼働中のため、月次再学習も residual=True を
    指定していることをソースレベルで確認する（市場コピー型モデルへの
    サイレントな回帰を防ぐ）。"""
    src_path = os.path.join(os.path.dirname(__file__), '..',
                             'scripts', 'monthly_retrain.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert 'residual=True' in src


def test_monthly_retrain_copies_residual_files_to_production_paths():
    """train_xgb(residual=True) の出力（*_residual.pkl/.json）を、
    engine.py が実際にロードするサフィックス無しの本番ファイルへコピーする
    処理が存在することをソースレベルで確認する。"""
    src_path = os.path.join(os.path.dirname(__file__), '..',
                             'scripts', 'monthly_retrain.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert 'xgb_fukusho_model_residual.pkl' in src
    assert 'xgb_fukusho_model.pkl' in src
    assert 'xgb_feature_cols_residual.json' in src
    assert 'xgb_feature_cols.json' in src

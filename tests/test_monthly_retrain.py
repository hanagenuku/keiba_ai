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


def test_monthly_retrain_runs_calibration_and_checks_result():
    """キャリブレータを自動更新し、戻り値を必ず確認していること。

    2026-07-31以前はここを丸ごと飛ばしていた（残差モデル非対応のため）。
    run_xgb_calibration は失敗しても例外ではなく None を返す作りなので、
    戻り値を見ずに「完了」と表示すると、新モデルに旧モデル用の較正器が
    付いたまま残り cal_prob だけが静かに誤る。
    """
    src_path = os.path.join(os.path.dirname(__file__), '..',
                            'scripts', 'monthly_retrain.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert 'run_xgb_calibration' in src, 'キャリブレーションを実行していません'
    assert 'cal is not None' in src, '戻り値(None)の確認をしていません'
    # 失敗時に旧較正器を残さないこと
    assert '.stale' in src, '失敗時に旧較正器を退避していません'


def test_shap_diagnosis_uses_format_agnostic_model_loader():
    """SHAP診断が残差モデル(UBJ)を pickle で読もうとしないこと（2026-08-03）。

    旧実装は pickle.load() で必ず失敗し、shap がインストール済みでも
    「SHAP: 未インストール」と誤表示していた。原因の切り分けを誤らせるため、
    失敗理由を区別して出すことも合わせて固定する。
    """
    src_path = os.path.join(os.path.dirname(__file__), '..',
                            'src', 'tools', 'shap_diagnosis.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert '_load_xgb_model_any' in src, '共通ローダーを使っていない'
    assert 'model = pickle.load(f)' not in src, 'pickle.load が残っている'
    assert 'shap_error' in src and '利用できません' in src, \
        '失敗理由を区別して表示していない'


def test_monthly_retrain_schedule_is_disabled():
    """月次自動実行は停止していること（2026-08-03）。

    GitHub側の history.db は Drive側の半分以下（5,457 vs 11,554レース）で、
    本番より良いモデルを作れる見込みが構造的に無い。閾値0.75はリーク除去前の
    遺物で毎月必ず失敗し、下げれば半分のデータのモデルが本番を上書きする。
    再開するなら history.db の是正と閾値の引き直しが前提。
    """
    import yaml
    p = os.path.join(os.path.dirname(__file__), '..',
                     '.github', 'workflows', 'monthly-retrain.yml')
    with open(p, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    trig = d[True] if True in d else d.get('on')
    assert 'schedule' not in (trig or {}), '自動実行が有効に戻っている'
    assert 'workflow_dispatch' in (trig or {}), '手動実行まで消えている'


def test_monthly_retrain_restores_model_when_rejected():
    """AUC不足で中止したとき、残差モデルを更新前に戻すこと。

    train_xgb は閾値判定より前に *_residual.pkl を新モデルで上書きするため、
    戻さないと「中止したのにファイルは新モデル」という状態が残る。
    """
    src_path = os.path.join(os.path.dirname(__file__), '..',
                            'scripts', 'monthly_retrain.py')
    with open(src_path, encoding='utf-8') as f:
        src = f.read()
    assert '_restore_rejected_residual' in src
    assert 'xgb_fukusho_model_residual_old.pkl' in src

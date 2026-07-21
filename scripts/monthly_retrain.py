#!/usr/bin/env python3
"""月次モデル再学習スクリプト。GitHub Actions の monthly-retrain.yml から呼び出される。"""
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.tools.build_training_data import build_training_data
from src.tools.train_xgb import train_xgb

JST = timezone(timedelta(hours=9))
AUC_THRESHOLD = 0.75


def main():
    jst_now = datetime.now(JST)

    # 学習期間: 2年前〜先月末
    val_end_dt = (jst_now.replace(day=1) - timedelta(days=1))
    val_start_dt = val_end_dt.replace(day=1) - timedelta(days=30)
    train_end_dt = val_start_dt - timedelta(days=1)

    train_end  = train_end_dt.strftime('%Y-%m-%d')
    val_start  = val_start_dt.strftime('%Y-%m-%d')
    val_end    = val_end_dt.strftime('%Y-%m-%d')

    print(f'📅 月次再学習: train_end={train_end}  val={val_start}〜{val_end}')

    print('📊 特徴量データ再構築中...')
    build_training_data(ROOT)

    print('🤖 XGBoost 残差学習モデルを再学習中...')
    # 本番は2026-07-14以降、残差学習モデル（f_popularityを特徴量から除外し、
    # 市場確率をbase_marginに固定）で稼働している。residual=True を指定しないと
    # 市場コピー型の旧方式で学習してしまい、本番モデルの設計思想（AIが市場を
    # コピーしない）を無効化してしまう。
    metrics = train_xgb(
        ROOT,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        use_optuna=False,
        residual=True,
    )

    auc = metrics.get('auc', 0.0)
    print(f'✅ AUC={auc:.4f}  Brier={metrics.get("brier", "?"):.4f}')

    if auc < AUC_THRESHOLD:
        print(f'❌ AUC {auc:.4f} < {AUC_THRESHOLD} — モデル更新を中止します')
        sys.exit(1)

    # train_xgb(residual=True) は xgb_fukusho_model_residual.pkl /
    # xgb_feature_cols_residual.json に保存・採用する。本番（engine.py の
    # init_engine）が読み込むのはサフィックス無しの xgb_fukusho_model.pkl /
    # xgb_feature_cols.json のため、ここで明示的にコピーして反映する
    # （2026-07-14の残差学習モデル本番投入時にColabで手動実行した手順と同じ）。
    data_dir = os.path.join(ROOT, 'data')
    shutil.copy2(os.path.join(data_dir, 'xgb_fukusho_model_residual.pkl'),
                 os.path.join(data_dir, 'xgb_fukusho_model.pkl'))
    shutil.copy2(os.path.join(data_dir, 'xgb_feature_cols_residual.json'),
                 os.path.join(data_dir, 'xgb_feature_cols.json'))
    print('✅ 本番ファイル（xgb_fukusho_model.pkl / xgb_feature_cols.json）に反映')

    # ⚠ xgb_calibrator.pkl の自動更新は今回未対応（意図的）。
    # src/tools/calibrate_xgb.py の run_xgb_calibration() は
    # _XGB_FUKUSHO_MODEL.predict_proba() を前提とした実装で、残差学習モデル
    # （xgb.Booster形式、predict()のみでpredict_proba()を持たない）を渡すと
    # AttributeErrorで失敗する。誤ったキャリブレーションを自動生成するリスクを
    # 避けるため、次回のColabセッションで手動キャリブレーション
    # （run_xgb_calibration経由）を実行するまで xgb_calibrator.pkl は更新しない。
    # 本番の予測順位（RL/CL）自体には影響しない。表示用の複勝確率較正が
    # やや古いデータのまま据え置かれるのみ。
    print('⚠ xgb_calibrator.pkl は自動更新していません（残差モデル非対応のため）。'
          'Colabで手動キャリブレーションを実行してください')

    print('✅ 月次再学習完了')


if __name__ == '__main__':
    main()

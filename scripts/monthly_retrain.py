#!/usr/bin/env python3
"""月次モデル再学習スクリプト。GitHub Actions の monthly-retrain.yml から呼び出される。"""
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.tools.build_training_data import build_training_data
from src.tools.train_xgb import train_xgb
from src.tools.calibrate_xgb import calibrate_xgb

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

    print('🤖 XGBoost 再学習中...')
    metrics = train_xgb(
        ROOT,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        use_optuna=False,
    )

    auc = metrics.get('auc', 0.0)
    print(f'✅ AUC={auc:.4f}  Brier={metrics.get("brier", "?"):.4f}')

    if auc < AUC_THRESHOLD:
        print(f'❌ AUC {auc:.4f} < {AUC_THRESHOLD} — モデル更新を中止します')
        sys.exit(1)

    print('🎯 キャリブレーション中...')
    calibrate_xgb(ROOT)

    print('✅ 月次再学習完了')


if __name__ == '__main__':
    main()

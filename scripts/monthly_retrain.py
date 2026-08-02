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


def _restore_rejected_residual(data_dir):
    """AUC不足で不採用にしたモデルを、退避済みの旧モデルへ戻す。

    train_xgb は評価より前に *_residual.pkl を新モデルで上書きするため、
    ここで戻さないと「中止したのにファイルは新モデル」という状態が残る。
    """
    old = os.path.join(data_dir, 'xgb_fukusho_model_residual_old.pkl')
    cur = os.path.join(data_dir, 'xgb_fukusho_model_residual.pkl')
    if os.path.exists(old):
        shutil.copy2(old, cur)
        print('   ↩ 残差モデルを更新前の状態に戻しました')


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
        # ⚠ train_xgb は閾値判定より前に *_residual.pkl を新モデルで
        #   置き換えてしまう（「新モデルを正式採用」のログが出る）。
        #   中止したのに残差モデルだけ差し替わった状態を残すと、
        #   将来この後段に push が足された時に不採用モデルが本番へ出る。
        #   退避済みの *_residual_old.pkl から戻して整合を保つ。
        _restore_rejected_residual(os.path.join(ROOT, 'data'))
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

    # キャリブレータ再作成。特徴量が変わると raw_prob の分布も変わるため、
    # 旧モデル用の較正器を残したままにすると cal_prob だけが静かに誤る。
    #
    # 2026-07-31以前は「残差モデル非対応」としてここを丸ごと飛ばしていた。
    # 実際には run_xgb_calibration() がレース単位の try/except で
    # predict_proba の AttributeError を吸ってしまい、例外ではなく
    # 「0ペア → None」を返す作りだった（＝呼び出し側では検知できない）。
    # 残差モデル対応は 2026-07-31 に入れたので、ここで自動更新する。
    print('🎯 キャリブレータ再作成中...')
    try:
        from src.tools.calibrate_xgb import run_xgb_calibration
        cal = run_xgb_calibration(ROOT)
    except Exception as e:
        cal = None
        print(f'⚠ キャリブレーション実行中に例外: {e}')

    if cal is not None:
        print('✅ xgb_calibrator.pkl を更新しました')
    else:
        # 新モデルに旧モデル用の較正器を付けたままにしない。
        # 較正器が無ければ engine は生シグモイド確率を使う（RL順位は不変）。
        stale = os.path.join(data_dir, 'xgb_calibrator.pkl')
        if os.path.exists(stale):
            shutil.move(stale, stale + '.stale')
            print('⚠ キャリブレーション失敗。旧較正器は新モデルと不整合なため退避しました'
                  '（xgb_calibrator.pkl → .stale）')
        print('⚠ cal_prob は生シグモイド確率になります（RL順位・買い目には影響なし）')

    print('✅ 月次再学習完了')


if __name__ == '__main__':
    main()

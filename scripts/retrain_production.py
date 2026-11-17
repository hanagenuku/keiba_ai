#!/usr/bin/env python3
"""本番のモデル一式を、いまの特徴量の作り方で作り直す（手動・GitHub ランナーで実行）。

    python3 scripts/retrain_production.py --reason "<なぜ作り直すか>"

使う場面: 特徴量の**作り方**を直したとき（例 2026-11-18 スピード指数の馬場差の修正）。
学習データと本番の推論で特徴量の作り方が揃っていないと、片方だけ値がずれる（North Star #3）。

やること（学習の窓は現行の本番と同じ。変えるのは特徴量の作り方だけ）:
  1. speed_index_cache.pkl を作り直す
  2. build_training_data（web 指数は data/private/web_signals.db から付く）
  3. 残差モデルを学習 → 本番ファイルへ
  4. 較正器（xgb_calibrator.pkl・ability_calibrator.pkl）を作り直す
  5. ☆のモデル（star_model.ubj ほか）を作り直す（列が本番モデルと同じなので一緒に直す）

🔴 壊れ検知: 新しい検証 AUC が、作り直す前の本番の検証 AUC（同じ窓・同じ設定）より 0.003 以上低ければ止める。
   これは効果の判定ではなく、作り直しが壊れていないかの確認。
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')
MAX_DROP = 0.003
OUTPUTS = ('speed_index_cache.pkl', 'xgb_fukusho_model_residual.pkl', 'xgb_feature_cols_residual.json',
           'xgb_fukusho_model.pkl', 'xgb_feature_cols.json', 'xgb_calibrator.pkl', 'ability_calibrator.pkl',
           'star_model.ubj', 'star_calibrator.pkl', 'star_feature_cols.json')


def fail(msg):
    print(f'::error::{msg}')
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reason', required=True)
    ap.add_argument('--train-end', default='2026-06-30')
    ap.add_argument('--val-start', default='2026-07-01')
    ap.add_argument('--val-end', default='2026-08-31')
    a = ap.parse_args()
    print('作り直す理由:', a.reason)

    with open(os.path.join(DATA, 'xgb_feature_cols.json'), encoding='utf-8') as f:
        old = json.load(f)
    old_cols, old_auc = list(old['feature_cols']), float(old['val_auc'])
    print(f'作り直す前の本番: {len(old_cols)}列  val_auc {old_auc}  trained_at {old.get("trained_at")}')

    db = os.path.join(DATA, 'private', 'web_signals.db')
    if any(c.startswith('f_ws_') for c in old_cols):
        if not os.path.exists(db):
            fail('web 指数の列があるのに data/private/web_signals.db が無い（artifact の復元に失敗）')
        n = sqlite3.connect(db).execute('SELECT COUNT(*) FROM web_signals').fetchone()[0]
        print(f'指数DB: {n:,}頭')

    from src.features.speed_index import rebuild_speed_index_cache
    calc = rebuild_speed_index_cache(ROOT)
    print('スピード指数キャッシュ:', calc)

    import pandas as pd
    from src.tools.build_training_data import build_training_data
    build_training_data(ROOT)
    df = pd.read_csv(os.path.join(DATA, 'horse_features.csv'), usecols=lambda c: c in ('date', 'f_ws_kayochinkeiba_rank', 'f_speed_fig_last'))
    if 'f_ws_kayochinkeiba_rank' in df:
        cov = df.groupby(df['date'].astype(str).str[:4])['f_ws_kayochinkeiba_rank'].apply(lambda s: round(s.notna().mean(), 3))
        print('指数が付いた行の割合（年別）:', cov.to_dict())
    print('f_speed_fig_last が入った行の割合:', round(df['f_speed_fig_last'].notna().mean(), 3))

    from src.tools.train_xgb import train_xgb
    m = train_xgb(ROOT, train_end=a.train_end, val_start=a.val_start, val_end=a.val_end,
                  use_optuna=False, residual=True)
    new_auc = float(m['auc'])
    print(f'新しい検証 AUC {new_auc}  作り直す前 {old_auc}  差 {new_auc - old_auc:+.4f}')
    if new_auc < old_auc - MAX_DROP:
        fail(f'検証 AUC が {MAX_DROP} 以上下がった（{old_auc} → {new_auc}）。作り直しが壊れていないか確かめること')

    # train_xgb は「旧モデルより下がると本番名へコピーしない」ので、ここで必ず新しいほうを置く
    for src, dst in (('xgb_fukusho_model_residual_new.pkl', 'xgb_fukusho_model_residual.pkl'),
                     ('xgb_feature_cols_residual_new.json', 'xgb_feature_cols_residual.json')):
        if os.path.exists(os.path.join(DATA, src)):
            shutil.copy2(os.path.join(DATA, src), os.path.join(DATA, dst))
    shutil.copy2(os.path.join(DATA, 'xgb_fukusho_model_residual.pkl'), os.path.join(DATA, 'xgb_fukusho_model.pkl'))
    shutil.copy2(os.path.join(DATA, 'xgb_feature_cols_residual.json'), os.path.join(DATA, 'xgb_feature_cols.json'))
    with open(os.path.join(DATA, 'xgb_feature_cols.json'), encoding='utf-8') as f:
        new_cols = list(json.load(f)['feature_cols'])
    if new_cols != old_cols:
        fail(f'列が変わった（{len(old_cols)} → {len(new_cols)}）。特徴量の作り方だけを直す作業のはず')
    print(f'✅ 本番ファイルに反映（{len(new_cols)}列）')

    from src.features import engine
    engine.init_engine(ROOT)
    from src.tools.calibrate_xgb import run_xgb_calibration
    if run_xgb_calibration(ROOT) is None:
        fail('較正器（xgb_calibrator.pkl）を作れなかった')
    from scripts.build_ability_calibrator import build_ability_calibrator
    if build_ability_calibrator(ROOT) is None:
        fail('ability 較正器を作れなかった')

    r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'tools', 'train_star.py')], cwd=ROOT)
    if r.returncode != 0:
        fail('☆のモデルを作り直せなかった')

    missing = [f for f in OUTPUTS if not os.path.exists(os.path.join(DATA, f))]
    if missing:
        fail(f'出力が欠けている: {missing}')
    with open(os.path.join(ROOT, 'retrain_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(reason=a.reason, old_val_auc=old_auc, new_val_auc=new_auc, n_cols=len(new_cols),
                       speed_cache=repr(calc)), f, ensure_ascii=False, indent=1)
    print('✅ 完了。コミット対象:', ', '.join(OUTPUTS))


if __name__ == '__main__':
    main()

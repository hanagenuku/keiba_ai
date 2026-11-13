#!/usr/bin/env python3
"""web 情報源を採用して、本番モデル一式を作り直す（GitHub のランナーで実行する）。

`websig/ADOPTION_STANDARD.md` §4 の手順を1本にしたもの。指数DB（data/private/web_signals.db）は
非公開の artifact にしか無いので、`.github/workflows/adopt-web-source.yml` から呼ぶ。

やること（順番が大事）:
  1. 台帳を adopted にする（card / results / measured.delta_auc_mean / feature_cols を埋める）
  2. build_training_data（採用済みの列が CSV に入る）→ 列と被覆率を検査
  3. train_xgb(residual=True)。**学習の窓は現行の本番と同じ**（指数を足す以外は変えない）
  4. サフィックス無しの本番ファイルへ反映
  5. 較正器（xgb_calibrator.pkl）と ability 較正器を作り直す
  6. 旧モデルとの比較を印字する

🔴 途中で失敗したら exit 1。ワークフローは失敗時に何もコミットしない。
🔴 horse_features.csv（128MB）と data/private/ はコミットしない。
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

JST = timezone(timedelta(hours=9))
DATA = os.path.join(ROOT, 'data')
REGISTRY = os.path.join(DATA, 'source_registry.json')
#: ワークフローがコミットするファイル（これ以外は触っても出さない）
OUTPUTS = ('source_registry.json', 'xgb_fukusho_model_residual.pkl', 'xgb_feature_cols_residual.json',
           'xgb_fukusho_model.pkl', 'xgb_feature_cols.json', 'xgb_calibrator.pkl',
           'ability_calibrator.pkl', 'xgb_solo_model.pkl', 'xgb_solo_cols.json')


def fail(msg):
    print(f'::error::{msg}')
    sys.exit(1)


def adopt_in_registry(source, card, results, result_json):
    from src.features.web_signals import feature_cols_for
    with open(REGISTRY, encoding='utf-8') as f:
        reg = json.load(f)
    with open(os.path.join(ROOT, result_json), encoding='utf-8') as f:
        res = json.load(f)
    entry = next((e for e in reg['sources'] if e.get('source') == source), None)
    if entry is None:
        fail(f'台帳に {source} が無い')
    for rel in (card, results):
        if not os.path.exists(os.path.join(ROOT, rel)):
            fail(f'{rel} が無い')
    prev = dict(entry.get('measured') or {})
    entry.update(
        status='adopted', card=card, results=results,
        feature_cols=feature_cols_for(source),
        decided_at=datetime.now(JST).strftime('%Y-%m-%d'),
        decided_by='ユーザー決定 2026-11-13「(a) 採用」／基準 websig/ADOPTION_STANDARD.md',
        measured=dict(
            measured_at='2026-11-11',
            delta_auc_mean=round(float(res['pooled_delta_mean']), 5),
            delta_auc_windows=[round(float(res['windows_result'][w]['delta_mean']), 5)
                               for w in ('W1', 'W2', 'W3')],
            axis_fukusho_hit=[res['arms']['A']['42']['fuku']['rate'], res['arms']['X']['42']['fuku']['rate']],
            control_a_mean=round(float(res['control_a']['mean']), 5),
            note=('効果の基準はすべて通過。対照群A の合格線（帰無が0中心）だけ不合格＝カードの線の'
                  '置き方の誤りで、ずれは採用に不利な側。ユーザーがそれを認めたうえで採用を決定'),
            previous=prev,
        ))
    with open(REGISTRY, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)   # 元の書式（indent=1）を保つ
        f.write('\n')
    from src.utils.source_registry import adopted_sources
    if not any(e.get('source') == source for e in adopted_sources(ROOT)):
        fail('台帳を書き換えたが採用として認識されない（裏付けの4点が欠けている）')
    print(f'✅ 台帳: {source} を adopted にした  列 {entry["feature_cols"]}')
    return entry['feature_cols']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='kayochinkeiba')
    ap.add_argument('--card', default='websig/CRITERIA_CONFIRM_kayochin.md')
    ap.add_argument('--results', default='websig/RESULTS_CONFIRM_kayochin.md')
    ap.add_argument('--result-json', default='websig/confirm_kayochin_result.json')
    # 現行の本番（2026-09-15 作成）と同じ窓。変えるのは「指数の2列を足す」ことだけ
    ap.add_argument('--train-end', default='2026-06-30')
    ap.add_argument('--val-start', default='2026-07-01')
    ap.add_argument('--val-end', default='2026-08-31')
    a = ap.parse_args()

    db = os.path.join(DATA, 'private', 'web_signals.db')
    if not os.path.exists(db):
        fail(f'{db} が無い（artifact の復元に失敗している）')
    n = sqlite3.connect(db).execute(
        'SELECT COUNT(*), MIN(date), MAX(date) FROM web_signals WHERE source=?', (a.source,)).fetchone()
    print(f'指数DB: {n[0]:,}頭  {n[1]}〜{n[2]}')
    if n[0] < 100000:
        fail('指数DBが小さすぎる（取り直しが終わっていない）')

    with open(os.path.join(DATA, 'xgb_feature_cols.json'), encoding='utf-8') as f:
        old_meta = json.load(f)
    old_cols = list(old_meta['feature_cols'])
    print(f'旧モデル: {len(old_cols)}列  trained_at {old_meta.get("trained_at")}  val_auc {old_meta.get("val_auc")}')

    web_cols = adopt_in_registry(a.source, a.card, a.results, a.result_json)

    # ── 学習データ ──
    import pandas as pd
    from src.tools.build_training_data import build_training_data
    build_training_data(ROOT)
    df = pd.read_csv(os.path.join(DATA, 'horse_features.csv'))
    for c in web_cols:
        if c not in df.columns:
            fail(f'学習データに {c} が無い（採用済みなのに列が作られていない）')
    df['year'] = df['date'].astype(str).str[:4]
    cov = df.groupby('year')[web_cols[0]].apply(lambda s: s.notna().mean())
    print('指数が付いた行の割合（年別）:', {k: f'{v:.1%}' for k, v in cov.items()})
    if df[web_cols[0]].notna().mean() < 0.5:
        fail('指数が付いた行が全体の50%未満（突合が壊れている）')
    # レース内順位は 1 以上・z は 0 中心のはず
    r, z = df[web_cols[0]].dropna(), df[web_cols[1]].dropna()
    print(f'  {web_cols[0]}: min {r.min():.1f} / max {r.max():.1f}   {web_cols[1]}: mean {z.mean():+.3f} / sd {z.std():.3f}')
    if r.min() < 1 or abs(z.mean()) > 0.05:
        fail('指数の列の値がおかしい')

    # ── 学習 ──
    from src.tools.train_xgb import train_xgb
    m = train_xgb(ROOT, train_end=a.train_end, val_start=a.val_start, val_end=a.val_end,
                  use_optuna=False, residual=True)
    print('学習結果:', {k: m[k] for k in ('auc', 'brier', 'logloss', 'n_features', 'n_train', 'n_val')},
          ' 旧モデル（同じ検証データ）:', m.get('old_model'))
    with open(os.path.join(DATA, 'xgb_feature_cols_residual.json'), encoding='utf-8') as f:
        new_meta = json.load(f)
    new_cols = list(new_meta['feature_cols'])
    if set(new_cols) != set(old_cols) | set(web_cols):
        # train_xgb は「旧モデルより検証AUCが低いと正式採用しない」。その場合ここで止まる
        fail(f'新モデルが本番ファイルに反映されていない、または列が想定と違う '
             f'（新 {len(new_cols)}列 / 旧 {len(old_cols)}列＋{len(web_cols)}）。'
             f'学習ログの「正式採用スキップ」を確認すること')
    shutil.copy2(os.path.join(DATA, 'xgb_fukusho_model_residual.pkl'), os.path.join(DATA, 'xgb_fukusho_model.pkl'))
    shutil.copy2(os.path.join(DATA, 'xgb_feature_cols_residual.json'), os.path.join(DATA, 'xgb_feature_cols.json'))
    print(f'✅ 本番ファイルに反映（{len(new_cols)}列）')

    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(os.path.join(DATA, 'xgb_fukusho_model.pkl'))
    gain = b.get_score(importance_type='gain')
    tot = sum(gain.values()) or 1.0
    print(f'木の本数 {b.num_boosted_rounds()}  best_iteration {b.attributes().get("best_iteration")}  '
          + '  '.join(f'{c} の重要度 {gain.get(c, 0) / tot:.2%}' for c in web_cols))

    # ── 較正器 ──
    from src.features import engine
    engine.init_engine(ROOT)          # 新モデルで初期化し直す
    from src.tools.calibrate_xgb import run_xgb_calibration
    if run_xgb_calibration(ROOT) is None:
        fail('較正器（xgb_calibrator.pkl）を作れなかった')
    from scripts.build_ability_calibrator import build_ability_calibrator
    if build_ability_calibrator(ROOT) is None:
        fail('ability 較正器を作れなかった')

    # ── 「AI単」列の単独モデル（列が本番と1つでも違うと engine は使わないので、必ず一緒に作り直す）──
    from src.tools.train_solo import train_solo
    _, solo_meta = train_solo(ROOT, train_end=a.train_end, val_start=a.val_start, val_end=a.val_end)
    print('単独モデル:', {k: v for k, v in solo_meta.items() if k != 'feature_cols'})

    missing = [f for f in OUTPUTS if not os.path.exists(os.path.join(DATA, f))]
    if missing:
        fail(f'出力が欠けている: {missing}')
    print('✅ 完了。コミット対象:', ', '.join(OUTPUTS))


if __name__ == '__main__':
    main()

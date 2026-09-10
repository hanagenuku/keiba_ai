"""EV用の絶対確率を作る Isotonic 較正器を生成する（`data/ability_calibrator.pkl`）。

## なぜ要るか

本番の残差モデルは `raw_margin = base_margin(当日の市場人気) + f_AI` なので、
画面の勝率も cal_prob も**当日のオッズを土台にしている**。
「AI独自確率 → 必要オッズ → 実オッズと比較」という設計を成立させるには、
当日のオッズを直接使わない確率が要る。それが

    E0 = sigmoid(ability_margin + フラット土台)          ← 未補正
    E1 = Isotonic(E0)                                  ← ここで作る較正器

🔴 E0 をそのまま確率として読んではいけない（2026-09-11 に実測）:
    40-50%帯で 予測43.8% / 実測62.7% ＝ **+18.9pt 過小**
🔴 softmax(T=3.5) を通したもの（2026-09-10 に画面へ入れた版）も駄目:
    ECE 0.0757。10-20%帯で -7.8pt、50-60%帯で +18.0pt
✅ E1 は確認期 24,993頭で ECE **0.0080**（cal_prob 0.0094 / 市場 0.0097 より良い）。
   70%までの全帯で ±1.6pt 以内。詳細は `calib/RESULTS_ability.md`

⚠ **ROIは改善しない。** 必要オッズを超える馬だけを買った実測は
   単勝 8,703頭で **73.7%**（全部買う 73.1%）。これは
   「画面の数字が嘘をつかなくなる」変更であって「儲かる」変更ではない。

## 較正器の学習データ

本番モデルの `train_end` より**後**の行だけを使う（モデルにとって out-of-sample）。
⚠ 本番モデルを差し替えたら、この較正器も作り直すこと。
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_ROWS = 3000     # これを下回るなら作らない（薄い較正器は害になる）


def build_ability_calibrator(base_dir, features_csv=None, min_rows=MIN_ROWS):
    """本番モデルで ability経路の確率を作り、Isotonic 較正器を保存する。

    Returns: dict（保存した内容）／作れなければ None
    """
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    from src.features.engine import _flat_base_margin
    from src.tools.train_xgb import _popularity_to_base_margin

    meta_path = os.path.join(base_dir, 'data', 'xgb_feature_cols.json')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    if not meta.get('residual'):
        print('⚠ 残差モデルではないので ability経路が定義できない。作らない')
        return None
    cols = meta['feature_cols']

    csv = features_csv or os.path.join(base_dir, 'data', 'horse_features.csv')
    if not os.path.exists(csv):
        print(f'⚠ {csv} が無い。build_training_data を先に実行すること')
        return None
    df = pd.read_csv(csv)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj']).reset_index(drop=True)

    # 🔴 本番モデルが学習に使った期間は除く（較正器が in-sample にならないように）
    train_end = pd.Timestamp(meta.get('train_end') or '2026-06-30')
    m = df['date_obj'] > train_end
    if m.sum() < min_rows:
        print(f'⚠ train_end({train_end.date()}) 以降が {int(m.sum()):,}行しかない'
              f'（最低 {min_rows:,}行）。作らない')
        return None
    d = df[m].reset_index(drop=True)

    field = d.groupby('race_id')['horse_num'].transform('count').to_numpy(float)
    pop = d['f_popularity'].to_numpy(float)
    pop = np.where(np.isfinite(pop), pop, field / 2.0)     # train_xgb と同じ埋め方
    bm = _popularity_to_base_margin(pd.Series(pop), pd.Series(field))
    # ✅ ability = predict(output_margin) − base_margin は base_margin の取り方に依存しない
    #    （実測: ±3 ずらしても最大差 2.9e-06。2026-08-25 D-2② と同じ検算）。
    #    したがってドリフト注入の有無はここでは効かない。bm は形式的に置くだけ。

    booster = xgb.Booster()
    booster.load_model(os.path.join(base_dir, 'data', 'xgb_fukusho_model.pkl'))
    # ⚠ 列名を渡さないと Booster が「feature names が無い」と拒否する
    dm = xgb.DMatrix(d[cols].fillna(5.0).astype(np.float32), feature_names=list(cols))
    dm.set_base_margin(bm)
    ability = booster.predict(dm, output_margin=True) - bm

    flat = np.array([_flat_base_margin(int(n)) for n in field])
    e0 = 1.0 / (1.0 + np.exp(-(ability + flat)))

    y3 = d['is_fukusho'].to_numpy(int)
    y1 = (d['place'] == 1).to_numpy(int)
    out = {
        'fuku': IsotonicRegression(out_of_bounds='clip').fit(e0, y3),
        'win': IsotonicRegression(out_of_bounds='clip').fit(e0, y1),
        'n_rows': int(len(d)),
        'fitted_on': f'{d.date_obj.min().date()}〜{d.date_obj.max().date()}',
        'model_trained_at': meta.get('trained_at'),
        'train_end': str(train_end.date()),
    }
    path = os.path.join(base_dir, 'data', 'ability_calibrator.pkl')
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    p3, p1 = out['fuku'].predict(e0), out['win'].predict(e0)
    print(f'✅ ability_calibrator.pkl 生成: {out["n_rows"]:,}行 ({out["fitted_on"]})')
    print(f'   3着内 予測平均 {p3.mean()*100:.2f}% / 実測 {y3.mean()*100:.2f}%'
          f'   （未補正なら {e0.mean()*100:.2f}%）')
    print(f'   1着   予測平均 {p1.mean()*100:.2f}% / 実測 {y1.mean()*100:.2f}%')
    return out


if __name__ == '__main__':
    build_ability_calibrator(ROOT, features_csv=(sys.argv[1] if len(sys.argv) > 1 else None))

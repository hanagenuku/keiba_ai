"""D-2: 残差学習の base_margin を「人気順位のZipf近似」→「実オッズ」にしたら効くか。

2026-08-06 に +0.008〜0.020 AUC と実測されたが、history.db の win_odds が全年0%
充足でブロックされていた案。netkeiba から実オッズを取れるようになったので測る。

🔑 再学習は要らない。残差学習は
      raw_margin = base_margin + ability_margin
   で、`ability_margin` は base_margin に**依存しない**（2026-08-03に検算・
   最大ずれ 2.8e-06）。なので学習済みモデルのまま、評価時にアンカーだけ差し替えて
   比べられる。⚠ この前提が崩れると以降が全部無意味なので、最初に検算する。

🔴 判定は「確定オッズをそのまま」では**やってはいけない**。
   netkeiba のオッズは確定オッズだが、本番が見るのは朝のオッズ。確定オッズで測ると
   本番では手に入らない情報を使うので必ず良く出る。
   2026-07-31 と 2026-08-21 に2回踏んだ罠（`simulate_serving_popularity`）。
   そのため3つ測って**Cで判定する**:

     A) 人気順位アンカー + ドリフト注入   ← 本番の現状。これを上回る必要がある
     B) 実オッズアンカー（確定そのまま）   ← 到達不能な上限。参考値
     C) 実オッズアンカー + オッズドリフト  ← 本番相当。**判定はこれ**

⚠ AUCが上がっても回収率が上がるとは書かない（North Star #9）。
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.train_xgb import (_CLIP_PROB, _apply_popularity_drift,
                                 _load_xgb_model_any, _popularity_to_base_margin,
                                 load_popularity_drift)

# 単勝オッズ盤の Σ(1/オッズ) の上限。控除率20%なら約1.25。
# 2026-08-20 C-2 の実測: 正常な盤 1,828件は中央値1.259・最大1.577、
# 壊れた盤 213件は最小1.714。空白帯の中間を取る。モデルを使わない検査。
MAX_VALID_ODDS_BOOK_SUM = 1.65
MIN_HORSES_FOR_BOOK = 5
# 窓ごとの最低行数。これを下回る窓は判定に使わない（AUCが不安定なため）
MIN_ROWS_PER_WINDOW = 500


def _odds_to_base_margin(odds, race_ids):
    """実オッズ→レース内で正規化した勝率→logit。

    `_popularity_to_base_margin` と同じ「レース内で合計1に正規化した勝率のlogit」
    を返す。違いは中身が Zipf近似か実測かだけ（apples-to-apples）。
    Σ(1/オッズ) で割るので控除率は自動的に落ちる。
    """
    inv = 1.0 / np.asarray(odds, dtype=float)
    s = pd.Series(inv).groupby(np.asarray(race_ids)).transform('sum').values
    p = np.clip(inv / s, _CLIP_PROB, 1 - _CLIP_PROB)
    return np.log(p / (1 - p))


def load_odds_drift(base_dir):
    """「朝のオッズ / 確定オッズ」の実測比を、確定人気ごとに集めた分布。

    `load_popularity_drift` が順位のズレを集めているのに対し、こちらは
    **値のズレ**を集める。実オッズをアンカーにする場合、順位だけ揺らしても
    本番の不確かさを再現できないため。
    2026-08-06 の実測では |log比| 平均0.532・24.4%の馬が2倍以上動く。

    ⚠ この分布はドリフトを**過小に見積もる**。`race_predictions` は同一race_idを
      上書きするため（2026-07-27⑩）、2026-08-17に当日refreshを3回に増やして以降、
      ここに残るのは朝ではなく**最後のrefresh(14:00)のオッズ**になっている。
      実測 |log比| 平均は 0.470 まで下がっており、当時の0.532より小さい。
      過小なドリフト＝実オッズアンカーに有利に出るので、Cが基準を満たした時ほど
      この点を割り引いて読むこと。順位ドリフト（アンカーA側）も同じ影響を受ける。

    Returns: dict {確定人気: np.ndarray(log比の標本)} / 足りなければ None
    """
    db = os.path.join(base_dir, 'data', 'keiba.db')
    if not os.path.exists(db):
        return None
    try:
        conn = sqlite3.connect(db)
        rows = conn.execute("""
            SELECT p.race_id, p.horse_num, p.tansho_odds, s.tansho
            FROM race_predictions p
            JOIN (SELECT race_id, horse_num, tansho, MAX(captured_at)
                  FROM odds_snapshots WHERE tansho >= 1.0
                  GROUP BY race_id, horse_num) s
              ON s.race_id = p.race_id AND s.horse_num = p.horse_num
            WHERE p.tansho_odds >= 1.0
        """).fetchall()
        conn.close()
    except Exception:
        return None

    by_race = defaultdict(list)
    for rid, hn, morn, late in rows:
        by_race[rid].append((hn, morn, late))

    pairs = []
    for hs in by_race.values():
        if len(hs) < 8:
            continue
        late_rank = {h[0]: i for i, h in enumerate(sorted(hs, key=lambda x: x[2]), 1)}
        for hn, morn, late in hs:
            pairs.append((late_rank[hn], np.log(morn / late)))
    if len(pairs) < 500:
        return None
    arr = np.array(pairs)
    out = {int(p): arr[arr[:, 0] == p][:, 1]
           for p in np.unique(arr[:, 0]) if (arr[:, 0] == p).sum() >= 30}
    out['_all'] = arr[:, 1]
    return out


def _apply_odds_drift(odds, pop, drift, seed=0):
    """確定オッズに実測の値ズレを注入して「朝のオッズ」を再現する。"""
    rng = np.random.default_rng(seed)
    fallback = drift['_all']
    jitter = np.array([rng.choice(drift.get(int(p), fallback)) for p in pop])
    return np.asarray(odds, dtype=float) * np.exp(jitter)


def require_drift(pop_drift, odds_drift):
    """ドリフト分布が無ければ**評価を中止する**。

    🔴 ここで「じゃあ確定オッズだけで測ろう」としてはいけない。
       netkeiba のオッズは確定オッズで、本番が見るのは朝のオッズ。確定オッズで
       測ると本番では手に入らない情報を使うことになり、実オッズアンカーが必ず
       良く出る。2026-07-31（ドリフトOFFが有利に見えた）と 2026-08-21（新旧
       モデルの優劣が反転した）で2回踏んだ罠。
       測れないなら「測れない」と言う方がまし。
    """
    if pop_drift is None or odds_drift is None:
        raise SystemExit(
            '🔴 ドリフト分布が作れない（keiba.db の朝/直前オッズが不足）。\n'
            '   確定オッズだけで判定すると実オッズアンカーが不当に良く出るので中止する。'
        )


def _metrics(margin, y):
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    p = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
    return (roc_auc_score(y, p), brier_score_loss(y, p), log_loss(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--features', default=None,
                    help='horse_features.csv のパス（既定は data/ 配下）')
    ap.add_argument('--start', default='2025-07-01')
    ap.add_argument('--end', default='2026-08-23')
    ap.add_argument('--windows', type=int, default=3,
                    help='検定窓の分割数。全窓でプラスかを見る（事前登録の基準①）')
    args = ap.parse_args()
    base = args.base_dir

    # ── 特徴量 ────────────────────────────────────────────────────────────
    csv = args.features or os.path.join(base, 'data', 'horse_features.csv')
    print(f'features: {csv}')
    df = pd.read_csv(csv, low_memory=False)
    df['date'] = df['date'].astype(str).str[:10]
    df = df[(df['date'] >= args.start) & (df['date'] <= args.end)].copy()
    print(f'  検定窓 {args.start}〜{args.end}: {len(df):,}行 / '
          f'{df["race_id"].nunique():,}レース')

    import json
    with open(os.path.join(base, 'data', 'xgb_feature_cols.json')) as f:
        meta = json.load(f)
    cols = meta['feature_cols']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f'🔴 CSVに特徴量が {len(missing)} 個足りない: {missing[:8]}\n'
                         '   build_training_data を先に流すこと')
    if not meta.get('residual'):
        raise SystemExit('🔴 残差学習モデルでないとアンカーの差し替えができない')

    # ── netkeiba の実オッズを結合 ────────────────────────────────────────
    nk = os.path.join(base, 'data', 'private', 'netkeiba.db')
    if not os.path.exists(nk):
        raise SystemExit(f'🔴 {nk} が無い。fetch-netkeiba-odds の artifact を復元すること')
    conn = sqlite3.connect(nk)
    odds = pd.read_sql('SELECT race_id, horse_num, win_odds FROM netkeiba_odds '
                       'WHERE win_odds IS NOT NULL AND win_odds >= 1.0', conn)
    conn.close()
    print(f'  netkeiba: {len(odds):,}頭 / {odds["race_id"].nunique():,}レース')

    df = df.merge(odds, on=['race_id', 'horse_num'], how='left')

    # 🔑 レース内の全馬にオッズが揃っていない盤は使わない。
    #    半端に残すと正規化が狂い「オッズ不明」より悪くなる（2026-08-20 C-2）。
    g = df.groupby('race_id')['win_odds']
    full = g.transform(lambda s: s.notna().all())
    # 盤の妥当性: Σ(1/オッズ) が控除率から決まる範囲に収まるか（モデル不要の検査）
    book = df.assign(inv=1.0 / df['win_odds']).groupby('race_id')['inv'].transform('sum')
    n_in_race = df.groupby('race_id')['horse_num'].transform('count')
    ok = full & ((book <= MAX_VALID_ODDS_BOOK_SUM) | (n_in_race < MIN_HORSES_FOR_BOOK))
    dropped_partial = df.loc[~full, 'race_id'].nunique()
    dropped_book = df.loc[full & ~ok, 'race_id'].nunique()
    df = df[ok].copy()
    print(f'  除外: オッズ欠けの盤 {dropped_partial:,}レース / '
          f'Σ(1/オッズ)が異常な盤 {dropped_book:,}レース')
    print(f'  評価対象 {len(df):,}頭 / {df["race_id"].nunique():,}レース')
    if df.empty:
        raise SystemExit('🔴 評価できるレースが残らなかった')

    df['_n'] = df.groupby('race_id')['horse_num'].transform('count')
    y = df['is_fukusho'].values

    # ── ability_margin（アンカーに依存しない部分）を取り出す ────────────
    import xgboost as xgb
    booster = _load_xgb_model_any(
        os.path.join(base, 'data', 'xgb_fukusho_model.pkl'), is_residual=True)
    X = df[cols].astype(float)

    d0 = xgb.DMatrix(X, feature_names=cols)
    d0.set_base_margin(np.zeros(len(df)))
    ability = booster.predict(d0, output_margin=True)

    bm_rank = _popularity_to_base_margin(df['f_popularity'], df['_n'])
    d1 = xgb.DMatrix(X, feature_names=cols)
    d1.set_base_margin(bm_rank)
    raw = booster.predict(d1, output_margin=True)

    # 🔑 North Star #8: 前提を別経路で作って突合する。ここが合わなければ以降は嘘。
    resid = np.abs((raw - bm_rank) - ability)
    print(f'\n■ 検算: ability_margin が base_margin に依存しないこと')
    print(f'  |(raw - bm) - ability| 最大 {resid.max():.3e} / 平均 {resid.mean():.3e}')
    if resid.max() > 1e-4:
        raise SystemExit('🔴 ability_margin がアンカーに依存している。'
                         'アンカーの差し替えでは測れない')

    # ── 3つのアンカー ────────────────────────────────────────────────────
    pop_drift = load_popularity_drift(base)
    odds_drift = load_odds_drift(base)
    require_drift(pop_drift, odds_drift)
    print(f'  ドリフト標本: 順位 {len(pop_drift["_all"]):,} / '
          f'オッズ {len(odds_drift["_all"]):,}')

    rid = df['race_id'].values
    pop_d = _apply_popularity_drift(df['f_popularity'], rid, pop_drift, seed=12)
    odds_d = _apply_odds_drift(df['win_odds'].values, df['f_popularity'].values,
                               odds_drift, seed=12)

    anchors = {
        'A 人気順位 + ドリフト（本番の現状）': _popularity_to_base_margin(
            pd.Series(pop_d), df['_n']),
        'B 実オッズ（確定・到達不能な上限）': _odds_to_base_margin(df['win_odds'], rid),
        'C 実オッズ + ドリフト（本番相当・判定はこれ）': _odds_to_base_margin(odds_d, rid),
    }

    print(f'\n■ 全期間 N={len(df):,}頭')
    print(f'{"アンカー":<40} {"AUC":>8} {"Brier":>8} {"LogLoss":>9}')
    base_auc = None
    for name, bm in anchors.items():
        auc, br, ll = _metrics(bm + ability, y)
        if base_auc is None:
            base_auc = auc
            print(f'{name:<40} {auc:>8.4f} {br:>8.4f} {ll:>9.4f}')
        else:
            print(f'{name:<40} {auc:>8.4f} {br:>8.4f} {ll:>9.4f}   '
                  f'(A比 {auc - base_auc:+.4f})')

    # ── 窓ごと（事前登録の基準①: 全窓でプラス）──────────────────────────
    dates = np.sort(df['date'].unique())
    edges = np.array_split(dates, args.windows)
    print(f'\n■ 窓ごと（全窓でプラスが基準①）')
    print(f'{"窓":<26} {"N":>7} {"A":>8} {"B":>8} {"C":>8} {"C-A":>9}')
    deltas = []
    skipped = 0
    for w in edges:
        m = df['date'].isin(w).values
        if m.sum() < MIN_ROWS_PER_WINDOW:
            # 黙って飛ばすと「窓が1つも無い＝判定不能」に気づけない
            print(f'{w[0]}〜{w[-1]:<12} {m.sum():>7,}  ← {MIN_ROWS_PER_WINDOW}行未満のため除外')
            skipped += 1
            continue
        a = _metrics(anchors['A 人気順位 + ドリフト（本番の現状）'][m] + ability[m], y[m])[0]
        b = _metrics(anchors['B 実オッズ（確定・到達不能な上限）'][m] + ability[m], y[m])[0]
        c = _metrics(anchors['C 実オッズ + ドリフト（本番相当・判定はこれ）'][m] + ability[m], y[m])[0]
        deltas.append(c - a)
        print(f'{w[0]}〜{w[-1]:<12} {m.sum():>7,} {a:>8.4f} {b:>8.4f} {c:>8.4f} {c - a:>+9.4f}')

    print(f'\n■ 判定（CRITERIA_netkeiba.md の事前登録）')
    if not deltas:
        # 🔴 ここで黙ると「測ったつもりで何も測っていない」になる
        print(f'  🔴 判定不能: 有効な窓が0個（{skipped}個が{MIN_ROWS_PER_WINDOW}行未満）。')
        print('     データが足りないか --windows が多すぎる。数字を採用しないこと')
        return
    print(f'  ① 全窓でプラス      : {"✅" if all(d > 0 for d in deltas) else "❌"} '
          f'({sum(d > 0 for d in deltas)}/{len(deltas)}窓)')
    print(f'  ② 平均 +0.01 以上   : '
          f'{"✅" if np.mean(deltas) >= 0.01 else "❌"} ({np.mean(deltas):+.4f})')
    if skipped:
        print(f'  ⚠ {skipped}窓が行数不足で除外されている')
    print('  ⚠ AUCが上がっても回収率が上がるとは限らない（North Star #9）')
    print('  ⚠ Bは本番では手に入らない情報を使った上限。Bを根拠にしないこと')


if __name__ == '__main__':
    main()

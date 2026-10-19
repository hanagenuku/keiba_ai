"""B（較正のパリティ違反）の測定ハーネス。カード retrain/CRITERIA_B.md の設計どおり。

OLD（確定人気で当てる・現行実装）と NEW（ドリフトを注入して当てる・修正版）の
較正器を、**本番の朝オッズ条件で記録された実際の予想行**に対して比較する。

🔴 評価行の raw_prob は両アームで同一。違うのは較正器だけ。
🔴 base_margin は `race_predictions.popularity`（engine が朝オッズから導いた実際の値）。
   シミュレートしたドリフトではなく本番の条件を使う（North Star #10）。
🔴 ビン化と PAV は `calibrate_xgb._fit_isotonic` をそのまま使う（自前で書き直さない）。

使い方: python3 retrain/calib_ab.py <workspace_dir>
⚠ workspace_dir は隔離コピーを渡すこと。
"""
import os, sys, json, math, sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb

TRAIN_END = '2026-06-30'     # 評価行と重ならないこと（本番モデルの学習期間末と同じ）
DRIFT_SEED = 13              # train_xgb の 11/12 と衝突させない
N_BINS = 20


def ece_and_bins(p, y, k=5):
    o = np.argsort(p); p, y = np.asarray(p)[o], np.asarray(y)[o]
    n = len(p); ece = 0.0; rows = []
    for i in range(k):
        b = slice(i * n // k, (i + 1) * n // k)
        pp, aa, m = p[b].mean(), y[b].mean(), len(p[b])
        ece += abs(pp - aa) * m / n
        rows.append((m, pp, aa))
    return ece, rows


def main(W):
    sys.path.insert(0, W)
    import src.tools.train_xgb as T
    from src.tools.calibrate_xgb import _fit_isotonic
    from src.models.calibration_xgb import load_xgb_calibrator

    # ── 特徴量 ──
    df = pd.read_csv(os.path.join(W, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj'])
    meta = json.load(open(os.path.join(W, 'data', 'xgb_feature_cols_residual.json')))
    feat = list(meta['feature_cols'])
    P = T._load_xgb_model_any(os.path.join(W, 'data', 'xgb_fukusho_model_residual.pkl'), True)
    cal_now = load_xgb_calibrator(W)
    drift = T.load_popularity_drift(W)
    print(f'CSV {len(df):,}行 / 特徴量 {len(feat)} / 現行較正器 {"OK" if cal_now else "なし"} / '
          f'drift {"ON" if drift is not None else "OFF"}')

    def raw_from(X, pop, n_h):
        bm = np.asarray(T._popularity_to_base_margin(pd.Series(pop), pd.Series(n_h)), dtype=float)
        d = xgb.DMatrix(X.reindex(columns=feat, fill_value=5.0), feature_names=feat)
        d.set_base_margin(bm)
        return 1.0 / (1.0 + np.exp(-P.predict(d, output_margin=True)))

    # ── 評価行: 本番が実際に出した予想（朝オッズ条件）──
    con = sqlite3.connect(os.path.join(W, 'data', 'keiba.db'))
    rp = pd.read_sql_query(
        "SELECT date, race_id, horse_num, popularity, cal_prob, actual_place FROM race_predictions "
        f"WHERE date > '{TRAIN_END}' AND actual_place IS NOT NULL", con)
    con.close()
    rp = rp[(rp.popularity > 0) & (rp.popularity != 99)]
    rp['_n'] = rp.groupby('race_id')['horse_num'].transform('count')   # 朝時点の出走頭数
    ev = rp.merge(df, on=['race_id', 'horse_num'], how='inner', suffixes=('', '_csv'))
    print(f'評価行: race_predictions {len(rp):,} → CSV と結合できた {len(ev):,}')

    raw_ev = raw_from(ev[feat], ev['popularity'].values, ev['_n'].values)
    y_ev = (ev['actual_place'] <= 3).astype(int).values

    # ── 中止条件2（カード §9 v2）──
    # 2a: base_margin の分母（出走頭数）が CSV と一致するか。ここがズレると base_margin が丸ごと狂う。
    n_csv = df.groupby('race_id')['horse_num'].count().rename('n_csv')
    chk = ev[['race_id', '_n']].merge(n_csv, on='race_id', how='left')
    ok_n = float((chk['_n'] == chk['n_csv']).mean())
    print(f'\n中止条件2a 出走頭数の一致: {ok_n:.4f}  → {"✅ 通過" if ok_n >= 1.0 else "🔴 不通過"}')

    # 2b: 9/15 以降（いまの data/ のモデル＋較正器が実際に本番で動いていた期間）で、
    #     再構成に現行 pkl を当てた ECE が保存 cal_prob の ECE を再現するか。
    #     合格線は手で決めず ECE 自身の帰無分布の std×2 から導く。
    # engine.py:3112 は transform の結果を [0.01, 0.99] にクリップして保存する。
    rec = np.clip(np.asarray(cal_now.transform(raw_ev.tolist()), dtype=float), 0.01, 0.99)
    m15 = (ev['date'].astype(str) >= '2026-09-15').values
    e_rec15 = ece_and_bins(rec[m15], y_ev[m15])[0]
    e_sav15 = ece_and_bins(ev['cal_prob'].values[m15], y_ev[m15])[0]
    rng = np.random.default_rng(20261019)
    null = np.array([ece_and_bins(rec[m15], rng.binomial(1, rec[m15]))[0] for _ in range(400)])
    lim = float(null.std() * 2)
    gap = abs(e_rec15 - e_sav15)
    print(f'中止条件2b 9/15以降 N={int(m15.sum()):,}: ECE 保存 {e_sav15:.4f} / 再構成 {e_rec15:.4f} '
          f'差 {gap:.4f}  判定線（帰無std×2）{lim:.4f}  → {"✅ 通過" if gap <= lim else "🔴 不通過"}')
    if ok_n < 1.0 or gap > lim:
        print('   破棄。raw_prob の再構成が本番と一致していない（ECEを読まずに破棄）。')
        return
    d_all = float(np.median(np.abs(rec - ev['cal_prob'].values)))
    print(f'   参考: 全期間の中央値差 {d_all:.5f}（世代交代ぶん。カード §9 の階段）')

    # ── 学習行（評価行と重ならない）──
    tr = df[df.date_obj <= pd.Timestamp(TRAIN_END)].copy()
    tr['_n'] = tr.groupby('race_id')['horse_num'].transform('count')
    pop_conf = tr['f_popularity'].fillna(tr['_n'] / 2)
    pop_drift = pd.Series(T._apply_popularity_drift(
        pop_conf, tr['race_id'].values, drift, seed=DRIFT_SEED))
    print(f'\n学習行: {len(tr):,}（〜{TRAIN_END}） / '
          f'人気のズレ 平均 {float(np.abs(pop_drift.values - pop_conf.values).mean()):.2f}位')

    arms = {}
    for lbl, pop in (('OLD（確定人気・現行実装）', pop_conf), ('NEW（ドリフト注入・修正版）', pop_drift)):
        raw_tr = raw_from(tr[feat], pop.values, tr['_n'].values)
        c = _fit_isotonic(pd.DataFrame({'raw_prob': raw_tr,
                                        'is_fukusho': tr['is_fukusho'].values}), n_bins=N_BINS)
        arms[lbl] = c
        print(f'  {lbl}: raw_prob 平均 {raw_tr.mean():.4f} → 較正器のビン {len(c.x_bins)}個')

    # ── 評価（両アーム同一の raw_ev に当てる）──
    print(f'\n{"":34}{"ECE":>8}{"平均予測":>10}{"平均実測":>10}{"水準差":>9}')
    out = {}
    for lbl, c in arms.items():
        p = np.asarray(c.transform(raw_ev.tolist()), dtype=float)
        e, rows = ece_and_bins(p, y_ev)
        out[lbl] = (e, rows, p)
        print(f'  {lbl:<32}{e:>8.4f}{p.mean():>10.3f}{y_ev.mean():>10.3f}{p.mean()-y_ev.mean():>+9.3f}')
    e_cur, rows_cur = ece_and_bins(rec, y_ev)[:2]
    print(f'  {"（参考）現行 pkl そのもの":<32}{e_cur:>8.4f}{rec.mean():>10.3f}{y_ev.mean():>10.3f}'
          f'{rec.mean()-y_ev.mean():>+9.3f}')

    kO, kN = list(arms)[0], list(arms)[1]
    eO, rO, _ = out[kO]; eN, rN, _ = out[kN]
    print(f'\n■ 5分位の |予測−実測|')
    print(f'  {"帯":<5}{"N":>7}{"OLD予測":>9}{"OLD実測":>9}{"OLD差":>8}{"NEW予測":>9}{"NEW実測":>9}{"NEW差":>8}  改善')
    better = 0
    for i, ((m, pO, aO), (_, pN, aN)) in enumerate(zip(rO, rN), 1):
        ok = abs(pN - aN) <= abs(pO - aO)
        better += ok
        print(f'  Q{i:<4}{m:>7}{pO:>9.3f}{aO:>9.3f}{pO-aO:>+8.3f}{pN:>9.3f}{aN:>9.3f}{pN-aN:>+8.3f}  '
              f'{"✅" if ok else "❌"}')

    print(f'\n■ 採用基準（カード §5・4つすべて必要）')
    c1 = eN < eO
    c2 = (eO - eN) >= 0.010
    c3 = better >= 4
    c4 = abs(rN[0][1] * 0 + np.mean([r[1] for r in rN]) - y_ev.mean()) <= \
         abs(np.mean([r[1] for r in rO]) - y_ev.mean())
    for lbl, ok in [(f'1 ECE 改善（NEW {eN:.4f} < OLD {eO:.4f}）', c1),
                    (f'2 改善幅 >= 0.010（実測 {eO-eN:+.4f}）', c2),
                    (f'3 5分位の 4帯以上で悪化しない（{better}/5）', c3),
                    ('4 全体の水準を壊さない', c4)]:
        print(f'   {"✅" if ok else "❌"} 基準{lbl}')
    print(f'\n  判定: {"採用（コード修正を本番へ）" if all([c1,c2,c3,c4]) else "不採用（較正器は差し替えない）"}')
    print(f'  中止条件3（OLD が 0.0682 ±0.01 か）: OLD {eO:.4f} → '
          f'{"✅" if abs(eO-0.0682)<=0.01 else "⚠ 離れている（fit の再現に注意）"}')


if __name__ == '__main__':
    main(sys.argv[1])

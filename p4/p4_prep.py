"""Phase 4 準備: cid82 の out-of-sample 予測 + レース条件メタデータ。

窓の作り方は mf1/prep.py と完全に同一（行順・内側HO42日・train_end）。
違いは「評価行の元インデックスを保持し、history.db の条件を貼る」ことだけ。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)

INNER_DAYS = 42
MARKET = {'f_popularity', 'f_pop_last', 'f_pop_avg', 'f_beat_market_rate'}
EXCL = {'race_id', 'date', 'horse_name', 'horse_num', 'place', 'is_fukusho', 'date_obj'}
# 探索期(2025) / 確認期(2026)  ← mf1/prep.py と同一
TUNE = [('T1', '2025-02-27', '2025-03-01', '2025-04-30'),
        ('T2', '2025-05-30', '2025-06-01', '2025-07-31'),
        ('T3', '2025-09-29', '2025-10-01', '2025-11-30')]
CONF = [('W1', '2026-06-30', '2026-07-04', '2026-09-06'),
        ('W2', '2026-04-30', '2026-05-02', '2026-06-28'),
        ('W3', '2026-02-28', '2026-03-01', '2026-04-26')]


def main():
    df = pd.read_csv(f'{BASE}/data/horse_features.csv')
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj']).reset_index(drop=True)
    feat = [c for c in df.columns
            if c not in EXCL | MARKET and df[c].dtype in ('float64','int64','float32','int32')]

    X = df[feat].fillna(5.0).to_numpy(dtype=np.float32)
    y3 = df['is_fukusho'].to_numpy(dtype=np.int8)
    y1 = (df['place'] == 1).to_numpy(dtype=np.int8)
    pop = df['f_popularity'].to_numpy(dtype=np.float32)
    d = df['date_obj'].to_numpy()

    # ── レース条件（history.db から。特徴量ではなく分析軸としてのみ使う）──
    con = sqlite3.connect(f'{BASE}/data/history.db')
    rh = pd.read_sql('SELECT race_id, racecourse, distance, surface, track_condition,'
                     ' race_class, race_name, weather FROM race_history', con)
    con.close()
    meta = df[['race_id', 'horse_num', 'date_obj']].merge(rh, on='race_id', how='left')
    meta['field'] = meta.groupby('race_id')['horse_num'].transform('count')
    assert len(meta) == len(df)
    print(f'{len(df):,}行 / 特徴量 {len(feat)}列 / '
          f'条件欠損 racecourse {meta.racecourse.isna().mean()*100:.2f}% '
          f'distance {meta.distance.isna().mean()*100:.2f}% '
          f'surface {meta.surface.isna().mean()*100:.2f}% '
          f'track_condition {meta.track_condition.isna().mean()*100:.2f}% '
          f'class {meta.race_class.isna().mean()*100:.2f}%')

    out = {'feat': feat, 'windows': {}, 'meta': meta}
    for name, te, v0, v1 in TUNE + CONF:
        te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
        inner_from = te - pd.Timedelta(days=INNER_DAYS)
        m_fit = d <= np.datetime64(inner_from)
        m_in = (d > np.datetime64(inner_from)) & (d <= np.datetime64(te))
        m_va = (d >= np.datetime64(v0)) & (d <= np.datetime64(v1))
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
        out['windows'][name] = dict(
            Xf=X[m_fit], y3f=y3[m_fit], Xi=X[m_in], y3i=y3[m_in],
            Xv=X[m_va], y3v=y3[m_va], y1v=y1[m_va], popv=pop[m_va],
            popi=pop[m_in], y1i=y1[m_in],
            va_idx=np.where(m_va)[0],
            span=(str(te.date()), str(v0.date()), str(v1.date())))
        print(f'{name}: 学習 {m_fit.sum():,} / 内側HO {m_in.sum():,} / 評価 {m_va.sum():,}')
    with open(f'{BASE}/p4_data.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('-> p4_data.pkl')


if __name__ == '__main__':
    main()

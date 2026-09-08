"""M-1 前処理: 窓ごとに (学習 / 内側ホールドアウト / 評価) の行列を作って1本のpklに固める。

🔴 前ハーネス(pl/stage2.py)の欠陥を直した点:
    early stopping の eval_set に外側の検証窓を渡していた。
    ここでは学習期間の末尾 INNER_DAYS 日を内側ホールドアウトに切り出し、
    early stopping も Isotonic 較正もそれだけで行う。外側の窓は評価にしか使わない。
"""
import pickle, sys
import numpy as np, pandas as pd

CSV   = sys.argv[1] if len(sys.argv) > 1 else 'pl/work/data/hf.csv'
OUT   = sys.argv[2] if len(sys.argv) > 2 else 'mf1/data.pkl'
INNER_DAYS = 42

MARKET = {'f_popularity', 'f_pop_last', 'f_pop_avg', 'f_beat_market_rate'}
EXCL   = {'race_id', 'date', 'horse_name', 'horse_num', 'place', 'is_fukusho', 'date_obj'}

# (名前, train_end, v0, v1)
TUNE = [('T1', '2025-02-27', '2025-03-01', '2025-04-30'),
        ('T2', '2025-05-30', '2025-06-01', '2025-07-31'),
        ('T3', '2025-09-29', '2025-10-01', '2025-11-30')]
CONF = [('W1', '2026-06-30', '2026-07-04', '2026-09-06'),
        ('W2', '2026-04-30', '2026-05-02', '2026-06-28'),
        ('W3', '2026-02-28', '2026-03-01', '2026-04-26')]


def main():
    df = pd.read_csv(CSV)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj']).reset_index(drop=True)

    feat = [c for c in df.columns
            if c not in EXCL | MARKET and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    print(f'{len(df):,}行  特徴量 {len(feat)}列  '
          f'{df["date_obj"].min().date()}〜{df["date_obj"].max().date()}')

    X = df[feat].fillna(5.0).to_numpy(dtype=np.float32)
    y3 = df['is_fukusho'].to_numpy(dtype=np.int8)
    y1 = (df['place'] == 1).to_numpy(dtype=np.int8)
    pop = df['f_popularity'].to_numpy(dtype=np.float32)
    d = df['date_obj'].to_numpy()

    out = {'feat': feat, 'windows': {}}
    for name, te, v0, v1 in TUNE + CONF:
        te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
        inner_from = te - pd.Timedelta(days=INNER_DAYS)
        m_fit = d <= np.datetime64(inner_from)
        m_in = (d > np.datetime64(inner_from)) & (d <= np.datetime64(te))
        m_va = (d >= np.datetime64(v0)) & (d <= np.datetime64(v1))
        assert not (m_fit & m_va).any() and not (m_in & m_va).any()
        out['windows'][name] = dict(
            Xf=X[m_fit], y3f=y3[m_fit],
            Xi=X[m_in], y3i=y3[m_in],
            Xv=X[m_va], y3v=y3[m_va], y1v=y1[m_va], popv=pop[m_va],
            span=(str(te.date()), str(v0.date()), str(v1.date())),
            inner_from=str(inner_from.date()))
        print(f'{name}: 学習 {m_fit.sum():,} / 内側HO {m_in.sum():,} '
              f'({inner_from.date()}〜{te.date()}) / 評価 {m_va.sum():,}  '
              f'複勝率 学習{y3[m_fit].mean():.3f} 評価{y3[m_va].mean():.3f}')

    with open(OUT, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()

"""較正検証: 4つの較正腕 × 2つのモデル腕 を、同じOOSデータで比べる。

腕は calib/CRITERIA.md で結果を見る前に確定済み:
  C0 未補正 / C1 Platt / C2 Isotonic(本番) / C3 Isotonic+レース内正規化(合計3.0)
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
pd.set_option('display.width', 250)

az_src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(az_src, 'analyze.py', 'exec'), az.__dict__)

D = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))
P = pickle.load(open(f'{BASE}/calib/pred.pkl', 'rb'))
META = D['meta']
TUNE, CONF = ['T1', 'T2', 'T3'], ['W1', 'W2', 'W3']
EPS = 1e-6
BANDS = [(0, .05), (.05, .10), (.10, .20), (.20, .30), (.30, .40),
         (.40, .50), (.50, .60), (.60, .70), (.70, 1.01)]

# ── 実配当（複勝）。3着内でも欠損する行があるので join して欠損は評価不能にする ──
con = sqlite3.connect(f'{BASE}/data/history.db')
PAY = pd.read_sql('SELECT race_id, horse_num, fukusho_payout, place FROM horse_history', con)
con.close()
PAY = PAY.set_index(['race_id', 'horse_num'])


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibrate(pi, yi, pv, rid_v):
    """4つの較正腕を作る。較正器は内側HOだけで学習し、評価行には一切触れない。"""
    out = {}
    out['C0 未補正'] = np.clip(pv, EPS, 1 - EPS)
    lr = LogisticRegression(C=1e6, solver='lbfgs').fit(logit(pi).reshape(-1, 1), yi)
    out['C1 Platt'] = np.clip(lr.predict_proba(logit(pv).reshape(-1, 1))[:, 1], EPS, 1 - EPS)
    iso = IsotonicRegression(out_of_bounds='clip').fit(pi, yi)
    c2 = np.clip(iso.predict(pv), EPS, 1 - EPS)
    out['C2 Isotonic'] = c2
    # C3: 「1レース3頭が3着内」は厳密に既知の制約。C2 をレース内で合計3.0にする
    s = pd.Series(c2).groupby(pd.Series(rid_v)).transform('sum').to_numpy()
    out['C3 Iso+正規化'] = np.clip(c2 * 3.0 / np.maximum(s, EPS), EPS, 1 - EPS)
    return out


def ece(p, y, nb=10):
    """10等頻度ビンの Expected Calibration Error。"""
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False, duplicates='drop')
    d = pd.DataFrame({'p': p, 'y': y, 'q': q})
    g = d.groupby('q').agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())


def metrics(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return dict(brier=float(np.mean((p - y) ** 2)),
                logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                ece=ece(p, y), auc=float(roc_auc_score(y, p)),
                mean=float(p.mean()))


def build(period):
    """その期間の全窓を縦に積んで、較正腕ごとの確率と付帯情報を返す。"""
    wins = TUNE if period == '探索' else CONF
    acc = {}
    ys, rids, hns, pops, fields = [], [], [], [], []
    for w in wins:
        r = P[w]; win = D['windows'][w]
        vi = r['va_idx']
        rid_v = META.iloc[vi]['race_id'].to_numpy()
        rid_i = META.iloc[r['in_idx']]['race_id'].to_numpy()
        for arm in ('A', 'B'):
            cal = calibrate(r[arm]['pi'], r['y3i'], r[arm]['pv'], rid_v)
            for k, v in cal.items():
                acc.setdefault((arm, k), []).append(v)
        # 市場（参考）: 人気×頭数の実測表を内側HOで作り、評価行に当てる
        base, gp = az.market_table(win['popi'].astype(float),
                                   META.iloc[r['in_idx']]['field'].to_numpy(float), r['y3i'])
        mkt = az.apply_market(base, gp, win['popv'].astype(float),
                              META.iloc[vi]['field'].to_numpy(float))
        acc.setdefault(('M', '市場(参考)'), []).append(mkt)
        ys.append(r['y3v']); rids.append(rid_v)
        hns.append(META.iloc[vi]['horse_num'].to_numpy())
        pops.append(win['popv'].astype(float))
        fields.append(META.iloc[vi]['field'].to_numpy(float))
    return ({k: np.concatenate(v) for k, v in acc.items()},
            np.concatenate(ys), np.concatenate(rids), np.concatenate(hns),
            np.concatenate(pops), np.concatenate(fields))


ARMS = ['C0 未補正', 'C1 Platt', 'C2 Isotonic', 'C3 Iso+正規化']
MODEL = {'A': 'A 市場フリー131（画面のAI勝率）', 'B': 'B 本番同型134+residual（cal_prob/EV）'}


def main():
    for period in ('確認', '探索'):
        cal, y, rid, hn, pop, field = build(period)
        ok = np.isfinite(pop) & (pop > 0) & (pop < 99)
        print(f'\n{"="*118}')
        print(f'■ {period}期  {len(y):,}頭 / {len(set(rid)):,}レース   実測3着内 {y.mean()*100:.2f}%')
        print(f'{"="*118}')
        for m in ('A', 'B'):
            print(f'\n  【{MODEL[m]}】')
            print(f'    {"腕":<16}{"予測平均":>9}{"Brier":>10}{"LogLoss":>10}{"ECE":>9}{"AUC":>9}')
            for a in ARMS:
                r = metrics(cal[(m, a)], y)
                print(f'    {a:<16}{r["mean"]*100:>8.2f}%{r["brier"]:>10.4f}'
                      f'{r["logloss"]:>10.4f}{r["ece"]:>9.4f}{r["auc"]:>9.4f}')
            r = metrics(cal[('M', '市場(参考)')][ok], y[ok])
            print(f'    {"（参考）市場":<15}{r["mean"]*100:>8.2f}%{r["brier"]:>10.4f}'
                  f'{r["logloss"]:>10.4f}{r["ece"]:>9.4f}{r["auc"]:>9.4f}   ← 人気×頭数の実測表')
        if period != '確認':
            continue

        # ── 確率帯別 予測 vs 実測 ────────────────────────────────────────
        for m in ('A', 'B'):
            print(f'\n{"="*118}\n■ 確率帯別 予測 vs 実測 — {MODEL[m]}\n{"="*118}')
            print(f'  {"帯":<10}' + ''.join(f'{a:>26}' for a in ARMS))
            print(f'  {"":<10}' + ''.join(f'{"N":>7}{"予測":>7}{"実測":>7}{"差":>5}' for _ in ARMS))
            for lo, hi in BANDS:
                line = f'  {int(lo*100)}-{min(int(hi*100),100)}%'.ljust(12)
                for a in ARMS:
                    p = cal[(m, a)]
                    s = (p >= lo) & (p < hi)
                    if s.sum() < 30:
                        line += f'{int(s.sum()):>7}{"-":>7}{"-":>7}{"-":>5}'
                    else:
                        line += (f'{int(s.sum()):>7}{p[s].mean()*100:>6.1f}%'
                                 f'{y[s].mean()*100:>6.1f}%{(y[s].mean()-p[s].mean())*100:>+5.1f}')
                print(line)

        # ── レース内合計 ────────────────────────────────────────────────
        print(f'\n{"="*118}\n■ 1レース内の予測確率の合計（理論値は厳密に 3.0）\n{"="*118}')
        print(f'  {"モデル":<10}{"腕":<16}{"中央値":>9}{"平均":>9}{"5%点":>9}{"95%点":>9}{"3.0±0.3の割合":>15}')
        for m in ('A', 'B'):
            for a in ARMS:
                s = pd.Series(cal[(m, a)]).groupby(pd.Series(rid)).sum()
                print(f'  {m:<10}{a:<16}{s.median():>9.3f}{s.mean():>9.3f}'
                      f'{s.quantile(.05):>9.3f}{s.quantile(.95):>9.3f}'
                      f'{((s - 3).abs() <= .3).mean()*100:>14.1f}%')

        # ── EV参考値 ────────────────────────────────────────────────────
        # 複勝オッズは実データが無いので人気×頭数の実測表から作った**合成オッズ**（代理）
        mkt = cal[('M', '市場(参考)')]
        odds = 0.8 / np.clip(mkt, 0.02, 0.999)
        key = pd.MultiIndex.from_arrays([rid, hn])
        pay = PAY.reindex(key)['fukusho_payout'].to_numpy(float)
        # 🔴 3着内なのに配当が取れていない行は「0円」ではなく評価不能として外す
        evaluable = ok & ~((y == 1) & (~np.isfinite(pay) | (pay <= 0)))
        print(f'\n{"="*118}')
        print('■ EV参考値（複勝・合成オッズ×実配当）— 🔴 参考のみ。閾値を選ばない・採用しない')
        print(f'{"="*118}')
        print(f'  オッズは人気×頭数の実測表からの**代理**（history.db の win_odds は全年0%充足）')
        print(f'  評価可能 {evaluable.sum():,} / {len(y):,}頭'
              f'（3着内なのに配当欠損 {int((ok & ~evaluable).sum()):,}頭を除外）')
        for m in ('A', 'B'):
            print(f'\n  【{MODEL[m]}】  全部買う: '
                  f'N={evaluable.sum():,} 的中{y[evaluable].mean()*100:.1f}% '
                  f'回収{pay[evaluable & (y==1)].sum()/evaluable.sum():.1f}%')
            print(f'    {"腕":<16}{"閾値":>6}{"買い目":>9}{"的中率":>9}{"回収率":>9}')
            for a in ARMS:
                p = cal[(m, a)]
                ev = p * odds
                for th in (1.0, 1.2, 1.3, 1.5):
                    s = evaluable & (ev >= th)
                    if s.sum() < 50:
                        print(f'    {a:<16}{th:>6.1f}{int(s.sum()):>9,}{"-":>9}{"-":>9}')
                        continue
                    roi = pay[s & (y == 1)].sum() / s.sum()
                    print(f'    {a:<16}{th:>6.1f}{int(s.sum()):>9,}'
                          f'{y[s].mean()*100:>8.1f}%{roi:>8.1f}%')


if __name__ == '__main__':
    main()

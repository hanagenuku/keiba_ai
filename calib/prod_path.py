"""🔴 事前登録の4腕に足りなかった1腕: **本番のEVが実際に使っている確率**。

CRITERIA.md を書いた時点で見落としていた。実装を読み直すと:

    cal_prob = Isotonic(sigmoid(raw_margin))            ← C2。表示の一部と穴マーク
    win_prob = softmax(raw_prob*10, T=3.5)              ← 画面の勝率
    fuku_prob(top3_prob) = Harville(win_prob)           ← 画面の複勝
    h.ev     = win_prob × オッズ                          ← **EVはこちら**

つまり EV が使うのは Isotonic を通した確率ではなく、**softmax T=3.5 で
意図的に平坦化した方**。較正を測るなら、この経路も測らないと問いに答えられない。

⚠ 後から足した腕であることを明記する（結果を見て有利な腕を追加したのではなく、
   設計の抜けを埋めたもの）。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai')
from src.models.predict import softmax_probs
from src.features.engine import calc_harville_probs

D = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))
P = pickle.load(open(f'{BASE}/calib/pred.pkl', 'rb'))
META, EPS, CONF = D['meta'], 1e-6, ['W1', 'W2', 'W3']
BANDS = [(0,.05),(.05,.10),(.10,.20),(.20,.30),(.30,.40),(.40,.50),(.50,.60),(.60,.70),(.70,1.01)]


def ece(p, y, nb=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())


def mets(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return (p.mean(), float(np.mean((p - y) ** 2)),
            float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p))), ece(p, y), roc_auc_score(y, p))


rows_c2, rows_c4, ys, rids = [], [], [], []
for w in CONF:
    r = P[w]; vi = r['va_idx']
    rid = META.iloc[vi]['race_id'].to_numpy()
    iso = IsotonicRegression(out_of_bounds='clip').fit(r['B']['pi'], r['y3i'])
    rows_c2.append(np.clip(iso.predict(r['B']['pv']), EPS, 1-EPS))
    # 本番経路: total = raw_prob*10 → softmax(T=3.5) → Harville → top3_prob
    top3 = np.empty(len(vi))
    s = pd.Series(range(len(vi))).groupby(pd.Series(rid))
    for _, ix in s.indices.items():
        wp = softmax_probs([r['B']['pv'][i] * 10 for i in ix], temperature=3.5)
        top3[list(ix)] = [t3 for _, t3 in calc_harville_probs(wp)]
    rows_c4.append(top3)
    ys.append(r['y3v']); rids.append(rid)

c2, c4 = np.concatenate(rows_c2), np.concatenate(rows_c4)
y, rid = np.concatenate(ys), np.concatenate(rids)
c0 = np.concatenate([P[w]['B']['pv'] for w in CONF])

print(f'確認期 {len(y):,}頭 / {len(set(rid)):,}レース  実測3着内 {y.mean()*100:.2f}%\n')
print(f'  {"腕":<40}{"予測平均":>9}{"Brier":>10}{"LogLoss":>10}{"ECE":>9}{"AUC":>9}')
for nm, p in [('C0 未補正 sigmoid(raw_margin)', c0),
              ('C2 Isotonic → cal_prob（穴マーク等）', c2),
              ('★C4 Harville(softmax T=3.5) → EVが使う値', c4)]:
    m = mets(p, y)
    print(f'  {nm:<40}{m[0]*100:>8.2f}%{m[1]:>10.4f}{m[2]:>10.4f}{m[3]:>9.4f}{m[4]:>9.4f}')

print(f'\n  確率帯別  予測 vs 実測')
print(f'  {"帯":<10}{"C2 Isotonic":>28}{"C4 本番のEV経路":>28}')
print(f'  {"":<10}' + ''.join(f'{"N":>8}{"予測":>7}{"実測":>7}{"差":>6}' for _ in range(2)))
for lo, hi in BANDS:
    line = f'  {int(lo*100)}-{min(int(hi*100),100)}%'.ljust(12)
    for p in (c2, c4):
        m = (p >= lo) & (p < hi)
        line += (f'{int(m.sum()):>8}{"-":>7}{"-":>7}{"-":>6}' if m.sum() < 30 else
                 f'{int(m.sum()):>8}{p[m].mean()*100:>6.1f}%{y[m].mean()*100:>6.1f}%'
                 f'{(y[m].mean()-p[m].mean())*100:>+6.1f}')
    print(line)

print(f'\n  レース内合計')
for nm, p in [('C2 Isotonic', c2), ('C4 本番のEV経路', c4)]:
    s = pd.Series(p).groupby(pd.Series(rid)).sum()
    print(f'    {nm:<18}中央値 {s.median():.3f}  5%点 {s.quantile(.05):.3f}  '
          f'95%点 {s.quantile(.95):.3f}  3.0±0.3 {((s-3).abs()<=.3).mean()*100:.1f}%')

print(f'\n  🔑 平坦化の度合い（レース内の最大−最小）')
for nm, p in [('C2 Isotonic', c2), ('C4 本番のEV経路', c4)]:
    g = pd.Series(p).groupby(pd.Series(rid))
    print(f'    {nm:<18}幅の中央値 {(g.max()-g.min()).median():.3f}  '
          f'レース内最大の中央値 {g.max().median():.3f}  最小 {g.min().median():.3f}')

print(f'\n  検算: Platt/Isotonic は単調変換なので順位を保つはず')
c1_rho = spearmanr(c0, c2).statistic
print(f'    corr(C0, C2) スピアマン = {c1_rho:+.6f}  '
      f'（1.0 からのズレは Isotonic の段差＝同値化による）')

# ── EV参考値: C4(本番のEV経路) を C2 と並べる ─────────────────────────────
print(f'\n  EV参考値（複勝・合成オッズ×実配当）🔴 参考のみ・閾値を選ばない')
az_src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(az_src, 'analyze.py', 'exec'), az.__dict__)
con = sqlite3.connect(f'{BASE}/data/history.db')
PAY = pd.read_sql('SELECT race_id, horse_num, fukusho_payout FROM horse_history', con).set_index(
    ['race_id', 'horse_num'])
con.close()
mk, hn, pp, fl = [], [], [], []
for w in CONF:
    r = P[w]; win = D['windows'][w]; vi = r['va_idx']
    base, gp = az.market_table(win['popi'].astype(float),
                               META.iloc[r['in_idx']]['field'].to_numpy(float), r['y3i'])
    mk.append(az.apply_market(base, gp, win['popv'].astype(float),
                              META.iloc[vi]['field'].to_numpy(float)))
    hn.append(META.iloc[vi]['horse_num'].to_numpy()); pp.append(win['popv'].astype(float))
mkt, hn, pp = np.concatenate(mk), np.concatenate(hn), np.concatenate(pp)
odds = 0.8 / np.clip(mkt, 0.02, 0.999)
pay = PAY.reindex(pd.MultiIndex.from_arrays([rid, hn]))['fukusho_payout'].to_numpy(float)
ok = np.isfinite(pp) & (pp > 0) & (pp < 99)
ev_ok = ok & ~((y == 1) & (~np.isfinite(pay) | (pay <= 0)))
print(f'    全部買う: N={ev_ok.sum():,}  的中{y[ev_ok].mean()*100:.1f}%  '
      f'回収{pay[ev_ok & (y==1)].sum()/ev_ok.sum():.1f}%')
print(f'    {"腕":<24}{"閾値":>6}{"買い目":>9}{"的中率":>9}{"回収率":>9}')
for nm, p in [('C2 Isotonic(較正済み)', c2), ('★C4 本番のEV経路', c4)]:
    for th in (1.0, 1.2, 1.3, 1.5):
        s = ev_ok & (p * odds >= th)
        if s.sum() < 50:
            print(f'    {nm:<24}{th:>6.1f}{int(s.sum()):>9,}{"-":>9}{"-":>9}'); continue
        print(f'    {nm:<24}{th:>6.1f}{int(s.sum()):>9,}{y[s].mean()*100:>8.1f}%'
              f'{pay[s & (y==1)].sum()/s.sum():>8.1f}%')

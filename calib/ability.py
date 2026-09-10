"""ability経路（当日の現在オッズを直接使わないAI確率）の較正をOOSで測る。

腕は calib/CRITERIA_ability.md で結果を見る前に確定:
  E0 未補正 / E1 Isotonic / E2 E1+レース内正規化
  参考S 現行画面(softmax T=3.5) / 参考C cal_prob / 参考M 市場

🔴 ability_margin には過去人気(f_pop_last等)が含まれる。「市場フリー」ではなく
   **当日の現在オッズを直接使わない** 確率、という位置づけ。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
BASE = os.path.abspath('.'); SCR = os.path.dirname(BASE)
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)
from src.features.engine import _flat_base_margin, calc_harville_probs
from src.models.predict import softmax_probs
az_src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(az_src, 'analyze.py', 'exec'), az.__dict__)

D = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))
P = pickle.load(open(f'{BASE}/calib/pred.pkl', 'rb'))
META, EPS, CONF = D['meta'], 1e-6, ['W1', 'W2', 'W3']
FIELD, RID = META['field'].to_numpy(float), META['race_id'].to_numpy()
HN = META['horse_num'].to_numpy()
df = pd.read_csv(f'{BASE}/data/horse_features.csv')
POP = df['f_popularity'].to_numpy(float)
POP = np.where(np.isfinite(POP), POP, FIELD / 2)
DR = load_popularity_drift(BASE)
BANDS = [(0,.05),(.05,.10),(.10,.20),(.20,.30),(.30,.40),(.40,.50),(.50,.60),(.60,.70),(.70,1.01)]
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))


def ece(p, y, nb=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        p=('p','mean'), y=('y','mean'), n=('y','size'))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())


def mets(p, y):
    p = np.clip(p, EPS, 1-EPS)
    return (p.mean(), float(np.mean((p-y)**2)),
            float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), ece(p, y), roc_auc_score(y, p))


def ability_probs(w, target):
    """その窓の ability経路の確率を腕ごとに返す。target は 'y3'(3着内) か 'y1'(1着)。"""
    r, win = P[w], D['windows'][w]
    ii, vi = r['in_idx'], r['va_idx']
    out = {}
    e0 = {}
    for tag, idx, seed in (('i', ii, 12), ('v', vi, 12)):
        bm = _popularity_to_base_margin(pd.Series(_apply_popularity_drift(
            pd.Series(POP[idx]), RID[idx], DR, seed=seed)), pd.Series(FIELD[idx]))
        ab = lg(r['B']['p' + tag]) - bm
        flat = np.array([_flat_base_margin(int(n)) for n in FIELD[idx]])
        e0[tag] = 1.0 / (1.0 + np.exp(-(ab + flat)))
    yi = win['y3i'] if target == 'y3' else win['y1i']
    tot = 3.0 if target == 'y3' else 1.0
    out['E0 未補正'] = np.clip(e0['v'], EPS, 1-EPS)
    iso = IsotonicRegression(out_of_bounds='clip').fit(e0['i'], yi)
    e1 = np.clip(iso.predict(e0['v']), EPS, 1-EPS)
    out['E1 Isotonic'] = e1
    s = pd.Series(e1).groupby(pd.Series(RID[vi])).transform('sum').to_numpy()
    out['E2 E1+正規化'] = np.clip(e1 * tot / np.maximum(s, EPS), EPS, 1-EPS)
    # 参考S: 現行画面の経路（softmax T=3.5 → 複勝はHarville）
    sc = np.empty(len(vi))
    for _, ix in pd.Series(range(len(vi))).groupby(pd.Series(RID[vi])).indices.items():
        ix = list(ix)
        wp = softmax_probs([e0['v'][i] * 10 for i in ix], temperature=3.5)
        sc[ix] = [t3 for _, t3 in calc_harville_probs(wp)] if target == 'y3' else wp
    out['参考S 現行画面'] = np.clip(sc, EPS, 1-EPS)
    # 参考C: cal_prob（市場アンカー込み・3着内のみ）
    if target == 'y3':
        isoc = IsotonicRegression(out_of_bounds='clip').fit(r['B']['pi'], win['y3i'])
        out['参考C cal_prob'] = np.clip(isoc.predict(r['B']['pv']), EPS, 1-EPS)
    return out


def gather(wins, target):
    acc, ys, rid, hn, pop = {}, [], [], [], []
    for w in wins:
        r, win = P[w], D['windows'][w]; vi = r['va_idx']
        for k, v in ability_probs(w, target).items():
            acc.setdefault(k, []).append(v)
        base, gp = az.market_table(win['popi'].astype(float),
                                   FIELD[r['in_idx']], win['y3i'] if target=='y3' else win['y1i'])
        acc.setdefault('参考M 市場', []).append(
            az.apply_market(base, gp, win['popv'].astype(float), FIELD[vi]))
        ys.append(win['y3v'] if target=='y3' else win['y1v'])
        rid.append(RID[vi]); hn.append(HN[vi]); pop.append(win['popv'].astype(float))
    return ({k: np.concatenate(v) for k, v in acc.items()},
            *[np.concatenate(x) for x in (ys, rid, hn, pop)])


ARMS3 = ['E0 未補正','E1 Isotonic','E2 E1+正規化','参考S 現行画面','参考C cal_prob','参考M 市場']
ARMS1 = ['E0 未補正','E1 Isotonic','E2 E1+正規化','参考S 現行画面','参考M 市場']


def section3():
    cal, y, rid, hn, pop = gather(CONF, 'y3')
    print(f'\n{"="*112}\n■ ①〜⑥ 3着内 — 確認期 {len(y):,}頭 / {len(set(rid)):,}レース  '
          f'実測 {y.mean()*100:.2f}%\n{"="*112}')
    ok = np.isfinite(pop) & (pop > 0) & (pop < 99)
    print(f'  {"腕":<18}{"予測平均":>9}{"①Brier":>10}{"②LogLoss":>11}{"③ECE":>9}{"④AUC":>9}')
    for a in ARMS3:
        m = np.isfinite(cal[a]) & (ok if a == '参考M 市場' else True)
        r = mets(cal[a][m], y[m])
        star = ' ★' if a == 'E1 Isotonic' else ''
        print(f'  {a:<18}{r[0]*100:>8.2f}%{r[1]:>10.4f}{r[2]:>11.4f}{r[3]:>9.4f}{r[4]:>9.4f}{star}')

    print(f'\n■ ⑤ 確率帯別 予測 vs 実測（3着内）')
    show = ['E0 未補正','E1 Isotonic','参考S 現行画面','参考C cal_prob']
    print(f'  {"帯":<10}' + ''.join(f'{a:>25}' for a in show))
    print(f'  {"":<10}' + ''.join(f'{"N":>7}{"予測":>6}{"実測":>6}{"差":>6}' for _ in show))
    for lo, hi in BANDS:
        line = f'  {int(lo*100)}-{min(int(hi*100),100)}%'.ljust(12)
        for a in show:
            p = cal[a]; m = (p >= lo) & (p < hi)
            line += (f'{int(m.sum()):>7}{"-":>6}{"-":>6}{"-":>6}' if m.sum() < 30 else
                     f'{int(m.sum()):>7}{p[m].mean()*100:>5.1f}%{y[m].mean()*100:>5.1f}%'
                     f'{(y[m].mean()-p[m].mean())*100:>+6.1f}')
        print(line)

    print(f'\n■ ⑥ レース内合計（理論値 3.0）')
    print(f'  {"腕":<18}{"中央値":>9}{"5%点":>9}{"95%点":>9}{"3.0±0.3":>10}')
    for a in ARMS3:
        s = pd.Series(cal[a]).groupby(pd.Series(rid)).sum()
        print(f'  {a:<18}{s.median():>9.3f}{s.quantile(.05):>9.3f}{s.quantile(.95):>9.3f}'
              f'{((s-3).abs()<=.3).mean()*100:>9.1f}%')
    print(f'\n  市場からの独立性（logit の R²・1に近いほど市場のコピー）')
    om = np.isfinite(cal['参考M 市場']) & (cal['参考M 市場'] > .005) & (cal['参考M 市場'] < .999)
    for a in ['E1 Isotonic', '参考C cal_prob']:
        print(f'    {a:<18}R² = {np.corrcoef(lg(cal["参考M 市場"][om]), lg(cal[a][om]))[0,1]**2:.4f}')


def section_real():
    """⑦⑧⑨ 単勝・実オッズ（W1のみ）。買う判定=朝オッズ / 払戻=確定配当。"""
    cal, y, rid, hn, pop = gather(['W1'], 'y1')
    c = sqlite3.connect(f'{SCR}/keiba.db')
    rp = pd.read_sql('SELECT race_id, horse_num, tansho_odds FROM race_predictions '
                     'WHERE tansho_odds >= 1.0', c); c.close()
    c2 = sqlite3.connect(f'{BASE}/data/history.db')
    pay = pd.read_sql('SELECT race_id, horse_num, tansho_payout FROM horse_history', c2); c2.close()
    key = pd.MultiIndex.from_arrays([rid, hn])
    odds = rp.set_index(['race_id','horse_num'])['tansho_odds'].reindex(key).to_numpy(float)
    pyt = pay.set_index(['race_id','horse_num'])['tansho_payout'].reindex(key).to_numpy(float)
    m = np.isfinite(odds) & (odds >= 1.0)
    # 勝ったのに確定配当が無い行は評価不能（0円として混ぜない）
    m &= ~((y == 1) & (~np.isfinite(pyt) | (pyt <= 0)))
    print(f'\n{"="*112}\n■ ⑦⑧⑨ 単勝・実オッズ（W1・2026-07-04〜09-06）\n{"="*112}')
    print(f'  評価可能 {m.sum():,}頭 / {len(y):,}頭   実勝率 {y[m].mean()*100:.2f}%')
    print(f'  🔴 買う判定は**朝オッズ**、払戻は**確定配当**（朝で払うと人気薄が有利に出る）')
    print(f'  検算: 確定配当/朝オッズ の中央値（勝ち馬）= '
          f'{np.median(pyt[m&(y==1)]/100/odds[m&(y==1)]):.3f}')
    base_roi = pyt[m & (y == 1)].sum() / m.sum()
    print(f'  全部買う: N={m.sum():,}  的中{y[m].mean()*100:.2f}%  回収{base_roi:.1f}%\n')

    print(f'  ⑦ 必要オッズ = 1.2 / p  と  ⑧ 実オッズ の比較')
    print(f'  {"腕":<18}{"必要オッズ中央値":>16}{"実オッズ中央値":>15}{"超える馬":>11}{"割合":>8}')
    for a in ARMS1:
        p = cal[a][m]; need = 1.2 / p
        cl = odds[m] >= need
        print(f'  {a:<18}{np.median(need):>15.1f}倍{np.median(odds[m]):>14.1f}倍'
              f'{int(cl.sum()):>11,}{cl.mean()*100:>7.1f}%')

    print(f'\n  ⑨ 固定閾値の実配当ROI（🔴 閾値は事前に 1.0/1.2/1.5 に固定。最適化しない）')
    print(f'  {"腕":<18}{"閾値":>6}{"買い目":>9}{"的中率":>9}{"回収率":>9}{"オッズ中央値":>13}')
    for a in ARMS1:
        p = cal[a][m]; ev = p * odds[m]
        for th in (1.0, 1.2, 1.5):
            s = ev >= th
            if s.sum() < 50:
                print(f'  {a:<18}{th:>6.1f}{int(s.sum()):>9,}{"-":>9}{"-":>9}{"-":>13}'); continue
            roi = pyt[m][s & (y[m] == 1)].sum() / s.sum()
            print(f'  {a:<18}{th:>6.1f}{int(s.sum()):>9,}{y[m][s].mean()*100:>8.2f}%'
                  f'{roi:>8.1f}%{np.median(odds[m][s]):>12.1f}倍')

    print(f'\n  参考: 単勝の較正そのもの（1着に対して）')
    print(f'  {"腕":<18}{"予測平均":>9}{"Brier":>10}{"LogLoss":>11}{"ECE":>9}{"AUC":>9}')
    for a in ARMS1:
        r = mets(cal[a][m], y[m])
        print(f'  {a:<18}{r[0]*100:>8.2f}%{r[1]:>10.4f}{r[2]:>11.4f}{r[3]:>9.4f}{r[4]:>9.4f}')


if __name__ == '__main__':
    section3()
    section_real()

"""EV用確率を頭数帯ごとに較正する案を、既存の3分割OOSで測る。

現状(P)は全頭数をプールして Isotonic 1本。絶対勝率なのにレース内合計が
7-9頭で約160% / 14-18頭で約80% と頭数で系統的にずれる。

🔴 判定基準（結果を見る前に固定）:
  ① ECE が P 以下（悪化させない）
  ② すべての頭数帯でレース内合計の中央値が [0.90, 1.10]（単勝）
  ③ 同値になる馬の数が P より減る
  3つすべて満たしたときだけ採用する。
"""
import os, sys, pickle
import numpy as np, pandas as pd
from collections import Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.features.engine import _flat_base_margin
Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb')); EPS = 1e-6
BANDS = [(2, 9), (10, 13), (14, 24)]          # 頭数帯（事前に3つで固定）

def e0_of(b):
    return 1/(1+np.exp(-((b['raw']-b['bm']) +
              np.array([_flat_base_margin(int(n)) for n in b['field']]))))

def ece(p, y, nb=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(
        p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((g.n*(g.p-g.y).abs()).sum()/g.n.sum())

CT, OOS = Q['CT'], Q['OOS']
e_ct, e_oos = e0_of(CT), e0_of(OOS)
out = {}
for tgt in ('y1', 'y3'):
    yct, yo = CT[tgt], OOS[tgt]
    P = np.clip(IsotonicRegression(out_of_bounds='clip').fit(e_ct, yct).predict(e_oos), EPS, 1-EPS)
    B = np.empty(len(e_oos))
    for lo, hi in BANDS:
        mc = (CT['field'] >= lo) & (CT['field'] <= hi)
        mo = (OOS['field'] >= lo) & (OOS['field'] <= hi)
        if mc.sum() < 300 or mo.sum() == 0:
            B[mo] = P[mo]; continue
        B[mo] = IsotonicRegression(out_of_bounds='clip').fit(e_ct[mc], yct[mc]).predict(e_oos[mo])
    B = np.clip(B, EPS, 1-EPS)
    out[tgt] = (P, B, yo)

print(f'{"="*92}\n■ 母集団: 最終OOS 2026-01-04〜09-06  {len(e_oos):,}頭 / '
      f'{len(set(OOS["rid"])):,}レース（較正は 2025-07〜10 の 14,910頭のみで学習）\n{"="*92}')
for tgt, name, tot in (('y1', '単勝(1着)', 1.0), ('y3', '3着内', 3.0)):
    P, B, y = out[tgt]
    print(f'\n■ {name}   理論上のレース内合計 = {tot}')
    print(f'  {"腕":<20}{"予測平均":>9}{"Brier":>9}{"LogLoss":>10}{"ECE":>9}{"AUC":>8}')
    for lab, p in (('P プール（現状）', P), ('B 頭数帯別（案）', B)):
        pc = np.clip(p, EPS, 1-EPS)
        print(f'  {lab:<20}{pc.mean()*100:>8.2f}%{np.mean((pc-y)**2):>9.4f}'
              f'{-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc)):>10.4f}{ece(pc,y):>9.4f}'
              f'{roc_auc_score(y,pc):>8.4f}')
    print(f'  {"頭数帯":<12}{"R数":>6}   {"P 合計中央値":>13}{"B 合計中央値":>13}   '
          f'{"P 同値最大":>11}{"B 同値最大":>11}')
    for lo, hi in BANDS:
        mo = (OOS['field'] >= lo) & (OOS['field'] <= hi)
        if mo.sum() == 0: continue
        rid = np.asarray(OOS['rid'])[mo]
        res = []
        for p in (P[mo], B[mo]):
            s = pd.Series(p).groupby(pd.Series(rid)).sum()
            tie = pd.Series(np.round(p, 4)).groupby(pd.Series(rid)).apply(
                lambda g: Counter(g).most_common(1)[0][1])
            res.append((s.median()/tot, tie.mean()))
        print(f'  {f"{lo}-{hi}頭":<12}{len(set(rid)):>6}   {res[0][0]*100:>12.0f}%{res[1][0]*100:>12.0f}%'
              f'   {res[0][1]:>11.2f}{res[1][1]:>11.2f}')

print(f'\n{"="*92}\n■ 追加: レース内で正規化する案（BN = B + Σを理論値に合わせる）\n{"="*92}')
print('  🔴 表示としては合計が必ず合うが、較正(ECE)を犠牲にする。両方出して比べる。')
for tgt, name, tot in (('y1', '単勝(1着)', 1.0), ('y3', '3着内', 3.0)):
    P, B, y = out[tgt]
    rid = pd.Series(np.asarray(OOS['rid']))
    s = pd.Series(B).groupby(rid).transform('sum').to_numpy()
    BN = np.clip(B * tot / np.maximum(s, EPS), EPS, 1-EPS)
    print(f'\n  {name}')
    print(f'  {"腕":<22}{"予測平均":>9}{"Brier":>9}{"LogLoss":>10}{"ECE":>9}{"AUC":>8}{"合計が±10%内":>13}')
    for lab, p in (('P プール（現状）', P), ('B 頭数帯別', B), ('BN 帯別+正規化', BN)):
        pc = np.clip(p, EPS, 1-EPS)
        ss = pd.Series(pc).groupby(rid).sum()
        ok = ((ss/tot - 1).abs() <= 0.10).mean()*100
        print(f'  {lab:<22}{pc.mean()*100:>8.2f}%{np.mean((pc-y)**2):>9.4f}'
              f'{-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc)):>10.4f}{ece(pc,y):>9.4f}'
              f'{roc_auc_score(y,pc):>8.4f}{ok:>12.1f}%')

"""AI勝率の品質評価: 較正(A) と 識別(B) を分離して測る。基準は q/CRITERIA_quality.md。"""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.features.engine import _flat_base_margin
az_src = open(f'{BASE}/analyze.py').read().rsplit("\nif __name__", 1)[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(az_src, 'analyze.py', 'exec'), az.__dict__)

Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb'))
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
BANDS = [(0,.05),(.05,.10),(.10,.15),(.15,.20),(.20,.25),(.25,.30),
         (.30,.40),(.40,.50),(.50,.60),(.60,.70),(.70,1.01)]
TGT = sys.argv[1] if len(sys.argv) > 1 else 'y1'
TNAME = '1着(単勝)' if TGT == 'y1' else '3着内'

def e0_of(b):
    ab = b['raw'] - b['bm']
    flat = np.array([_flat_base_margin(int(n)) for n in b['field']])
    return 1.0 / (1.0 + np.exp(-(ab + flat)))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

def ece_q(p, y, nb=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), nb, labels=False, duplicates='drop')
    g = pd.DataFrame({'p':p,'y':y,'q':q}).groupby('q').agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
    return float((g.n*(g.p-g.y).abs()).sum()/g.n.sum())

def ece_band(p, y):
    tot = num = 0.0
    for lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0: continue
        num += m.sum()*abs(p[m].mean()-y[m].mean()); tot += m.sum()
    return num/tot

def mets(p, y):
    p = np.clip(p, EPS, 1-EPS)
    return dict(mean=p.mean(), brier=float(np.mean((p-y)**2)),
                ll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                ece=ece_q(p, y), eceb=ece_band(p, y),
                auc=roc_auc_score(y, p), pr=average_precision_score(y, p), sd=p.std())

def brier_decomp(p, y):
    """Murphy: BS = REL - RES + UNC（ユーザー指定の11帯で）"""
    o = y.mean(); N = len(y); rel = res = 0.0
    for lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0: continue
        rel += m.sum()*(p[m].mean()-y[m].mean())**2
        res += m.sum()*(y[m].mean()-o)**2
    return rel/N, res/N, o*(1-o)

# ── 腕を作る（Isotonic と 市場表は CT だけで学習する）────────────────
CT, CV, OOS = Q['CT'], Q['CV'], Q['OOS']
e0 = {k: e0_of(b) for k, b in (('CT',CT),('CV',CV),('OOS',OOS))}
yv = {k: b[TGT] for k, b in (('CT',CT),('CV',CV),('OOS',OOS))}
iso = IsotonicRegression(out_of_bounds='clip').fit(e0['CT'], yv['CT'])
base, gp = az.market_table(CT['pop'].astype(float), CT['field'], yv['CT'])   # 関門4: CTのみ
ARMS = {}
for k in ('CV','OOS'):
    ARMS[k] = {'E0 未補正': np.clip(e0[k], EPS, 1-EPS),
               'E1 Isotonic': np.clip(iso.predict(e0[k]), EPS, 1-EPS),
               '参考M 市場': az.apply_market(base, gp, Q[k]['pop'].astype(float), Q[k]['field'])}
A = ARMS['OOS']; Y = yv['OOS']

W = 108
print(f'\n{"="*W}\n■ ② 対象期間と母集団   目的変数 = {TNAME}\n{"="*W}')
print(f'  評価する確率: Isotonic( sigmoid( ability_margin + flat_base_margin ) )  '
      f'＝ 画面の ev_tan_pct / ev_fuku_pct')
print(f'  モデル学習   〜{Q["span"]["train_end"]}  103,031頭（内側HO 5,343頭は early stopping のみ）'
      f'  木 {Q["trees"]}本')
for k, b in (('較正train(Isotonicを学習)', CT), ('較正val (過学習チェック)', CV), ('最終OOS  (本表の数字)', OOS)):
    n = len(b['y1']); print(f'  {k:<26}{b["date"].min()}〜{b["date"].max()}  '
        f'{n:>7,}頭 / {len(set(b["rid"])):>5,}R  1着率 {b["y1"].mean()*100:5.2f}%  '
        f'3着内 {b["y3"].mean()*100:5.2f}%')

print(f'\n{"="*W}\n■ ③④ Calibration curve（最終OOS・{TNAME}）\n{"="*W}')
print(f'  {"帯":<10}{"N":>7}{"平均予測":>9}{"中央値":>8}{"実測":>8}{"実測-予測":>10}{"Wilson 95%CI":>20}  判定')
for lo, hi in BANDS:
    p = A['E1 Isotonic']; m = (p >= lo) & (p < hi)
    lab = f'{int(lo*100)}-{min(int(hi*100),100)}%'
    if m.sum() == 0: print(f'  {lab:<10}{0:>7}      該当なし'); continue
    k_, n_ = int(Y[m].sum()), int(m.sum()); lo_, hi_ = wilson(k_, n_)
    d = (Y[m].mean()-p[m].mean())*100
    j = 'N<200 参考' if n_ < 200 else ('OK' if abs(d) <= 2.0 else f'ズレ{d:+.1f}pt')
    print(f'  {lab:<10}{n_:>7,}{p[m].mean()*100:>8.2f}%{np.median(p[m])*100:>7.2f}%'
          f'{Y[m].mean()*100:>7.2f}%{d:>+9.2f}pt{f"[{lo_*100:.1f}, {hi_*100:.1f}]":>20}  {j}')

print(f'\n{"="*W}\n■ ⑤ 校正前 → 校正後（同じ最終OOS・{TNAME}）\n{"="*W}')
print(f'  {"腕":<16}{"予測平均":>9}{"Brier":>9}{"LogLoss":>10}{"ECE(10分位)":>13}{"ECE(11帯)":>11}'
      f'{"AUC":>8}{"PR-AUC":>9}{"予測SD":>8}')
for a in ('E0 未補正','E1 Isotonic','参考M 市場'):
    r = mets(A[a], Y)
    print(f'  {a:<16}{r["mean"]*100:>8.2f}%{r["brier"]:>9.4f}{r["ll"]:>10.4f}'
          f'{r["ece"]:>13.4f}{r["eceb"]:>11.4f}{r["auc"]:>8.4f}{r["pr"]:>9.4f}{r["sd"]:>8.4f}')
print(f'  （参考）実測 {Y.mean()*100:.2f}% を全馬に割り当てる無情報予測: '
      f'Brier {np.mean((Y.mean()-Y)**2):.4f}  AUC 0.5000')

print(f'\n  校正前の帯別ズレ（E0 未補正）')
print(f'  {"帯":<10}{"N":>7}{"平均予測":>9}{"実測":>8}{"実測-予測":>10}')
for lo, hi in BANDS:
    p = A['E0 未補正']; m = (p >= lo) & (p < hi)
    if m.sum() < 30: continue
    print(f'  {f"{int(lo*100)}-{min(int(hi*100),100)}%":<10}{int(m.sum()):>7,}'
          f'{p[m].mean()*100:>8.2f}%{Y[m].mean()*100:>7.2f}%{(Y[m].mean()-p[m].mean())*100:>+9.2f}pt')

print(f'\n{"="*W}\n■ ⑩ 過学習チェック（較正val と 最終OOS でズレの符号が一致するか）\n{"="*W}')
print(f'  {"帯":<10}{"CV N":>7}{"CV 差":>10}{"OOS N":>8}{"OOS 差":>10}  符号')
agree = tot = 0
for lo, hi in BANDS:
    pc, pv = ARMS['CV']['E1 Isotonic'], A['E1 Isotonic']
    mc, mo = (pc>=lo)&(pc<hi), (pv>=lo)&(pv<hi)
    if mc.sum() < 30 or mo.sum() < 30: continue
    dc = (yv['CV'][mc].mean()-pc[mc].mean())*100; do = (Y[mo].mean()-pv[mo].mean())*100
    ok = (dc>0) == (do>0); agree += ok; tot += 1
    print(f'  {f"{int(lo*100)}-{min(int(hi*100),100)}%":<10}{int(mc.sum()):>7,}{dc:>+9.2f}pt'
          f'{int(mo.sum()):>8,}{do:>+9.2f}pt  {"一致" if ok else "🔴不一致"}')
print(f'  符号一致 {agree}/{tot} 帯（N>=30 の帯のみ）')
for k in ('CV','OOS'):
    r = mets(ARMS[k]['E1 Isotonic'], yv[k])
    print(f'  {k:<5} ECE {r["ece"]:.4f}  Brier {r["brier"]:.4f}  AUC {r["auc"]:.4f}')

print(f'\n{"="*W}\n■ ⑥⑦ Calibration と Resolution の分解（Murphy・11帯・最終OOS）\n{"="*W}')
print(f'  {"腕":<16}{"REL(小さいほど良)":>18}{"RES(大きいほど良)":>18}{"UNC":>10}{"RES/UNC":>10}{"BS=REL-RES+UNC":>17}')
for a in ('E0 未補正','E1 Isotonic','参考M 市場'):
    rel, res, unc = brier_decomp(A[a], Y)
    print(f'  {a:<16}{rel:>18.5f}{res:>18.5f}{unc:>10.5f}{res/unc*100:>9.1f}%{rel-res+unc:>17.5f}')
print(f'\n  予測確率の分布（最終OOS）')
print(f'  {"腕":<16}{"5%点":>8}{"25%":>8}{"中央値":>8}{"75%":>8}{"95%点":>8}{"最大":>8}{"SD":>8}')
for a in ('E0 未補正','E1 Isotonic','参考M 市場'):
    p = A[a]; q = np.percentile(p, [5,25,50,75,95])
    print(f'  {a:<16}' + ''.join(f'{x*100:>7.2f}%' for x in q) + f'{p.max()*100:>7.2f}%{p.std():>8.4f}')
print(f'\n  帯を上がるにつれ実勝率は上がるか（E1 Isotonic・単調性）')
prev, mono = -1, True
for lo, hi in BANDS:
    p = A['E1 Isotonic']; m = (p>=lo)&(p<hi)
    if m.sum() < 30: continue
    r = Y[m].mean()*100
    if r < prev - 0.01: mono = False
    prev = r
p = A['E1 Isotonic']
lowm, highm = p < 0.05, p >= 0.30
if lowm.sum() and highm.sum():
    print(f'  最下位帯(0-5%) 実測 {Y[lowm].mean()*100:.2f}% → 30%以上 実測 {Y[highm].mean()*100:.2f}%'
          f'  = {Y[highm].mean()/max(Y[lowm].mean(),1e-9):.1f}倍   単調: {"はい" if mono else "いいえ"}')

print(f'\n{"="*W}\n■ ⑧ 市場との比較（同じ最終OOS）\n{"="*W}')
om = np.isfinite(A['参考M 市場'])
print(f'  logit の相関 R²（AI E1 vs 市場）: '
      f'{np.corrcoef(lg(A["E1 Isotonic"][om]), lg(A["参考M 市場"][om]))[0,1]**2:.4f}')
Xc = np.column_stack([lg(ARMS['CT']['参考M 市場'])]) if False else None
mk_ct = az.apply_market(base, gp, CT['pop'].astype(float), CT['field'])
e1_ct = iso.predict(e0['CT'])
for lab, cols in (('市場のみ', ['m']), ('市場 + AI', ['m','a'])):
    Xtr = np.column_stack([lg(mk_ct)] + ([lg(e1_ct)] if 'a' in cols else []))
    Xte = np.column_stack([lg(A['参考M 市場'])] + ([lg(A['E1 Isotonic'])] if 'a' in cols else []))
    lr = LogisticRegression(max_iter=1000).fit(Xtr, yv['CT'])
    pr = lr.predict_proba(Xte)[:,1]
    print(f'  {lab:<12} AUC {roc_auc_score(Y, pr):.4f}  Brier {np.mean((pr-Y)**2):.4f}  '
          f'LogLoss {-np.mean(Y*np.log(np.clip(pr,EPS,1-EPS))+(1-Y)*np.log(np.clip(1-pr,EPS,1-EPS))):.4f}')
print(f'\n  市場の Calibration curve（比較用）')
print(f'  {"帯":<10}{"N":>7}{"平均予測":>9}{"実測":>8}{"実測-予測":>10}')
for lo, hi in BANDS:
    p = A['参考M 市場']; m = (p>=lo)&(p<hi)
    if m.sum() < 30: continue
    print(f'  {f"{int(lo*100)}-{min(int(hi*100),100)}%":<10}{int(m.sum()):>7,}'
          f'{p[m].mean()*100:>8.2f}%{Y[m].mean()*100:>7.2f}%{(Y[m].mean()-p[m].mean())*100:>+9.2f}pt')

print(f'\n{"="*W}\n■ ⑨ AIが市場から離れたときの精度（補助分析・ここから条件は拾わない）\n{"="*W}')
gap = (A['E1 Isotonic'] - A['参考M 市場'])*100
print(f'  {"AI - 市場":<16}{"N":>7}{"AI予測":>9}{"市場予測":>10}{"実測":>8}'
      f'{"実測-AI":>10}{"実測-市場":>11}  どちらが近いか')
for lo, hi, lab in [(-99,-10,'-10pt以下'),(-10,-5,'-10〜-5pt'),(-5,0,'-5〜0pt'),
                    (0,5,'0〜+5pt'),(5,10,'+5〜+10pt'),(10,99,'+10pt以上')]:
    m = (gap>=lo)&(gap<hi)
    if m.sum() < 30: print(f'  {lab:<16}{int(m.sum()):>7}  N不足'); continue
    da, dm = (Y[m].mean()-A['E1 Isotonic'][m].mean())*100, (Y[m].mean()-A['参考M 市場'][m].mean())*100
    print(f'  {lab:<16}{int(m.sum()):>7,}{A["E1 Isotonic"][m].mean()*100:>8.2f}%'
          f'{A["参考M 市場"][m].mean()*100:>9.2f}%{Y[m].mean()*100:>7.2f}%{da:>+9.2f}pt{dm:>+10.2f}pt'
          f'  {"AI" if abs(da)<abs(dm) else "市場"}')

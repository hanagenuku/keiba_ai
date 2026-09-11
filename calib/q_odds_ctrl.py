"""⑧⑨ の対照: 市場を「人気テーブル」ではなく**実オッズ**にして測り直す。

🔴 人気×頭数帯のテーブルはオッズの大きさを捨てた粗い市場（D-2③で +0.0254 AUC 相当の
   情報損失が実測済み）。⑨で「市場が過小」と見えたのが代理変数の粗さの産物でないかを確かめる。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.dirname(BASE)
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.features.engine import _flat_base_margin
az_src = open(f'{BASE}/analyze.py').read().rsplit("\nif __name__", 1)[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(az_src, 'analyze.py', 'exec'), az.__dict__)
Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb'))
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS)/(1-np.clip(p, EPS, 1-EPS)))

def e0_of(b):
    return 1/(1+np.exp(-((b['raw']-b['bm']) + np.array([_flat_base_margin(int(n)) for n in b['field']]))))
CT, OOS = Q['CT'], Q['OOS']
iso = IsotonicRegression(out_of_bounds='clip').fit(e0_of(CT), CT['y1'])
ai = np.clip(iso.predict(e0_of(OOS)), EPS, 1-EPS)
base, gp = az.market_table(CT['pop'].astype(float), CT['field'], CT['y1'])
mk_tab = az.apply_market(base, gp, OOS['pop'].astype(float), OOS['field'])
Y = OOS['y1']

c = sqlite3.connect(f'{SCR}/keiba.db')
rp = pd.read_sql('SELECT race_id, horse_num, tansho_odds FROM race_predictions '
                 'WHERE tansho_odds >= 1.0', c); c.close()
key = pd.MultiIndex.from_arrays([OOS['rid'], np.asarray(OOS['field'])*0 + 0])  # placeholder
hn = pickle.load(open(f'{BASE}/cut_data.pkl','rb'))['meta']['horse_num'].to_numpy()[OOS['idx']]
key = pd.MultiIndex.from_arrays([OOS['rid'], hn])
odds = rp.set_index(['race_id','horse_num'])['tansho_odds'].reindex(key).to_numpy(float)
have = np.isfinite(odds) & (odds >= 1.0)
# レース内で Σ(1/オッズ)=1 に正規化＝控除率を除いた市場示唆確率
raw = np.where(have, 1.0/np.where(have, odds, 1), np.nan)
s = pd.Series(raw).groupby(pd.Series(OOS['rid'])).transform('sum').to_numpy()
cnt = pd.Series(have.astype(int)).groupby(pd.Series(OOS['rid'])).transform('sum').to_numpy()
full = pd.Series(np.ones(len(Y))).groupby(pd.Series(OOS['rid'])).transform('sum').to_numpy()
ok = have & (cnt == full) & np.isfinite(s) & (s > 0)      # 全頭のオッズが揃ったレースだけ
mk_odds = np.where(ok, raw/np.where(ok, s, 1), np.nan)
print(f'■ 実オッズが全頭そろったレース: {len(set(np.asarray(OOS["rid"])[ok])):,}R / {ok.sum():,}頭'
      f'  期間 {OOS["date"][ok].min()}〜{OOS["date"][ok].max()}')
print(f'  検算: Σ(1/オッズ) の中央値 = '
      f'{pd.Series(s[ok]).groupby(pd.Series(np.asarray(OOS["rid"])[ok])).first().median():.3f}（控除率20%なら約1.25）')
print(f'  この部分集合の実勝率 {Y[ok].mean()*100:.2f}%（全OOS {Y.mean()*100:.2f}%）\n')

def row(lab, p, y):
    p = np.clip(p, EPS, 1-EPS)
    e = []
    for lo, hi in [(0,.05),(.05,.10),(.10,.15),(.15,.20),(.20,.25),(.25,.30),(.30,.40),(.40,.50),(.50,1.01)]:
        m = (p>=lo)&(p<hi)
        if m.sum(): e.append((m.sum(), abs(p[m].mean()-y[m].mean())))
    ece = sum(n*d for n,d in e)/sum(n for n,_ in e)
    print(f'  {lab:<22}{p.mean()*100:>8.2f}%{np.mean((p-y)**2):>9.4f}'
          f'{-np.mean(y*np.log(p)+(1-y)*np.log(1-p)):>10.4f}{ece:>9.4f}{roc_auc_score(y,p):>8.4f}{p.std():>8.4f}')

print(f'■ 同じ部分集合での比較（{ok.sum():,}頭）')
print(f'  {"腕":<22}{"予測平均":>9}{"Brier":>9}{"LogLoss":>10}{"ECE":>9}{"AUC":>8}{"SD":>8}')
row('AI E1 Isotonic', ai[ok], Y[ok])
row('市場 人気テーブル', mk_tab[ok], Y[ok])
row('市場 実オッズ(正規化)', mk_odds[ok], Y[ok])

print(f'\n■ ⑧ 市場を統制したときAIに残る情報（学習は較正train・評価はこの部分集合）')
ct_tab = az.apply_market(base, gp, CT['pop'].astype(float), CT['field'])
ai_ct = iso.predict(e0_of(CT))
for lab, use_ai in (('市場(テーブル)のみ', False), ('市場(テーブル) + AI', True)):
    Xtr = np.column_stack([lg(ct_tab)] + ([lg(ai_ct)] if use_ai else []))
    Xte = np.column_stack([lg(mk_tab[ok])] + ([lg(ai[ok])] if use_ai else []))
    lr = LogisticRegression(max_iter=1000).fit(Xtr, CT['y1'])
    print(f'  {lab:<22}AUC {roc_auc_score(Y[ok], lr.predict_proba(Xte)[:,1]):.4f}')
# 実オッズ版は較正trainに実オッズが無いので、その場で2変数ロジスティックを
# **この部分集合の中で**5分割CVして測る（探索はしない・閾値も選ばない）
from sklearn.model_selection import cross_val_predict
for lab, cols in (('市場(実オッズ)のみ', [lg(mk_odds[ok])]),
                  ('市場(実オッズ) + AI', [lg(mk_odds[ok]), lg(ai[ok])])):
    X = np.column_stack(cols)
    pr = cross_val_predict(LogisticRegression(max_iter=1000), X, Y[ok],
                           cv=5, method='predict_proba')[:, 1]
    print(f'  {lab:<22}AUC {roc_auc_score(Y[ok], pr):.4f}  (5分割CV)')

print(f'\n■ ⑨ 乖離帯を「実オッズの市場」で測り直す')
gap = (ai[ok] - mk_odds[ok])*100; yo = Y[ok]; a_, m_ = ai[ok], mk_odds[ok]
print(f'  {"AI - 市場":<16}{"N":>7}{"AI予測":>9}{"市場予測":>10}{"実測":>8}{"実測-AI":>10}{"実測-市場":>11}  近い方')
for lo, hi, lab in [(-99,-10,'-10pt以下'),(-10,-5,'-10〜-5pt'),(-5,0,'-5〜0pt'),
                    (0,5,'0〜+5pt'),(5,10,'+5〜+10pt'),(10,99,'+10pt以上')]:
    m = (gap>=lo)&(gap<hi)
    if m.sum() < 30: print(f'  {lab:<16}{int(m.sum()):>7}  N不足'); continue
    da, dm = (yo[m].mean()-a_[m].mean())*100, (yo[m].mean()-m_[m].mean())*100
    print(f'  {lab:<16}{int(m.sum()):>7,}{a_[m].mean()*100:>8.2f}%{m_[m].mean()*100:>9.2f}%'
          f'{yo[m].mean()*100:>7.2f}%{da:>+9.2f}pt{dm:>+10.2f}pt  {"AI" if abs(da)<abs(dm) else "市場"}')

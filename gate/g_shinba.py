"""問い(ユーザー・2026-09-11): 新馬戦を推奨レースにするのはおかしくないか。

推奨ゲートは1本だけ（軸の top3_prob >= 0.55）。それが**新馬で較正されているか**を測る。
E-1(2026-08-24) はクラス別の回収率は測ったが**較正は測っていない**ので、ここは未検証。

🔴 対照群が決定的: 市場1番人気の同じ数字を併記する。
   新馬が「誰にも当てられない」のか「AIだけ外す」のかを分けないと結論できない。

判定（結果を見る前に固定）: 次のどれかが成立すれば「新馬は別扱いすべき」
  ① 軸の |予測−実測| が他クラス平均より 5pt 以上大きい
  ② ゲート通過後の軸の実測3着内率が他クラスより 5pt 以上低い
  ③ ゲート通過後の複勝回収率が他クラスより 5pt 以上低い
どれも成立しなければ「現状どおり除外しない」。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.dirname(BASE)
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.models.predict import softmax_probs
from src.features.engine import calc_harville_probs
GATE = 0.55
Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb'))
META = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))['meta']
B = Q['OOS']
# 🔴 本番の total は raw_prob*10 で、raw_prob は**市場アンカー込みの完全な確率**
#    （ability だけの e0 ではない）。ここを取り違えるとゲート通過率が 83%→1.8% になる。
p_full = 1/(1+np.exp(-B['raw']))
md = META.iloc[B['idx']]
d = pd.DataFrame({'rid': B['rid'], 'hn': md['horse_num'].to_numpy(), 'e0': p_full,
                  'y1': B['y1'], 'y3': B['y3'], 'pop': B['pop'],
                  'cls': md['race_class'].fillna('').to_numpy()})
# 本番の経路: softmax(total, T=3.5) → Harville（相対ブレンドは近似で省略・参考Sと同じ）
rows = []
for rid, g in d.groupby('rid', sort=False):
    wp = softmax_probs([v*10 for v in g['e0']], temperature=3.5)
    t3 = [b for _, b in calc_harville_probs(wp)]
    i = int(np.argmax(g['e0'].to_numpy()))                 # 軸 = RL1
    pp = g['pop'].to_numpy(float)
    mi = int(np.nanargmin(np.where(np.isfinite(pp) & (pp > 0), pp, 99)))  # 市場1番人気
    rows.append(dict(rid=rid, cls=g['cls'].iloc[0], n=len(g),
                     ax_p=t3[i], ax_y3=int(g['y3'].iloc[i]), ax_hn=int(g['hn'].iloc[i]),
                     mk_y3=int(g['y3'].iloc[mi]), mk_hn=int(g['hn'].iloc[mi])))
R = pd.DataFrame(rows)
# 複勝配当
h = sqlite3.connect(f'{BASE}/data/history.db')
pay = pd.read_sql('SELECT race_id, horse_num, fukusho_payout FROM horse_history', h); h.close()
pk = pay.set_index(['race_id', 'horse_num'])['fukusho_payout']
for tag, hc in (('ax', 'ax_hn'), ('mk', 'mk_hn')):
    R[tag+'_pay'] = pk.reindex(pd.MultiIndex.from_arrays([R['rid'], R[hc]])).to_numpy(float)
    # 🔴 3着内なのに配当が欠損している行は評価不能（0円として混ぜない・2026-08-18③）
    R[tag+'_ok'] = ~((R[tag+'_y3'] == 1) & (~np.isfinite(R[tag+'_pay']) | (R[tag+'_pay'] <= 0)))

def roi(df, tag):
    m = df[df[tag+'_ok']]
    return (m.loc[m[tag+'_y3'] == 1, tag+'_pay'].sum() / len(m) if len(m) else np.nan)

W = 104
print(f'{"="*W}\n■ 母集団: 最終OOS 2026-01-04〜09-06  {len(R):,}レース（モデル・較正器は2025年までのデータのみ）')
print(f'  推奨ゲートは1本だけ: 軸(RL1)の top3_prob >= {GATE}\n{"="*W}')
print(f'\n■ ① クラス別 — 軸の確度は較正されているか（全レース）')
print(f'  {"クラス":<8}{"R数":>6}{"平均頭数":>9}{"予測":>8}{"実測":>8}{"実測-予測":>11}   '
      f'{"市場1人気の実測":>15}{"AI-市場":>9}')
order = ['新馬', '未勝利', '1勝', '2勝', '3勝', 'OP']
for c in order:
    g = R[R['cls'] == c]
    if len(g) < 30: print(f'  {c:<8}{len(g):>6}  N不足'); continue
    dp = (g['ax_y3'].mean()-g['ax_p'].mean())*100
    print(f'  {c:<8}{len(g):>6,}{g["n"].mean():>9.1f}{g["ax_p"].mean()*100:>7.1f}%'
          f'{g["ax_y3"].mean()*100:>7.1f}%{dp:>+10.1f}pt   {g["mk_y3"].mean()*100:>14.1f}%'
          f'{(g["ax_y3"].mean()-g["mk_y3"].mean())*100:>+8.1f}pt')
oth = R[R['cls'].isin(['1勝','2勝','3勝','OP'])]
print(f'  {"（1勝以上）":<8}{len(oth):>6,}{oth["n"].mean():>9.1f}{oth["ax_p"].mean()*100:>7.1f}%'
      f'{oth["ax_y3"].mean()*100:>7.1f}%{(oth["ax_y3"].mean()-oth["ax_p"].mean())*100:>+10.1f}pt'
      f'   {oth["mk_y3"].mean()*100:>14.1f}%{(oth["ax_y3"].mean()-oth["mk_y3"].mean())*100:>+8.1f}pt')

print(f'\n■ ② クラス別 — ゲート(>={GATE})の通過率と、通過したレースの成績')
print(f'  {"クラス":<8}{"全R":>6}{"通過":>6}{"通過率":>8}{"軸3着内":>9}{"軸複勝回収":>11}   '
      f'{"市場1人気 3着内":>16}{"市場1人気 回収":>15}')
for c in order + ['（1勝以上）']:
    g = oth if c == '（1勝以上）' else R[R['cls'] == c]
    p = g[g['ax_p'] >= GATE]
    if len(p) < 30: print(f'  {c:<8}{len(g):>6,}{len(p):>6}  N不足'); continue
    print(f'  {c:<8}{len(g):>6,}{len(p):>6,}{len(p)/len(g)*100:>7.1f}%'
          f'{p["ax_y3"].mean()*100:>8.1f}%{roi(p,"ax"):>10.1f}%   '
          f'{p["mk_y3"].mean()*100:>15.1f}%{roi(p,"mk"):>14.1f}%')

print(f'\n■ ③ 新馬の中で確度帯ごとに見る（過信がどこで出るか）')
print(f'  {"帯":<12}{"R数":>6}{"予測":>8}{"実測":>8}{"差":>9}   {"（1勝以上）R数":>14}{"予測":>8}{"実測":>8}{"差":>9}')
for lo, hi in [(0,.50),(.50,.55),(.55,.60),(.60,.65),(.65,.70),(.70,1.01)]:
    s = R[(R['cls']=='新馬') & (R['ax_p']>=lo) & (R['ax_p']<hi)]
    o = oth[(oth['ax_p']>=lo) & (oth['ax_p']<hi)]
    f = lambda g: ('  N不足'.ljust(25) if len(g) < 30 else
        f'{g["ax_p"].mean()*100:>7.1f}%{g["ax_y3"].mean()*100:>7.1f}%'
        f'{(g["ax_y3"].mean()-g["ax_p"].mean())*100:>+8.1f}pt')
    print(f'  {f"{lo*100:.0f}-{min(hi*100,100):.0f}%":<12}{len(s):>6}{f(s)}   {len(o):>14,}{f(o)}')

print(f'\n■ ④ 推奨は1日6レースまで。新馬がどれだけ枠を取っているか')
day = pd.Series([r[:8] for r in R['rid']])
p = R[R['ax_p'] >= GATE].copy(); p['day'] = [r[:8] for r in p['rid']]
top = p.sort_values('ax_p', ascending=False).groupby('day').head(6)
print(f'  ゲート通過 {len(p):,}R → 1日上位6本に絞ると {len(top):,}R / {top["day"].nunique()}開催日')
vc = top['cls'].value_counts()
for c in order:
    n = int(vc.get(c, 0)); base = int((R['cls']==c).sum())
    print(f'  {c:<8}{n:>5}本 ({n/len(top)*100:>5.1f}%)   母集団での割合 {base/len(R)*100:>5.1f}%')
print(f'\n  実際に選ばれた6本の成績: 軸3着内 {top["ax_y3"].mean()*100:.1f}% / '
      f'複勝回収 {roi(top,"ax"):.1f}%   （市場1人気 {top["mk_y3"].mean()*100:.1f}% / {roi(top,"mk"):.1f}%）')
s6 = top[top['cls']=='新馬']; o6 = top[~top['cls'].isin(['新馬'])]
if len(s6) >= 30:
    print(f'  └ うち新馬  {len(s6):>4}本  軸3着内 {s6["ax_y3"].mean()*100:.1f}% / 複勝回収 {roi(s6,"ax"):.1f}%')
    print(f'  └ 新馬以外  {len(o6):>4}本  軸3着内 {o6["ax_y3"].mean()*100:.1f}% / 複勝回収 {roi(o6,"ax"):.1f}%')

print(f'\n{"="*W}\n■ ⑤ 「新馬でAIが市場に最も負ける」は頑健か（日ブロック bootstrap 2,000回）\n{"="*W}')
rng = np.random.default_rng(42)
R['day'] = [r[:8] for r in R['rid']]
print(f'  {"クラス":<10}{"R数":>6}{"AI軸":>8}{"市場1人気":>10}{"差":>9}{"95%CI(差)":>18}{"前半":>9}{"後半":>9}')
for c in order + ['（1勝以上）']:
    g = oth if c == '（1勝以上）' else R[R['cls'] == c]
    if len(g) < 30: continue
    days = g['day'].unique(); bs = []
    for _ in range(2000):
        s = g[g['day'].isin([])] if False else pd.concat(
            [g[g['day'] == x] for x in rng.choice(days, len(days))])
        bs.append((s['ax_y3'].mean()-s['mk_y3'].mean())*100)
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    mid = np.sort(days)[len(days)//2]
    a, b = g[g['day'] < mid], g[g['day'] >= mid]
    da = (a['ax_y3'].mean()-a['mk_y3'].mean())*100 if len(a) else np.nan
    db = (b['ax_y3'].mean()-b['mk_y3'].mean())*100 if len(b) else np.nan
    print(f'  {c:<10}{len(g):>6,}{g["ax_y3"].mean()*100:>7.1f}%{g["mk_y3"].mean()*100:>9.1f}%'
          f'{(g["ax_y3"].mean()-g["mk_y3"].mean())*100:>+8.1f}pt{f"[{lo_:+.1f}, {hi_:+.1f}]":>18}'
          f'{da:>+8.1f}pt{db:>+8.1f}pt')
print(f'\n■ ⑥ なぜ新馬でAIが負けるか — 軸が市場1番人気と一致した割合')
for c in order + ['（1勝以上）']:
    g = oth if c == '（1勝以上）' else R[R['cls'] == c]
    if len(g) < 30: continue
    same = (g['ax_hn'] == g['mk_hn'])
    print(f'  {c:<10}{len(g):>6,}  軸=市場1人気 {same.mean()*100:>5.1f}%   '
          f'一致時の軸3着内 {g.loc[same,"ax_y3"].mean()*100:>5.1f}%   '
          f'不一致時 {g.loc[~same,"ax_y3"].mean()*100:>5.1f}%'
          f'（その時の市場1人気 {g.loc[~same,"mk_y3"].mean()*100:>5.1f}%）')

"""②-A: 上位分位そのものの残差が +6pt を超える軸はあるか。

mispricing.py は 10分位の「最大−最小」（スプレッド）しか出していない。
スプレッド 6.6pt でも、上位分位の残差そのものは その半分程度でしかない。
合格線（mispricing/CRITERIA.md §1）は残差そのもので +6pt なので、そこを直接見る。

🔴 ルール（ユーザー指示 2026-09-10）
   ・新しい特徴量を追加しない  ・新しい条件を思いつきで探索しない
   ・読み取り専用・既存データのみ  ・見つかっても採用・実戦投入としない
   ・多重比較で偶然出た可能性を必ず併記する
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
pd.set_option('display.width', 250)

src = open(f'{BASE}/analyze.py').read().split("if __name__")[0]
az = type(sys)('az'); az.__dict__['__file__'] = f'{BASE}/analyze.py'
exec(compile(src, 'analyze.py', 'exec'), az.__dict__)

D = pickle.load(open(f'{BASE}/cut_data.pkl','rb'))
FEAT = D['feat']
df = pd.read_csv(f'{BASE}/data/horse_features.csv')

TUNE, CONF = ['T1','T2','T3'], ['W1','W2','W3']
POPB = [(1,1),(2,3),(4,5),(6,9),(10,99)]
NB = 10
MIN_N   = 300     # 関門1（事前登録）
BAR_PT  = 6.0     # 関門2（事前登録）合格線 +6pt
POOL_ROI, POOL_RES, RATE = 69.9, -0.37, 4.75   # §1 の交換レート


def build():
    rows = []
    for w in TUNE + CONF:
        win = D['windows'][w]; vi = win['va_idx']
        pop = win['popv'].astype(float); fs = D['meta'].iloc[vi]['field'].to_numpy(float)
        base, gp = az.market_table(win['popi'].astype(float),
                                   D['meta'].iloc[win['in_idx']]['field'].to_numpy(float),
                                   win['y3i'])
        mkt = az.apply_market(base, gp, pop, fs)
        ok = np.isfinite(pop) & (pop > 0) & (pop < 99)
        sub = df.iloc[vi][FEAT].copy()
        sub['__y3'] = win['y3v'].astype(int); sub['__mkt'] = mkt
        sub['__pop'] = pop; sub['__per'] = '探索' if w in TUNE else '確認'
        rows.append(sub[ok])
    return pd.concat(rows, ignore_index=True)


def cell(a, mask):
    """そのセルの N / 実測3着内率 / 市場示唆 / 残差 / 残差の95%CI下限（pt）"""
    s = a[mask]
    n = len(s)
    if n == 0: return None
    y, m = s['__y3'].mean(), s['__mkt'].mean()
    res = (y - m) * 100
    se = np.sqrt(max(y*(1-y), 1e-9) / n) * 100      # 市場示唆はセル内でほぼ定数
    return dict(n=n, y3=y*100, mkt=m*100, res=res, lo=res - 1.96*se, hi=res + 1.96*se)


def roi_lin(res):   # (i) 事前登録した線形の交換レート
    return POOL_ROI + RATE * (res - POOL_RES)

def roi_mul(y3, mkt):  # (ii) 比で見る感度チェック
    return POOL_ROI * (y3 / mkt)


def main():
    f = build()
    print(f'行 {len(f):,}（探索 {(f.__per=="探索").sum():,} / 確認 {(f.__per=="確認").sum():,}）')
    for per in ['探索','確認']:
        a = f[f.__per==per]
        print(f'  検算 {per}: 実測3着内 {a.__y3.mean()*100:.2f}%  市場示唆 {a.__mkt.mean()*100:.2f}%  '
              f'ズレ {(a.__y3.mean()-a.__mkt.mean())*100:+.2f}pt  ← 0 付近なら計測器は正常')

    f['__pb'] = 0
    for i,(lo,hi) in enumerate(POPB):
        f.loc[f.__pop.between(lo,hi), '__pb'] = i

    ex, cf = f[f.__per=='探索'], f[f.__per=='確認']
    out = []
    for c in FEAT:
        if f[c].nunique() < 4: continue
        try:
            q = f.groupby('__pb')[c].transform(
                lambda s: pd.qcut(s.rank(method='first'), NB, labels=False, duplicates='drop'))
        except Exception:
            continue
        qe, qc = q[f.__per=='探索'], q[f.__per=='確認']
        nq = int(q.max())
        if nq < 5: continue
        # (a) 事前指定: 最上位分位 q=nq と 最下位分位 q=0 の2セルだけ
        for tag, qi in (('最上位', nq), ('最下位', 0)):
            a = cell(cf, qc == qi); b = cell(ex, qe == qi)
            if a is None or b is None or a['n'] < MIN_N: continue
            out.append(dict(feat=c, side=tag, mode='事前指定', **a,
                            res_ex=b['res'], n_ex=b['n']))
        # (b) データで選んだ最大／最小残差の分位（多重比較あり）
        mm = cf.groupby(qc)['__res'].agg(['mean','size']) if '__res' in cf else None
        r_by = cf.assign(__r=(cf['__y3']-cf['__mkt'])).groupby(qc)['__r'].agg(['mean','size'])
        r_by = r_by[r_by['size'] >= MIN_N]
        if len(r_by) == 0: continue
        for tag, qi in (('最大残差', r_by['mean'].idxmax()), ('最小残差', r_by['mean'].idxmin())):
            a = cell(cf, qc == qi); b = cell(ex, qe == qi)
            if a is None or b is None or a['n'] < MIN_N: continue
            out.append(dict(feat=c, side=f'{tag}(q{int(qi)})', mode='データ選択', **a,
                            res_ex=b['res'], n_ex=b['n']))

    r = pd.DataFrame(out)
    r['roi_lin'] = roi_lin(r.res); r['roi_mul'] = roi_mul(r.y3, r.mkt)
    r['abs_res'] = r.res.abs()
    r.to_csv(f'{BASE}/gap_a.csv', index=False)

    def show(sub, title, k=15):
        print(f'\n{"="*128}\n■ {title}\n{"="*128}')
        print(f'  {"軸":<30}{"分位":<14}{"N":>7}{"実測":>8}{"市場":>8}{"残差":>9}{"CI下限":>9}'
              f'{"探索期":>9}{"回収(i)":>9}{"回収(ii)":>9}')
        for _, x in sub.head(k).iterrows():
            print(f'  {x.feat:<30}{x.side:<14}{x.n:>7,}{x.y3:>7.1f}%{x.mkt:>7.1f}%'
                  f'{x.res:>+8.2f}pt{x.lo:>+8.2f}pt{x.res_ex:>+8.2f}pt'
                  f'{x.roi_lin:>8.1f}%{x.roi_mul:>8.1f}%')

    A = r[r['mode']=='事前指定'].sort_values('abs_res', ascending=False)
    B = r[r['mode']=='データ選択'].sort_values('abs_res', ascending=False)
    show(A, f'(a) 事前指定の分位（最上位・最下位のみ／{len(A)}セル）  残差の絶対値降順')
    show(B, f'(b) データで選んだ分位（{len(B)}セル・多重比較あり）  残差の絶対値降順')

    print(f'\n{"="*128}\n■ 関門（mispricing/CRITERIA.md §4・結果を見る前に固定）\n{"="*128}')
    for nm, sub in (('(a) 事前指定', A), ('(b) データ選択', B)):
        g1 = sub[sub.n >= MIN_N]
        g2 = g1[g1.lo > BAR_PT]
        g3 = g2[(g2.res_ex > BAR_PT) & (np.sign(g2.res_ex) == np.sign(g2.res))]
        print(f'  {nm}')
        print(f'    関門1  N>={MIN_N}                      : {len(g1)}/{len(sub)} セル')
        print(f'    関門2  残差の95%CI下限 > +{BAR_PT}pt      : {len(g2)} セル'
              f'   {"🔴 ゼロ" if len(g2)==0 else "✅"}')
        print(f'    関門3  探索期でも同符号で +{BAR_PT}pt 級  : {len(g3)} セル')
        print(f'    参考   残差そのものが +{BAR_PT}pt を超える : '
              f'{(g1.res > BAR_PT).sum()} セル（CI下限は問わない）')
        print(f'    参考   |残差| の最大 {g1.abs_res.max():.2f}pt  '
              f'/ 95%点 {g1.abs_res.quantile(.95):.2f}pt / 中央値 {g1.abs_res.median():.2f}pt')

    print(f'\n{"="*128}\n■ 🔴 多重比較: 乱数列でも同じ大きさが出るか（帰無分布）\n{"="*128}')
    rng = np.random.default_rng(0)
    n_pre, n_sel = [], []
    for k in range(200):
        f['__rnd'] = rng.normal(size=len(f))
        q = f.groupby('__pb')['__rnd'].transform(
            lambda s: pd.qcut(s.rank(method='first'), NB, labels=False, duplicates='drop'))
        qc = q[f.__per=='確認']; nq = int(q.max())
        rb = cf.assign(__r=(cf['__y3']-cf['__mkt'])).groupby(qc)['__r'].agg(['mean','size'])
        rb = rb[rb['size'] >= MIN_N]
        if len(rb) == 0: continue
        for qi in (nq, 0):
            if qi in rb.index: n_pre.append(abs(rb.loc[qi,'mean'])*100)
        n_sel.append(max(abs(rb['mean'].max()), abs(rb['mean'].min()))*100)
    n_pre, n_sel = np.array(n_pre), np.array(n_sel)
    print(f'  (a) 事前指定セル 乱数{len(n_pre)}個: |残差| 中央値 {np.median(n_pre):.2f}pt / '
          f'95%点 {np.percentile(n_pre,95):.2f}pt / 最大 {n_pre.max():.2f}pt')
    print(f'  (b) データ選択   乱数{len(n_sel)}本: |残差| 中央値 {np.median(n_sel):.2f}pt / '
          f'95%点 {np.percentile(n_sel,95):.2f}pt / 最大 {n_sel.max():.2f}pt')
    for nm, sub, nl in (('(a)', A, n_pre), ('(b)', B, n_sel)):
        g1 = sub[sub.n >= MIN_N]
        p95 = np.percentile(nl, 95)
        exp = len(g1) * 0.05
        print(f'  {nm} 帰無95%点 {p95:.2f}pt を超えた実セル {int((g1.abs_res>p95).sum())} / {len(g1)}'
              f'   （偶然の期待 {exp:.0f}セル）')

    print(f'\n{"="*128}\n■ 参考: 地図のスプレッド上位軸を、残差そのもので見ると\n{"="*128}')
    for c in ['rl_f_speed_fig_max_vs_field','rl_f_pl_rating','f_pl_rating',
              'rl_f_time_diff_vs_field','f_agari_ability','f_last1_rank']:
        s = A[A.feat==c]
        if not len(s): continue
        for _, x in s.iterrows():
            print(f'  {c:<30}{x.side:<10}N={x.n:>6,}  実測{x.y3:>5.1f}% 市場{x.mkt:>5.1f}%  '
                  f'残差{x.res:>+6.2f}pt  探索期{x.res_ex:>+6.2f}pt  回収(i){x.roi_lin:>6.1f}%')


if __name__ == '__main__':
    main()

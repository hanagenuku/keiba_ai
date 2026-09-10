"""「切り馬」からレース構造を読めるか。基準は race_select/CRITERIA_cut.md に凍結済み。

🔴 ROI は一切見ない（ユーザー指示）。着順のみ。
"""
import os, sys, pickle
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 230)

TUNE, CONF = ['T1','T2','T3'], ['W1','W2','W3']
THR = [0.05, 0.10, 0.15, 0.20]
# 探索期でだけ探す格子（事前宣言・確認期では動かさない）
GRID_A = [0.45, 0.50, 0.55, 0.60, 0.65]
GRID_B = [3, 4, 5, 6, 7]
MIN_COV = 0.10          # 探索期での最低カバレッジ

D = pickle.load(open(f'{BASE}/cut_data.pkl','rb'))
P = pickle.load(open(f'{BASE}/cut_pred.pkl','rb'))
meta = D['meta']


def market_table(pop, fs, y3):
    """人気×頭数帯 → 実測3着内率。学習側の行だけで作る（リーク無し）。

    🔴 y3 は int8。numpy スカラのまま加算すると 127 で溢れて集計が壊れる
       （2026-09-09 に実際に踏み、MKT が「予測2.0%/実測11.5%」という
         あり得ない較正を出した）。int() に落としてから足す。
    ⚠ 人気が NaN の行は表に入れない（astype(int) がゴミ値になる）。
    """
    good = np.isfinite(pop) & (pop > 0) & (pop < 99) & np.isfinite(fs)
    pop, fs, y3 = pop[good], fs[good], np.asarray(y3)[good]
    fb = np.clip(((fs.astype(int) - 8) // 3), 0, 3)
    pc = np.clip(pop.astype(int), 1, 18)
    tab, cnt = {}, {}
    for p_, f_, y_ in zip(pc, fb, y3):
        p_, f_, y_ = int(p_), int(f_), int(y_)
        tab[(p_, f_)] = tab.get((p_, f_), 0) + y_
        cnt[(p_, f_)] = cnt.get((p_, f_), 0) + 1
    base = {}
    for k in cnt:
        base[k] = tab[k] / cnt[k] if cnt[k] >= 30 else None
    gp = {}
    for p_ in range(1, 19):
        s = sum(tab[k] for k in tab if k[0] == p_); c = sum(cnt[k] for k in cnt if k[0] == p_)
        gp[p_] = s / c if c else np.nan
    return base, gp


def apply_market(base, gp, pop, fs):
    safe = np.where(np.isfinite(pop) & (pop > 0) & (pop < 99), pop, 99.0)
    fb = np.clip(((np.nan_to_num(fs, nan=12).astype(int) - 8) // 3), 0, 3)
    pc = np.clip(safe.astype(int), 1, 18)
    out = np.empty(len(pop))
    for i, (p_, f_) in enumerate(zip(pc, fb)):
        v = base.get((p_, f_))
        out[i] = v if v is not None else gp.get(p_, np.nan)
    return np.clip(np.nan_to_num(out, nan=0.2), 1e-4, 1-1e-4)


def build_frame():
    rows = []
    for w in TUNE + CONF:
        win = D['windows'][w]
        vi, ii = win['va_idx'], win['in_idx']
        mv = meta.iloc[vi]
        fsv = mv['field'].to_numpy(float)
        popv = win['popv'].astype(float)
        # 市場表は「その窓の学習側（内側HOより前）＋内側HO」で作る＝評価窓は使わない
        fi = meta.iloc[ii]['field'].to_numpy(float)
        base, gp = market_table(win['popi'].astype(float), fi, win['y3i'])
        mkt_v = apply_market(base, gp, popv, fsv)
        mkt_i = apply_market(base, gp, win['popi'].astype(float), fi)
        ai_v, ai_i = P[w]['cal'], P[w]['cal_in']
        # AI+MKT: 内側HOで2変数ロジスティック
        from sklearn.linear_model import LogisticRegression
        lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
        Zi = np.c_[lg(ai_i), lg(mkt_i)]; Zv = np.c_[lg(ai_v), lg(mkt_v)]
        blend = LogisticRegression(max_iter=1000).fit(Zi, win['y3i']).predict_proba(Zv)[:,1]
        ok = np.isfinite(popv) & (popv > 0) & (popv < 99)
        rows.append(pd.DataFrame(dict(
            win=w, period=('探索' if w in TUNE else '確認'),
            race_id=mv['race_id'].to_numpy(), field=fsv, pop=popv,
            y3=win['y3v'].astype(int), AI=ai_v, MKT=mkt_v, AIMKT=blend, ok=ok)))
    f = pd.concat(rows, ignore_index=True)
    # 🔴 関門4: 人気取得行に揃えると Phase 4 の記録(確認期 22,399行 / 1,731レース)に一致するか
    cf = f[(f.period=='確認') & f.ok]
    print(f'関門4  確認期(人気取得行) {len(cf):,}行 / {cf.race_id.nunique():,}レース'
          f'   Phase4記録 22,399行 / 1,731レース  '
          f'{"✅一致" if abs(len(cf)-22399)<=50 else "🔴不一致"}')
    return f[f.ok].drop(columns=['ok']).reset_index(drop=True)


def q1_calibration(f):
    print('\n' + '='*100); print('■ Q1  「切り馬」性能: 低確率と判定した馬は実際に来ないか'); print('='*100)
    for arm in ['AI','MKT','AIMKT']:
        print(f'\n  【{arm}】')
        print(f'    {"閾値":<10}{"該当頭数":>10}{"全体比":>8}{"予測平均":>10}{"実測3着内率":>12}{"ズレ":>8}')
        for t in THR:
            for per in ['探索','確認']:
                s = f[(f.period==per) & (f[arm] <= t)]
                a = f[f.period==per]
                if len(s) < 50: continue
                print(f'    {per} <={t:.2f}{len(s):>10,}{len(s)/len(a)*100:>7.1f}%'
                      f'{s[arm].mean()*100:>9.1f}%{s.y3.mean()*100:>11.1f}%{(s.y3.mean()-s[arm].mean())*100:>+7.1f}pt')
        # ECE（10分位）
        for per in ['探索','確認']:
            a = f[f.period==per]
            q = pd.qcut(a[arm].rank(method='first'), 10, labels=False)
            ece = sum(abs(a.y3[q==k].mean()-a[arm][q==k].mean())*(q==k).mean() for k in range(10))
            print(f'    {per} ECE {ece:.4f}')


def race_struct(f, arm):
    g = f.sort_values([ 'win','race_id',arm], ascending=[True,True,False])
    out = []
    for (w, rid), d in g.groupby(['win','race_id'], sort=False):
        p = d[arm].to_numpy(); y = d.y3.to_numpy(); n = len(d)
        if n < 8: continue
        top3 = p[:3]
        rec = dict(win=w, period=d.period.iloc[0], race_id=rid, field=n,
                   p1=p[0], top3sum=top3.sum(), resid=p[3:].sum(),
                   n_eff=(p.sum()**2)/(p**2).sum(),
                   ax_hit=int(y[0]), top3_hits=int(y[:3].sum()),
                   rand3=9.0/n)
        for t in THR: rec[f'cut{int(t*100)}'] = int((p <= t).sum())
        out.append(rec)
    return pd.DataFrame(out)


def q2(rs, arm):
    print('\n' + '='*100); print(f'■ Q2  切り馬が多いレースほど軸が来るか  [{arm}]'); print('='*100)
    for t in [10, 15]:
        col = f'cut{t}'
        print(f'\n  【{col} = {t}%以下の頭数】')
        for per in ['探索','確認']:
            a = rs[rs.period==per]
            b = pd.cut(a[col], [-1,1,3,5,7,99], labels=['0-1','2-3','4-5','6-7','8+'])
            line = []
            for k in ['0-1','2-3','4-5','6-7','8+']:
                s = a[b==k]
                line.append(f'{k}:{len(s):>4}R 軸{s.ax_hit.mean()*100:>4.1f}% 上位3頭{s.top3_hits.mean():.2f}' if len(s)>30 else f'{k}: -')
            print(f'    {per}  ' + ' | '.join(line))
        # 頭数で統制
        print('    --- 頭数で統制（確認期）---')
        a = rs[rs.period=='確認']
        for lo,hi,lab in [(8,11,'8-11頭'),(12,14,'12-14頭'),(15,18,'15-18頭')]:
            m = a.field.between(lo,hi)
            if m.sum() < 80: continue
            med = a[col][m].median()
            hi_ = a[m & (a[col] > med)]; lo_ = a[m & (a[col] <= med)]
            if len(hi_)<40 or len(lo_)<40: continue
            print(f'      {lab} (中央値{med:.0f})  多い {len(hi_):>4}R 軸{hi_.ax_hit.mean()*100:>5.1f}% '
                  f'| 少ない {len(lo_):>4}R 軸{lo_.ax_hit.mean()*100:>5.1f}%  差 {(hi_.ax_hit.mean()-lo_.ax_hit.mean())*100:>+5.1f}pt')


def q3(rs_ai, rs_mkt):
    print('\n' + '='*100); print('■ Q3  「強い軸 ＋ 切り馬多数」 (本命)'); print('='*100)
    ex = rs_ai[rs_ai.period=='探索']
    base_ex = ex.ax_hit.mean()
    best = None
    for a in GRID_A:
        for b in GRID_B:
            m = (ex.p1 >= a) & (ex.cut15 >= b)
            if m.mean() < MIN_COV: continue
            v = ex.ax_hit[m].mean()
            if best is None or v > best[2]: best = (a, b, v, m.mean())
    if best is None:
        print('  探索期でカバレッジ10%以上の条件が存在しない → 打ち切り'); return
    A, B, v_ex, cov = best
    print(f'\n  探索期で選んだ条件: 軸のp >= {A}  かつ  15%以下の馬 >= {B}頭')
    print(f'    探索期  カバレッジ {cov*100:.1f}%  軸3着内 {v_ex*100:.1f}%  (全体 {base_ex*100:.1f}%  差 {(v_ex-base_ex)*100:+.1f}pt)')
    print(f'\n  {"":<6}{"":<10}{"R数":>7}{"軸3着内":>10}{"全体":>9}{"差":>8}{"上位3頭":>9}{"無作為":>8}')
    for per in ['探索','確認']:
        a = rs_ai[rs_ai.period==per]; m = (a.p1>=A)&(a.cut15>=B)
        for lab, s in [('該当', a[m]), ('非該当', a[~m])]:
            if len(s)<30: continue
            print(f'  {per:<6}{lab:<10}{len(s):>7,}{s.ax_hit.mean()*100:>9.1f}%{a.ax_hit.mean()*100:>8.1f}%'
                  f'{(s.ax_hit.mean()-a.ax_hit.mean())*100:>+7.1f}pt{s.top3_hits.mean():>9.2f}{s.rand3.mean():>8.2f}')
    print('\n  --- 確認期3窓それぞれ（基準③: 全窓で同符号）---')
    for w in CONF:
        a = rs_ai[rs_ai.win==w]; m = (a.p1>=A)&(a.cut15>=B)
        if m.sum()<20: print(f'    {w}: N不足'); continue
        print(f'    {w}  該当{m.sum():>4}R 軸{a.ax_hit[m].mean()*100:>5.1f}%  '
              f'全体{a.ax_hit.mean()*100:>5.1f}%  差{(a.ax_hit[m].mean()-a.ax_hit.mean())*100:>+5.1f}pt')
    # 基準④: 市場側で同じ形の条件
    print('\n  --- 🔴 基準④: 市場だけで同じ形の条件を作った場合 ---')
    exm = rs_mkt[rs_mkt.period=='探索']; bestm=None
    for a_ in GRID_A:
        for b_ in GRID_B:
            m = (exm.p1>=a_)&(exm.cut15>=b_)
            if m.mean()<MIN_COV: continue
            v = exm.ax_hit[m].mean()
            if bestm is None or v>bestm[2]: bestm=(a_,b_,v,m.mean())
    if bestm:
        Am,Bm,_,_ = bestm
        cm = rs_mkt[rs_mkt.period=='確認']; mm=(cm.p1>=Am)&(cm.cut15>=Bm)
        ca = rs_ai[rs_ai.period=='確認']; ma=(ca.p1>=A)&(ca.cut15>=B)
        print(f'    MKT条件 p>={Am} cut15>={Bm}:  確認期 該当{mm.sum():>4}R  '
              f'軸(市場1位)3着内 {cm.ax_hit[mm].mean()*100:.1f}%  全体{cm.ax_hit.mean()*100:.1f}%  '
              f'差{(cm.ax_hit[mm].mean()-cm.ax_hit.mean())*100:+.1f}pt')
        print(f'    AI 条件:                    確認期 該当{ma.sum():>4}R  '
              f'軸(AI1位)3着内 {ca.ax_hit[ma].mean()*100:.1f}%  全体{ca.ax_hit.mean()*100:.1f}%  '
              f'差{(ca.ax_hit[ma].mean()-ca.ax_hit.mean())*100:+.1f}pt')
    # 逆側（難解レース）
    print('\n  --- 逆側: 上位が弱く切り馬も少ない（難解レース）---')
    for per in ['探索','確認']:
        a = rs_ai[rs_ai.period==per]
        m = (a.p1 < a.p1.median()) & (a.cut15 <= a.cut15.median())
        print(f'    {per}  該当{m.sum():>5}R  軸{a.ax_hit[m].mean()*100:>5.1f}%  '
              f'全体{a.ax_hit.mean()*100:>5.1f}%  差{(a.ax_hit[m].mean()-a.ax_hit.mean())*100:>+5.1f}pt  '
              f'上位3頭{a.top3_hits[m].mean():.2f}')
    return A, B


def q4(rs):
    print('\n' + '='*100); print('■ Q4  切り馬の合計確率 / 裾の形'); print('='*100)
    for per in ['探索','確認']:
        a = rs[rs.period==per]
        print(f'\n  {per}期  corr(上位3頭合計, 残余) = {a.top3sum.corr(a.resid):+.4f}   '
              f'レース内合計の中央値 {(a.top3sum+a.resid).median():.3f}')
        print(f'         corr(残余, 頭数) = {a.resid.corr(a.field):+.3f}   '
              f'corr(N_eff, 頭数) = {a.n_eff.corr(a.field):+.3f}   '
              f'corr(N_eff, 上位3頭合計) = {a.n_eff.corr(a.top3sum):+.3f}')
    print('\n  --- N_eff（有効頭数）5分位ごと・確認期 ---')
    a = rs[rs.period=='確認']
    q = pd.qcut(a.n_eff.rank(method='first'), 5, labels=False)
    for k in range(5):
        s = a[q==k]
        print(f'    Q{k+1}  N={len(s):>5}  N_eff中央 {s.n_eff.median():>5.2f}  頭数中央 {s.field.median():>4.0f}  '
              f'軸3着内 {s.ax_hit.mean()*100:>5.1f}%  上位3頭 {s.top3_hits.mean():.2f} (無作為 {s.rand3.mean():.2f})')


def placebo(f, A, B):
    print('\n  --- 関門: レース内で確率をシャッフルした対照群（確認期）---')
    rng = np.random.default_rng(0)
    g = f[f.period=='確認'].copy()
    g['AIs'] = g.groupby(['win','race_id'])['AI'].transform(lambda s: rng.permutation(s.values))
    rs = race_struct(g.rename(columns={'AI':'AI_orig','AIs':'AI'}), 'AI')
    m = (rs.p1>=A)&(rs.cut15>=B)
    if m.sum()<20: print('    N不足'); return
    print(f'    該当{m.sum():>4}R  軸3着内 {rs.ax_hit[m].mean()*100:.1f}%  '
          f'全体 {rs.ax_hit.mean()*100:.1f}%  差 {(rs.ax_hit[m].mean()-rs.ax_hit.mean())*100:+.1f}pt  ← 0付近であるべき')


if __name__ == '__main__':
    f = build_frame()
    print(f'行 {len(f):,} / レース {f.race_id.nunique():,} / 探索 {(f.period=="探索").sum():,} 確認 {(f.period=="確認").sum():,}')
    q1_calibration(f)
    rs_ai  = race_struct(f, 'AI')
    rs_mkt = race_struct(f, 'MKT')
    print(f'\nレース数  AI {len(rs_ai):,}  MKT {len(rs_mkt):,}')
    q2(rs_ai, 'AI'); q2(rs_mkt, 'MKT')
    ab = q3(rs_ai, rs_mkt)
    q4(rs_ai)
    if ab: placebo(f, *ab)
    rs_ai.to_csv(f'{BASE}/cut_races.csv', index=False)

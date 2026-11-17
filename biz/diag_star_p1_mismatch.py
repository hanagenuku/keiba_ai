# 事後診断（カード外）。biz/RESULTS_HONMEI_STAR.md §2 の数字を出したコード。隔離ワークスペースで実行。
import sys,importlib.util,numpy as np,sqlite3
sys.argv=['x','/home/claude/keiba_ai']
s=importlib.util.spec_from_file_location('h','/home/claude/keiba_ai/biz/eval_honmei_star.py');H=importlib.util.module_from_spec(s);s.loader.exec_module(H)
S=H.S
df, cols, X, d, rid, hn, pop_tr, pop_ev, place, pay = S.load()
_, pairs, fits = S.fit_period(H.WINS, cols, X, d, rid, hn, pop_tr, pop_ev, place, '2026')
hc=sqlite3.connect('data/history.db')
fav={r:n for r,n in hc.execute("select race_id,horse_num from horse_history where popularity=1")}
pl={(r,n):p for r,n,p in hc.execute("select race_id,horse_num,place from horse_history")}
for name,_,_,_ in H.WINS:
    sel=S.select(S.picks(pairs,fits,name,42),'S1')
    same=[fav.get(r)==int(p1) for r,p1 in zip(sel['r'],sel['p1'])]
    t3=lambda r,n:(pl.get((r,int(n))) or 99)<=3
    st=[t3(r,n) for r,n in zip(sel['r'],sel['n'])]
    p1=[t3(r,p) for r,p in zip(sel['r'],sel['p1'])]
    f1=[t3(r,fav[r]) for r in sel['r'] if r in fav]
    print(name,'☆レース',len(sel),'評価のP1=実際の1番人気',round(np.mean(same)*100,1),'☆3着内',round(np.mean(st)*100,1),'評価のP1 3着内',round(np.mean(p1)*100,1),'実際の1番人気3着内',round(np.mean(f1)*100,1))

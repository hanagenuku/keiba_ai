# 事後診断（カード外）。biz/RESULTS_HONMEI_STAR.md §2 の数字を出したコード。隔離ワークスペースで実行。
import sys,os,importlib.util,numpy as np
sys.argv=['x','/home/claude/keiba_ai']
s=importlib.util.spec_from_file_location('h','/home/claude/keiba_ai/biz/eval_honmei_star.py');H=importlib.util.module_from_spec(s);s.loader.exec_module(H)
stars=H.star_set()
H.E.DB='data/keiba.db'
races=[r for r in H.E.load() if H.FROM<=r['date']<=H.TO]
import sqlite3
c=sqlite3.connect('data/keiba.db')
pop1={rid:n for rid,n in c.execute("select race_id,horse_num from race_predictions where popularity=1")}
place={(rid,n):p for rid,n,p in c.execute("select race_id,horse_num,actual_place from race_predictions")}
rows=[]
for r in races:
    s=stars.get(r['rid'])
    if not s: continue
    st=s[0]; ax=r['axis']; p1=pop1.get(r['rid'])
    f=lambda n: (place.get((r['rid'],n)) or 99)<=3
    rows.append((st==ax, f(st), f(ax), f(p1) if p1 else None, ax==p1))
a=np.array([x for x in rows],dtype=object)
for lab,m in (('全☆レース',np.ones(len(a),bool)),('☆=◎',a[:,0]==True),('☆≠◎',a[:,0]==False)):
    sub=a[m]; p1v=[x for x in sub[:,3] if x is not None]
    print(lab,len(sub),'☆3着内',round(np.mean(sub[:,1].astype(float))*100,1),'◎3着内',round(np.mean(sub[:,2].astype(float))*100,1),'1番人気3着内',round(np.mean(p1v)*100,1),'◎=1番人気',round(np.mean(sub[:,4].astype(float))*100,1))

"""M-1 関門3: 探索期で同じ配置を3シード回してシードstdを出す。

改善幅がシードstd以下なら、その差はシードの揺らぎと区別できない。
"""
import json, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import fit_one
from search import configs

TUNE = ['T1', 'T2', 'T3']
SEEDS = [42, 7, 2026]
W = None


def job(a):
    cid, cfg, wn, sd = a
    t = time.time()
    r = fit_one(W[wn], cfg, sd)
    r.update(cid=cid, win=wn, seed=sd, sec=round(time.time() - t, 1))
    return r


def main():
    global W
    cids = sorted({0} | {int(x) for x in sys.argv[1].split(',') if x.strip()})
    outp = sys.argv[2] if len(sys.argv) > 2 else 'mf1/seedstd.jsonl'
    W = pickle.load(open('mf1/data.pkl', 'rb'))['windows']
    CFG = configs()
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            done.add((r['cid'], r['win'], r['seed']))
    # seed42 は search.jsonl に既にあるので 7 と 2026 だけ
    jobs = [(c, CFG[c], wn, sd) for c in cids for wn in TUNE for sd in SEEDS[1:]
            if (c, wn, sd) not in done]
    print(f'{len(jobs)} フィット（配置 {cids}）', flush=True)
    with open(outp, 'a') as f, Pool(4) as p:
        for r in p.imap_unordered(job, jobs, chunksize=1):
            f.write(json.dumps(r, default=float) + '\n')
            f.flush()
            print(f'cid{r["cid"]} {r["win"]} seed{r["seed"]}: auc3 {r["auc3"]:.4f}',
                  flush=True)


if __name__ == '__main__':
    main()

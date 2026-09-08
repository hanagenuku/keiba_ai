"""M-1 確認: 探索期で選んだ配置を、確認期3窓(2026年)で一度だけ測る。

使い方: python mf1/confirm.py <cid,cid,...> [out.jsonl]
cid=0（現行既定）は常に含める。各配置 3窓 × 3シード。
"""
import json, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import fit_one, market_auc
from search import configs

CONF = ['W1', 'W2', 'W3']
SEEDS = [42, 7, 2026]
W = None


def job(a):
    cid, cfg, wn, sd = a
    t = time.time()
    r = fit_one(W[wn], cfg, sd)
    r.update(cid=cid, win=wn, seed=sd, cfg=cfg, sec=round(time.time() - t, 1))
    return r


def main():
    global W
    cids = sorted({0} | {int(x) for x in sys.argv[1].split(',') if x.strip()})
    outp = sys.argv[2] if len(sys.argv) > 2 else 'mf1/confirm.jsonl'
    W = pickle.load(open('mf1/data.pkl', 'rb'))['windows']
    CFG = configs()

    print('■ 市場だけの基準線（同じ窓・同じ母集団）')
    for wn in CONF:
        m = market_auc(W[wn])
        print(f'  {wn}: 3着内AUC {m["auc3"]:.4f}  1着AUC {m["auc1"]:.4f}  '
              f'N {m["n"]:,}/{m["n_all"]:,}')

    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            done.add((r['cid'], r['win'], r['seed']))
    jobs = [(c, CFG[c], wn, sd) for c in cids for wn in CONF for sd in SEEDS
            if (c, wn, sd) not in done]
    print(f'\n{len(jobs)} フィット（配置 {cids}）', flush=True)

    with open(outp, 'a') as f, Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=1), 1):
            f.write(json.dumps(r, default=float) + '\n')
            f.flush()
            print(f'{i}/{len(jobs)} cid{r["cid"]} {r["win"]} seed{r["seed"]}: '
                  f'auc3 {r["auc3"]:.4f} auc1 {r["auc1"]:.4f} ll {r["ll"]:.4f} '
                  f'ece {r["ece"]:.4f} 木{r["best"]}', flush=True)


if __name__ == '__main__':
    main()

"""M-1 探索: 探索期3窓(2025年)だけを見てハイパーパラメータを選ぶ。

確認期(2026年)には触れない。1配置ずつ JSONL に追記するので、
途中で止まっても再実行すれば続きから走る。
"""
import json, os, pickle, random, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import fit_one, DEFAULT

TUNE = ['T1', 'T2', 'T3']
SEED = 42
N_CFG = 120

SPACE = dict(
    max_depth=[3, 4, 5, 6, 8],
    learning_rate=[0.02, 0.05, 0.10],
    min_child_weight=[10, 30, 100, 300],
    subsample=[0.6, 0.8, 1.0],
    colsample_bytree=[0.4, 0.6, 0.8],
    reg_lambda=[1.0, 5.0, 20.0],
    reg_alpha=[0.0, 0.1, 1.0],
    spw=['prior', 'one'],
    es_metric=['logloss', 'auc'],
)

W = None


def configs():
    """現行既定を cid=0 に置き、以降はランダム探索（seed固定・重複除去）。"""
    out, seen = [dict(DEFAULT)], {json.dumps(DEFAULT, sort_keys=True)}
    rng = random.Random(20260908)
    while len(out) < N_CFG + 1:
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        k = json.dumps(c, sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def job(a):
    cid, cfg, wn = a
    t = time.time()
    r = fit_one(W[wn], cfg, SEED)
    r.update(cid=cid, win=wn, cfg=cfg, sec=round(time.time() - t, 1))
    return r


def main():
    global W
    data = sys.argv[1] if len(sys.argv) > 1 else 'mf1/data.pkl'
    outp = sys.argv[2] if len(sys.argv) > 2 else 'mf1/search.jsonl'
    W = pickle.load(open(data, 'rb'))['windows']

    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            done.add((r['cid'], r['win']))

    jobs = [(cid, cfg, wn) for cid, cfg in enumerate(configs())
            for wn in TUNE if (cid, wn) not in done]
    print(f'{len(jobs)} フィット残り（済 {len(done)}）', flush=True)

    t0 = time.time()
    with open(outp, 'a') as f, Pool(4) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs, chunksize=1), 1):
            f.write(json.dumps(r, default=float) + '\n')
            f.flush()
            if i % 15 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f'{i}/{len(jobs)}  経過{el/60:.0f}分  '
                      f'残り約{el/i*(len(jobs)-i)/60:.0f}分', flush=True)


if __name__ == '__main__':
    main()

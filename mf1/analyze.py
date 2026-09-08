"""M-1 探索結果の集計。探索期3窓の平均で並べ、現行既定との差を出す。"""
import json, sys
from collections import defaultdict
import numpy as np

TUNE = ['T1', 'T2', 'T3']
KEYS = ['auc3', 'auc1', 'll', 'brier', 'ece', 'best']


def load(path):
    by = defaultdict(dict)
    cfgs = {}
    for line in open(path):
        r = json.loads(line)
        by[r['cid']][r['win']] = r
        cfgs[r['cid']] = r['cfg']
    return {c: v for c, v in by.items() if len(v) == len(TUNE)}, cfgs


def agg(v):
    return {k: float(np.mean([v[w][k] for w in TUNE])) for k in KEYS}


def fmt(c):
    return (f"d{c['max_depth']} lr{c['learning_rate']} mcw{c['min_child_weight']} "
            f"sub{c['subsample']} col{c['colsample_bytree']} "
            f"L2:{c['reg_lambda']:g} L1:{c['reg_alpha']:g} "
            f"{'spw' if c['spw']=='prior' else 'spw1'} ES:{c['es_metric']}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'mf1/search.jsonl'
    by, cfgs = load(path)
    A = {c: agg(v) for c, v in by.items()}
    base = A.get(0)
    print(f'完走 {len(A)} 配置 / 121\n')
    if base:
        print('現行既定(cid=0) 探索期3窓平均: ' +
              '  '.join(f'{k} {base[k]:.4f}' for k in KEYS))
        for w in TUNE:
            print(f'  {w}: auc3 {by[0][w]["auc3"]:.4f}  auc1 {by[0][w]["auc1"]:.4f}  '
                  f'木{by[0][w]["best"]}')
        print()

    order = sorted(A, key=lambda c: -A[c]['auc3'])
    print(f'{"順":>3} {"cid":>4} {"auc3":>7} {"Δ既定":>8} {"auc1":>7} {"LogLoss":>8} '
          f'{"ECE":>7} {"木":>6}  配置')
    for i, c in enumerate(order[:25], 1):
        a = A[c]
        d = a['auc3'] - base['auc3'] if base else float('nan')
        print(f'{i:>3} {c:>4} {a["auc3"]:.4f} {d:+8.4f} {a["auc1"]:.4f} '
              f'{a["ll"]:8.4f} {a["ece"]:7.4f} {a["best"]:6.0f}  {fmt(cfgs[c])}')
    if base:
        r = order.index(0) + 1
        print(f'\n現行既定の順位: {r}/{len(A)}')

    # 各軸の周辺平均（どのパラメータが効いているか）
    print('\n■ パラメータごとの探索期 auc3 平均')
    for k in ['max_depth', 'learning_rate', 'min_child_weight', 'subsample',
              'colsample_bytree', 'reg_lambda', 'reg_alpha', 'spw', 'es_metric']:
        g = defaultdict(list)
        for c in A:
            g[cfgs[c][k]].append(A[c]['auc3'])
        s = '  '.join(f'{v}:{np.mean(g[v]):.4f}(n{len(g[v])})' for v in sorted(g, key=str))
        print(f'  {k:18s} {s}')


if __name__ == '__main__':
    main()

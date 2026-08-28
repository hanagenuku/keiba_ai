"""collect_odds.py が書いた JSONL を keiba.db に取り込む。

■ なぜ収集と取り込みを分けるのか
collect-odds は土日 09:20〜14:50 JST の5.5時間走る。その間に weekend.yml の
refresh が 11:30 / 14:00 に走り、同じ data/keiba.db を push する。
収集ジョブがジョブ開始時の keiba.db を最後に上書き push すると、
**その間の race_predictions / prediction_snapshots が黙って消える**
（SQLiteはバイナリなので git がマージできない）。

そこで収集側は自分専用の追記JSONLだけを書き、keiba.db への取り込みは
keiba.db の持ち主である sunday-results.yml から一度だけ行う。

■ 冪等
save_odds_snapshots / save_race_schedule はどちらも UNIQUE 制約を持つので
同じ行を二度入れても増えない。取り込みに成功した JSONL のみ削除する。
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import init_db, save_odds_snapshots, save_race_schedule  # noqa: E402

TS_DIR = 'odds_ts'


def merge_odds_ts(base_dir='.', keep=False):
    d = os.path.join(base_dir, 'data', TS_DIR)
    files = sorted(glob.glob(os.path.join(d, '*.jsonl')))
    if not files:
        print('📥 取り込む JSONL はありません')
        return {'files': 0, 'odds': 0, 'schedule': 0}
    init_db(base_dir)
    tot = {'files': 0, 'odds': 0, 'schedule': 0}
    for path in files:
        odds, sched, bad = [], [], 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # 収集中に落ちると最終行が欠けることがある。1行捨てて続ける。
                    bad += 1
                    continue
                kind = r.pop('kind', None)
                if kind == 'odds':
                    odds.append(r)
                elif kind == 'schedule':
                    sched.append(r)
                else:
                    bad += 1
        n_s = save_race_schedule(sched, base_dir=base_dir) if sched else 0
        n_o = save_odds_snapshots(odds, base_dir=base_dir) if odds else 0
        tot['files'] += 1
        tot['odds'] += n_o
        tot['schedule'] += n_s
        print(f'  {os.path.basename(path)}: オッズ {len(odds)}行→新規{n_o} / '
              f'発走時刻 {len(sched)}行→新規{n_s}' + (f' / 壊れた行 {bad}' if bad else ''))
        if not keep:
            os.remove(path)
    print(f'📥 取り込み完了: {tot["files"]}ファイル / '
          f'オッズ新規 {tot["odds"]} / 発走時刻新規 {tot["schedule"]}')
    return tot


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--keep', action='store_true', help='取り込んでもJSONLを消さない')
    a = ap.parse_args()
    merge_odds_ts(a.base_dir, keep=a.keep)

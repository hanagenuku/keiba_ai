#!/usr/bin/env python3
"""data/latest.json のgit履歴から displayed_bets を埋め戻す。

displayed_bets を導入する前（2026-08-31以前）の表示買い目は、
`latest.json` のgit履歴にしか残っていない。1開催日に複数世代あるので、
**最もレース数が多い世代**を採る（refreshで一部だけ書き換わった世代より
その日の全レースが揃っている世代のほうが記録として正しい）。

一度だけ流す想定。既存行は UNIQUE で上書きされるので再実行しても安全。
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betting.displayed_bets import rows_from_app_json          # noqa: E402
from src.utils.db import (init_db, save_displayed_bets,             # noqa: E402
                          settle_displayed_bets)


def _generations(repo):
    shas = subprocess.run(['git', '-C', repo, 'log', '--format=%H', '--all',
                           '--', 'data/latest.json'],
                          capture_output=True, text=True).stdout.split()
    best = {}
    for sha in shas:
        out = subprocess.run(['git', '-C', repo, 'show', f'{sha}:data/latest.json'],
                             capture_output=True, text=True).stdout
        try:
            data = json.loads(out)
        except Exception:
            continue
        raw = data.get('races') or {}
        flat = ([r for v in raw.values() for r in v]
                if isinstance(raw, dict) else raw)
        if not flat:
            continue
        day = (flat[0].get('race_id') or '')[:8]
        if not day:
            continue
        if day not in best or len(flat) > best[day][0]:
            best[day] = (len(flat), data)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--repo', default=None,
                    help='git履歴を読むリポジトリ（既定: --base-dir）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    init_db(args.base_dir)
    best = _generations(args.repo or args.base_dir)
    print(f'📚 latest.json のgit履歴: {len(best)}開催日')

    total = 0
    for day in sorted(best):
        n_races, data = best[day]
        rows = rows_from_app_json(data, snapshot='initial')
        if not rows:
            print(f'  {day}  {n_races:>3}R  gumbel_bets なし（旧フォーマット）')
            continue
        if not args.dry_run:
            save_displayed_bets(rows, args.base_dir)
        total += len(rows)
        print(f'  {day}  {n_races:>3}R  {len(rows):>4}点')

    print(f'\n✅ 合計 {total}点')
    if args.dry_run:
        return
    r = settle_displayed_bets(args.base_dir)
    roi = (r['recovered'] / r['invested'] * 100) if r['invested'] else 0
    print(f'決済: {r["hit"]}/{r["settled"]}的中  '
          f'¥{r["invested"]:,.0f} → ¥{r["recovered"]:,.0f} = {roi:.1f}%'
          + (f'  ⚠配当引けず{r["no_payout"]}件' if r['no_payout'] else ''))


if __name__ == '__main__':
    main()

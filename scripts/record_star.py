#!/usr/bin/env python3
"""☆（1番人気を上回る可能性）と、その後の着順を `data/star_log.json` に残す。

☆は 2026-11-11 から本番に出ているが、予想時点の確率を残す仕組みが無いと
「☆は当たっているのか」に後から答えられない（`latest.json` は上書きされる）。

🔴 本番の予想・モデル・買い目には触らない。読むのは `data/latest.json` と
   `data/history.db`（着順）と `data/keiba.db`（複勝払戻・発走時刻）だけ。
🔴 ☆の判定は再実装しない（`latest.json` の `race.star` をそのまま読む）。
🔴 回収率は出さない。残すのは着順・3着内か・複勝払戻・1番人気の着順という事実だけ。
⚠ **台帳が溜まるまで「☆の成績」を名乗らない**（`biz/S4_HITRATE_AXIS.md` §5）。

使い方:
    python3 scripts/record_star.py                  # 朝: ☆を記録
    python3 scripts/record_star.py --attach-results  # 夜: 着順を追記
    python3 scripts/record_star.py --summary         # 集計（事実だけ）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import results as results_mod, star_log  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--latest', default='',
                    help='読む latest.json（既定は <base-dir>/data/latest.json）')
    ap.add_argument('--attach-results', action='store_true',
                    help='確定した着順・複勝払戻を追記する（記録はしない）')
    ap.add_argument('--summary', action='store_true', help='集計を出して終わる')
    ap.add_argument('--include-late', action='store_true',
                    help='--summary で発走後に記録された分も含める（既定は除く）')
    ap.add_argument('--backfill-source', default='',
                    help='過去分を取り込むときの出典')
    ap.add_argument('--dry-run', action='store_true', help='書かずに件数だけ出す')
    args = ap.parse_args(argv)
    base_dir = args.base_dir

    if args.summary:
        s = star_log.summarize(base_dir, include_late=args.include_late)
        print(f'☆ {s["races"]}レース（{s["days"]}開催日・'
              f'{s["date_from"]}〜{s["date_to"]}）/ 結果あり {s["with_result"]}件 / '
              f'結果待ち {s["pending"]}件 / 発走後の記録で除外 {s["late_excluded"]}件')
        if s['with_result']:
            print(f'1番人気を上回った: {s["beat_p1"]}/{s["with_result"]}'
                  f'（{s["beat_p1_rate"]}%）')
            print(f'  参考: ☆の3着内 {s["star_top3"]}/{s["with_result"]}'
                  f'（{s["star_top3_rate"]}%）/ '
                  f'1番人気の3着内 {s["p1_top3"]}/{s["with_result"]}'
                  f'（{s["p1_top3_rate"]}%）')
        print('⚠ 記録であって成績ではない。N が小さいうちの偏りを読まないこと'
              '（North Star #7）。回収率はここでは出さない', file=sys.stderr)
        return 0

    if args.attach_results:
        keiba = results_mod.open_keiba(base_dir)
        hist = results_mod.open_history(base_dir)
        if keiba is None or hist is None:
            # 🔴 払戻・着順が引けない状態で「結果」を書くと全部外れとして残る
            print('⚠ data/keiba.db または data/history.db の実体が無いため'
                  '結果追記をスキップしました（LFS 実体を取得してから実行）',
                  file=sys.stderr)
            return 0
        r = star_log.attach_results(base_dir, hist, keiba, dry_run=args.dry_run)
        print(f'☆の結果追記 {r["attached"]}件 / 既存 {r["already"]}件 / '
              f'結果待ち {r["pending"]}件', file=sys.stderr)
        if r['pending_races']:
            print('結果待ちのレース: '
                  + ', '.join(sorted(set(r['pending_races']))), file=sys.stderr)
        if r['late']:
            print('🔴 発走後に記録された☆（集計から除くこと）: '
                  + ', '.join(r['late']), file=sys.stderr)
        return 0

    path = args.latest or os.path.join(base_dir, 'data', 'latest.json')
    if not os.path.exists(path):
        print(f'⚠ {path} が無いため記録をスキップしました', file=sys.stderr)
        return 0
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if args.dry_run:
        found = star_log.collect_from_latest(data)
        print(f'（dry-run）☆ {len(found)}件', file=sys.stderr)
        for e in found:
            print(f'  {e["date"]} {e["venue"]}{e["race_num"]}R #{e["num"]}'
                  f' {e["name"]}（{e["pop"]}番人気{e["odds"]}倍・'
                  f'上回る確率{e["prob"]}・1番人気は#{e["p1_num"]}）', file=sys.stderr)
        return 0
    r = star_log.record(base_dir, data,
                        backfill_source=args.backfill_source or None)
    print(f'☆を記録: 追記 {r["added"]}件 / 既存 {r["already"]}件 / '
          f'台帳 {r["total"]}件', file=sys.stderr)
    for e in r['entries']:
        print(f'  {e["date"]} {e["venue"]}{e["race_num"]}R #{e["num"]} {e["name"]}'
              f'（{e["pop"]}番人気{e["odds"]}倍・上回る確率{e["prob"]}）',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

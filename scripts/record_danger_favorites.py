#!/usr/bin/env python3
"""「危険な人気馬」の警告と、その後の着順を `data/danger_favorites.json` に残す。

商品の2本柱の片方（市場1番人気のうち素のAIが下位＝複勝率 68.5%→51.4%）が
当たったか外れたかを、開催のたびに自動で積む。2026-10-10〜12 の初運用では
この器が無く、「4戦4敗」は手作業で突き合わせて出した数字だった。

🔴 本番の予想・モデル・買い目には触らない。読むのは `data/latest.json` と
   `data/history.db`（着順）と `data/keiba.db`（複勝払戻・発走時刻）だけ。
🔴 判定条件は再実装しない（`src/product/marks.py::danger_favorites` を呼ぶ）。
🔴 回収率は出さない。残すのは着順・3着内か・複勝払戻という事実だけ。

使い方:
    python3 scripts/record_danger_favorites.py                 # 朝: 警告を記録
    python3 scripts/record_danger_favorites.py --attach-results # 夜: 着順を追記
    python3 scripts/record_danger_favorites.py --summary        # 集計（事実だけ）
    python3 scripts/record_danger_favorites.py --latest path.json \
        --backfill-source 'latest.json@f906c6e'                 # 過去分の取り込み
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import danger_log, results as results_mod  # noqa: E402


def _load_latest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--latest', default='',
                    help='読む latest.json（既定は <base-dir>/data/latest.json）')
    ap.add_argument('--attach-results', action='store_true',
                    help='確定した着順・複勝払戻を追記する（記録はしない）')
    ap.add_argument('--summary', action='store_true', help='集計を出して終わる')
    ap.add_argument('--scope-only', action='store_true',
                    help='--summary で記事の対象範囲（9R〜11R＋自信のある平場）だけ数える')
    ap.add_argument('--backfill-source', default='',
                    help='過去分を取り込むときの出典（entry に backfilled 印を付ける）')
    ap.add_argument('--dry-run', action='store_true', help='書かずに件数だけ出す')
    args = ap.parse_args(argv)

    base_dir = args.base_dir

    if args.summary:
        s = danger_log.summarize(base_dir, scope_only=args.scope_only)
        print(f'警告 {s["total"]}件（{len(s["dates"])}開催日）'
              f' / 結果あり {s["with_result"]}件 / 結果待ち {s["pending"]}件'
              f' / 発走後の記録で除外 {s["excluded_late"]}件')
        print(f'3着内 {s["top3"]}/{s["with_result"]}')
        # 🔴 ◎ 自身だった分を必ず分けて出す（2026-10-15 の実測で 9/11 が ◎ 自身）
        print(f'  うち ◎と同じ馬（rl_rank=1） {s["axis_top3"]}/{s["axis_n"]}'
              f' ／ ◎ではない {s["other_top3"]}/{s["other_n"]}')
        if s['places']:
            print(f'着順: {s["places"]}')
        if s['fukusho_payouts']:
            print(f'3着内だった馬の複勝払戻(100円あたり): {s["fukusho_payouts"]}')
        # 🔴 回収率は出さない（賭け金の仮定が必要になる）。N の小ささも毎回書く
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
        r = danger_log.attach_results(base_dir, hist, keiba, dry_run=args.dry_run)
        print(f'結果追記 {r["attached"]}件 / 既存 {r["already"]}件 / '
              f'結果待ち {r["pending"]}件', file=sys.stderr)
        if r['pending_races']:
            print('結果待ちのレース: '
                  + ', '.join(sorted(set(r['pending_races']))), file=sys.stderr)
        if r['late']:
            # 発走後に条件を満たした＝予告ではない。集計から外すための印
            print('🔴 発走後に記録された警告（集計から除くこと）: '
                  + ', '.join(r['late']), file=sys.stderr)
        return 0

    path = args.latest or os.path.join(base_dir, 'data', 'latest.json')
    if not os.path.exists(path):
        print(f'⚠ {path} が無いため記録をスキップしました', file=sys.stderr)
        return 0
    data = _load_latest(path)
    if args.dry_run:
        found = danger_log.collect_from_latest(data)
        print(f'（dry-run）危険な人気馬 {len(found)}件', file=sys.stderr)
        for e in found:
            print(f'  {e["date"]} {e["venue"]}{e["race_num"]}R #{e["num"]}'
                  f' {e["name"]}（{e["pop"]}番人気{e["odds"]}倍・AI単独{e["solo_rank"]}位'
                  f'・{"範囲内" if e["in_scope"] else "範囲外"}）', file=sys.stderr)
        return 0
    r = danger_log.record(base_dir, data,
                          backfill_source=args.backfill_source or None)
    print(f'危険な人気馬を記録: 追記 {r["added"]}件 / 既存 {r["already"]}件',
          file=sys.stderr)
    for e in r['entries']:
        print(f'  {e["date"]} {e["venue"]}{e["race_num"]}R #{e["num"]} {e["name"]}'
              f'（{e["pop"]}番人気{e["odds"]}倍・AI単独{e["solo_rank"]}位）',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

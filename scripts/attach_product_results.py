#!/usr/bin/env python3
"""公開済みの予想（`prediction_archive/`）に確定結果を追記する。

🔴 予想部分は1バイトも変えない（`archive.attach_result()` が凍結キーを検算し、
   変わっていたら `AssertionError` を投げる）。足すのは `result` だけ。
🔴 **回収率は出さない。** 保存するのは着順と払戻（100円あたり）という事実だけ。
   理由は `src/product/results.py` の冒頭に書いてある（金額は記事では換算例なので、
   回収率には賭け金の仮定が必要になり、それを「結果」として永久保存できない）。
🔴 結果がまだ取れていないレースは**何も書かない**（空の結果で埋めない）。
   次回の実行で改めて拾われる。

本番の予想・モデル・買い目には触らない。読むのは
`prediction_archive/` と `data/history.db`（着順）と `data/keiba.db`（払戻）だけ。

使い方:
    python3 scripts/attach_product_results.py              # 公開済み全日付
    python3 scripts/attach_product_results.py --date 2026-10-04
    python3 scripts/attach_product_results.py --dry-run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import results as results_mod  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--date', help='YYYY-MM-DD（省略時は公開済みの全日付）')
    ap.add_argument('--dry-run', action='store_true',
                    help='追記せず件数だけ出す')
    args = ap.parse_args(argv)

    base_dir = args.base_dir
    dates = [args.date] if args.date else results_mod.published_dates(base_dir)
    if not dates:
        # 公開実績ゼロ（2026-10-02 時点の状態）。これは異常ではない
        print('公開済みの予想がありません（prediction_archive/ が空）',
              file=sys.stderr)
        return 0

    hist = results_mod.open_history(base_dir)
    keiba = results_mod.open_keiba(base_dir)
    if keiba is None:
        # 払戻が引けないのに「結果」を書くと全部外れとして記録される
        print('🔴 中止: data/keiba.db の実体が無い（払戻が引けない）。'
              'LFS 実体を取得してから実行すること', file=sys.stderr)
        return 1
    if hist is None:
        print('⚠ data/history.db の実体が無い。着順を記録できないため中止',
              file=sys.stderr)
        return 1

    total = {'attached': 0, 'already': 0, 'pending': 0}
    mismatch = []
    for date_str in dates:
        r = results_mod.attach_for_date(base_dir, date_str, hist, keiba,
                                       dry_run=args.dry_run)
        for k in total:
            total[k] += r[k]
        mismatch += [(date_str, rid) for rid in r['mismatch']]
        print(f'{date_str}: 追記 {r["attached"]} / 既存 {r["already"]} / '
              f'結果待ち {r["pending"]}', file=sys.stderr)

    if mismatch:
        # 着順（history.db）と払戻（keiba.db）が食い違った。どちらかが壊れている
        print('🔴 検算が合わないレース（着順とワイド払戻の3着内が不一致）: '
              + ', '.join(f'{d}/{r}' for d, r in mismatch), file=sys.stderr)
    print(f'合計: 追記 {total["attached"]} / 既存 {total["already"]} / '
          f'結果待ち {total["pending"]}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

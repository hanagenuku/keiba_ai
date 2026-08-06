#!/usr/bin/env python3
"""列崩れで壊れた popularity / trainer / body_weight を再スクレイプで復旧する。

## なぜ要るか

2026-07-04 以降、JRAの結果ページの列構成が変わり（2026-08-03②③で発見・修正）、
`popularity` / `trainer` / `body_weight` の3列が取得できなくなっていた。
パーサー自体は 2026-08-03③ で修正済みだが、**既にDBに入ってしまった行は
直っていない**。

特に `popularity` は深刻で、取得失敗時に NULL ではなく **99（不明を表す
センチネル値）** で保存される。`engine.py` は
`float(pop) if 0 < pop < 99 else nan` で 99 を NaN に落とすため、
残差学習モデルの **base_margin（市場アンカー）が完全に死ぬ**。

実測した影響（2026-08-05調査、walk-forward検証 N=28,247）:

    アンカー健全な行  N=20,952  AUC 0.8207
    アンカー死亡行    N= 7,295  AUC 0.7091   ← 差 -0.11

しかも本番モデル（PR #143）の検証窓 2026-06-06〜08-02 は
**7,733行中4,484行＝58%がアンカー死亡状態**だった。つまり現行モデルを
採用した根拠の数字自体が壊れたデータの上で測られている。

## 前提となるコード修正（同時に入っている）

`save_history_db()` の UPDATE 節には元々 `popularity` / `trainer` が
含まれておらず、INSERT も `INSERT OR IGNORE` なので既存行はスキップされる。
つまり **再スクレイプしても永久に直らない**状態だった。
`src/utils/db.py` 側を先に修正してあり、

  - `popularity` は「新しい値が正当な人気(1〜98)の時だけ上書き」
    （取得失敗の99で既存の正しい値を壊さない）
  - `trainer` は `COALESCE(NULLIF(?, ''), trainer)`

としている。このスクリプトはその修正が入っていることが前提。

## 安全設計

- **history.db にしか書き込まない。** keiba.db・モデル・latest.json は無変更
- `save_history_db()` は `INSERT OR IGNORE` + `UPDATE` なので、
  **既存の正しいデータを壊さない**（新規レースが増えることもない）
- **1回の実行に上限**（`--budget` 日数 と `--max-minutes` の両方。
  先に来たほうで打ち切る）。上限なしのスクレイピングでCIタイムアウトを
  起こしその回のデータを失った事故が 2026-07-18 に実際にある（North Star #4）
- **1日ぶん保存するたびに時間を確認**するので、途中で強制終了されて
  成果を失うことがない
- **復旧済みの日はスキップ**するので何回でも追加実行できる
- 1日分が失敗しても次の日に進む
- `fetch_results` に `hist_db_path` を渡さない = 血統の追加リクエストをしない
  （復旧が目的なので余計なリクエストを増やさない）

## 使い方

    python scripts/repair_popularity.py --dry-run      # 対象日を数えるだけ
    python scripts/repair_popularity.py --budget 10
"""
import argparse
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._session import create_session
from src.scraper.jra_scraper import fetch_results
from src.utils.db import save_history_db

DEFAULT_BUDGET = 10
DEFAULT_MAX_MINUTES = 25
# この割合以上の馬が popularity=99（不明）ならその開催日は「壊れている」と見なす。
# 単発の取消馬などで数頭が99になるのは正常なので、高めに取る。
BROKEN_RATIO = 0.5


def find_broken_dates(hist_path, broken_ratio=BROKEN_RATIO):
    """popularity が壊れている開催日を YYYYMMDD で返す（古い順）。

    Returns: [(date_yyyymmdd, n_rows, n_broken), ...]
    """
    conn = sqlite3.connect(hist_path)
    try:
        rows = conn.execute("""
            SELECT REPLACE(date, '-', '') AS d,
                   COUNT(*) AS n,
                   SUM(CASE WHEN popularity = 99 OR popularity IS NULL
                            THEN 1 ELSE 0 END) AS bad
            FROM horse_history
            WHERE date IS NOT NULL AND date != ''
            GROUP BY d
            ORDER BY d
        """).fetchall()
    finally:
        conn.close()

    out = []
    for d, n, bad in rows:
        d = str(d)[:8]
        if len(d) != 8 or not d.isdigit() or not n:
            continue
        if bad / n >= broken_ratio:
            out.append((d, n, bad))
    return out


def repair(base_dir, budget=DEFAULT_BUDGET, dry_run=False,
           max_minutes=DEFAULT_MAX_MINUTES, broken_ratio=BROKEN_RATIO,
           oldest_first=False):
    hist_path = os.path.join(base_dir, 'data', 'history.db')
    if not os.path.exists(hist_path):
        raise SystemExit(f'history.db が見つかりません: {hist_path}')

    todo = find_broken_dates(hist_path, broken_ratio)
    # 既定は新しい順。JRAの結果一覧ページは直近ぶんしか載っておらず
    # （2026-08-03の配当埋め戻しで「2026-06-06より前に行けない」と実測済み）、
    # 古い日から処理すると取得できない日でbudgetを使い切ってしまう。
    # 直近の日ほど復旧できる可能性が高く、かつ検証窓に直結するので優先する。
    if not oldest_first:
        todo = list(reversed(todo))
    total_bad = sum(b for _, _, b in todo)
    print(f'popularity が壊れている開催日: {len(todo)}日 / 対象行 {total_bad:,}行')
    if not todo:
        print('✅ 復旧が必要な開催日はない')
        return {'processed': 0, 'repaired': 0, 'remaining': 0}

    for d, n, bad in todo[:budget]:
        print(f'    {d}: {bad}/{n}行が不明({bad/n*100:.0f}%)')
    if len(todo) > budget:
        print(f'    …ほか{len(todo)-budget}日')
    print(f'  今回の対象: 先頭{min(budget, len(todo))}日 / 時間上限 {max_minutes}分')

    if dry_run:
        print('  --dry-run のため取得は行わない')
        return {'processed': 0, 'repaired': 0, 'remaining': len(todo)}

    sess = create_session()
    processed = 0
    t0 = time.time()
    for date, _n, _bad in todo[:budget]:
        try:
            # hist_db_path を渡さない = 血統の追加リクエストをしない
            results = fetch_results(sess, date)
        except Exception as e:
            print(f'  ⚠ {date}: 結果取得に失敗 — {type(e).__name__}: {e}')
            continue
        if not results:
            print(f'  ⚠ {date}: 0レース（次回また対象になる）')
            continue

        before = _count_broken(hist_path, date)
        try:
            # INSERT OR IGNORE + UPDATE。既存行の popularity/trainer/body_weight を
            # 修正後のパーサーで取り直した値で埋め直す
            save_history_db(results, db_path=hist_path)
        except Exception as e:
            print(f'  ⚠ {date}: 保存に失敗 — {type(e).__name__}: {e}')
            continue
        after = _count_broken(hist_path, date)

        if after >= before:
            # 取り直しても直っていない＝済み扱いにせず次回また対象にする
            print(f'  ⚠ {date}: {len(results)}レース取得したが復旧せず '
                  f'({before}→{after}行)（次回また対象になる）')
            continue
        processed += 1
        print(f'  ✅ {date}: {len(results)}レース → 不明 {before}→{after}行 '
              f'({before - after}行を復旧)')

        elapsed = (time.time() - t0) / 60
        if elapsed >= max_minutes:
            print(f'  ⏱ 時間上限 {max_minutes}分に到達したのでここで止める'
                  f'（復旧済みぶんは残る）')
            break

    remaining = len(find_broken_dates(hist_path, broken_ratio))
    print(f'\n完了: {processed}日を復旧 / 残り{remaining}日 '
          f'({(time.time()-t0)/60:.1f}分)')
    if remaining:
        print('  ※ 残りは次回の実行で埋まる（同じコマンドをもう一度）')
    return {'processed': processed, 'repaired': total_bad, 'remaining': remaining}


def _count_broken(hist_path, date_yyyymmdd):
    """その開催日で popularity が不明のままの行数。"""
    conn = sqlite3.connect(hist_path)
    try:
        row = conn.execute("""
            SELECT SUM(CASE WHEN popularity = 99 OR popularity IS NULL
                            THEN 1 ELSE 0 END)
            FROM horse_history
            WHERE REPLACE(date, '-', '') = ?
        """, (date_yyyymmdd,)).fetchone()
    finally:
        conn.close()
    return int(row[0] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                    help=f'1回に処理する開催日数の上限（既定{DEFAULT_BUDGET}日）')
    ap.add_argument('--max-minutes', type=float, default=DEFAULT_MAX_MINUTES,
                    help=f'1回の実行の時間上限（既定{DEFAULT_MAX_MINUTES}分）')
    ap.add_argument('--broken-ratio', type=float, default=BROKEN_RATIO,
                    help=f'この割合以上が不明なら壊れていると判定（既定{BROKEN_RATIO}）')
    ap.add_argument('--oldest-first', action='store_true',
                    help='古い開催日から処理する（既定は新しい順＝取得できる可能性が高い順）')
    ap.add_argument('--dry-run', action='store_true',
                    help='対象日を表示するだけで取得しない')
    args = ap.parse_args()
    repair(ROOT, budget=args.budget, dry_run=args.dry_run,
           max_minutes=args.max_minutes, broken_ratio=args.broken_ratio,
           oldest_first=args.oldest_first)


if __name__ == '__main__':
    main()

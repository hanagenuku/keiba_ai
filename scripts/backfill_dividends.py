#!/usr/bin/env python3
"""過去レースの全券種配当を `race_dividends` に埋め戻す。

## なぜ要るか

`parse_dividends()` は昔から8券種すべて（単勝・複勝・枠連・馬連・馬単・ワイド・
三連複・三連単）を解析していたのに、DBに保存していたのは単勝・複勝だけだった。
2026-07-31 に `race_dividends` を新設して以降のレースは保存されるようになったが、
**それ以前の約5,400レースぶんは配当が残っていない**。

そのため「馬連・ワイド・三連複で回収率100%を超える買い方があるか」を
過去データで検証できない。実測できたのは `shadow_bets` が持つ
「AI上位3頭で1点ずつ」という**特定の1通り**だけで、しかも271レースしかなかった。

このスクリプトは過去の結果ページを取り直して配当だけを保存する。
埋まれば `horse_history` の着順（1〜3着が99.2%完備）と突き合わせて、
**任意の組み合わせ**の回収率を5,400レース規模で測れるようになる。

## 安全設計

- **`race_dividends` 以外には一切書き込まない。** history.db・モデル・latest.json は無変更
- **1回の実行に上限を設ける**（`--budget` 日数 と `--max-minutes` 時間の両方。
  先に来たほうで打ち切る）。上限なしで新規スクレイピングを走らせてCIタイムアウトを
  起こし、その回のデータを丸ごと失った事故が 2026-07-18 に実際に起きている
  （North Star #4）。**1日ぶん保存するたびに時間を確認して、超えていれば
  きれいに止まる**ので、途中で強制終了されて成果を失うことがない
- **既に配当が入っている開催日はスキップ**するので、何回でも追加実行できる。
  数回に分けて少しずつ埋める運用を想定している
- 1日分が失敗しても次の日に進む（1日の失敗で全体を止めない）

## 使い方

    python scripts/backfill_dividends.py --budget 8
    python scripts/backfill_dividends.py --budget 8 --oldest-first
    python scripts/backfill_dividends.py --dry-run     # 対象日を数えるだけ
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
from src.utils.db import save_dividends_db

DEFAULT_BUDGET = 30
DEFAULT_MAX_MINUTES = 25


def _hist_dates(hist_path):
    """history.db にあるレース開催日を YYYYMMDD の集合で返す。"""
    conn = sqlite3.connect(hist_path)
    try:
        rows = conn.execute('SELECT DISTINCT date FROM race_history').fetchall()
    finally:
        conn.close()
    out = set()
    for (d,) in rows:
        if not d:
            continue
        s = str(d).replace('-', '')[:8]
        if len(s) == 8 and s.isdigit():
            out.add(s)
    return out


def _done_dates(keiba_path):
    """race_dividends に既に配当が入っている開催日を返す。

    race_id は 'YYYYMMDD_VV_RR' 形式なので先頭8文字が開催日。
    """
    if not os.path.exists(keiba_path):
        return set()
    conn = sqlite3.connect(keiba_path)
    try:
        tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='race_dividends'"
        ).fetchone()
        if not tbl:
            return set()
        rows = conn.execute(
            'SELECT DISTINCT substr(race_id, 1, 8) FROM race_dividends').fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows if r[0]}


def backfill(base_dir, budget=DEFAULT_BUDGET, oldest_first=False, dry_run=False,
             max_minutes=DEFAULT_MAX_MINUTES):
    hist_path = os.path.join(base_dir, 'data', 'history.db')
    keiba_path = os.path.join(base_dir, 'data', 'keiba.db')
    if not os.path.exists(hist_path):
        raise SystemExit(f'history.db が見つかりません: {hist_path}')

    have = _hist_dates(hist_path)
    done = _done_dates(keiba_path)
    todo = sorted(have - done, reverse=not oldest_first)

    print(f'history.db の開催日 {len(have)}日 / 配当済み {len(done)}日 '
          f'→ 未取得 {len(todo)}日')
    if not todo:
        print('✅ すべての開催日で配当を取得済み')
        return {'processed': 0, 'saved': 0, 'remaining': 0}
    print(f'  今回の対象: 先頭{min(budget, len(todo))}日 '
          f'({"古い順" if oldest_first else "新しい順"}) / 時間上限 {max_minutes}分')

    if dry_run:
        print('  --dry-run のため取得は行わない')
        print(f'  対象日の例: {todo[:budget]}')
        return {'processed': 0, 'saved': 0, 'remaining': len(todo)}

    sess = create_session()
    processed = saved = 0
    t0 = time.time()
    for date in todo[:budget]:
        try:
            # hist_db_path は渡さない = 血統の追加リクエストをしない（配当だけが目的）
            results = fetch_results(sess, date)
        except Exception as e:
            print(f'  ⚠ {date}: 結果取得に失敗 — {type(e).__name__}: {e}')
            continue
        if not results:
            # 開催がない・ページが落ちている等。次回の実行で再挑戦される
            print(f'  ⚠ {date}: 0レース（次回また対象になる）')
            continue
        try:
            # {'races': 保存対象レース数, 'rows': 追加行数} を返す
            stat = save_dividends_db(results, base_dir=base_dir)
        except Exception as e:
            print(f'  ⚠ {date}: 配当の保存に失敗 — {type(e).__name__}: {e}')
            continue
        if not stat.get('races'):
            # 配当が1件も取れていない＝取り直す価値があるので済み扱いにしない
            print(f'  ⚠ {date}: {len(results)}レース取得したが配当が空'
                  f'（次回また対象になる）')
            continue
        processed += 1
        saved += stat['rows']
        print(f'  ✅ {date}: {len(results)}レース → '
              f'{stat["races"]}レースぶん・配当{stat["rows"]}行を保存')

        # 1日ぶん保存し終えた区切りでだけ時間を見る。
        # 途中で強制終了されるとその日の取得が無駄になるため、必ず保存後に判定する。
        elapsed = (time.time() - t0) / 60
        if elapsed >= max_minutes:
            print(f'  ⏱ 時間上限 {max_minutes}分に到達したのでここで止める'
                  f'（保存済みぶんは残る）')
            break

    remaining = len(todo) - processed
    print(f'\n完了: {processed}日 / 配当{saved:,}行を保存 / 残り{remaining}日 '
          f'({(time.time()-t0)/60:.1f}分)')
    if remaining:
        print('  ※ 残りは次回の実行で埋まる（同じコマンドをもう一度）')
    return {'processed': processed, 'saved': saved, 'remaining': remaining}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                    help=f'1回の実行で処理する開催日の上限（既定{DEFAULT_BUDGET}）')
    ap.add_argument('--max-minutes', type=float, default=DEFAULT_MAX_MINUTES,
                    help=f'1回の実行の時間上限（既定{DEFAULT_MAX_MINUTES}分）')
    ap.add_argument('--oldest-first', action='store_true',
                    help='古い開催日から埋める（既定は新しい順）')
    ap.add_argument('--dry-run', action='store_true', help='対象日を数えるだけ')
    ap.add_argument('--base-dir', default=ROOT)
    a = ap.parse_args()
    if a.budget < 1:
        raise SystemExit('--budget は1以上を指定してください')
    if a.max_minutes <= 0:
        raise SystemExit('--max-minutes は0より大きい値を指定してください')
    backfill(a.base_dir, budget=a.budget, oldest_first=a.oldest_first,
             dry_run=a.dry_run, max_minutes=a.max_minutes)


if __name__ == '__main__':
    main()

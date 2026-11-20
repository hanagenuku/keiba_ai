#!/usr/bin/env python3
"""アプリの出走表の下に出す「note 用テキスト」を作る（data/note_blocks.json）。

ユーザー依頼（2026-11-14）:
    「商品になる予定のレースの予想根拠コメントと買い目をアプリのレース出走表の
      下部に置いてくれないか。そのままコピーして note に貼り付けられるように」

🔴 本文は記事と**同じ1つの実装**（`src.product.article.build_race_section`）で作る。
   アプリ側（JS）で文章を組み立てない。組み立てを2か所に分けると、
   `commentary.FORBIDDEN_PATTERNS`（陣営コメント・調教・パドックに触れない）や
   誇大表現の禁止がアプリ側だけ素通りする。

🔴 対象は `scope.article_kind()` が有料／無料と判定したレースだけ。
   「商品になる予定のレース」の定義をここで作り直さない。

🔴 予想は何も変えない。`latest.json` と `history.db` を読むだけで、
   ◎・印・買い目・確率・`article_kind` に一切触れない。

⚠ web 見解（別枠）は `data/claude_web_findings.json` の日付が対象日と一致した
  ときだけ入る。朝いちばんの生成では前回の開催日のものしか無いので入らず、
  その日の見解が main に入ったあと（11:30 / 14:00 の更新）から入る。

使い方:
    python3 scripts/build_note_blocks.py --base-dir .
    python3 scripts/build_note_blocks.py --stdout 京都8     # 確認用
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_product_article import (  # noqa: E402
    _iter_races, _load_latest, _target_date,
)
from src.features import claude_web  # noqa: E402
from src.product import article, facts as facts_mod, scope  # noqa: E402

JST = timezone(timedelta(hours=9))
OUT_NAME = 'note_blocks.json'
SCHEMA_VERSION = 1


def build_blocks(base_dir, *, comment_marks=3):
    """{race_id: ブロック} を作る。対象外のレースは入れない。"""
    data = _load_latest(base_dir)
    target_date = _target_date(data)
    web_findings = claude_web.load_findings(base_dir)
    conn = facts_mod.open_history(base_dir)
    if conn is None:
        print('⚠ history.db の実体が無いため、過去走に基づく記述は省略されます',
              file=sys.stderr)

    blocks, skipped = {}, 0
    day = scope.day_selection(_iter_races(data))   # 2026-11-21: その日の全レースから振り分ける
    for race in _iter_races(data):
        horses = race.get('horses') or []
        if not horses:
            continue
        kind, why = scope.article_kind(race, day)
        if kind not in (scope.PAID, scope.FREE):
            skipped += 1
            continue

        facts_by_num = {}
        if conn is not None:
            for h in horses:
                facts_by_num[h.get('n')] = facts_mod.build_facts(
                    h, race, target_date, conn)

        # ⚠ `claude_web` は馬番キーが `num`。`latest.json` は `n` なので
        #   ここで渡し替える（build_product_article.py と同じ扱い）。
        web_view = claude_web.build_claude_view(
            race.get('race_id'), race.get('r'),
            [{'num': h.get('n')} for h in horses], web_findings,
            race_date=target_date.replace('-', '')) if web_findings else None

        # 🔴 無料も中身は有料と同じ（2026-11-04 ユーザー指示）。paid=True 固定。
        sec = article.build_race_section(race, horses, facts_by_num,
                                         paid=True,
                                         n_comment_marks=comment_marks,
                                         web_view=web_view)
        blocks[race.get('race_id')] = {
            'kind':   kind,
            'why':    why,
            'venue':  race.get('_venue'),
            'r':      race.get('r'),
            'name':   race.get('name'),
            'conf':   race.get('conf'),
            'heading': sec['heading'],
            'grade':  sec['grade'],
            'md':     sec['markdown'],
        }
    if conn is not None:
        conn.close()
    return {
        'schema_version': SCHEMA_VERSION,
        'generated_at':   datetime.now(JST).isoformat(),
        'date':           target_date,
        'source_generated_at': data.get('generated_at'),
        'races':          blocks,
        'skipped':        skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--comment-marks', type=int, default=3,
                    help='根拠を書く印の数（1=◎ / 2=◎○ / 3=◎○▲）')
    ap.add_argument('--stdout', metavar='会場R',
                    help='保存せず、そのレースの本文だけ標準出力に出す（確認用）')
    args = ap.parse_args()

    out = build_blocks(args.base_dir, comment_marks=args.comment_marks)

    if args.stdout:
        for b in out['races'].values():
            if f"{b['venue']}{b['r']}" == args.stdout:
                print(b['md'])
                return 0
        print(f"⚠ {args.stdout} は商品の対象ではありません", file=sys.stderr)
        return 1

    path = os.path.join(args.base_dir, 'data', OUT_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    paid = sum(1 for b in out['races'].values() if b['kind'] == scope.PAID)
    free = sum(1 for b in out['races'].values() if b['kind'] == scope.FREE)
    print(f"✅ {OUT_NAME}: 有料{paid}本 / 無料{free}本（対象外 {out['skipped']}本）→ {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

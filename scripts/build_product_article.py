#!/usr/bin/env python3
"""latest.json から商品（記事＋公開スナップショット）を作る。

🔴 本番の予想ロジックを一切呼ばない。`data/latest.json` と `data/history.db` を読むだけ。
🔴 公開スナップショットは上書きしない（予想を後から変更しないため）。

使い方:
    python3 scripts/build_product_article.py                  # 記事を標準出力へ
    python3 scripts/build_product_article.py --free           # 無料版
    python3 scripts/build_product_article.py --publish        # スナップショットも保存
    python3 scripts/build_product_article.py --races 中山9,阪神9
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import (  # noqa: E402
    archive, article, facts as facts_mod, marks, scope, tickets,
)


def _load_latest(base_dir):
    path = os.path.join(base_dir, 'data', 'latest.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _target_date(data):
    """対象日を 'YYYY-MM-DD' で返す。履歴のリーク防止に使う。"""
    for key in ('date', 'display_date'):
        v = data.get(key)
        if v and len(str(v)) >= 10:
            return str(v)[:10]
    gen = data.get('generated_at') or ''
    if len(gen) >= 10:
        return gen[:10]
    return datetime.now().strftime('%Y-%m-%d')


def _iter_races(data, wanted=None):
    races = data.get('races') or {}
    if isinstance(races, dict):
        pairs = [(v, r) for v, rs in races.items() for r in rs]
    else:  # 念のため（将来 list に変わった場合）
        pairs = [(r.get('venue') or '', r) for r in races]
    out = []
    for venue, r in pairs:
        r = dict(r)
        r['_venue'] = venue
        key = f"{venue}{r.get('r')}"
        if wanted and key not in wanted:
            continue
        out.append(r)
    out.sort(key=lambda r: (r['_venue'], r.get('r') or 0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--free', action='store_true', help='無料版を作る')
    ap.add_argument('--publish', action='store_true',
                    help='公開スナップショットを prediction_archive/ に保存する')
    ap.add_argument('--races', default='', help='例: 中山9,阪神9')
    ap.add_argument('--all-races', action='store_true',
                    help='対象範囲（9R〜11R＋自信のある平場）を無視して全レース出す')
    ap.add_argument('--only-recommended', action='store_true',
                    help='見送りにならないレースだけ')
    ap.add_argument('--out', default='', help='記事の出力先ファイル')
    args = ap.parse_args(argv)

    base_dir = args.base_dir
    data = _load_latest(base_dir)
    target_date = _target_date(data)
    wanted = {s.strip() for s in args.races.split(',') if s.strip()} or None

    conn = facts_mod.open_history(base_dir)
    if conn is None:
        print('⚠ history.db の実体が無いため、過去走に基づく記述は省略されます',
              file=sys.stderr)

    if args.publish:
        # 🔴 公開記録は後から直せない。保存する前に2つだけ確かめる。
        # ① 発走が終わったレースを「公開」として残さない（記録自体が嘘になる）
        try:
            archive.assert_publishable(target_date)
        except archive.StalePublication as e:
            print(f'🔴 公開を中止: {e}', file=sys.stderr)
            return 1
        # ② 過去走を引けない状態で保存すると、見解が薄いまま永久に固定される
        if conn is None:
            print('🔴 公開を中止: history.db の実体が無い。過去走に基づく見解が'
                  '欠けたまま永久保存されるため、実体を取得してから公開すること'
                  '（CLAUDE.md の media.githubusercontent.com 経由の手順）',
                  file=sys.stderr)
            return 1

    sections = []
    published = []
    skipped = []
    out_of_scope = []
    for race in _iter_races(data, wanted):
        horses = race.get('horses') or []
        if not horses:
            continue
        # 対象範囲（ユーザー指示: 各競馬場9R〜11R ＋ 自信のある平場）。
        # --races で明示指定された場合と --all-races では適用しない。
        if not wanted and not args.all_races:
            ok, why = scope.in_scope(race)
            if not ok:
                out_of_scope.append(f"{race['_venue']}{race.get('r')}R({why})")
                continue
        verdict = tickets.verdict(race.get('conf'))
        if args.only_recommended and verdict == 'skip':
            skipped.append(f"{race['_venue']}{race.get('r')}R")
            continue

        facts_by_num = {}
        if conn is not None:
            for h in horses:
                facts_by_num[h.get('n')] = facts_mod.build_facts(
                    h, race, target_date, conn)

        sec = article.build_race_section(race, horses, facts_by_num,
                                         paid=not args.free)
        sections.append(sec)

        if args.publish:
            horse_comments = {}
            for h in sec['marked']:
                if h.get('product_mark'):
                    from src.product import commentary as cm
                    horse_comments[h.get('n')] = cm.horse_comment(
                        h, facts_by_num.get(h.get('n'), {}), race)
            from src.product import commentary as cm
            snap = archive.build_snapshot(
                race, sec['marked'], sec['rows'], sec['tickets'], sec['dangers'],
                horse_comments,
                cm.race_comment(race, sec['marked'], sec['tickets'], sec['dangers']),
            )
            try:
                path = archive.save_snapshot(base_dir, target_date, snap)
                published.append(path)
            except archive.AlreadyPublished as e:
                print(f'⚠ {e}', file=sys.stderr)

    if not sections:
        print('対象レースがありません', file=sys.stderr)
        return 1

    label = target_date
    text = article.build_article(label, sections, paid=not args.free)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'記事を書き出しました: {args.out}', file=sys.stderr)
    else:
        print(text)

    if published:
        print(f'公開スナップショット {len(published)} 件を保存しました', file=sys.stderr)
    if skipped:
        print(f'見送り: {", ".join(skipped)}', file=sys.stderr)
    if out_of_scope:
        print(f'対象外 {len(out_of_scope)}件: {", ".join(out_of_scope)}',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

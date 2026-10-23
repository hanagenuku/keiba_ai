#!/usr/bin/env python3
"""朝の web 見解と、その後の結果を `data/web_findings_log.json` に情報源ごとに残す。

2026-09-25 に作った `data/source_registry.json` は「測った結果」を機械可読にする
側はできていたが、**実運用の当たり外れを情報源ごとに積む層が1つも無かった**。
`data/claude_web_findings.json` は開催日ごとに上書きされ、結果と突合するコードは
1行も存在しなかった。この器が無いと「どこの情報が有効か」は永久に N=0 から始まる。

🔴 本番の予想・モデル・買い目・表示には触らない。読むのは
   `data/claude_web_findings.json` と `data/history.db`（確定馬場・着順）と
   `data/keiba.db`（複勝払戻）だけ。
🔴 見解の判定は再実装しない（`claude_web.py` が検証して書いた形をそのまま読む）。
🔴 一度記録したものは上書きしない。結果が無いレースは空で埋めない。
🔴 回収率は出さない。残すのは確定馬場・着順・3着内か・複勝払戻という事実だけ。

使い方:
    python3 scripts/record_web_findings.py                 # 朝/夜: 見解を記録
    python3 scripts/record_web_findings.py --attach-results # 夜: 結果を追記
    python3 scripts/record_web_findings.py --summary        # 情報源ごとの集計
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import results as results_mod, web_log  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--attach-results', action='store_true',
                    help='確定した馬場・着順・複勝払戻を追記する（夜）')
    ap.add_argument('--summary', action='store_true', help='集計を出して終わる')
    ap.add_argument('--dry-run', action='store_true', help='書かずに件数だけ出す')
    args = ap.parse_args(argv)
    base_dir = args.base_dir

    if args.summary:
        s = web_log.summarize(base_dir)
        print(f'記録 {s["n_entries"]}件（{len(s["dates"])}開催日）'
              f' / 結果あり {s["n_with_result"]}件')
        for host, kinds in sorted(s['by_host'].items()):
            print(f'  {host}')
            for kind, k in sorted(kinds.items()):
                extra = ''
                if kind == 'going':
                    bits = []
                    if k['stable_of']:
                        # 🔴 これは予想の精度ではない。馬場が動かなかった率
                        bits.append(f'収集時点の馬場＝確定馬場 '
                                    f'{k["stable"]}/{k["stable_of"]}')
                    bits.append(f'想定の言い当て {k["expected_match"]}/{k["expected_of"]}'
                                if k['expected_of'] else '想定は状態名を名乗らず採点不能')
                    extra = ' / ' + ' / '.join(bits)
                elif k['top3_of']:
                    # 🔴 これは信頼度ではない。全馬にメモが付く情報源では
                    #   母集団の3着内率そのものになる。必ず言及頭数と並べて読む
                    extra = f' / 3着内 {k["top3"]}/{k["top3_of"]}'
                print(f'    {kind:6s} {k["n"]:4d}件（結果あり {k["with_result"]:4d}）{extra}')
                if kind != 'going':
                    mpr = k.get('mentions_per_race')
                    print(f'           言及レース {k["n_races"]:3d} / '
                          f'1レースあたり {mpr if mpr is not None else "-"}頭'
                          f' / 序列あり {k["with_grade"]:4d}'
                          f' / 方向あり {k["with_direction"]:4d}'
                          f'  ← 測れる形か（websig/CRITERIA_RELIABILITY.md）')
        print(f'⚠ {s["caveat"]}', file=sys.stderr)
        return 0

    if args.attach_results:
        keiba = results_mod.open_keiba(base_dir)
        hist = results_mod.open_history(base_dir)
        if hist is None:
            # 🔴 着順が引けない状態で「結果」を書くと事実でないものが永久に残る
            print('⚠ data/history.db の実体が無いため結果追記をスキップしました'
                  '（LFS 実体を取得してから実行）', file=sys.stderr)
            return 0
        r = web_log.attach_results(base_dir, hist, keiba, dry_run=args.dry_run)
        print(f'結果追記 {r["attached"]}件 / 既存 {r["already"]}件 / '
              f'結果待ち {r["pending"]}件', file=sys.stderr)
        if r['pending_races']:
            print('結果待ちのレース: ' + ', '.join(sorted(r['pending_races'])),
                  file=sys.stderr)
        return 0

    findings = web_log.load_findings_raw(base_dir)
    if findings is None:
        print('⚠ data/claude_web_findings.json が無い/読めないため記録をスキップ',
              file=sys.stderr)
        return 0
    entries = web_log.collect_from_findings(findings)
    if args.dry_run:
        print(f'（dry-run）{findings.get("date")} の見解 {len(entries)}件',
              file=sys.stderr)
        for e in entries:
            print(f'  {e["date"]} {e["venue"]}{e["race_num"]}R {e["kind"]}'
                  f'[{e["key"]}] 出典={",".join(e["hosts"]) or "なし"}',
                  file=sys.stderr)
        return 0
    r = web_log.record(base_dir, findings)
    print(f'web 見解を記録: 追記 {r["added"]}件 / 台帳 計 {r["total"]}件',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

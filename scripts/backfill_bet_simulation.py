#!/usr/bin/env python3
"""蓄積済みの未決済 bet_simulation を、DB内の既存レース結果で一括決済する。

背景:
    log_bet_simulation() は全6券種を「買った想定」で毎レース記録するが、
    settle_bet_simulation() は fetch_results() が返す「その回スクレイプした
    結果」しか参照しない。そのため過去に溜まった未決済行は、週次ワークフローを
    何度実行しても決済されないまま残り続ける（2026-07-28に発覚）。

    このスクリプトは keiba.db / history.db に既に保存されている結果から
    決済に必要な情報を組み立て、未決済行(is_hit=-1)だけを決済する。
    何度実行しても二重計上されない。

⚠ 算出できるもの / できないもの:
    単勝・複勝は実配当が保存されているため **回収率まで** 算出できる。
    ワイド・馬連・馬単・三連複の組み合わせ配当はどのテーブルにも
    保存されていないため **的中率のみ**（回収率は算出不可）。
    的中はしたが配当が取れない行は payout=0 のまま残るので、
    単純な SUM(payout)/点数 は過小評価になる点に注意すること。

使い方:
    python scripts/backfill_bet_simulation.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.db import backfill_bet_simulation, get_db_path, backup_db


def _is_lfs_pointer(path):
    """git-lfs が展開されていないポインタファイルかどうか。

    LFS未展開のクローンでそのまま実行すると sqlite3 が
    'file is not a database' で落ちるため、先に分かりやすく弾く。
    """
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, 'rb') as f:
            return f.read(64).startswith(b'version https://git-lfs')
    except OSError:
        return False


def main():
    db_path = get_db_path(ROOT)
    if not os.path.exists(db_path):
        print(f'✗ keiba.db が見つかりません: {db_path}')
        return 1

    if _is_lfs_pointer(db_path):
        print('✗ data/keiba.db が git-lfs のポインタのままです。')
        print('  実データを取得してから再実行してください:')
        print('    git lfs pull')
        print('  もしくは Colab / GitHub Actions 上で実行してください。')
        return 1

    backup_db(db_path)
    print(f'バックアップ作成: {db_path}.bak')

    out = backfill_bet_simulation(base_dir=ROOT)

    if not out['races_total']:
        print('未決済の bet_simulation はありません。')
        return 0

    print(f'\n決済 {out["settled"]}行 / 的中 {out["hit"]}件')
    print(f'対象レース {out["races_total"]}  '
          f'結果あり {out["races_covered"]}  '
          f'結果なし {out["races_missing"]}（未決済のまま据え置き）')

    print('\n券種別:')
    for bt, st in sorted(out['by_type'].items(), key=lambda x: -x[1]['n']):
        n = st['n']
        if not n:
            continue
        na = out['payout_unavailable'].get(bt, 0)
        head = f'  {bt:<5} {n:>4}点  的中{st["hit"]:>3} ({100 * st["hit"] / n:>5.1f}%)'
        if na:
            print(f'{head}  回収率=算出不可（実配当が無い的中 {na}件）')
        else:
            print(f'{head}  回収率={st["payout"] / n:>6.1f}%')

    print('\n※ ワイド/馬連/馬単/三連複は組み合わせ配当が未保存のため的中率のみ。')
    return 0


if __name__ == '__main__':
    sys.exit(main())

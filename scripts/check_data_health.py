#!/usr/bin/env python3
"""history.db の「静かに壊れた列」を検知する。

## なぜ要るか

2026-07-04 の結果ページ列崩れで `popularity` が取れなくなり、**3週間気づかな
かった**。原因は取得失敗時に NULL ではなく **99（不明のセンチネル）** で保存され
ることで、`popularity IS NOT NULL AND popularity > 0` という素朴な充足率チェック
を素通りしていたため。実際は `engine.py` が 99 を NaN に落とすので残差学習モデルの
base_margin（市場アンカー）が完全に死んでいた（実測: 健全行 AUC 0.8207 /
死亡行 0.7091、差 -0.11）。

**「NULLでない＝健全」では判定できない。** このスクリプトは
  ① センチネル値（99等）の混入率
  ② 直近月が過去平均から急落していないか（列崩れの兆候）
を見て、同型の事故を早期に検知する。

## 使い方

    python scripts/check_data_health.py              # 直近3ヶ月を点検
    python scripts/check_data_health.py --months 12
    python scripts/check_data_health.py --strict     # 異常があれば exit 1（CI用）
"""
import argparse
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 列名 → (実質欠損とみなすSQL条件, 説明)
# ⚠ センチネル値を使う列は「IS NOT NULL」では健全性を判定できないため、
#   値そのものの範囲で判定する。
COLUMN_CHECKS = [
    ('popularity',  "popularity IS NULL OR popularity <= 0 OR popularity >= 99",
     '市場アンカー(base_margin)の唯一の入力。死ぬと残差学習が機能しない'),
    ('agari_rank',  "agari_rank IS NULL OR agari_rank <= 0 OR agari_rank >= 99",
     'f_agari_ability の入力'),
    ('place',       "place IS NULL OR place <= 0 OR place >= 99",
     '目的変数 is_fukusho の元。全ての学習に影響'),
    ('trainer',     "trainer IS NULL OR trainer = ''", 'f_trainer の入力'),
    ('jockey',      "jockey IS NULL OR jockey = ''",   'f_jockey の入力'),
    ('corner_all',  "corner_all IS NULL OR corner_all = ''",
     'corner_3 の導出元。f_pos_avg_3 等12特徴量に影響'),
    ('finish_time', "finish_time IS NULL OR finish_time <= 0",
     'スピード指数の入力'),
    ('agari3f',     "agari3f IS NULL OR agari3f <= 0", '上がり3F'),
    ('weight_load', "weight_load IS NULL OR weight_load <= 0", '斤量'),
    ('body_weight', "body_weight IS NULL",
     'f_weight_trend_avg の入力（2025-01〜2026-02は元々0%・復旧不可）'),
    ('sex',         "sex IS NULL OR sex = ''", 'f_sex'),
    ('age',         "age IS NULL OR age <= 0", 'f_age'),
    # ⚠ 配当2列は回収率（このプロジェクトの唯一のKPI）の**分子そのもの**なのに、
    #   2026-08-09の監査まで一度も点検対象に入っていなかった。壊れても
    #   「回収率が下がった」としか見えず、データ破損だと気づけない。
    #
    #   4要素目で分母を絞るのが必須。1着馬しか単勝配当を持たず、3着内馬しか
    #   複勝配当を持たないため、全行を分母にすると**健全なDBでも
    #   単勝90%・複勝70%が「欠損」**と出て毎回「⚠ 半分以上が欠損」が鳴る
    #   （実測値。この検知器は普段黙っていることに価値があるので、
    #   常時誤報すると本物の異常が埋もれる）。
    ('tansho_payout',
     "tansho_payout IS NULL OR tansho_payout <= 0",
     '単勝回収率の分子', 'place = 1'),
    ('fukusho_payout',
     "fukusho_payout IS NULL OR fukusho_payout <= 0",
     '複勝回収率の分子', 'place BETWEEN 1 AND 3'),
]

# 元々ほぼ入っていないと分かっている列（警告しても意味がないので除外）
# docs/history_db_schema.md と CLAUDE.md の既知事項に対応
KNOWN_EMPTY = {'win_odds', 'bracket', 'trainer_affiliation', 'sire', 'dam_sire',
               'field_size', 'corner_3'}

# 直近月の欠損率が「過去の中央値 + この値」を超えたら急落とみなす
SPIKE_THRESHOLD = 0.30


def check(db_path, months=3, strict=False):
    if not os.path.exists(db_path):
        raise SystemExit(f'history.db が見つかりません: {db_path}')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT SUBSTR(date,1,7) FROM horse_history "
                "WHERE date IS NOT NULL AND date != '' ORDER BY 1")
    all_months = [r[0] for r in cur.fetchall()]
    if len(all_months) < 2:
        print('データが少なすぎて点検できない')
        return 0
    recent = all_months[-months:]
    past = all_months[:-months]
    print(f'点検対象: 直近{len(recent)}ヶ月 {recent[0]}〜{recent[-1]}'
          f'（比較対象: {past[0]}〜{past[-1]} の{len(past)}ヶ月）\n')

    problems = []
    print(f'{"列":<14}{"直近欠損率":>10}{"過去中央値":>11}   判定')
    print('-' * 62)
    for entry in COLUMN_CHECKS:
        # 4要素目は「その列が入っているべき行」の絞り込み（省略時は全行）。
        # ⚠ 配当のように一部の着順にしか入らない列は、分母を全行にすると
        #   健全な状態でも大半が「欠損」と数えられ、毎回誤報になる。
        col, cond, desc = entry[0], entry[1], entry[2]
        scope = entry[3] if len(entry) > 3 else '1=1'
        if col in KNOWN_EMPTY:
            continue
        rates = {}
        for ym in all_months:
            cur.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN {cond} THEN 1 ELSE 0 END) "
                f"FROM horse_history WHERE SUBSTR(date,1,7) = ? AND ({scope})",
                (ym,))
            n, bad = cur.fetchone()
            if n:
                rates[ym] = (bad or 0) / n

        recent_vals = [rates[m] for m in recent if m in rates]
        past_vals = sorted(rates[m] for m in past if m in rates)
        if not recent_vals or not past_vals:
            continue
        recent_rate = sum(recent_vals) / len(recent_vals)
        past_med = past_vals[len(past_vals) // 2]

        verdict, flag = 'OK', False
        if recent_rate >= 0.99:
            verdict, flag = '🔴 直近が全滅', True
        elif recent_rate - past_med >= SPIKE_THRESHOLD:
            verdict, flag = '🔴 急落（列崩れの疑い）', True
        elif recent_rate >= 0.50:
            verdict, flag = '⚠ 半分以上が欠損', True
        print(f'{col:<14}{recent_rate*100:9.1f}%{past_med*100:10.1f}%   {verdict}')
        if flag:
            problems.append((col, recent_rate, past_med, desc))

    conn.close()
    if problems:
        print(f'\n{"="*62}\n🔴 {len(problems)}件の異常\n{"="*62}')
        for col, r, p, desc in problems:
            print(f'\n[{col}] 直近{r*100:.1f}% ← 過去{p*100:.1f}%')
            print(f'  影響: {desc}')
        print('\n対処: popularity/trainer/body_weight なら '
              '`python scripts/repair_popularity.py` で再取得できる場合がある')
        print('      （JRAの結果一覧は直近しか遡れないため古い月は復旧不可）')
        if strict:
            return 1
    else:
        print('\n✅ 直近に列崩れの兆候なし')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(ROOT, 'data', 'history.db'))
    ap.add_argument('--months', type=int, default=3, help='直近何ヶ月を点検するか')
    ap.add_argument('--strict', action='store_true',
                    help='異常があれば exit 1（CI用）')
    args = ap.parse_args()
    sys.exit(check(args.db, months=args.months, strict=args.strict))


if __name__ == '__main__':
    main()

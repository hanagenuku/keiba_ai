"""本番134特徴量が「何に依存しているか」を実測で仕分ける。

設計指示書 Step 1 の「馬能力 / 市場由来 / レース由来 / コース由来 / 未定義」への
分類を、人の目で名前を見て決めるのではなく**摂動して測る**。

  A 馬を入れ替えても値が変わらない → レース共通（馬の情報ではない）
  B 今回のコース(会場/馬場/距離)を変えると動く → コース由来
  C 市場の値(人気/オッズ)を変えると動く → 市場由来
  D 過去走を空にすると動く → その馬の履歴由来

⚠ これは「値の出どころ」の仕分けであって、有用性の評価ではない。

使い方: python3 noryoku/inv1.py <db_path> [n_races]
"""
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, '.')
import src.features.engine as eng
from src.features.engine import (init_engine, calc_features_for_xgb,
                                 add_relative_features)
from src.scraper.jra_scraper import _infer_running_style, HISTORY_LIMIT
from src.tools.build_training_data import _get_history_before

ALT_COURSE = {'racecourse': '新潟', 'surface': 'ダート', 'distance': 1200}


def build_race(conn, race_id):
    r = conn.execute("SELECT * FROM race_history WHERE race_id=?", (race_id,)).fetchone()
    rc, surf = r['racecourse'], (r['surface'] or '芝')
    race = {
        'race_id': race_id, 'date': r['date'], 'racecourse': rc,
        'distance': int(r['distance'] or 1600), 'surface': surf,
        'first_3f': 0.0, 'race_class': r['race_class'] or '1勝',
        'race_name': r['race_name'] or '',
        'track_condition': r['track_condition'] or '良',
        'pace_label': 'mid', 'horses': [],
    }
    rows = conn.execute("""SELECT horse_name, horse_num, jockey, trainer, corner_3,
                                  weight_load, sex, age, win_odds, popularity
                           FROM horse_history WHERE race_id=? ORDER BY horse_num""",
                        (race_id,)).fetchall()
    hs = []
    for d in rows:
        jn = (d['jockey'] or '').replace(' ', '').replace('　', '')
        tn = (d['trainer'] or '').replace(' ', '').replace('　', '')
        h = {'name': d['horse_name'], 'horse_num': int(d['horse_num'] or 1),
             'place': 99, 'running_style': '差し', 'agari3f': None,
             'jockey': d['jockey'] or '', 'trainer': d['trainer'] or '',
             'corner_3': d['corner_3'], 'weight_load': float(d['weight_load'] or 56.0),
             'win_odds': float(d['win_odds'] or 10.0),
             'popularity': int(d['popularity'] or 0),
             'age': int(d['age'] or 4), 'sex': d['sex'] or '牡',
             'jockey_rate': (eng._jockey_dict.get((jn, rc, surf))
                             or eng._jockey_dict.get((jn, '', '')) or 0.15),
             'trainer_rate': eng._trainer_dict.get(tn, 0.12), 'history': []}
        hs.append(h)
    race['horses'] = hs
    for h in hs:
        h['history'] = _get_history_before(conn, h['name'], r['date'], limit=HISTORY_LIMIT)
    for h in hs:
        h['running_style'] = _infer_running_style(h['name'], h['history'], h.get('horse_num'))
    return race


def _fresh(race, **over):
    # 🔑 race には '_pace_dist_cache' が生える（engine.py:2507）。
    #    浅いコピーだと摂動しても前の値を使い回し、
    #    「ペース確率はコースで動かない」という嘘の結論になる。
    r = {k: v for k, v in race.items() if not k.startswith('_')}
    r.update(over)
    return r


def feats_of(race, cols):
    xs = []
    for h in race['horses']:
        try:
            xs.append(calc_features_for_xgb(h, race))
        except Exception as e:      # 落ちたら仕分け不能として残す
            print('  !! calc失敗', h['name'], e)
            xs.append({})
    add_relative_features(xs)
    return [{c: x.get(c) for c in cols} for x in xs]


def diff_keys(a, b, cols):
    out = set()
    for xa, xb in zip(a, b):
        for c in cols:
            va, vb = xa.get(c), xb.get(c)
            if va is None or vb is None:
                if va is not vb:
                    out.add(c)
                continue
            try:
                fa, fb = float(va), float(vb)
            except (TypeError, ValueError):
                if va != vb:
                    out.add(c)
                continue
            na, nb = fa != fa, fb != fb      # NaN 判定
            # 🔑 NaN 同士の比較は必ず False。素の不等号だと
            #    「値 → NaN に変わった」を見逃す（2026-08-25 D-2④ と同型）
            if na != nb or (not na and abs(fa - fb) > 1e-9):
                out.add(c)
    return out


def run(db, n_races=6):
    import json
    cols = json.load(open('data/xgb_feature_cols.json'))['feature_cols']
    init_engine('.')
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    ids = [r[0] for r in conn.execute(
        """SELECT race_id FROM race_history
           WHERE date >= '2026-01-01' ORDER BY race_id LIMIT ?""", (n_races,))]

    varies_horse, varies_course, varies_market, varies_hist = set(), set(), set(), set()
    seen = defaultdict(set)

    for rid in ids:
        race = build_race(conn, rid)
        if len(race['horses']) < 6:
            continue
        base = feats_of(race, cols)
        for c in cols:                       # A: 馬による違いがあるか
            vals = {repr(x.get(c)) for x in base}
            seen[c] |= vals
            if len(vals) > 1:
                varies_horse.add(c)

        r2 = _fresh(race, **ALT_COURSE)          # B: コースを差し替える
        varies_course |= diff_keys(base, feats_of(r2, cols), cols)

        r3 = _fresh(race)                        # C: 市場だけ差し替える
        r3['horses'] = [dict(h) for h in race['horses']]
        for i, h in enumerate(r3['horses']):
            h['popularity'] = len(r3['horses']) - i
            h['win_odds'] = 2.0 + 5.0 * i
        varies_market |= diff_keys(base, feats_of(r3, cols), cols)

        r4 = _fresh(race)                        # D: 過去走を消す
        r4['horses'] = [dict(h) for h in race['horses']]
        for h in r4['horses']:
            h['history'] = []
        varies_hist |= diff_keys(base, feats_of(r4, cols), cols)

    conn.close()

    # ── 感度検査（結果を読む前に通す）──────────────────────
    #   分かりきった依存を検出できない計測器の出力は読まない。
    gate = [('f_recent', varies_hist, '過去走'),
            ('f_same_course_rate', varies_course, 'コース'),
            ('f_jockey', varies_horse, '馬'),
            ('f_pace_prob_fast', varies_course, 'コース')]
    bad = [f'{c}が{w}に反応しない' for c, s_, w in gate if c not in s_]
    if bad:
        print('🔴 計測器が壊れている: ' + ' / '.join(bad))
        return

    const = [c for c in cols if c not in varies_horse]
    print(f'■ 対象 {len(cols)}列 / {len(ids)}レース\n')
    print(f'A レース内で全馬同じ値（馬の情報を含まない） : {len(const)}列')
    for c in const:
        tag = []
        if c in varies_course: tag.append('コース/条件で動く')
        if c in varies_market: tag.append('市場で動く')
        if c in varies_hist: tag.append('履歴で動く')
        vals = sorted(seen[c])[:3]
        print(f'   {c:34s} {",".join(tag) or "どれでも動かない=定数"}  例{vals}')

    print(f'\nB 今回のコースを変えると動く : {len(varies_course)}列')
    print('   ' + ' '.join(sorted(varies_course)))
    print(f'\nC 市場(人気/オッズ)を変えると動く : {len(varies_market)}列')
    print('   ' + ' '.join(sorted(varies_market)))
    print(f'\nD 過去走を消すと動く : {len(varies_hist)}列')

    # 交差分類
    horse_and_course = sorted((varies_horse & varies_course) - varies_market)
    horse_only = sorted(varies_horse - varies_course - varies_market)
    print(f'\n■ 仕分け')
    print(f'   レース共通（馬に依らない）        {len(const):3d}列')
    print(f'   馬 × 今回のコース（交互作用）      {len(horse_and_course):3d}列')
    print('      ' + ' '.join(horse_and_course))
    print(f'   馬のみ（今回の条件に依らない）     {len(horse_only):3d}列')
    print(f'   市場由来                        {len(sorted(varies_market)):3d}列')
    json.dump({c: {'varies_horse': c in varies_horse,
                   'varies_course': c in varies_course,
                   'varies_market': c in varies_market,
                   'varies_hist': c in varies_hist} for c in cols},
              open('noryoku/inventory.json', 'w'), indent=1, ensure_ascii=False)
    print('\n   → noryoku/inventory.json に書き出した')

    print(f'   履歴なしでも変わらない（静的）     '
          f'{len([c for c in cols if c not in varies_hist]):3d}列')
    print('      ' + ' '.join([c for c in cols if c not in varies_hist]))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)

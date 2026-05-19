import sqlite3
import json
import os


def get_db_path(base_dir):
    return os.path.join(base_dir, 'data', 'keiba.db')


def get_history_db_path(base_dir):
    return os.path.join(base_dir, 'data', 'history.db')


def init_db(base_dir=None, db_path=None):
    """keiba.db の初期化。テーブルがなければ作成"""
    path = db_path or get_db_path(base_dir)
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS races (
            id TEXT PRIMARY KEY, date TEXT, racecourse TEXT,
            race_name TEXT, distance INTEGER, surface TEXT,
            condition TEXT, num_horses INTEGER, raw_json TEXT);
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT, place INTEGER, horse_num INTEGER,
            horse_name TEXT, running_style TEXT,
            agari3f REAL, tansho_payout INTEGER, fukusho_payout INTEGER);
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, race_id TEXT, bet_type TEXT,
            horse_num INTEGER, horse_name TEXT,
            odds_est REAL, amount INTEGER,
            is_hit INTEGER DEFAULT -1, payout INTEGER DEFAULT 0,
            horse_num2 INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS bet_simulation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, race_id TEXT, racecourse TEXT,
            race_num INTEGER, bet_type TEXT,
            horse_num TEXT, horse_name TEXT,
            odds_est REAL, ai_prob REAL, ev REAL,
            num_horses INTEGER, chaos REAL,
            is_tanzen INTEGER, is_2kyou INTEGER, is_konsen INTEGER,
            pop_rank INTEGER, score_gap REAL,
            is_hit INTEGER DEFAULT -1, payout REAL DEFAULT 0
        );
    ''')
    conn.commit()
    conn.close()


def save_race_db(race, base_dir=None, db_path=None):
    path = db_path or get_db_path(base_dir)
    conn = sqlite3.connect(path)
    conn.execute(
        'INSERT OR REPLACE INTO races VALUES (?,?,?,?,?,?,?,?,?)',
        (race['id'], race['date'], race['racecourse'], race['race_name'],
         race['distance'], race['surface'], race.get('condition', '良'),
         race['num_horses'], json.dumps(race, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def save_bets_db(date_str, race_id, bets, base_dir=None, db_path=None):
    """ベットをDBに保存（重複スキップ方式）"""
    path = db_path or get_db_path(base_dir)
    conn = sqlite3.connect(path)
    for b in bets:
        if b['type'] == '三連複' and 'tickets' in b:
            for t in b['tickets']:
                existing = conn.execute(
                    'SELECT id FROM bets WHERE race_id=? AND bet_type=? AND horse_num=? AND horse_num2=?',
                    (race_id, '三連複', t[0], t[1]),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    'INSERT INTO bets (date,race_id,bet_type,horse_num,horse_name,odds_est,amount,horse_num2) VALUES (?,?,?,?,?,?,?,?)',
                    (date_str, race_id, '三連複', t[0], b.get('horse_name', ''), b.get('odds_est', 0), 100, t[1]),
                )
            continue
        existing = conn.execute(
            'SELECT id FROM bets WHERE race_id=? AND bet_type=? AND horse_num=?',
            (race_id, b['type'], b['nums'][0]),
        ).fetchone()
        if existing:
            continue
        horse_num2 = b['nums'][1] if len(b['nums']) > 1 else 0
        conn.execute(
            'INSERT INTO bets (date,race_id,bet_type,horse_num,horse_name,odds_est,amount,horse_num2) VALUES (?,?,?,?,?,?,?,?)',
            (date_str, race_id, b['type'], b['nums'][0],
             b.get('horse_name', ''), b.get('odds_est', 0), b['amount'], horse_num2),
        )
    conn.commit()
    conn.close()


def save_history_db(all_results, base_dir=None, db_path=None):
    """レース結果を history.db の horse_history / race_history に追記する。

    毎週末の結果取得後に呼ぶことで学習データが自動蓄積される。
    race_id が既に存在する場合は INSERT OR IGNORE でスキップ。
    """
    path = db_path or get_history_db_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS race_history (
            race_id   TEXT PRIMARY KEY,
            date      TEXT,
            racecourse TEXT,
            distance  INTEGER,
            surface   TEXT,
            first_3f  REAL
        );
        CREATE TABLE IF NOT EXISTS horse_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id      TEXT,
            date         TEXT,
            racecourse   TEXT,
            horse_name   TEXT,
            horse_num    INTEGER,
            place        INTEGER,
            running_style TEXT,
            agari3f      REAL,
            jockey       TEXT,
            trainer      TEXT,
            corner_3     INTEGER,
            distance     INTEGER,
            surface      TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_horse_history_uniq
            ON horse_history (race_id, horse_num);
    ''')

    new_races = 0
    new_horses = 0
    for r in all_results:
        race_id = r.get('race_id', '')
        if not race_id:
            continue
        # race_id から日付を復元（形式: YYYYMMDD_XX_NN）
        raw_date = race_id.split('_')[0]
        date_str = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}' if len(raw_date) == 8 else raw_date

        cur = conn.execute(
            'INSERT OR IGNORE INTO race_history (race_id,date,racecourse,distance,surface,first_3f) VALUES (?,?,?,?,?,?)',
            (race_id, date_str, r.get('racecourse', ''),
             r.get('distance', 0), r.get('surface', ''), None),
        )
        new_races += cur.rowcount

        for h in r.get('finishers', []):
            cur2 = conn.execute(
                '''INSERT OR IGNORE INTO horse_history
                   (race_id,date,racecourse,horse_name,horse_num,place,
                    running_style,agari3f,jockey,trainer,corner_3,distance,surface)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (race_id, date_str, r.get('racecourse', ''),
                 h.get('name', ''), h.get('num', 0), h.get('place', 99),
                 h.get('running_style', ''), h.get('agari3f', 0.0),
                 h.get('jockey', ''), h.get('trainer', ''),
                 None,
                 h.get('distance', r.get('distance', 0)),
                 h.get('surface', r.get('surface', ''))),
            )
            new_horses += cur2.rowcount

    conn.commit()
    conn.close()
    print(f'📚 history.db に追記: {new_races}レース / {new_horses}頭 (重複スキップ済み)')


def update_bet_results(race_id, results, base_dir=None, db_path=None):
    """レース結果でbetsテーブルのis_hit/payoutを更新"""
    path = db_path or get_db_path(base_dir)
    conn = sqlite3.connect(path)
    rows = conn.execute(
        'SELECT id, bet_type, horse_num, horse_num2 FROM bets WHERE race_id=? AND is_hit=-1',
        (race_id,),
    ).fetchall()
    placed = {r['place']: r for r in results} if results and isinstance(results[0], dict) else {}
    top3_nums = {r.get('horse_num') for r in results[:3]} if results else set()
    for row_id, bet_type, h1, h2 in rows:
        is_hit = 0
        payout = 0
        if bet_type == '複勝' and h1 in top3_nums:
            is_hit = 1
            r = next((r for r in results if r.get('horse_num') == h1), None)
            payout = (r.get('fukusho_payout', 0) or 0) if r else 0
        elif bet_type == '単勝':
            winner = next((r for r in results if r.get('place') == 1), None)
            if winner and winner.get('horse_num') == h1:
                is_hit = 1
                payout = winner.get('tansho_payout', 0) or 0
        conn.execute(
            'UPDATE bets SET is_hit=?, payout=? WHERE id=?',
            (is_hit, payout, row_id),
        )
    conn.commit()
    conn.close()

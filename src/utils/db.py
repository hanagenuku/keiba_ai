import sqlite3
import json
import os
import shutil


def get_db_path(base_dir):
    return os.path.join(base_dir, 'data', 'keiba.db')


def get_history_db_path(base_dir):
    return os.path.join(base_dir, 'data', 'history.db')


def _connect(path):
    """WALモード・busy_timeout付きでDBに接続する（並行アクセス対策）"""
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=5000')
    return conn


def backup_db(path):
    """DBファイルの .bak バックアップを作成する（処理前に呼ぶ）"""
    if os.path.exists(path):
        shutil.copy2(path, path + '.bak')


def checkpoint_db(path):
    """WALの内容をDB本体に統合し、-wal/-shmファイルを解消する（commit/push前に呼ぶ）"""
    conn = _connect(path)
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.close()


def init_db(base_dir=None, db_path=None):
    """keiba.db の初期化。テーブルがなければ作成"""
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
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
        CREATE TABLE IF NOT EXISTS shadow_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            race_id TEXT,
            racecourse TEXT,
            race_num INTEGER,
            race_class TEXT,
            num_horses INTEGER,
            surface TEXT,
            distance INTEGER,
            chaos_grade TEXT,
            rl1_num INTEGER,
            rl1_name TEXT,
            rl1_win_prob REAL,
            rl1_cal_prob REAL,
            rl2_num INTEGER,
            rl2_name TEXT,
            rl3_num INTEGER,
            rl3_name TEXT,
            winner_num INTEGER,
            winner_pop INTEGER,
            winner_odds REAL,
            second_num INTEGER,
            third_num INTEGER,
            shadow_tansho_hit INTEGER,
            shadow_tansho_payout REAL,
            shadow_fukusho_hit INTEGER,
            shadow_fukusho_payout REAL,
            shadow_umaren_hit INTEGER,
            shadow_umaren_payout REAL,
            shadow_wide_hit INTEGER,
            shadow_wide_payout REAL,
            shadow_sanrenp_hit INTEGER,
            shadow_sanrenp_payout REAL,
            was_recommended INTEGER DEFAULT 0,
            actual_bet_type TEXT,
            actual_bet_hit INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    conn.close()


def save_race_db(race, base_dir=None, db_path=None):
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
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
    conn = _connect(path)
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
    conn = _connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS race_history (
            race_id         TEXT PRIMARY KEY,
            date            TEXT,
            racecourse      TEXT,
            distance        INTEGER,
            surface         TEXT,
            first_3f        REAL,
            race_name       TEXT,
            race_class      TEXT,
            track_condition TEXT,
            num_finishers   INTEGER
        );
        CREATE TABLE IF NOT EXISTS horse_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id         TEXT,
            date            TEXT,
            racecourse      TEXT,
            horse_name      TEXT,
            horse_num       INTEGER,
            place           INTEGER,
            running_style   TEXT,
            agari3f         REAL,
            jockey          TEXT,
            trainer         TEXT,
            corner_3        INTEGER,
            distance        INTEGER,
            surface         TEXT,
            popularity      INTEGER,
            tansho_payout   INTEGER,
            fukusho_payout  INTEGER,
            margin          REAL,
            agari_rank      INTEGER
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_horse_history_uniq
            ON horse_history (race_id, horse_num);
    """)

    # Migrations for existing DBs (idempotent: errors on duplicate column are ignored)
    migrations = [
        "ALTER TABLE race_history ADD COLUMN race_name TEXT",
        "ALTER TABLE race_history ADD COLUMN race_class TEXT",
        "ALTER TABLE race_history ADD COLUMN track_condition TEXT",
        "ALTER TABLE race_history ADD COLUMN num_finishers INTEGER",
        "ALTER TABLE horse_history ADD COLUMN popularity INTEGER",
        "ALTER TABLE horse_history ADD COLUMN tansho_payout INTEGER",
        "ALTER TABLE horse_history ADD COLUMN fukusho_payout INTEGER",
        "ALTER TABLE horse_history ADD COLUMN margin REAL",
        "ALTER TABLE horse_history ADD COLUMN agari_rank INTEGER",
        # 新フィールド（着差・馬場・タイム・クラス整備）
        "ALTER TABLE race_history ADD COLUMN race_num INTEGER",
        "ALTER TABLE race_history ADD COLUMN lap_times TEXT",
        "ALTER TABLE race_history ADD COLUMN first_3f REAL",
        "ALTER TABLE race_history ADD COLUMN last_3f REAL",
        "ALTER TABLE horse_history ADD COLUMN class_grade TEXT",
        "ALTER TABLE horse_history ADD COLUMN field_size INTEGER",
        "ALTER TABLE horse_history ADD COLUMN corner_4 INTEGER",
        "ALTER TABLE horse_history ADD COLUMN finish_time REAL",
        "ALTER TABLE horse_history ADD COLUMN time_diff_sec REAL",
        "ALTER TABLE horse_history ADD COLUMN chakusa_text TEXT",
        # Stage 3 で追加（事前確定情報＋過去走履歴の充実）
        "ALTER TABLE horse_history ADD COLUMN weight_load REAL",
        "ALTER TABLE horse_history ADD COLUMN sex TEXT",
        "ALTER TABLE horse_history ADD COLUMN age INTEGER",
        "ALTER TABLE horse_history ADD COLUMN body_weight INTEGER",
        "ALTER TABLE horse_history ADD COLUMN body_weight_diff INTEGER",
        "ALTER TABLE horse_history ADD COLUMN bracket INTEGER",
        "ALTER TABLE horse_history ADD COLUMN corner_all TEXT",
        "ALTER TABLE horse_history ADD COLUMN win_odds REAL",
        "ALTER TABLE race_history ADD COLUMN weather TEXT",
        "ALTER TABLE race_history ADD COLUMN pace_label TEXT",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()

    new_races = 0
    new_horses = 0
    for r in all_results:
        race_id = r.get('race_id', '')
        if not race_id:
            continue
        raw_date = race_id.split('_')[0]
        date_str = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}' if len(raw_date) == 8 else raw_date

        cur = conn.execute(
            "INSERT OR IGNORE INTO race_history "
            "(race_id,date,racecourse,distance,surface,first_3f,race_name,race_class,"
            " track_condition,num_finishers,weather,pace_label) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (race_id, date_str, r.get('racecourse', ''),
             r.get('distance', 0), r.get('surface', ''), None,
             r.get('race_name', ''), r.get('race_class', ''),
             r.get('track_condition', '良'), r.get('num_finishers', 0),
             r.get('weather'), r.get('pace_label')),
        )
        new_races += cur.rowcount
        # Stage 3 rescrape 用：既存行の新フィールドを UPDATE で充填
        conn.execute(
            "UPDATE race_history SET "
            "  race_name      = COALESCE(NULLIF(?, ''), race_name), "
            "  race_class     = COALESCE(NULLIF(?, ''), race_class), "
            "  track_condition= COALESCE(?, track_condition), "
            "  num_finishers  = COALESCE(?, num_finishers), "
            "  weather        = COALESCE(?, weather), "
            "  pace_label     = COALESCE(?, pace_label) "
            "WHERE race_id = ?",
            (r.get('race_name', ''), r.get('race_class', ''),
             r.get('track_condition'), r.get('num_finishers'),
             r.get('weather'), r.get('pace_label'),
             race_id),
        )

        for h in r.get('finishers', []):
            cur2 = conn.execute(
                "INSERT OR IGNORE INTO horse_history "
                "(race_id,date,racecourse,horse_name,horse_num,place,"
                " running_style,agari3f,jockey,trainer,corner_3,distance,surface,"
                " popularity,tansho_payout,fukusho_payout,margin,agari_rank,"
                " class_grade,finish_time,time_diff_sec,chakusa_text,"
                " weight_load,sex,age,body_weight,body_weight_diff,"
                " bracket,corner_all,win_odds) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (race_id, date_str, r.get('racecourse', ''),
                 h.get('name', ''), h.get('num', 0), h.get('place', 99),
                 h.get('running_style', ''), h.get('agari3f', 0.0),
                 h.get('jockey', ''), h.get('trainer', ''),
                 None,
                 h.get('distance', r.get('distance', 0)),
                 h.get('surface', r.get('surface', '')),
                 h.get('popularity', 99),
                 h.get('tansho_payout', 0), h.get('fukusho_payout', 0),
                 h.get('margin', 0.0), h.get('agari_rank', 99),
                 r.get('race_class', ''),
                 h.get('finish_time'),
                 h.get('time_diff_sec'),
                 h.get('chakusa_text', ''),
                 h.get('weight_load'),
                 h.get('sex', ''), h.get('age'),
                 h.get('body_weight'), h.get('body_weight_diff'),
                 h.get('bracket'), h.get('corner_all', ''),
                 h.get('win_odds')),
            )
            new_horses += cur2.rowcount
            # Stage 3 rescrape 用：既存行の新フィールドを UPDATE で充填
            conn.execute(
                "UPDATE horse_history SET "
                "  finish_time      = COALESCE(?, finish_time), "
                "  time_diff_sec    = COALESCE(?, time_diff_sec), "
                "  chakusa_text     = COALESCE(NULLIF(?, ''), chakusa_text), "
                "  margin           = COALESCE(?, margin), "
                "  class_grade      = COALESCE(NULLIF(?, ''), class_grade), "
                "  agari_rank       = COALESCE(?, agari_rank), "
                "  weight_load      = COALESCE(?, weight_load), "
                "  sex              = COALESCE(NULLIF(?, ''), sex), "
                "  age              = COALESCE(?, age), "
                "  body_weight      = COALESCE(?, body_weight), "
                "  body_weight_diff = COALESCE(?, body_weight_diff), "
                "  bracket          = COALESCE(?, bracket), "
                "  corner_all       = COALESCE(NULLIF(?, ''), corner_all), "
                "  win_odds         = COALESCE(?, win_odds), "
                "  surface          = COALESCE(NULLIF(?, ''), surface) "
                "WHERE race_id = ? AND horse_num = ?",
                (h.get('finish_time'), h.get('time_diff_sec'),
                 h.get('chakusa_text', ''), h.get('margin'),
                 r.get('race_class', ''), h.get('agari_rank'),
                 h.get('weight_load'), h.get('sex', ''), h.get('age'),
                 h.get('body_weight'), h.get('body_weight_diff'),
                 h.get('bracket'), h.get('corner_all', ''),
                 h.get('win_odds'),
                 h.get('surface', r.get('surface', '')),
                 race_id, h.get('num', 0)),
            )

    conn.commit()
    conn.close()
    print(f'📚 history.db に追記: {new_races}レース / {new_horses}頭 (重複スキップ済み)')

def save_results_db(all_results, base_dir=None, db_path=None):
    """レース結果を keiba.db の results テーブルに保存する。"""
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    for r in all_results:
        race_id = r.get('race_id', '')
        divs = r.get('dividends', {})
        tp = divs.get('tansho', {}).get('payout', 0)
        for h in r.get('finishers', [])[:6]:
            fp = next((f['payout'] for f in divs.get('fukusho', []) if f['num'] == h['num']), 0)
            exists = conn.execute(
                'SELECT 1 FROM results WHERE race_id=? AND horse_num=?',
                (race_id, h['num']),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                '''INSERT INTO results
                   (race_id, place, horse_num, horse_name, running_style,
                    agari3f, tansho_payout, fukusho_payout)
                   VALUES (?,?,?,?,?,?,?,?)''',
                (race_id, h['place'], h['num'], h['name'],
                 h.get('running_style', ''),
                 h.get('agari3f', 0),
                 tp if h['place'] == 1 else 0, fp),
            )
    conn.commit()
    conn.close()


def check_and_update_bets(all_results, base_dir=None, db_path=None):
    """全レース結果でbetsテーブルのis_hit/payoutを更新し照合サマリを返す。

    Args:
        all_results : fetch_results が返すレース結果リスト（dividends含む）

    Returns:
        dict: {hit, total, invested, recovered, roi, details}
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    conn.row_factory = sqlite3.Row

    # 未照合ベットを取得
    bets = conn.execute(
        'SELECT * FROM bets WHERE is_hit=-1'
    ).fetchall()

    hit = total = invested = recovered = 0
    details = []

    for bet in bets:
        race_id  = bet['race_id']
        bet_type = bet['bet_type']
        h1       = int(bet['horse_num'])
        h2       = int(bet['horse_num2'] or 0)
        amount   = int(bet['amount'])

        result = next((r for r in all_results if r.get('race_id') == race_id), None)
        if not result:
            continue

        divs  = result.get('dividends', {})
        fin   = result.get('finishers', [])
        top3  = [h['num'] for h in fin[:3]]
        top1  = top3[0] if top3 else 0

        is_hit = False
        payout = 0

        if bet_type == '複勝' and h1 in top3:
            is_hit = True
            for f in divs.get('fukusho', []):
                if f['num'] == h1:
                    payout = int(amount * f['payout'] / 100)
                    break
        elif bet_type == '単勝' and h1 == top1:
            is_hit = True
            payout = int(amount * divs.get('tansho', {}).get('payout', 0) / 100)
        elif bet_type == 'ワイド' and h2:
            for w in divs.get('wide', []):
                if h1 in w['nums'] and h2 in w['nums']:
                    is_hit = True
                    payout = int(amount * w['payout'] / 100)
                    break
        elif bet_type in ('馬連', '馬単') and h2:
            is_hit = (h1 in top3[:2] and h2 in top3[:2])
            if is_hit:
                key = 'umaren' if bet_type == '馬連' else 'umatan'
                payout = int(amount * divs.get(key, {}).get('payout', 0) / 100)

        conn.execute(
            'UPDATE bets SET is_hit=?, payout=? WHERE id=?',
            (1 if is_hit else 0, payout, bet['id']),
        )
        total    += 1
        invested += amount
        recovered += payout
        if is_hit:
            hit += 1

        rc   = result.get('racecourse', '')
        rnum = result.get('race_num', 0)
        rname = result.get('race_name', '')[:6]
        mark = '✅' if is_hit else '❌'
        suffix = f'→¥{payout:,}' if is_hit else '→外れ'
        details.append(f'  {mark} {rc}R{rnum:02d} {rname} {bet_type}#{h1} ¥{amount:,}{suffix}')

    conn.commit()
    conn.close()

    roi = recovered / invested * 100 if invested > 0 else 0
    return {'hit': hit, 'total': total, 'invested': invested,
            'recovered': recovered, 'roi': roi, 'details': details}


def update_bet_results(race_id, results, base_dir=None, db_path=None):
    """レース結果でbetsテーブルのis_hit/payoutを更新"""
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
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

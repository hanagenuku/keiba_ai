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
        CREATE TABLE IF NOT EXISTS race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            race_id TEXT,
            racecourse TEXT,
            race_num INTEGER,
            horse_num INTEGER,
            bracket INTEGER,
            horse_name TEXT,
            popularity INTEGER,
            tansho_odds REAL,
            rl_rank INTEGER,
            win_prob REAL,
            cal_prob REAL,
            fuku_prob REAL,
            rl_rank_raw INTEGER,
            win_prob_raw REAL,
            cal_prob_raw REAL,
            fuku_prob_raw REAL,
            rl_rank_corrected INTEGER,
            win_prob_corrected REAL,
            cal_prob_corrected REAL,
            fuku_prob_corrected REAL,
            correction_factor REAL,
            correction_enabled INTEGER,
            actual_place INTEGER,
            prediction_gap INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_rp_horse ON race_predictions(horse_name, date);
        CREATE INDEX IF NOT EXISTS idx_rp_race  ON race_predictions(race_id);
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id     TEXT,
            horse_num   INTEGER,
            tansho      REAL,
            fukusho     REAL,
            captured_at TEXT,
            source      TEXT DEFAULT 'chokuzen',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, horse_num, captured_at)
        );
        CREATE INDEX IF NOT EXISTS idx_os_race ON odds_snapshots(race_id);
        CREATE TABLE IF NOT EXISTS race_notes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            race_id         TEXT,
            racecourse      TEXT,
            race_num        INTEGER,
            horse_num       INTEGER NOT NULL,
            horse_name      TEXT,
            notes_data      TEXT NOT NULL,
            total_handicap  REAL,
            schema_version  INTEGER,
            free_memo       TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at      TEXT,
            UNIQUE(date, race_id, horse_num)
        );
        CREATE INDEX IF NOT EXISTS idx_notes_horse ON race_notes(horse_name, date);
    ''')
    # 既存DB向けマイグレーション（重複カラムエラーは無視）
    for sql in [
        "ALTER TABLE race_predictions ADD COLUMN bracket INTEGER",
        # 市場補正レイヤー（2026-06-27）— 補正前後の両値を保存
        "ALTER TABLE race_predictions ADD COLUMN rl_rank_raw INTEGER",
        "ALTER TABLE race_predictions ADD COLUMN win_prob_raw REAL",
        "ALTER TABLE race_predictions ADD COLUMN cal_prob_raw REAL",
        "ALTER TABLE race_predictions ADD COLUMN fuku_prob_raw REAL",
        "ALTER TABLE race_predictions ADD COLUMN rl_rank_corrected INTEGER",
        "ALTER TABLE race_predictions ADD COLUMN win_prob_corrected REAL",
        "ALTER TABLE race_predictions ADD COLUMN cal_prob_corrected REAL",
        "ALTER TABLE race_predictions ADD COLUMN fuku_prob_corrected REAL",
        "ALTER TABLE race_predictions ADD COLUMN correction_factor REAL",
        "ALTER TABLE race_predictions ADD COLUMN correction_enabled INTEGER",
        # save_bets_db が書き込む拡張列（新規DBでは CREATE TABLE に無いため追加）
        "ALTER TABLE bets ADD COLUMN racecourse TEXT",
        "ALTER TABLE bets ADD COLUMN distance INTEGER",
        "ALTER TABLE bets ADD COLUMN surface TEXT",
        "ALTER TABLE bets ADD COLUMN running_style TEXT",
        "ALTER TABLE bets ADD COLUMN popularity INTEGER",
        "ALTER TABLE bets ADD COLUMN ai_score REAL",
        "ALTER TABLE bets ADD COLUMN ev_rank INTEGER",
    ]:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    # race_predictions の重複行を排除し (race_id, horse_num) に一意制約を張る。
    # 一意制約が無いと INSERT OR REPLACE が実質ただのINSERTになり、
    # 同一レースを複数回保存した際に重複行が溜まって乖離学習が二重カウントされる。
    try:
        conn.execute("""
            DELETE FROM race_predictions
            WHERE id NOT IN (
                SELECT MAX(id) FROM race_predictions GROUP BY race_id, horse_num
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_rp_uniq "
            "ON race_predictions(race_id, horse_num)"
        )
    except sqlite3.OperationalError:
        pass
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


def save_bets_db(date_str, race_id, bets, base_dir=None, db_path=None,
                 race=None, scored_by_num=None):
    """ベットをDBに保存（重複スキップ方式）

    race          : レース辞書（racecourse/distance/surface の取得に使用）
    scored_by_num : {horse_num: scored_horse} — ai_score/popularity/running_style の取得に使用
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    rc  = (race or {}).get('racecourse', '')
    dst = (race or {}).get('distance', 0)
    srf = (race or {}).get('surface', '')
    snb = scored_by_num or {}

    def _extra(horse_num):
        h = snb.get(horse_num, {})
        return (
            h.get('total', 0),          # ai_score
            h.get('rl_rank', 99),       # ev_rank（RL順位を代用）
            h.get('running_style', ''),
            h.get('popularity') or h.get('_pop') or 99,
        )

    for b in bets:
        if b['type'] == '三連複' and 'tickets' in b:
            for t in b['tickets']:
                existing = conn.execute(
                    'SELECT id FROM bets WHERE race_id=? AND bet_type=? AND horse_num=? AND horse_num2=?',
                    (race_id, '三連複', t[0], t[1]),
                ).fetchone()
                if existing:
                    continue
                ai_sc, ev_r, rs, pop = _extra(t[0])
                conn.execute(
                    'INSERT INTO bets (date,race_id,bet_type,horse_num,horse_name,odds_est,amount,horse_num2,'
                    'racecourse,distance,surface,running_style,popularity,ai_score,ev_rank) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (date_str, race_id, '三連複', t[0], b.get('horse_name', ''),
                     b.get('odds_est', 0), 100, t[1],
                     rc, dst, srf, rs, pop, ai_sc, ev_r),
                )
            continue
        existing = conn.execute(
            'SELECT id FROM bets WHERE race_id=? AND bet_type=? AND horse_num=?',
            (race_id, b['type'], b['nums'][0]),
        ).fetchone()
        if existing:
            continue
        horse_num2 = b['nums'][1] if len(b['nums']) > 1 else 0
        ai_sc, ev_r, rs, pop = _extra(b['nums'][0])
        conn.execute(
            'INSERT INTO bets (date,race_id,bet_type,horse_num,horse_name,odds_est,amount,horse_num2,'
            'racecourse,distance,surface,running_style,popularity,ai_score,ev_rank) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (date_str, race_id, b['type'], b['nums'][0],
             b.get('horse_name', ''), b.get('odds_est', 0), b['amount'], horse_num2,
             rc, dst, srf, rs, pop, ai_sc, ev_r),
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
            "(race_id,date,racecourse,distance,surface,first_3f,last_3f,lap_times,"
            " race_name,race_class,track_condition,num_finishers,weather,pace_label) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (race_id, date_str, r.get('racecourse', ''),
             r.get('distance', 0), r.get('surface', ''),
             r.get('first_3f'), r.get('last_3f'), r.get('lap_times', ''),
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
            "  pace_label     = COALESCE(?, pace_label), "
            "  first_3f       = COALESCE(?, first_3f), "
            "  last_3f        = COALESCE(?, last_3f), "
            "  lap_times      = COALESCE(NULLIF(?, ''), lap_times) "
            "WHERE race_id = ?",
            (r.get('race_name', ''), r.get('race_class', ''),
             r.get('track_condition'), r.get('num_finishers'),
             r.get('weather'), r.get('pace_label'),
             r.get('first_3f'), r.get('last_3f'), r.get('lap_times', ''),
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


def save_race_predictions(race, scored_horses, base_dir=None, db_path=None):
    """全レース・全馬の予測スナップショットを race_predictions に保存。

    予測時（金曜/土日の予想生成後）に呼ぶ。推奨・非推奨を問わず全レース保存。
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    for h in scored_horses:
        _race_id = race.get('id') or race.get('race_id', '')
        _fuku = (h.get('top3_prob') if h.get('top3_prob') is not None
                 else (h.get('fuku_pct', 0) or 0) / 100.0)
        _fuku_raw = (h.get('top3_prob_raw') if h.get('top3_prob_raw') is not None
                     else _fuku)
        conn.execute("""
            INSERT OR REPLACE INTO race_predictions
            (date, race_id, racecourse, race_num, horse_num, bracket, horse_name,
             popularity, tansho_odds, rl_rank, win_prob, cal_prob, fuku_prob,
             rl_rank_raw, win_prob_raw, cal_prob_raw, fuku_prob_raw,
             rl_rank_corrected, win_prob_corrected, cal_prob_corrected, fuku_prob_corrected,
             correction_factor, correction_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race.get('date', ''), _race_id,
            race.get('racecourse', ''), race.get('race_num', 0),
            h.get('horse_num', h.get('num', 0)), h.get('bracket'), h.get('name', ''),
            h.get('popularity', 99), h.get('win_odds') or h.get('odds'),
            # 実使用値（補正後）
            h.get('rl_rank', 99), h.get('win_prob', 0), h.get('cal_prob', 0), _fuku,
            # 補正前（AI素）
            h.get('rl_rank_raw', h.get('rl_rank', 99)),
            h.get('win_prob_raw', h.get('win_prob', 0)),
            h.get('cal_prob_raw', h.get('cal_prob', 0)),
            _fuku_raw,
            # 補正後（明示）
            h.get('rl_rank', 99), h.get('win_prob', 0), h.get('cal_prob', 0), _fuku,
            # 補正情報
            h.get('correction_factor', 1.0),
            1 if race.get('correction_enabled', True) else 0,
        ))
    conn.commit()
    conn.close()


def update_prediction_results(all_results, base_dir=None, db_path=None):
    """レース結果判明後に actual_place と prediction_gap を更新。

    結果取得時（土曜夜/日曜夜）に fetch_and_save_results の後で呼ぶ。
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    updated = 0
    for race in all_results:
        # results オブジェクトは 'race_id' キー、shutuba は 'id' キーを使う
        race_id = race.get('race_id') or race.get('id', '')
        for h in race.get('finishers', []):
            place = h.get('place')
            num   = h.get('num') or h.get('horse_num')
            if not race_id or place is None or num is None:
                continue
            # 枠順は結果ページが確定値なので、ここで race_predictions に充填する
            # （出馬表パースは枠を取得しないため、予測時点では NULL のまま）。
            conn.execute("""
                UPDATE race_predictions
                SET actual_place    = ?,
                    prediction_gap  = rl_rank - ?,
                    bracket         = COALESCE(?, bracket)
                WHERE race_id = ? AND horse_num = ?
            """, (place, place, h.get('bracket'), race_id, num))
            updated += conn.execute('SELECT changes()').fetchone()[0]
    conn.commit()
    conn.close()
    return updated


def get_latest_odds_snapshot_time(db_path):
    """odds_snapshots の最新 captured_at を返す（GAS取込の since に使う）。"""
    conn = _connect(db_path)
    try:
        row = conn.execute('SELECT MAX(captured_at) FROM odds_snapshots').fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return row[0] if row and row[0] else ''


def save_odds_snapshots(rows, base_dir=None, db_path=None):
    """直前オッズログ（GASの getOddsLog が返す行）を odds_snapshots に保存する。

    rows: [{race_id, horse_num, tansho, fukusho, captured_at}, ...]
    (race_id, horse_num, captured_at) の一意制約で重複取込は無視する。

    Returns: 新規保存した行数
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    n = 0
    for r in rows:
        race_id = str(r.get('race_id', ''))
        captured_at = str(r.get('captured_at', ''))
        num = r.get('horse_num')
        if not race_id or not captured_at or num is None:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO odds_snapshots "
            "(race_id, horse_num, tansho, fukusho, captured_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (race_id, int(num), r.get('tansho'), r.get('fukusho'),
             captured_at, r.get('source', 'chokuzen')),
        )
        n += cur.rowcount
    conn.commit()
    conn.close()
    return n


# ── 不利メモ（race_notes）: 手動入力の不利・出遅れ・展開ロスを蓄積する ──────────

def get_note_schema_path(base_dir):
    return os.path.join(base_dir, 'data', 'note_schema.json')


def load_note_schema(base_dir=None, schema_path=None):
    """note_schema.json を読み込む。無ければ空スキーマを返す（安全に no-op）。"""
    path = schema_path or (get_note_schema_path(base_dir) if base_dir else None)
    if not path or not os.path.exists(path):
        return {'version': 0, 'categories': [], 'free_memo': True}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def calc_handicap_from_notes(notes, schema):
    """notes_data(dict) とスキーマから補正値合計を計算する。

    feature=true の項目のみを value × weight で合算する。スキーマに無い
    キーや欠損キーは 0 扱い。スキーマが変わっても保存済み notes から再計算できる。
    """
    if not isinstance(notes, dict):
        return 0.0
    total = 0.0
    for cat in schema.get('categories', []):
        if not cat.get('feature'):
            continue
        try:
            val = float(notes.get(cat['id'], 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        total += val * float(cat.get('weight', 1.0))
    return round(total, 2)


def save_race_notes(rows, base_dir=None, db_path=None, schema=None):
    """不利メモログ（GASの getNotesLog が返す行）を race_notes に保存する。

    rows: [{date, race_id, racecourse, race_num, horse_num, horse_name,
            notes_data(JSON文字列 or dict), free_memo, captured_at}, ...]
    (date, race_id, horse_num) の一意制約で同じ馬は最新入力に上書きする。
    total_handicap はスキーマから自動計算してキャッシュする。

    Returns: 保存（新規 or 上書き）した行数
    """
    path = db_path or get_db_path(base_dir)
    if schema is None:
        schema = load_note_schema(base_dir) if base_dir else {'categories': []}
    schema_version = schema.get('version', 0)
    conn = _connect(path)
    n = 0
    for r in rows:
        date = str(r.get('date', '')).strip()
        num = r.get('horse_num')
        if not date or num is None or str(num) == '':
            continue
        raw = r.get('notes_data', {})
        if isinstance(raw, str):
            try:
                notes = json.loads(raw) if raw.strip() else {}
            except (ValueError, TypeError):
                notes = {}
        else:
            notes = raw or {}
        total = calc_handicap_from_notes(notes, schema)
        captured_at = str(r.get('captured_at', '')) or None
        conn.execute(
            "INSERT INTO race_notes "
            "(date, race_id, racecourse, race_num, horse_num, horse_name, "
            " notes_data, total_handicap, schema_version, free_memo, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date, race_id, horse_num) DO UPDATE SET "
            " racecourse=excluded.racecourse, race_num=excluded.race_num, "
            " horse_name=excluded.horse_name, notes_data=excluded.notes_data, "
            " total_handicap=excluded.total_handicap, "
            " schema_version=excluded.schema_version, "
            " free_memo=excluded.free_memo, updated_at=excluded.updated_at",
            (date, str(r.get('race_id', '')), r.get('racecourse', ''),
             r.get('race_num'), int(num), r.get('horse_name', ''),
             json.dumps(notes, ensure_ascii=False), total, schema_version,
             r.get('free_memo', ''), captured_at),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def get_latest_note_time(db_path):
    """race_notes の最新 updated_at を返す（増分取込の since に使う）。無ければ ''。"""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT MAX(updated_at) FROM race_notes"
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return row[0] if row and row[0] else ''


def recalc_all_handicaps(base_dir=None, db_path=None, schema_path=None):
    """note_schema.json 変更後、保存済み race_notes の total_handicap を再計算する。

    保存済みの notes_data(JSON) から再計算するので、weight や項目が変わっても
    過去データを壊さず追従できる。Returns: 更新した行数。
    """
    path = db_path or get_db_path(base_dir)
    schema = load_note_schema(base_dir, schema_path)
    conn = _connect(path)
    rows = conn.execute("SELECT id, notes_data FROM race_notes").fetchall()
    n = 0
    for row_id, notes_json in rows:
        try:
            notes = json.loads(notes_json) if notes_json else {}
        except (ValueError, TypeError):
            notes = {}
        total = calc_handicap_from_notes(notes, schema)
        conn.execute(
            "UPDATE race_notes SET total_handicap=?, schema_version=? WHERE id=?",
            (total, schema.get('version', 0), row_id),
        )
        n += 1
    conn.commit()
    conn.close()
    return n

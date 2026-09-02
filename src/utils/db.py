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
            actual_place INTEGER,
            prediction_gap INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_rp_horse ON race_predictions(horse_name, date);
        CREATE INDEX IF NOT EXISTS idx_rp_race  ON race_predictions(race_id);
        -- 予想の「時点別」履歴（2026-07-27⑩導入）。
        -- race_predictions は (race_id, horse_num) にUNIQUE制約があり
        -- INSERT OR REPLACE で上書きされるため、当日朝のrefresh実行後は
        -- 前夜の予想が完全に消える。実際に2026-07-26のオッズ障害を
        -- 「DBだけ見て起きていない」と誤判定する事故が起きた。
        -- 集計側(11箇所)を壊さないよう race_predictions はそのまま残し、
        -- 時点別の記録はこの別テーブルに追記する。
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT,
            race_id     TEXT,
            racecourse  TEXT,
            race_num    INTEGER,
            horse_num   INTEGER,
            horse_name  TEXT,
            snapshot    TEXT,     -- 'initial'(前夜/前日生成) | 'refresh'(当日朝)
            popularity  INTEGER,
            tansho_odds REAL,
            rl_rank     INTEGER,
            win_prob    REAL,
            cal_prob    REAL,
            fuku_prob   REAL,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, horse_num, snapshot)
        );
        CREATE INDEX IF NOT EXISTS idx_ps_race ON prediction_snapshots(race_id);
        CREATE INDEX IF NOT EXISTS idx_ps_date ON prediction_snapshots(date, snapshot);
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
        -- 発走時刻。オッズ時系列は「発走何分前か」で揃えないとレース間で比較できない。
        -- parse_header は start_time を取っていたが、どのDBにも保存されていなかった
        -- （races.raw_json に245行中6行だけ残っていた。2026-08-27発見）。
        CREATE TABLE IF NOT EXISTS race_schedule (
            race_id    TEXT PRIMARY KEY,
            date       TEXT,
            racecourse TEXT,
            race_num   INTEGER,
            post_time  TEXT,          -- 'HH:MM'（JST）
            n_horses   INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_os_race ON odds_snapshots(race_id);
        -- 全券種の実配当。parse_dividends は8券種すべてを取得しているのに、
        -- 保存していたのは単勝・複勝だけ（results テーブル）で、
        -- ワイド・馬連・馬単・三連複などの組み合わせ配当は
        -- パースした直後に捨てられていた（2026-07-31に判明）。
        -- そのため単勝・複勝以外は回収率を一度も検証できない状態が続いていた。
        CREATE TABLE IF NOT EXISTS race_dividends (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id    TEXT NOT NULL,
            date       TEXT,
            bet_type   TEXT NOT NULL,   -- tansho/fukusho/wakuren/umaren/umatan/wide/sanrenpuku/sanrentan
            combo      TEXT NOT NULL,   -- '3' / '3-8' / '3-8-10'（馬単・三連単は着順どおり）
            payout     INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, bet_type, combo)
        );
        CREATE INDEX IF NOT EXISTS idx_rd_race ON race_dividends(race_id);
        -- 画面に実際に出した買い目（gumbel_bets = 軸1頭ベース）。
        -- `bets` は旧 make_bets() の出力・推奨レースのみで、画面とは別物だった
        -- （2026-08-31 の棚卸しで判明。詳細は src/betting/displayed_bets.py）。
        CREATE TABLE IF NOT EXISTS displayed_bets (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            date           TEXT NOT NULL,
            race_id        TEXT NOT NULL,
            racecourse     TEXT,
            race_num       INTEGER,
            is_recommended INTEGER DEFAULT 0,
            snapshot       TEXT NOT NULL,   -- 'initial'(前夜/前日) | 'refresh'(当日)
            bet_type       TEXT NOT NULL,   -- race_dividends と同じ表記
            combo          TEXT NOT NULL,   -- 昇順ハイフン連結 '5' / '5-7' / '2-5-7'
            amount         REAL,            -- 1点あたりの金額（行の合計を点数で等分）
            is_hit         INTEGER DEFAULT -1,
            payout         REAL DEFAULT 0,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, snapshot, bet_type, combo)
        );
        CREATE INDEX IF NOT EXISTS idx_db_date ON displayed_bets(date);
        CREATE INDEX IF NOT EXISTS idx_db_race ON displayed_bets(race_id);
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
    # shadow_bets の重複行を排除し race_id に一意制約を張る。
    # 一意制約が無いと record_all_shadow_bets の INSERT OR IGNORE が実質ただの
    # INSERT になり、ワークフローの再実行（--force やリトライ）でレースが
    # 二重に記録され、rl_accuracy / 盲点パターン集計が二重カウントされる。
    try:
        conn.execute("""
            DELETE FROM shadow_bets
            WHERE id NOT IN (
                SELECT MAX(id) FROM shadow_bets GROUP BY race_id
            )
        """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_bets_uniq "
            "ON shadow_bets(race_id)"
        )
    except sqlite3.OperationalError:
        pass
    # cal_prob > 1.0 の修正（旧market_correction残骸）
    try:
        conn.execute("UPDATE race_predictions SET cal_prob = 0.99 WHERE cal_prob > 1.0")
    except sqlite3.OperationalError:
        pass
    # popularity=99 をオッズ順位で補填
    try:
        races_to_fix = conn.execute(
            "SELECT DISTINCT race_id FROM race_predictions "
            "WHERE (popularity = 99 OR popularity IS NULL) AND tansho_odds IS NOT NULL"
        ).fetchall()
        for (rid,) in races_to_fix:
            horses = conn.execute(
                "SELECT horse_num FROM race_predictions "
                "WHERE race_id = ? ORDER BY COALESCE(tansho_odds, 9999)", (rid,)
            ).fetchall()
            for rank, (hnum,) in enumerate(horses, 1):
                conn.execute(
                    "UPDATE race_predictions SET popularity = ? "
                    "WHERE race_id = ? AND horse_num = ? AND (popularity = 99 OR popularity IS NULL)",
                    (rank, rid, hnum))
    except sqlite3.OperationalError:
        pass
    # correction_enabled/factor の履歴をクリア（旧market_correction残骸）
    try:
        conn.execute(
            "UPDATE race_predictions SET correction_enabled = NULL, correction_factor = NULL "
            "WHERE correction_enabled IS NOT NULL")
    except sqlite3.OperationalError:
        pass
    _ensure_odds_snapshot_columns(conn)
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
        # 血統（父・母の父）。2026-07-17〜、accessU.htmlから取得
        "ALTER TABLE horse_history ADD COLUMN sire TEXT",
        "ALTER TABLE horse_history ADD COLUMN dam_sire TEXT",
        # 調教師所属（栗東/美浦）。2026-07-21〜、結果ページの調教師欄「名前(栗東)」表記から取得
        "ALTER TABLE horse_history ADD COLUMN trainer_affiliation TEXT",
        # コーナー通過順位（同着グルーピング表記込みの生テキスト）。2026-07-21〜
        "ALTER TABLE race_history ADD COLUMN corner_pass_3 TEXT",
        "ALTER TABLE race_history ADD COLUMN corner_pass_4 TEXT",
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
            " race_name,race_class,track_condition,num_finishers,weather,pace_label,"
            " corner_pass_3,corner_pass_4) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (race_id, date_str, r.get('racecourse', ''),
             r.get('distance', 0), r.get('surface', ''),
             r.get('first_3f'), r.get('last_3f'), r.get('lap_times', ''),
             r.get('race_name', ''), r.get('race_class', ''),
             r.get('track_condition', '良'), r.get('num_finishers', 0),
             r.get('weather'), r.get('pace_label'),
             r.get('corner_pass_3'), r.get('corner_pass_4')),
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
            "  lap_times      = COALESCE(NULLIF(?, ''), lap_times), "
            "  corner_pass_3  = COALESCE(?, corner_pass_3), "
            "  corner_pass_4  = COALESCE(?, corner_pass_4) "
            "WHERE race_id = ?",
            (r.get('race_name', ''), r.get('race_class', ''),
             r.get('track_condition'), r.get('num_finishers'),
             r.get('weather'), r.get('pace_label'),
             r.get('first_3f'), r.get('last_3f'), r.get('lap_times', ''),
             r.get('corner_pass_3'), r.get('corner_pass_4'),
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
                " bracket,corner_all,win_odds,sire,dam_sire,trainer_affiliation) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                 h.get('win_odds'), h.get('sire', ''), h.get('dam_sire', ''),
                 h.get('trainer_affiliation')),
            )
            new_horses += cur2.rowcount
            # Stage 3 rescrape 用：既存行の新フィールドを UPDATE で充填
            #
            # ⚠ popularity は「取得できなかった時に 99 というセンチネル値で
            # INSERT される」列であり NULL にならない（parse_result_soup:
            # `int(pop_m.group(1)) if pop_m else 99`）。そのため COALESCE では
            # 直せず、2026-07-04以降の列崩れ（2026-08-03②③）で 99 が入った
            # 4,484行は再スクレイプしても永久に復旧しなかった。
            # popularity は残差学習モデルの base_margin（市場アンカー）の
            # 唯一の入力であり、99 だと engine.py の
            # `float(pop) if 0 < pop < 99 else nan` で NaN に落ちて
            # アンカーが完全に死ぬ（実測: 健全行 AUC 0.8207 / 死亡行 0.7091）。
            # そのため「新しい値が正当な人気(1〜98)の時だけ上書きする」形にし、
            # 取得失敗(99)で既存の正しい値を壊さないようにする。
            _pop_new = h.get('popularity')
            _place_new = h.get('place')
            _a3f = h.get('agari3f') or 0.0
            _tan = h.get('tansho_payout') or 0
            _fuku = h.get('fukusho_payout') or 0
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
                "  surface          = COALESCE(NULLIF(?, ''), surface), "
                "  sire             = COALESCE(NULLIF(?, ''), sire), "
                "  dam_sire         = COALESCE(NULLIF(?, ''), dam_sire), "
                "  trainer_affiliation = COALESCE(?, trainer_affiliation), "
                "  trainer          = COALESCE(NULLIF(?, ''), trainer), "
                # ⚠ 以下は2026-08-09の「対になっている処理」監査で追加した列。
                #   trainer は 2026-08-06 の popularity 事故のときに追加されたが、
                #   結果ページの**同じ行から取る** jockey は追従していなかった。
                #   payout 2列は回収率の分子そのもので、壊れても直す手段が
                #   無い状態だった。place / agari3f / running_style も同様に
                #   「再スクレイプしても既存行は直らない」列だった。
                "  jockey           = COALESCE(NULLIF(?, ''), jockey), "
                "  running_style    = COALESCE(NULLIF(?, ''), running_style), "
                "  agari3f          = CASE WHEN ? > 0 THEN ? ELSE agari3f END, "
                "  tansho_payout    = CASE WHEN ? > 0 THEN ? ELSE tansho_payout END, "
                "  fukusho_payout   = CASE WHEN ? > 0 THEN ? ELSE fukusho_payout END, "
                # place は取得失敗時に 99 で INSERT されるセンチネル列。
                # popularity と同じ理由で COALESCE では直せない。
                "  place            = CASE WHEN ? BETWEEN 1 AND 98 "
                "                          THEN ? ELSE place END, "
                "  popularity       = CASE WHEN ? BETWEEN 1 AND 98 "
                "                          THEN ? ELSE popularity END "
                "WHERE race_id = ? AND horse_num = ?",
                (h.get('finish_time'), h.get('time_diff_sec'),
                 h.get('chakusa_text', ''), h.get('margin'),
                 r.get('race_class', ''), h.get('agari_rank'),
                 h.get('weight_load'), h.get('sex', ''), h.get('age'),
                 h.get('body_weight'), h.get('body_weight_diff'),
                 h.get('bracket'), h.get('corner_all', ''),
                 h.get('win_odds'),
                 h.get('surface', r.get('surface', '')),
                 h.get('sire', ''), h.get('dam_sire', ''),
                 h.get('trainer_affiliation'),
                 h.get('trainer', ''),
                 h.get('jockey', ''),
                 h.get('running_style', ''),
                 _a3f, _a3f,
                 _tan, _tan,
                 _fuku, _fuku,
                 _place_new, _place_new,
                 _pop_new, _pop_new,
                 race_id, h.get('num', 0)),
            )

    conn.commit()
    conn.close()
    print(f'📚 history.db に追記: {new_races}レース / {new_horses}頭 (重複スキップ済み)')

# parse_dividends が返すキー → 保存する券種名
# 値が dict（1組だけ）か list（複数組）かが券種で違うため両方を扱う
_DIVIDEND_KEYS = ('tansho', 'fukusho', 'wakuren', 'umaren',
                  'umatan', 'wide', 'sanrenpuku', 'sanrentan')


def _dividend_rows(divs):
    """parse_dividends の出力を (bet_type, combo, payout) の並びに正規化する。

    combo は馬番をハイフンでつないだ文字列。
    **馬単・三連単は着順に意味があるため並べ替えない**。
    それ以外（馬連・ワイド・三連複・枠連）は順不同なので昇順に正規化し、
    照合時に組の並び順で取りこぼさないようにする。
    """
    ordered = {'umatan', 'sanrentan'}
    out = []
    for bt in _DIVIDEND_KEYS:
        v = divs.get(bt)
        if not v:
            continue
        entries = v if isinstance(v, list) else [v]
        for e in entries:
            if not isinstance(e, dict):
                continue
            nums = e.get('nums')
            if nums is None:
                n = e.get('num')
                nums = [n] if n is not None else None
            if not nums:
                continue
            try:
                nums = [int(x) for x in nums]
            except (TypeError, ValueError):
                continue
            if bt not in ordered:
                nums = sorted(nums)
            payout = e.get('payout')
            try:
                payout = int(payout)
            except (TypeError, ValueError):
                continue
            out.append((bt, '-'.join(str(x) for x in nums), payout))
    return out


def save_dividends_db(all_results, base_dir=None, db_path=None):
    """全券種の実配当を race_dividends に保存する。

    ⚠ これ以降に取得したレースぶんしか貯まらない。過去分の組み合わせ配当は
    どこにも保存されていないため、遡って復元することはできない
    （必要なら結果ページの再スクレイプが要る）。

    Returns:
        dict: {'races': 保存対象レース数, 'rows': 追加行数}
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    races = rows = 0
    for r in all_results or []:
        race_id = r.get('race_id') or r.get('id')
        divs = r.get('dividends') or {}
        if not race_id or not divs:
            continue
        recs = _dividend_rows(divs)
        if not recs:
            continue
        races += 1
        date = r.get('date') or (r.get('info') or {}).get('date')
        for bt, combo, payout in recs:
            cur = conn.execute(
                'INSERT OR IGNORE INTO race_dividends '
                '(race_id, date, bet_type, combo, payout) VALUES (?,?,?,?,?)',
                (race_id, date, bt, combo, payout))
            rows += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    conn.close()
    return {'races': races, 'rows': rows}


def save_displayed_bets(rows, base_dir=None, db_path=None):
    """画面に出した買い目を保存する。同一 (race_id, snapshot, 券種, 組番) は上書き。

    再実行・リトライで点数が二重に積み上がらないよう UNIQUE で潰す。
    ただし金額は上書きしたい（オッズが動けば点数配分が変わる）ので
    INSERT OR IGNORE ではなく明示的に UPDATE する。
    """
    if not rows:
        return 0
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    n = 0
    for r in rows:
        conn.execute(
            """INSERT INTO displayed_bets
               (date, race_id, racecourse, race_num, is_recommended,
                snapshot, bet_type, combo, amount)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(race_id, snapshot, bet_type, combo) DO UPDATE SET
                 amount         = excluded.amount,
                 is_recommended = excluded.is_recommended,
                 racecourse     = excluded.racecourse,
                 race_num       = excluded.race_num""",
            (r['date'], r['race_id'], r.get('racecourse', ''), r.get('race_num', 0),
             r.get('is_recommended', 0), r.get('snapshot', 'initial'),
             r['bet_type'], r['combo'], r.get('amount', 0)),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def settle_displayed_bets(base_dir=None, db_path=None, hist_db_path=None):
    """displayed_bets を race_dividends と着順で決済する。

    未決済(is_hit=-1)の行だけを対象にするので、何度呼んでも二重計上しない。
    的中しているのに配当が引けない行は payout=0 のまま is_hit=1 にする
    （回収率の分母には入るが分子には入らない＝過大評価しない側に倒す）。
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    conn.row_factory = sqlite3.Row

    pending = conn.execute(
        'SELECT * FROM displayed_bets WHERE is_hit = -1'
    ).fetchall()
    if not pending:
        conn.close()
        return {'settled': 0, 'hit': 0, 'invested': 0.0, 'recovered': 0.0, 'no_payout': 0}

    race_ids = {r['race_id'] for r in pending}
    qmarks = ','.join('?' * len(race_ids))
    div = {}
    for row in conn.execute(
            f'SELECT race_id, bet_type, combo, payout FROM race_dividends '
            f'WHERE race_id IN ({qmarks})', tuple(race_ids)):
        div[(row['race_id'], row['bet_type'], row['combo'])] = row['payout']

    # 着順は history.db 側にしかない
    hp = hist_db_path or os.path.join(base_dir or '.', 'data', 'history.db')
    top3 = {}
    if os.path.exists(hp):
        hc = sqlite3.connect(hp)
        for rid, hn, pl in hc.execute(
                f'SELECT race_id, horse_num, place FROM horse_history '
                f'WHERE place BETWEEN 1 AND 3 AND race_id IN ({qmarks}) '
                f'ORDER BY race_id, place', tuple(race_ids)):
            top3.setdefault(rid, []).append(hn)
        hc.close()

    settled = hit = no_payout = 0
    invested = recovered = 0.0
    for b in pending:
        t = top3.get(b['race_id'])
        if not t or len(t) < 3:
            continue  # 結果がまだ入っていない。次回に持ち越す
        nums = [int(x) for x in b['combo'].split('-')]
        bt = b['bet_type']
        if bt == 'tansho':
            won = nums[0] == t[0]
        elif bt == 'fukusho':
            won = nums[0] in t
        elif bt == 'wide':
            won = len(set(nums) & set(t)) == 2
        elif bt == 'umaren':
            won = set(nums) <= set(t[:2])
        elif bt == 'sanrenpuku':
            won = set(nums) == set(t)
        else:
            continue

        payout = 0.0
        if won:
            p = div.get((b['race_id'], bt, b['combo']))
            if p is None:
                no_payout += 1
            else:
                payout = (b['amount'] or 0) * p / 100.0
        conn.execute('UPDATE displayed_bets SET is_hit=?, payout=? WHERE id=?',
                     (1 if won else 0, payout, b['id']))
        settled += 1
        hit += 1 if won else 0
        invested += b['amount'] or 0
        recovered += payout

    conn.commit()
    conn.close()
    return {'settled': settled, 'hit': hit, 'invested': invested,
            'recovered': recovered, 'no_payout': no_payout}


def get_dividends_for_races(race_ids, base_dir=None, db_path=None):
    """race_dividends から {race_id: parse_dividends 相当の dict} を組み立てる。

    settle_bet_simulation / build_results_from_db がそのまま使える形に戻す。
    """
    path = db_path or get_db_path(base_dir)
    if not os.path.exists(path):
        return {}
    multi = {'fukusho', 'wide'}          # 1レースに複数組ある券種
    out = {}
    try:
        conn = _connect(path)
        ids = list(race_ids) if race_ids is not None else None
        if ids:
            res = []
            for i in range(0, len(ids), 500):     # SQLite の変数上限対策
                chunk = ids[i:i + 500]
                marks = ','.join('?' * len(chunk))
                res += conn.execute(
                    f'SELECT race_id, bet_type, combo, payout FROM race_dividends '
                    f'WHERE race_id IN ({marks})', tuple(chunk)).fetchall()
        else:
            res = conn.execute(
                'SELECT race_id, bet_type, combo, payout FROM race_dividends').fetchall()
        conn.close()
    except Exception:
        return {}
    for race_id, bt, combo, payout in res:
        try:
            nums = [int(x) for x in str(combo).split('-') if x != '']
        except ValueError:
            continue
        d = out.setdefault(race_id, {})
        if bt == 'fukusho':
            d.setdefault('fukusho', []).append({'num': nums[0], 'payout': payout})
        elif bt in multi:
            d.setdefault(bt, []).append({'nums': nums, 'payout': payout})
        elif bt == 'tansho':
            d['tansho'] = {'num': nums[0], 'payout': payout}
        else:
            d[bt] = {'nums': nums, 'payout': payout}
    return out


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
            # 馬連は順不同、馬単は着順まで一致して初めて的中。
            # 2026-07-27⑤修正: 以前は両方とも `h1 in top3[:2] and h2 in top3[:2]`
            # で判定しており、馬単で着順が逆でも的中扱いになっていた
            top2 = top3[:2]
            if bet_type == '馬連':
                is_hit = (h1 in top2 and h2 in top2)
            else:
                is_hit = ([h1, h2] == top2)
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


def _parse_sim_horse_nums(bet_type, horse_num_str):
    """bet_simulation.horse_num の文字列から馬番リストを取り出す。

    log_bet_simulation が書き込む形式:
        単勝/複勝 : '3'
        ワイド/馬連: '3-8'
        馬単      : '3->8'   （順序に意味がある）
        三連複    : '3-8-10'
    """
    s = str(horse_num_str or '').strip()
    if not s:
        return []
    sep = '->' if '->' in s else '-'
    try:
        return [int(x) for x in s.split(sep) if x.strip()]
    except ValueError:
        return []


def settle_bet_simulation(all_results, base_dir=None, db_path=None):
    """bet_simulation の未決済行(is_hit=-1)をレース結果で決済する。

    log_bet_simulation は全券種を「買った想定」で記録するが、決済処理が
    どこにも実装されておらず 2,160行が is_hit=-1 のまま放置されていた
    （2026-07-27④で発覚）。実際に買った bets（数百点）より遥かに多い
    サンプルが得られるため、券種別の有効性を評価するために決済する。

    payout は odds_est（推定オッズ）ではなく**実際の配当**を使う。
    推定オッズで決済すると「推定が甘い券種ほど成績が良く見える」という
    バイアスが入るため。実配当が取得できない券種・組み合わせは
    is_hit のみ更新し payout=0 のままにする（回収率の分母には入るが
    分子には入らないため過大評価にならない）。

    Returns:
        dict: {settled, hit, by_type: {bet_type: {n, hit, payout}}}
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        'SELECT id, race_id, bet_type, horse_num FROM bet_simulation WHERE is_hit=-1'
    ).fetchall()

    by_race = {r.get('race_id'): r for r in all_results if r.get('race_id')}
    settled = hit = 0
    by_type = {}

    for row in rows:
        result = by_race.get(row['race_id'])
        if not result:
            continue

        fin  = result.get('finishers', [])
        divs = result.get('dividends', {})
        order = [h['num'] for h in fin]      # 着順どおりの馬番
        top3  = order[:3]
        top2  = order[:2]
        top1  = order[0] if order else 0

        # 複勝の配当対象は出走頭数で決まる
        # （JRA: 8頭以上→3着まで / 5〜7頭→2着まで / 4頭以下→複勝の発売なし）。
        # 7頭立ての3着馬を的中扱いにすると券種別評価が汚れるため、
        # num_runners が渡された場合のみ厳密に判定する。
        # 未指定の呼び出し元（fetch_results 経由の既存パス）は
        # 従来どおり3着までとみなし、挙動を変えない。
        n_run = result.get('num_runners')
        if n_run:
            fuku_n = 3 if n_run >= 8 else 2 if n_run >= 5 else 0
        else:
            fuku_n = 3
        fuku_zone = order[:fuku_n]

        nums = _parse_sim_horse_nums(row['bet_type'], row['horse_num'])
        if not nums:
            continue

        bt = row['bet_type']
        is_hit, payout = False, 0

        if bt == '単勝':
            if nums[0] == top1:
                is_hit = True
                payout = int(divs.get('tansho', {}).get('payout', 0))
        elif bt == '複勝':
            if nums[0] in fuku_zone:
                is_hit = True
                for f in divs.get('fukusho', []):
                    if f['num'] == nums[0]:
                        payout = int(f['payout'])
                        break
        elif bt == 'ワイド' and len(nums) >= 2:
            if nums[0] in top3 and nums[1] in top3:
                is_hit = True
                for w in divs.get('wide', []):
                    if set(w['nums']) == set(nums[:2]):
                        payout = int(w['payout'])
                        break
        elif bt == '馬連' and len(nums) >= 2:
            if set(nums[:2]) == set(top2):
                is_hit = True
                payout = int(divs.get('umaren', {}).get('payout', 0))
        elif bt == '馬単' and len(nums) >= 2:
            # 馬単は着順まで一致して初めて的中（順序を見る）
            if nums[:2] == top2:
                is_hit = True
                payout = int(divs.get('umatan', {}).get('payout', 0))
        elif bt == '三連複' and len(nums) >= 3:
            if set(nums[:3]) == set(top3):
                is_hit = True
                payout = int(divs.get('sanrenpuku', {}).get('payout', 0))
        else:
            continue  # 未知の券種は触らない

        conn.execute(
            'UPDATE bet_simulation SET is_hit=?, payout=? WHERE id=?',
            (1 if is_hit else 0, payout, row['id']),
        )
        settled += 1
        if is_hit:
            hit += 1
        st = by_type.setdefault(bt, {'n': 0, 'hit': 0, 'payout': 0})
        st['n'] += 1
        st['hit'] += 1 if is_hit else 0
        st['payout'] += payout

    conn.commit()
    conn.close()
    return {'settled': settled, 'hit': hit, 'by_type': by_type}


def build_results_from_db(race_ids=None, base_dir=None, db_path=None,
                          hist_db_path=None):
    """DBに保存済みのレース結果から all_results 形式を再構築する。

    settle_bet_simulation() は fetch_results() が返す「その回スクレイプした
    結果」しか参照しないため、過去に蓄積された未決済 bet_simulation は
    週次ワークフローを何度回しても決済されない（2026-07-28に発覚）。
    ここでは keiba.db / history.db に既に入っている結果から同じ形を
    組み立て、settle 側の判定ロジックをそのまま再利用する。

    ⚠ 単勝・複勝の実配当は保存されているが、ワイド・馬連・馬単・三連複の
    組み合わせ配当はどのテーブルにも存在しない。そのため dividends には
    tansho / fukusho のみを載せる。settle 側は該当配当が無ければ
    payout=0 のまま is_hit だけ確定するため、**的中率は正しく求まるが
    回収率は算出できない**（分子に入らないだけなので過大評価にはならない）。

    Args:
        race_ids: 対象race_idの集合。Noneなら全レース。

    Returns:
        list[dict]: {'race_id', 'finishers', 'dividends', 'num_runners'}
    """
    main_path = db_path or (get_db_path(base_dir) if base_dir else None)
    hist_path = hist_db_path or (get_history_db_path(base_dir) if base_dir else None)

    wanted = set(race_ids) if race_ids is not None else None
    by_race = {}

    def _ingest(rows):
        for rid, place, num, tp, fp in rows:
            if not rid or num is None:
                continue
            if wanted is not None and rid not in wanted:
                continue
            try:
                place = int(place or 0)
                num = int(num)
            except (TypeError, ValueError):
                continue
            if place <= 0:          # 取消・除外・失格は着順を持たない
                continue
            # 同一(race, horse)が複数ソースに存在する場合は先に読んだ方を採用
            by_race.setdefault(rid, {}).setdefault(num, (place, tp or 0, fp or 0))

    # 組み合わせ配当（ワイド・馬連・馬単・三連複）は race_dividends に貯まる。
    # 2026-07-31より前に取得したレースには存在しないため、取れた分だけ使う。
    combo_divs = get_dividends_for_races(wanted, db_path=main_path) if main_path else {}

    # history.db を優先（未決済レースの97%をカバー）、keiba.db.results で補完
    for path, sql in (
        (hist_path, 'SELECT race_id, place, horse_num, tansho_payout, '
                    'fukusho_payout FROM horse_history'),
        (main_path, 'SELECT race_id, place, horse_num, tansho_payout, '
                    'fukusho_payout FROM results'),
    ):
        if not path or not os.path.exists(path):
            continue
        try:
            c = _connect(path)
            _ingest(c.execute(sql).fetchall())
            c.close()
        except Exception:
            continue

    results = []
    for rid, horses in by_race.items():
        ordered = sorted(horses.items(), key=lambda kv: kv[1][0])   # 着順昇順
        tansho_payout = 0
        fukusho = []
        for num, (place, tp, fp) in ordered:
            if place == 1 and tp:
                tansho_payout = int(tp)
            if fp:
                fukusho.append({'num': num, 'payout': int(fp)})
        # 組み合わせ配当を先に載せ、単勝・複勝は horse_history 側の値を優先する
        # （こちらは馬ごとに紐づいており取りこぼしが少ないため）
        divs = dict(combo_divs.get(rid) or {})
        if tansho_payout:
            divs['tansho'] = {'payout': tansho_payout}
        if fukusho:
            divs['fukusho'] = fukusho
        results.append({
            'race_id':     rid,
            'finishers':   [{'num': num} for num, _ in ordered],
            'dividends':   divs,
            'num_runners': len(ordered),
        })
    return results


def backfill_bet_simulation(base_dir=None, db_path=None, hist_db_path=None):
    """過去に蓄積された未決済 bet_simulation を、DB内の既存結果で決済する。

    is_hit=-1 の行だけを対象にするため、何度実行しても二重計上されない。

    Returns:
        dict: settle_bet_simulation の戻り値に加えて
              races_total / races_covered / races_missing /
              payout_unavailable（的中したが実配当が保存されていない件数）
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    pending = [r[0] for r in conn.execute(
        'SELECT DISTINCT race_id FROM bet_simulation WHERE is_hit=-1'
    ).fetchall() if r[0]]
    conn.close()

    empty = {'settled': 0, 'hit': 0, 'by_type': {}, 'races_total': 0,
             'races_covered': 0, 'races_missing': 0, 'payout_unavailable': {}}
    if not pending:
        return empty

    results = build_results_from_db(race_ids=set(pending), base_dir=base_dir,
                                    db_path=path, hist_db_path=hist_db_path)
    covered = {r['race_id'] for r in results}
    out = settle_bet_simulation(results, base_dir=base_dir, db_path=path)

    # 今回決済した範囲で「的中したが実配当が無い」件数を券種別に数える
    unavail = {}
    if covered:
        conn = _connect(path)
        conn.row_factory = sqlite3.Row
        marks = ','.join('?' * len(covered))
        for r in conn.execute(
            f'SELECT bet_type, COUNT(*) n FROM bet_simulation '
            f'WHERE is_hit=1 AND payout=0 AND race_id IN ({marks}) '
            f'GROUP BY bet_type', tuple(covered)
        ):
            unavail[r['bet_type']] = r['n']
        conn.close()

    out.update({'races_total': len(set(pending)),
                'races_covered': len(covered),
                'races_missing': len(set(pending) - covered),
                'payout_unavailable': unavail})
    return out


def save_race_predictions(race, scored_horses, base_dir=None, db_path=None,
                          snapshot='initial'):
    """全レース・全馬の予測スナップショットを保存する。

    予測時（金曜/土日の予想生成後）に呼ぶ。推奨・非推奨を問わず全レース保存。

    2つのテーブルに書く:
      race_predictions    : (race_id, horse_num) UNIQUE。常に最新で上書きされる。
                            既存の集計コード（乖離分析・市場KPI・エラータグ等
                            11箇所）が「1頭1行」を前提にしているため構造は変えない
      prediction_snapshots: (race_id, horse_num, snapshot) UNIQUE。時点別に併存する。
                            当日朝のrefreshで前夜の予想が消える問題への対応
                            （2026-07-27⑩）

    Args:
        snapshot: 'initial'（前夜/前日の通常生成）または 'refresh'（当日朝の再生成）
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    for h in scored_horses:
        _race_id = race.get('id') or race.get('race_id', '')
        _fuku = (h.get('top3_prob') if h.get('top3_prob') is not None
                 else (h.get('fuku_pct', 0) or 0) / 100.0)
        _row = (
            race.get('date', ''), _race_id,
            race.get('racecourse', ''), race.get('race_num', 0),
            h.get('horse_num', h.get('num', 0)), h.get('name', ''),
            h.get('popularity', 99), h.get('win_odds') or h.get('odds'),
            h.get('rl_rank', 99), h.get('win_prob', 0), h.get('cal_prob', 0), _fuku,
        )
        conn.execute("""
            INSERT OR REPLACE INTO race_predictions
            (date, race_id, racecourse, race_num, horse_num, bracket, horse_name,
             popularity, tansho_odds, rl_rank, win_prob, cal_prob, fuku_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race.get('date', ''), _race_id,
            race.get('racecourse', ''), race.get('race_num', 0),
            h.get('horse_num', h.get('num', 0)), h.get('bracket'), h.get('name', ''),
            h.get('popularity', 99), h.get('win_odds') or h.get('odds'),
            h.get('rl_rank', 99), h.get('win_prob', 0), h.get('cal_prob', 0), _fuku,
        ))
        # 時点別の履歴。同一 snapshot の再実行（--force/リトライ）は上書きし、
        # 異なる snapshot は別行として併存させる
        conn.execute("""
            INSERT OR REPLACE INTO prediction_snapshots
            (date, race_id, racecourse, race_num, horse_num, horse_name, snapshot,
             popularity, tansho_odds, rl_rank, win_prob, cal_prob, fuku_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            _row[0], _row[1], _row[2], _row[3], _row[4], _row[5], snapshot,
            _row[6], _row[7], _row[8], _row[9], _row[10], _row[11],
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


def compare_prediction_snapshots(base_dir=None, db_path=None, date_from=None):
    """『前夜の予想』と『当日朝refreshの予想』を突合して差分を返す。

    当日朝のrefreshは前夜の予想を上書きするため、race_predictions だけでは
    「オッズ変動で予想がどう変わったか」「どちらが正しかったか」を後から
    検証できない。prediction_snapshots に両時点が残るようになったため、
    それを結果(actual_place)と突き合わせて評価する。

    Returns:
        dict: {
          'n_horses', 'n_races',
          'rank_changed'   : RL順位が変わった馬の数,
          'fav_changed'    : AI本命(RL1)が入れ替わったレース数,
          'initial_rl1_win': 前夜のRL1が勝った数,
          'refresh_rl1_win': 朝のRL1が勝った数,
          'rows'           : 差分が出た馬の明細（最大200件）
        }
        両時点が揃ったデータが無ければ None
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    where = "AND i.date >= ?" if date_from else ""
    args = (date_from,) if date_from else ()
    rows = conn.execute(f"""
        SELECT i.race_id, i.date, i.racecourse, i.race_num, i.horse_num, i.horse_name,
               i.popularity ip, r.popularity rp,
               i.tansho_odds io, r.tansho_odds ro,
               i.rl_rank irl, r.rl_rank rrl,
               i.win_prob iwp, r.win_prob rwp,
               p.actual_place
        FROM prediction_snapshots i
        JOIN prediction_snapshots r
          ON r.race_id = i.race_id AND r.horse_num = i.horse_num
         AND r.snapshot = 'refresh'
        LEFT JOIN race_predictions p
          ON p.race_id = i.race_id AND p.horse_num = i.horse_num
        WHERE i.snapshot = 'initial' {where}
    """, args).fetchall()
    conn.close()

    if not rows:
        return None

    races = {}
    rank_changed = 0
    detail = []
    for r in rows:
        if r['irl'] != r['rrl']:
            rank_changed += 1
            if len(detail) < 200:
                detail.append({
                    'date': r['date'], 'racecourse': r['racecourse'],
                    'race_num': r['race_num'], 'horse_num': r['horse_num'],
                    'horse_name': r['horse_name'],
                    'odds': (r['io'], r['ro']), 'pop': (r['ip'], r['rp']),
                    'rl': (r['irl'], r['rrl']),
                    'win_prob': (r['iwp'], r['rwp']),
                    'actual_place': r['actual_place'],
                })
        g = races.setdefault(r['race_id'], {'i1': None, 'r1': None})
        if r['irl'] == 1:
            g['i1'] = r
        if r['rrl'] == 1:
            g['r1'] = r

    fav_changed = sum(
        1 for g in races.values()
        if g['i1'] and g['r1'] and g['i1']['horse_num'] != g['r1']['horse_num'])
    i_win = sum(1 for g in races.values()
                if g['i1'] and g['i1']['actual_place'] == 1)
    r_win = sum(1 for g in races.values()
                if g['r1'] and g['r1']['actual_place'] == 1)

    return {
        'n_horses': len(rows), 'n_races': len(races),
        'rank_changed': rank_changed, 'fav_changed': fav_changed,
        'initial_rl1_win': i_win, 'refresh_rl1_win': r_win,
        'rows': detail,
    }


def get_latest_odds_snapshot_time(db_path):
    """odds_snapshots の最新 captured_at を返す（GAS取込の since に使う）。"""
    conn = _connect(db_path)
    try:
        row = conn.execute('SELECT MAX(captured_at) FROM odds_snapshots').fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    return row[0] if row and row[0] else ''


_ODDS_SNAPSHOT_EXTRA_COLS = (
    # 複勝は範囲(X.X-Y.Y)で出る。従来は (min+max)/2 に潰していたが、
    # 範囲の広さ自体が「市場がどれだけ決めかねているか」を表す情報なので両端を残す。
    ('fukusho_min', 'REAL'),
    ('fukusho_max', 'REAL'),
    # 発走まで何分か。post_time が分かる時だけ入る。
    ('minutes_to_post', 'REAL'),
    # そのスナップショットで何頭ぶん取れたか / 何頭いるはずか。
    # 一部しか取れていない回で市場シェアを正規化すると静かに誤った値になる。
    ('n_captured', 'INTEGER'),
    ('n_expected', 'INTEGER'),
    # 取消・除外。1頭消えると全馬のオッズが動くので、記録しないと
    # 「情報を持った資金流入」と誤読する。
    ('is_scratched', 'INTEGER DEFAULT 0'),
)


def _ensure_odds_snapshot_columns(conn):
    """odds_snapshots に後付け列を足す（既存行は NULL のまま残る）。"""
    have = {r[1] for r in conn.execute('PRAGMA table_info(odds_snapshots)')}
    for name, decl in _ODDS_SNAPSHOT_EXTRA_COLS:
        if name not in have:
            conn.execute(f'ALTER TABLE odds_snapshots ADD COLUMN {name} {decl}')


def save_race_schedule(rows, base_dir=None, db_path=None):
    """発走時刻を race_schedule に保存する。

    オッズ時系列は「発走何分前か」で揃えないとレース間で比較できない。
    parse_header は start_time を取っていたが、どのDBにも保存されておらず
    （races.raw_json に245行中6行だけ残っていた）、このままでは
    何ヶ月集めても時点別の分析ができない状態だった（2026-08-27発見）。

    rows: [{race_id, date, racecourse, race_num, post_time, n_horses}, ...]
    Returns: 新規保存した行数
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    n = 0
    for r in rows:
        rid = str(r.get('race_id') or '')
        if not rid:
            continue
        cur = conn.execute(
            "INSERT OR REPLACE INTO race_schedule "
            "(race_id, date, racecourse, race_num, post_time, n_horses) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, str(r.get('date') or ''), str(r.get('racecourse') or ''),
             r.get('race_num'), r.get('post_time'), r.get('n_horses')),
        )
        n += cur.rowcount
    conn.commit()
    conn.close()
    return n


def save_odds_snapshots(rows, base_dir=None, db_path=None):
    """直前オッズログ（GASの getOddsLog が返す行）を odds_snapshots に保存する。

    rows: [{race_id, horse_num, tansho, fukusho, captured_at}, ...]
    (race_id, horse_num, captured_at) の一意制約で重複取込は無視する。

    Returns: 新規保存した行数
    """
    path = db_path or get_db_path(base_dir)
    conn = _connect(path)
    _ensure_odds_snapshot_columns(conn)
    n = 0
    for r in rows:
        race_id = str(r.get('race_id', ''))
        captured_at = str(r.get('captured_at', ''))
        num = r.get('horse_num')
        if not race_id or not captured_at or num is None:
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO odds_snapshots "
            "(race_id, horse_num, tansho, fukusho, captured_at, source,"
            " fukusho_min, fukusho_max, minutes_to_post,"
            " n_captured, n_expected, is_scratched) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (race_id, int(num), r.get('tansho'), r.get('fukusho'),
             captured_at, r.get('source', 'chokuzen'),
             r.get('fukusho_min'), r.get('fukusho_max'), r.get('minutes_to_post'),
             r.get('n_captured'), r.get('n_expected'),
             int(bool(r.get('is_scratched', False)))),
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
    """notes_data(dict) とスキーマから「レース後の不利」の補正値合計を計算する。

    feature=true の項目のみを value × weight で合算する。スキーマに無い
    キーや欠損キーは 0 扱い。スキーマが変わっても保存済み notes から再計算できる。

    ⚠ phase='pre'（パドック気配など、レース**前**に入力するもの）は合算しない。
    レース前の見立てとレース後の不利は意味が違う量で、足すと総和が何も指さなく
    なる。パドック側を測る時は notes_data のキーを直接読むこと（total_handicap
    は回顧の不利だけを表す、という既存の意味を壊さないための除外）。
    """
    if not isinstance(notes, dict):
        return 0.0
    total = 0.0
    for cat in schema.get('categories', []):
        if not cat.get('feature'):
            continue
        if cat.get('phase') == 'pre':
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

"""Gate 0 — 能力と独立した「適性」軸を、現在の特徴量で作れているか。

事前登録: betting/CRITERIA.md（測定より前にコミット済み）。
🔴 閾値・構成列・符号・重み・期間・母集団を**結果を見てから変更しない**。

  0-a 縮退検査     レース内 Spearman(能力順位, 適性順位)        合格 ρ̄ < 0.80
  0-b 条件感応検査 今回条件を差し替えた T の Kendall τ          合格 τ̄ < 0.70
  0-c 対照群B      注入の復元（本データを読む前に通す）
  0-d 対照群A      並べ替え分布 200回

目的変数（着順・配当・市場）は一切使わない。特徴量の幾何だけを見る。

使い方: python3 betting/gate0.py <history.db> [n_races]
"""
import hashlib
import json
import sqlite3
import sys

import numpy as np

sys.path.insert(0, '.')
import src.features.engine as eng
from src.features.engine import init_engine
from src.features.pl_rating import PLRatings
from noryoku.inv1 import build_race, _fresh, feats_of

SEED = 20260918
N_RACES_DEFAULT = 2000
A_START, A_END = '2023-01-07', '2024-12-31'
MIN_HORSES = 6
N_SHUFFLE = 200

# 適性軸 T の構成列（事前登録）。sign は構築式から決まる（大きいほど適合）。
# 'absneg' は −|x|（両方向に外れるほど悪い）。
T_COLS = [
    ('f_style_course_fit',    'pos'),
    ('f_dist_vs_optimal',     'absneg'),
    ('f_speed_x_shortening',  'pos'),
    ('f_stamina_x_extension', 'pos'),
    ('f_course_fit_score',    'pos'),
]
ABILITY_COL = 'f_pl_rating'

# 0-a / 0-b の合格線（事前登録）
GATE_A = 0.80
GATE_B = 0.70


# ── 基本の道具 ────────────────────────────────────────────────
def _signed(vals, mode):
    a = np.asarray(vals, dtype=float)
    return -np.abs(a) if mode == 'absneg' else a


def _zrace(a):
    """レース内 z 化。std=0 なら None（その列はそのレースで使わない）。"""
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        return None
    sd = a.std()
    if sd < 1e-12:
        return None
    return (a - a.mean()) / sd


def t_score(xs):
    """採用5列のレース内 z 値の等重み平均。使える列が0なら None。"""
    parts = []
    used = []
    for c, mode in T_COLS:
        raw = [x.get(c) for x in xs]
        if any(v is None for v in raw):
            continue
        z = _zrace(_signed(raw, mode))
        if z is not None:
            parts.append(z)
            used.append(c)
    if not parts:
        return None, []
    return np.mean(parts, axis=0), used


def desc_rank(score, horse_num=None):
    """降順順位（1 = 最大）。**同値は平均順位**（tie-aware）。

    🔴 2026-09-18・対照群A が捕まえた計測器の欠陥の修正。
      当初は「同値は馬番の昇順で強制的に一意化する」としていたが、
      これだと **T が同値の馬では順位が必ず馬番順になる**。能力側も
      初出走馬（θ=0）の同値を馬番順で割るため、両者に**共通の馬番成分**が入り、
      T をシャッフルしても ρ が 0 にならない（実測 +0.0176・帰無が0中心にならず
      対照群A が落ちた）。合成例で機序を確認済み:
          同値なし E[ρ]=+0.005 / t に4頭同値 E[ρ]=+0.090 / t が全頭同値 E[ρ]=+1.000
      平均順位に直すと 0.005 / -0.002 / 未定義(NaN) になる。
    ⚠ これは閾値・構成列・符号・期間・母集団の変更ではない。
      「着順を絶対に使わない」という当初の意図は、恣意的なタイブレークを
      置かない平均順位のほうがより厳密に満たす。
    """
    v = np.asarray(score, dtype=float)
    order = np.argsort(-v, kind='stable')
    n = len(v)
    rk = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rk


def spearman(a, b):
    """厳密順位同士なので Pearson と一致する。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def kendall(a, b):
    """Kendall tau-b（同値を正しく扱う）。全同値なら NaN。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, 1)
    da = np.sign(a[:, None] - a[None, :])[iu]
    db = np.sign(b[:, None] - b[None, :])[iu]
    num = float((da * db).sum())
    n_a = float((da != 0).sum())     # a で同値でない対の数
    n_b = float((db != 0).sum())
    if n_a <= 0 or n_b <= 0:
        return np.nan
    return num / np.sqrt(n_a * n_b)


def _stable_seed(*parts):
    """🔴 Python の hash() は文字列に対してプロセスごとに変わる
       （PYTHONHASHSEED）。対照群が実行のたびに別物になるので使わない。"""
    raw = ('|'.join([str(SEED)] + [str(x) for x in parts])).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def rng_normal(*key):
    return float(np.random.default_rng(_stable_seed('n', *key)).normal())


def rng_uniform(*key):
    return float(np.random.default_rng(_stable_seed('u', *key)).uniform(-1.0, 1.0))


def dist_zone(d):
    d = int(d or 1600)
    if d <= 1400:
        return 'sp'
    if d <= 1800:
        return 'mi'
    if d <= 2200:
        return 'md'
    return 'lo'


def cell_of(race):
    return (race.get('racecourse', ''), race.get('surface', '芝'),
            dist_zone(race.get('distance')))


def synth_S(race, names):
    """合成適性 S = z( g1[cell]·v1[horse] + g2[cell]·v2[horse] )。

    v は馬に固定（条件に依存しない・能力を一切参照しない）、
    g は今回条件だけで決まる。→ 「馬の型 × 今回条件の要求」と同じ構造。
    ⚠ 1成分だと g のスケールが正負を変えるだけで順位が ±1 にしかならず、
      条件感応の計測器を検査できない（2026-09-18・本データを読む前に修正）。
    """
    c = cell_of(race)
    g1, g2 = rng_uniform('g1', *c), rng_uniform('g2', *c)
    v1 = np.array([rng_normal('v1', nm) for nm in names])
    v2 = np.array([rng_normal('v2', nm) for nm in names])
    return _zrace(g1 * v1 + g2 * v2)


# ── データ ───────────────────────────────────────────────────
def sample_races(conn):
    ids = [r[0] for r in conn.execute(
        """SELECT race_id FROM race_history
           WHERE date >= ? AND date <= ? ORDER BY date, race_id""",
        (A_START, A_END))]
    return ids


def preload_finish(conn):
    """race_id → 1着順の馬名リスト。θ 更新に使う。"""
    rows = conn.execute("""
        SELECT race_id, date, horse_name, place
        FROM horse_history
        WHERE horse_name != '' AND date IS NOT NULL AND date != ''
    """).fetchall()
    fin, day = {}, {}
    for r in rows:
        rid = r['race_id']
        day[rid] = str(r['date'])[:10]
        p = r['place']
        if p and 0 < int(p) < 99:
            fin.setdefault(rid, []).append((int(p), r['horse_name']))
    for rid in fin:
        fin[rid].sort(key=lambda x: x[0])
        fin[rid] = [nm for _, nm in fin[rid]]
    return fin, day


# ── 測定 ─────────────────────────────────────────────────────
def measure(races_data, t_of, s_of=None, swap=True):
    """races_data の各レースについて ρ（0-a）と τ（0-b）を返す。

    t_of(rec) -> 元条件での適性スコア
    s_of(rec) -> 差替後条件での適性スコア（None なら 0-b を測らない）
    """
    rhos, taus = [], []
    for rec in races_data:
        t = t_of(rec)
        if t is None:
            continue
        tr = desc_rank(t, rec['horse_num'])
        rhos.append(spearman(rec['a_rank'], tr))
        if s_of is not None:
            t2 = s_of(rec) if swap else t
            if t2 is None:
                continue
            tr2 = desc_rank(t2, rec['horse_num'])
            taus.append(kendall(tr, tr2))
    return np.array(rhos, float), np.array(taus, float)


def run(db_path, n_races=N_RACES_DEFAULT):
    cols = json.load(open('data/xgb_feature_cols.json'))['feature_cols']
    init_engine('.')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    all_ids = sample_races(conn)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(all_ids), size=min(n_races, len(all_ids)),
                             replace=False).tolist())
    ids = [all_ids[i] for i in pick]
    print(f'A期間の全レース {len(all_ids):,} → 抽出 {len(ids):,}（seed={SEED}）')

    print('θ を日付順に再現しながら特徴量を作る'
          '（未来の成績が能力順位に入らないように）...', flush=True)
    fin_of, day_of = preload_finish(conn)

    # 条件の差し替え先: race_id 昇順で k=1 の循環シフト（事前登録）
    ids_sorted = sorted(ids)
    swap_to = {rid: ids_sorted[(i + 1) % len(ids_sorted)]
               for i, rid in enumerate(ids_sorted)}
    cond = {}
    for rid in ids:
        r = conn.execute('SELECT racecourse, surface, distance FROM race_history '
                         'WHERE race_id=?', (rid,)).fetchone()
        cond[rid] = {'racecourse': r['racecourse'],
                     'surface': r['surface'] or '芝',
                     'distance': int(r['distance'] or 1600)}

    # 全レースを (日付, race_id) 順に走らせ、対象レースに来たらその場で測る。
    # 🔴 スナップショットを2,000件持つとメモリが持たないので1パスで処理する。
    all_rows = conn.execute("""SELECT race_id, date FROM race_history
                               ORDER BY date, race_id""").fetchall()
    want = set(ids)
    pl = PLRatings()
    eng._PL_RATINGS = pl
    cur_day = None

    data = []
    n_skip = 0
    same_cond = 0
    done = 0
    for row in all_rows:
        rid, d = row['race_id'], str(row['date'])[:10]
        if cur_day is not None and d != cur_day:
            pl.advance_day(cur_day)       # 前日ぶんの結果を反映
        cur_day = d

        if rid in want:
            done += 1
            if done % 200 == 1:
                print(f'  特徴量 {done}/{len(ids)}  ({d})', flush=True)
            try:
                race = build_race(conn, rid)
            except Exception:
                race = None
            if race is not None and len(race['horses']) >= MIN_HORSES:
                names = [h['name'] for h in race['horses']]
                hn = [h['horse_num'] for h in race['horses']]
                n_debut = int(sum(1 for nm in names if pl.get(nm)[1] == 0))

                base = feats_of(race, cols)
                alt_c = cond[swap_to[rid]]
                if (alt_c['racecourse'], alt_c['surface'], alt_c['distance']) == \
                   (race['racecourse'], race['surface'], race['distance']):
                    same_cond += 1
                alt = feats_of(_fresh(race, **alt_c), cols)

                abil = [x.get(ABILITY_COL) for x in base]
                t_base, used = t_score(base)
                t_alt, _ = t_score(alt)
                if any(v is None for v in abil) or t_base is None:
                    n_skip += 1
                else:
                    a_rank = desc_rank(np.asarray(abil, float), hn)
                    per_col = {}
                    for c, mode in T_COLS:
                        raw = [x.get(c) for x in base]
                        if any(v is None for v in raw):
                            continue
                        z = _zrace(_signed(raw, mode))
                        raw2 = [x.get(c) for x in alt]
                        za = (None if any(v is None for v in raw2)
                              else _zrace(_signed(raw2, mode)))
                        per_col[c] = (z, za)
                    data.append({
                        'race_id': rid, 'horse_num': hn, 'names': names,
                        'n': len(hn), 'a_rank': a_rank,
                        't': t_base, 't_alt': t_alt, 'used': used,
                        'per_col': per_col,
                        'race': {k2: v2 for k2, v2 in race.items()
                                 if k2 != 'horses' and not k2.startswith('_')},
                        'alt_cond': alt_c, 'n_debut': n_debut,
                    })
            else:
                n_skip += 1

        f = fin_of.get(rid)
        if f:
            pl.queue_result(f)
        if done == len(ids) and d > A_END:
            break
    if cur_day is not None:
        pl.advance_day(cur_day)

    conn.close()
    print(f'  使用 {len(data):,} レース / スキップ {n_skip:,} '
          f'/ 差替先が同一条件 {same_cond:,}')

    out = {'n_races': len(data), 'n_skipped': n_skip, 'seed': SEED,
           'period': [A_START, A_END], 'same_cond': same_cond,
           't_cols': [c for c, _ in T_COLS], 'gate_a': GATE_A, 'gate_b': GATE_B}

    # ── 0-c 対照群B（本データを読む前に通す）────────────────────
    print('\n=== 0-c 対照群B（注入の復元）===')
    c_ok, c_log = [], {}

    # c-4 代数恒等式: δ=0 は z(T_score) と完全一致
    d4 = max(float(np.max(np.abs((r['t'] + 0.0 * synth_S(r['race'], r['names'])) - r['t'])))
             for r in data)
    c_log['c4_max_diff'] = d4
    c_ok.append(('c-4 δ=0 が完全一致', d4 < 1e-9, f'最大差 {d4:.3e}'))

    # c-3 代数恒等式: 条件を差し替えずに 0-b → τ = 1.000
    _, tau_same = measure(data, lambda r: r['t'], lambda r: r['t'], swap=False)
    t3 = float(np.nanmin(tau_same))
    c_log['c3_min_tau'] = t3
    c_ok.append(('c-3 条件を差し替えなければ τ=1', abs(t3 - 1.0) < 1e-9,
                 f'最小 τ {t3:.9f}'))

    # c-1 能力を注入 → ρ̄ > 0.95
    def t_c1(r):
        az = _zrace(-r['a_rank'])
        return r['t'] + 4.0 * az if az is not None else None
    rho_c1, _ = measure(data, t_c1)
    m1 = float(np.nanmean(rho_c1))
    c_log['c1_rho'] = m1
    c_ok.append(('c-1 能力注入で ρ̄>0.95', m1 > 0.95, f'ρ̄ {m1:.4f}'))

    # c-2 合成適性に置換 → ρ̄<0.10 かつ τ̄<0.30
    rho_c2, tau_c2 = measure(
        data,
        lambda r: synth_S(r['race'], r['names']),
        lambda r: synth_S(_fresh(r['race'], **r['alt_cond']), r['names']))
    m2r, m2t = float(np.nanmean(rho_c2)), float(np.nanmean(tau_c2))
    c_log['c2_rho'], c_log['c2_tau'] = m2r, m2t
    c_ok.append(('c-2 合成適性で ρ̄<0.10', abs(m2r) < 0.10, f'ρ̄ {m2r:.4f}'))
    c_ok.append(('c-2 合成適性で τ̄<0.30', abs(m2t) < 0.30, f'τ̄ {m2t:.4f}'))

    for name, ok, msg in c_ok:
        print(f'  {"✅" if ok else "🔴"} {name}: {msg}')
    out['control_b'] = {'checks': [(n, bool(o), m) for n, o, m in c_ok], **c_log}

    if not all(o for _, o, _ in c_ok):
        print('\n🔴 対照群B に落ちた。本データの 0-a / 0-b は印字せず中止する。')
        out['status'] = 'ABORTED_CONTROL_B'
        json.dump(out, open('betting/gate0_report.json', 'w'),
                  ensure_ascii=False, indent=1, default=float)
        return out

    # ── 0-d 対照群A（並べ替え分布）──────────────────────────────
    print('\n=== 0-d 対照群A（レース内シャッフル 200回）===')
    shuf_rng = np.random.default_rng(SEED + 1)
    means = []
    for _ in range(N_SHUFFLE):
        vals = []
        for r in data:
            p = shuf_rng.permutation(r['n'])
            tr = desc_rank(r['t'][p], r['horse_num'])
            vals.append(spearman(r['a_rank'], tr))
        means.append(np.nanmean(vals))
    means = np.array(means, float)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    thr = 2 * sd / np.sqrt(N_SHUFFLE)
    ok_d = abs(mu) < thr
    print(f'  mean(ρ_shuffle) {mu:+.5f} / std {sd:.5f} / 合格線 {thr:.5f} '
          f'→ {"✅" if ok_d else "🔴"}')
    out['control_a'] = {'mean': mu, 'std': sd, 'thr': thr, 'pass': bool(ok_d)}

    # ── 0-a 縮退検査 ───────────────────────────────────────────
    print('\n=== 0-a 縮退検査 ===')
    rho, _ = measure(data, lambda r: r['t'])
    rho_bar = float(np.nanmean(rho))
    q = np.nanpercentile(rho, [5, 25, 50, 75, 95])
    perm_p = float((np.abs(means) >= abs(rho_bar)).mean())

    absdiff = [float(np.mean(np.abs(r['a_rank'] - desc_rank(r['t'], r['horse_num']))))
               for r in data]
    far = [float(np.mean(np.abs(r['a_rank'] - desc_rank(r['t'], r['horse_num'])) >= 3))
           for r in data]

    print(f'  ρ̄ = {rho_bar:+.4f}   合格線 < {GATE_A}  → '
          f'{"✅ 合格" if rho_bar < GATE_A else "🔴 落ちた"}')
    print(f'  ρ の分布  5% {q[0]:+.3f} / 25% {q[1]:+.3f} / 中央 {q[2]:+.3f} '
          f'/ 75% {q[3]:+.3f} / 95% {q[4]:+.3f}')
    print(f'  平均絶対順位差  {np.mean(absdiff):.3f}')
    print(f'  |順位差|>=3 の馬の割合  {np.mean(far)*100:.1f}%')
    print(f'  並べ替え P  {perm_p:.4f}')

    print('  列ごとの単独 ρ̄:')
    per_col_rho = {}
    for c, _m in T_COLS:
        vs = []
        for r in data:
            z = r['per_col'].get(c, (None, None))[0]
            if z is None:
                continue
            vs.append(spearman(r['a_rank'], desc_rank(z, r['horse_num'])))
        per_col_rho[c] = (float(np.nanmean(vs)) if vs else None, len(vs))
        v, n = per_col_rho[c]
        print(f'    {c:24s} ρ̄ {v:+.4f}  （{n:,}レース）' if v is not None
              else f'    {c:24s} —')

    # 初出走馬を含む / 含まない
    with_d = [x for x, r in zip(rho, data) if r['n_debut'] > 0]
    wo_d = [x for x, r in zip(rho, data) if r['n_debut'] == 0]
    print(f'  初出走馬を含むレース ρ̄ {np.nanmean(with_d):+.4f}（{len(with_d):,}） / '
          f'含まない ρ̄ {np.nanmean(wo_d):+.4f}（{len(wo_d):,}）')

    out['gate_0a'] = {
        'rho_bar': rho_bar, 'pass': bool(rho_bar < GATE_A),
        'pct': {str(p): float(v) for p, v in zip([5, 25, 50, 75, 95], q)},
        'mean_abs_rank_diff': float(np.mean(absdiff)),
        'frac_rank_diff_ge3': float(np.mean(far)),
        'perm_p': perm_p,
        'per_col_rho': {k: v[0] for k, v in per_col_rho.items()},
        'per_col_n': {k: v[1] for k, v in per_col_rho.items()},
        'rho_with_debut': float(np.nanmean(with_d)) if with_d else None,
        'rho_without_debut': float(np.nanmean(wo_d)) if wo_d else None,
        'n_with_debut': len(with_d), 'n_without_debut': len(wo_d),
    }

    # ── 0-b 条件感応検査 ───────────────────────────────────────
    print('\n=== 0-b 条件感応検査 ===')
    _, tau = measure(data, lambda r: r['t'], lambda r: r['t_alt'])
    tau_bar = float(np.nanmean(tau))
    tq = np.nanpercentile(tau, [5, 25, 50, 75, 95])
    top_changed = []
    for r in data:
        if r['t_alt'] is None:
            continue
        a = desc_rank(r['t'], r['horse_num'])
        b = desc_rank(r['t_alt'], r['horse_num'])
        top_changed.append(int(np.argmin(a)) != int(np.argmin(b)))
    print(f'  τ̄ = {tau_bar:+.4f}   合格線 < {GATE_B}  → '
          f'{"✅ 合格" if tau_bar < GATE_B else "🔴 落ちた"}')
    print(f'  τ の分布  5% {tq[0]:+.3f} / 25% {tq[1]:+.3f} / 中央 {tq[2]:+.3f} '
          f'/ 75% {tq[3]:+.3f} / 95% {tq[4]:+.3f}')
    print(f'  差し替えで適性1位が入れ替わったレース  {np.mean(top_changed)*100:.1f}%')

    print('  列ごとの τ̄:')
    per_col_tau = {}
    for c, _m in T_COLS:
        vs = []
        for r in data:
            z, za = r['per_col'].get(c, (None, None))
            if z is None or za is None:
                continue
            vs.append(kendall(desc_rank(z, r['horse_num']),
                              desc_rank(za, r['horse_num'])))
        per_col_tau[c] = float(np.nanmean(vs)) if vs else None
        print(f'    {c:24s} τ̄ {per_col_tau[c]:+.4f}' if per_col_tau[c] is not None
              else f'    {c:24s} —')

    out['gate_0b'] = {
        'tau_bar': tau_bar, 'pass': bool(tau_bar < GATE_B),
        'pct': {str(p): float(v) for p, v in zip([5, 25, 50, 75, 95], tq)},
        'frac_top_changed': float(np.mean(top_changed)),
        'per_col_tau': per_col_tau,
    }

    # ── 総合判定 ───────────────────────────────────────────────
    passed = out['gate_0a']['pass'] and out['gate_0b']['pass'] and ok_d
    out['status'] = 'PASS' if passed else 'FAIL'
    print('\n' + '=' * 60)
    print(f'Gate 0 総合: {"✅ 通過" if passed else "🔴 落ちた"}'
          f'  (0-a {"✅" if out["gate_0a"]["pass"] else "🔴"} / '
          f'0-b {"✅" if out["gate_0b"]["pass"] else "🔴"} / '
          f'0-d {"✅" if ok_d else "🔴"})')
    if not passed:
        print('\n現在の適性特徴量では、能力と独立した今回条件への適合度を')
        print('作れていないため、能力×適性の馬券変換研究には進まない。')
    json.dump(out, open('betting/gate0_report.json', 'w'),
              ensure_ascii=False, indent=1, default=float)
    print('\n→ betting/gate0_report.json')
    return out


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'data/history.db',
        int(sys.argv[2]) if len(sys.argv) > 2 else N_RACES_DEFAULT)

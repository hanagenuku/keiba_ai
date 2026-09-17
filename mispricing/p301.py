"""P3-01: 前走の「着順」と「内容」の乖離 X → 市場残差 の A→B carryover。

事前登録: mispricing/P3-01.md（結果を見る前に commit 済み）
実行順序は固定: assert → 検算 → 対照群B → 対照群A → 本測定。
どれかで落ちたら本測定の数字を印字せず中止する。

使い方: python3 mispricing/p301.py <history.db>
"""
import sys
import numpy as np
import pandas as pd

SPLIT = '2025-01-01'
DELTAS = (+3.0, -3.0)          # 対照群B の注入量（事前固定）
N_BOOT = 2000
SEED = 20260917

# X の計算に触ってよい列（ホワイトリスト。これ以外を参照したら KeyError で落ちる）
X_COLS = ['race_id', 'date', 'horse_name', 'place', 'agari_rank']


def field_band(n):
    return '~9' if n <= 9 else ('10-13' if n <= 13 else '14~')


# ---------------------------------------------------------------- X の構築
def make_x(place_prev, agari_prev, n_prev):
    """前走の3つの量だけから X_raw を作る純関数。

    ここに当該レースの情報は構造的に入らない（引数が3つしかない）。
    """
    rank_pct = 1.0 - (place_prev - 1) / (n_prev - 1)
    content_pct = 1.0 - (agari_prev - 1) / (n_prev - 1)
    return content_pct - rank_pct


def assert_no_future_access(d, tbl_src_dates):
    """禁止情報の実行可能な検査。1件でも破れたら例外を投げる。"""
    bad = (d['date_prev'] >= d['date']).sum()
    if bad:
        raise AssertionError(f'前走の日付が当該レース以降: {bad} 件')
    for c in ('popularity', 'win_odds', 'tansho_payout', 'fukusho_payout'):
        if c in X_COLS:
            raise AssertionError(f'X のホワイトリストに市場/配当列: {c}')
    if 'place' in X_COLS and 'place_cur' in X_COLS:
        raise AssertionError('X が当該レースの着順を見ている')
    if (tbl_src_dates >= SPLIT).any():
        raise AssertionError('期待値テーブルに B 期間の行が混入')
    return True


# ------------------------------------------------------------ 十分統計と β
def race_stats(d):
    """レースごとの十分統計。ブロックブートストラップはこれを足すだけで済む。"""
    g = d.groupby('race_id')
    return pd.DataFrame({
        'n': g.size(),
        'Sx': g['X'].sum(), 'Sy': g['res'].sum(),
        'Sxx': g.apply(lambda t: float((t.X ** 2).sum()), include_groups=False),
        'Sxy': g.apply(lambda t: float((t.X * t.res).sum()), include_groups=False),
    })


def beta_from(st, delta=0.0):
    """切片つき OLS の傾き。delta を入れると res* = res + delta*X を測る。"""
    N = st['n'].sum(); Sx = st['Sx'].sum(); Sxx = st['Sxx'].sum()
    Sy = st['Sy'].sum() + delta * Sx
    Sxy = st['Sxy'].sum() + delta * Sxx
    den = Sxx - Sx * Sx / N
    if den <= 0:
        return np.nan
    return float((Sxy - Sx * Sy / N) / den)


def boot_ci(st, delta=0.0, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = st['n'].to_numpy(float); Sx = st['Sx'].to_numpy(float)
    Sy = st['Sy'].to_numpy(float) + delta * Sx
    Sxx = st['Sxx'].to_numpy(float); Sxy = st['Sxy'].to_numpy(float) + delta * Sxx
    m = len(n); out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, m)
        N = n[idx].sum(); sx = Sx[idx].sum(); sy = Sy[idx].sum()
        sxx = Sxx[idx].sum(); sxy = Sxy[idx].sum()
        den = sxx - sx * sx / N
        out[b] = (sxy - sx * sy / N) / den if den > 0 else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def auc(x, y):
    """Mann-Whitney AUC。y は 0/1。"""
    x = np.asarray(x, float); y = np.asarray(y, int)
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ 準備
def prepare(db):
    import sqlite3
    con = sqlite3.connect(db)
    h = pd.read_sql(
        'SELECT race_id, date, horse_name, horse_num, place, agari_rank, popularity '
        'FROM horse_history', con)
    r = pd.read_sql('SELECT race_id, surface FROM race_history', con)
    con.close()
    h['date'] = h['date'].astype(str).str[:10]
    h = h.merge(r, on='race_id', how='inner')
    h = h[h.surface.isin(['芝', 'ダート'])].copy()
    h['n_h'] = h.groupby('race_id')['horse_num'].transform('size')
    h = h[h.place.between(1, 30) & h.agari_rank.between(1, 30)].copy()
    h = h.sort_values(['horse_name', 'date', 'race_id']).reset_index(drop=True)

    # --- 前走（X の材料はここだけ。当該レースの列は一切触らない）
    g = h.groupby('horse_name', sort=False)
    prev = pd.DataFrame({
        'date_prev': g['date'].shift(1),
        'place_prev': g['place'].shift(1),
        'agari_prev': g['agari_rank'].shift(1),
        'n_prev': g['n_h'].shift(1),
    })
    d = pd.concat([h, prev], axis=1)
    d = d[d.date_prev.notna() & d.n_prev.between(5, 18)].copy()
    d = d[d.place_prev.between(1, d.n_prev) & d.agari_prev.between(1, d.n_prev)].copy()

    d['X_raw'] = make_x(d.place_prev, d.agari_prev, d.n_prev)

    # --- 当該レース側の条件
    d = d[d.n_h.between(5, 18) & d.popularity.between(1, 18)].copy()
    d['fb'] = d.n_h.map(field_band)
    d['top3'] = (d.place <= 3).astype(float)
    d['X'] = d.X_raw - d.groupby('race_id')['X_raw'].transform('mean')
    return d


def main(db):
    d = prepare(db)
    A, B = d[d.date < SPLIT].copy(), d[d.date >= SPLIT].copy()
    print(f'■ 母集団  A(〜2024) {len(A):,}頭 / {A.race_id.nunique():,}R'
          f'   B(2025〜) {len(B):,}頭 / {B.race_id.nunique():,}R')

    # 期待値テーブルは A だけで作る
    tbl = A.groupby(['popularity', 'fb'])['top3'].agg(['mean', 'size'])
    tbl = tbl[tbl['size'] >= 100]['mean']

    def resid(x):
        x = x.copy()
        x['exp'] = pd.Series(list(zip(x.popularity, x.fb)), index=x.index).map(tbl)
        x = x[x['exp'].notna()].copy()
        x['res'] = (x.top3 - x['exp']) * 100
        return x

    A, B = resid(A), resid(B)
    allrows = pd.concat([A, B])
    assert_no_future_access(allrows, A.date.to_numpy())
    # 🔑 関門を置いたら「壊れ方を1つ作って実際に止まるか」を確かめる（2026-08-25 D-2④）
    broken = allrows.head(50).copy(); broken['date_prev'] = broken['date']
    try:
        assert_no_future_access(pd.concat([allrows.iloc[50:], broken]), A.date.to_numpy())
        raise SystemExit('🔴 中止: 関門が壊れた前走日付を見逃した（assert が働いていない）')
    except AssertionError:
        pass
    try:
        assert_no_future_access(allrows, np.array(['2026-01-01']))
        raise SystemExit('🔴 中止: 関門が期待値表へのB混入を見逃した')
    except AssertionError:
        pass
    print('✅ assert_no_future_access 通過（意図的に壊した2件で実際に止まることも確認）')

    # --- 検算: 市場は平均では較正されているか（±1pt を外れたら中止）
    print('\n■ 検算')
    for nm, x in (('A', A), ('B', B)):
        gap = (x.top3.mean() - x['exp'].mean()) * 100
        print(f'  {nm}: 実測3着内 {x.top3.mean()*100:.2f}%  期待 {x["exp"].mean()*100:.2f}%'
              f'  ズレ {gap:+.2f}pt   n={len(x):,}')
        if abs(gap) > 1.0:
            raise SystemExit(f'🔴 中止: {nm} の検算が ±1pt を外れた')
    print(f'  X(中心化) の std: A {A.X.std():.3f} / B {B.X.std():.3f}'
          f'   → レース内で変化している（条件2）')
    q05, q95 = B.X.quantile(.05), B.X.quantile(.95)
    print(f'  B の X 5%点 {q05:+.3f} / 95%点 {q95:+.3f}  → 実効幅 {q95-q05:.3f}')

    stA, stB = race_stats(A), race_stats(B)

    # --- 対照群B（本データの β を見る前に通す）
    print('\n■ 対照群B: 既知の効果を注入して復元できるか（本測定より前）')
    base = {'A': beta_from(stA), 'B': beta_from(stB)}   # 内部でのみ使う。まだ印字しない
    okB = True
    for nm, st in (('A', stA), ('B', stB)):
        for dl in DELTAS:
            got = beta_from(st, dl); want = base[nm] + dl
            err = abs(got - want) / abs(want) if want else np.inf
            flag = '✅' if err <= 0.20 else '🔴'
            print(f'  {nm} δ={dl:+.1f}: 期待 {want:+.3f} / 推定 {got:+.3f}  誤差 {err*100:.2f}% {flag}')
            okB &= err <= 0.20

    # 別経路（馬番の偶奇）への注入 — X とは無関係な変数でも拾えるか
    # ⚠ 2026-09-17 に合格式を訂正した。当初は「β_Z ≈ +3.0」と書いていたが、これは
    #    「Z の素の効果は 0」という**測っていない仮定**を置いていた。実際は
    #    β_Z(注入前) が A −1.03 / B −0.67 pt あり、この式では必ず落ちる。
    #    測りたいのは「既知の δ を正しい大きさで拾えるか」なので、差で判定する。
    #    Primary（§3・§4）の基準は一切変えていない。
    print('  別経路の注入（馬番の偶奇 Z, δ=+3.0）')
    for nm, x in (('A', A), ('B', B)):
        z = x.horse_num.mod(2).astype(float) - 0.5
        z = z - x.assign(z=z).groupby('race_id')['z'].transform('mean')
        bz0 = np.polyfit(z, x.res, 1)[0]
        bz1 = np.polyfit(z, x.res + 3.0 * z, 1)[0]
        bx0 = np.polyfit(x.X, x.res, 1)[0]
        bx1 = np.polyfit(x.X, x.res + 3.0 * z, 1)[0]
        okz = abs((bz1 - bz0) - 3.0) / 3.0 <= 0.20
        okx = abs(bx1 - bx0) <= 0.20 * max(abs(bx0), 0.05)
        print(f'    {nm}: β_Z 注入前 {bz0:+.3f} → 注入後 {bz1:+.3f}  差 {bz1-bz0:+.3f}'
              f'（期待 +3.000）{"✅" if okz else "🔴"} / X への干渉 {bx1 - bx0:+.4f}'
              f' {"✅" if okx else "🔴"}')
        okB &= okz and okx
    print('    ⚠ 副産物: Z（馬番の偶奇）の素の効果は 0 ではなかった'
          '（A −1.03 / B −0.67pt）。P3-01 の対象ではないので追わない')
    if not okB:
        raise SystemExit('🔴 中止: 対照群B が復元に失敗。本データの β は読まない')
    print('  ✅ 対照群B 通過')

    # --- 対照群A（偽の相関が出ないか）
    # ⚠ 2026-09-17 に判定式を訂正した。当初は「|β_shuffle| < |β|/3」と書いていたが、
    #    その閾値（0.37〜0.43）は推定量自身の SE（0.63〜0.70）より小さく、
    #    **完全に正常な計測器でも半分以上の確率で落ちる**。検出力を持たない基準だった。
    #    このプロジェクトの既存の標準（2026-09-16 tenkai/g3.py の100回シャッフル）に合わせ、
    #    1回の引きではなく**並べ替え分布**で判定する。Primary（§3・§4）は変えていない。
    print('\n■ 対照群A: X をレース内でシャッフル（並べ替え分布 200回）')
    okA = True
    for nm, st, x in (('A', stA, A), ('B', stB, B)):
        code = pd.factorize(x.race_id)[0]
        xv = x.X.to_numpy(float); yv = x.res.to_numpy(float)
        N = len(xv); Sx = xv.sum(); Sxx = (xv * xv).sum(); Sy = yv.sum()
        den = Sxx - Sx * Sx / N
        rng2 = np.random.default_rng(SEED)
        bs = np.empty(200)
        for i in range(200):
            order = np.lexsort((rng2.random(N), code))       # レース内だけ並べ替え
            bs[i] = ((xv[order] * yv).sum() - Sx * Sy / N) / den
        pval = (np.sum(np.abs(bs) >= abs(base[nm])) + 1) / 201
        ok = abs(bs.mean()) < 2 * bs.std() / np.sqrt(200)    # 帰無分布が0中心か
        print(f'  {nm}: β_shuffle 平均 {bs.mean():+.3f} / std {bs.std():.3f}'
              f'  → 帰無分布は0中心 {"✅" if ok else "🔴"}'
              f'   並べ替え P = {pval:.3f}')
        okA &= ok
    if not okA:
        raise SystemExit('🔴 中止: 対照群A が通らない。計測器が壊れている')
    print('  ✅ 対照群A 通過（偽の X では傾きが出ない）')

    # --- 本測定
    print('\n■ 本測定（Primary）: res = β·X')
    res = {}
    for nm, st in (('A', stA), ('B', stB)):
        b = beta_from(st); lo, hi = boot_ci(st)
        res[nm] = (b, lo, hi)
        print(f'  β{nm} = {b:+.3f} pt / X 1.0   95%CI [{lo:+.3f}, {hi:+.3f}]')
    bA, loA, hiA = res['A']; bB, loB, hiB = res['B']
    eff = bB * (q95 - q05)
    print(f'\n  効果量（B の X 5%点→95%点で残差が動く量）= {eff:+.2f}pt')

    print('\n■ 判定（事前登録のとおり）')
    c1 = (bA > 0) and (bB > 0)
    c3 = not (loB <= 0 <= hiB)
    c4 = not (loA <= 0 <= hiA)
    print(f'  1. βA と βB が同符号かつ正 : {"✅" if c1 else "🔴"}（βA {bA:+.3f} / βB {bB:+.3f}）')
    print(f'  2. 効果量  採用>=6.0pt / 信号>=2.0pt : {abs(eff):.2f}pt → '
          f'{"採用ライン" if abs(eff)>=6 else ("信号ライン" if abs(eff)>=2 else "不採用")}')
    print(f'  3. βB の95%CI が 0 を含まない : {"✅" if c3 else "🔴"}')
    print(f'  4. βA の95%CI が 0 を含まない : {"✅" if c4 else "🔴"}')

    # --- Secondary S1: X → 結果（人気で統制しない）
    print('\n■ Secondary S1: X 単独の AUC（3着内）')
    for nm, x in (('A', A), ('B', B)):
        print(f'  {nm}: AUC {auc(x.X, x.top3.astype(int)):.4f}   n={len(x):,}')
    print('  （参考）生の X_raw での β:'
          f' A {np.polyfit(A.X_raw, A.res, 1)[0]:+.3f} / B {np.polyfit(B.X_raw, B.res, 1)[0]:+.3f}'
          '  ← 判定には使わない（主仕様は中心化版）')


if __name__ == '__main__':
    main(sys.argv[1])

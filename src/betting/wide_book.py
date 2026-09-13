"""ワイド盤・複勝盤をパースし、恒等式 Σ_B P(A,B) = 2·P(A) で突き合わせる。

⚠ ここは「価格の読み方」だけを担当する。予測モデルも控除率の推定も出てこない。
   1レースで3着内に入るのは必ず3頭なので、両方とも合計3に正規化すれば
   控除率を知らなくても比較できる（`dprime/CRITERIA.md` Step 1〜2）。

🔴 8頭立て以上でしか使えない。
   5〜7頭は複勝が「2着以内」の払戻なので恒等式の右辺が別物になる。
   ここを混ぜると実在しないズレを測って「不整合を発見した」と誤報告する。

到達確認は 2026-09-12 の probe-wide-odds 実行で完了している（Step 0）:
  単勝・複勝 pw151ou... / 枠連 pw153 / 馬連 pw154 / **ワイド pw155** /
  馬単 pw156 / 3連複 pw157 / 3連単 pw158
  ワイド盤は「ワイドオッズ（馬番順）」で C(n,2) 組が範囲表記（min-max）で並ぶ。
"""
import math
import re
import unicodedata

from bs4 import BeautifulSoup

# 控除率（JRA公式）。理論値の目安を出すためだけに使う。判定には使わない。
TAKEOUT_FUKU = 0.20
TAKEOUT_WIDE = 0.225

# Σ(1/o) が理論値からどれだけ外れたら盤を疑うか（CRITERIA Step 1 で事前登録）
BOOK_TOLERANCE = 0.15

MIN_FIELD_SIZE = 8   # 複勝が3着払いになる下限

_NUM = re.compile(r'^\d{1,2}$')
_RANGE = re.compile(r'^(\d{1,5}(?:\.\d)?)\s*[-−~〜]\s*(\d{1,5}(?:\.\d)?)$')
_SINGLE = re.compile(r'^(\d{1,5}\.\d)$')


def _norm(s):
    return unicodedata.normalize('NFKC', s).replace('\xa0', ' ').strip()


def parse_odds_cell(text):
    """'5.9-6.6' → (5.9, 6.6) / '16.7' → (16.7, 16.7) / それ以外 → None。

    ⚠ ワイドも複勝も**範囲表記**で出る。min だけ・max だけを見ると
       系統的に片側へ寄った乖離が出て、それを市場の不整合と読み違える。
       min / max / 中点の3通りで測れるよう、必ず両方を持つ。
    """
    t = _norm(text)
    m = _RANGE.match(t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if 0 < lo <= hi else None
    m = _SINGLE.match(t)
    if m:
        v = float(m.group(1))
        return (v, v) if v > 0 else None
    return None


def expand_table(table):
    """rowspan / colspan を展開して、行ごとに同じ列数のセル文字列を返す。

    🔴 2026-09-13 の実測でここが決定打だった。単複ページは **枠に rowspan** が
    付いており（1枠に2頭）、2頭目の行は先頭セルが1つ足りない。列位置を
    そのまま使うと**2頭目が全部落ちて、どのレースでも必ず「8頭」になる**
    （枠の数＝8）。実際 R01〜R12 のすべてで len(fuku)==8 になり、
    16頭立てでも 10頭立てでも 8 が返っていた。

    見出しから列を引いていても、行側がずれていれば意味がない。
    「見出しで引いているから安全」という思い込みを潰した形。
    """
    grid = {}
    rows = table.find_all('tr')
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(['td', 'th']):
            while (r, c) in grid:
                c += 1
            try:
                rs = max(1, int(cell.get('rowspan', 1)))
                cs = max(1, int(cell.get('colspan', 1)))
            except (TypeError, ValueError):
                rs = cs = 1
            txt = _norm(cell.get_text(' '))
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = txt
            c += cs
    out = []
    for r in range(len(rows)):
        cols = [c for (rr, c) in grid if rr == r]
        out.append([grid.get((r, c), '') for c in range(max(cols) + 1)] if cols else [])
    return out


def parse_fukusho_book(html):
    """単複ページ（pw151…）から複勝オッズを読む。

    見出しは 枠/馬番/馬名/単勝/複勝(2着払い)/性齢/馬体重/負担重量/騎手名/調教師名。
    列位置を決め打ちせず**見出しから引く**（2026-08-03③で結果ページの列が
    まるごとズレて1ヶ月気づかなかった事故があるため）。

    🔴 かつ **rowspan を展開してから**引く。枠に rowspan が付いているので、
       展開しないと1枠の2頭目が丸ごと落ちる（2026-09-13 に実測・上記参照）。

    Returns: {馬番: (min, max)}  取消馬は入らない
    """
    soup = BeautifulSoup(html, 'lxml')
    out = {}
    for table in soup.find_all('table'):
        grid = expand_table(table)
        i_num = i_fuku = None
        for row in grid:
            for i, cell in enumerate(row):
                if i_num is None and '馬番' in cell:
                    i_num = i
                if i_fuku is None and '複勝' in cell:
                    i_fuku = i
            if i_num is not None and i_fuku is not None:
                break
        if i_num is None or i_fuku is None:
            continue
        for row in grid:
            if len(row) <= max(i_num, i_fuku):
                continue
            if not _NUM.match(row[i_num]):
                continue
            o = parse_odds_cell(row[i_fuku])
            if o:
                out[int(row[i_num])] = o
    return out


def parse_wide_book(html):
    """ワイド盤（pw155…）から組ごとのオッズを読む。

    🔴 ページは「軸馬ごとに1テーブル」で、各行は [相手の馬番, オッズ範囲] しか
    持たない（軸馬の番号は行に書かれていない）。テーブルの並び順に依存すると
    取消馬が出た瞬間にずれるので、**軸馬はテーブルの見出し（caption/th）から引き、
    取れないときだけ「その表の最小の相手番号 − 1」で補う**。
    どちらで決めたかは戻り値の `axis_source` に残す（推測が混ざったまま
    数字を出さないため）。

    Returns: (pairs, axis_source)
      pairs       : {(小さい馬番, 大きい馬番): (min, max)}
      axis_source : {'header': n件, 'inferred': n件}
    """
    soup = BeautifulSoup(html, 'lxml')
    pairs = {}
    src = {'header': 0, 'inferred': 0}
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [_norm(c.get_text(' ')) for c in tr.find_all(['td', 'th'])]
            if len(cells) < 2:
                continue
            # 行の中から「馬番」と「オッズ」の組を探す（列位置を決め打ちしない）
            for i in range(len(cells) - 1):
                if _NUM.match(cells[i]):
                    o = parse_odds_cell(cells[i + 1])
                    if o:
                        rows.append((int(cells[i]), o))
                        break
        if not rows:
            continue

        # 🔴 軸馬は caption だけから読む。実機（2026-09-13）は
        #    <caption>1</caption> と馬番の数字だけが入っている。
        #    行の <th> は「相手の馬番」なので、そこを軸馬と読むと全部ずれる。
        cap = table.find('caption')
        blob = _norm(cap.get_text(' ')) if cap else ''
        m = re.fullmatch(r'\s*(\d{1,2})\s*番?\s*', blob)
        if m:
            axis = int(m.group(1))
            src['header'] += 1
        else:
            # caption が無い形に変わった時の保険。並び順ではなく
            # 「その表の最小の相手番号 − 1」で補い、推測したことを必ず残す。
            axis = min(n for n, _ in rows) - 1
            src['inferred'] += 1
        if axis <= 0:
            continue
        for partner, o in rows:
            if partner == axis:
                continue
            pairs[(min(axis, partner), max(axis, partner))] = o
    return pairs, src


def _pick(rng, how):
    lo, hi = rng
    return lo if how == 'min' else hi if how == 'max' else (lo + hi) / 2.0


def book_sum(book, how='mid'):
    """Σ(1/o)。単勝盤の Σ≈1.25 と同型の健全性検査（2026-08-15 C-2）。"""
    return sum(1.0 / _pick(v, how) for v in book.values() if _pick(v, how) > 0)


def field_size_from_wide(wide_pairs):
    """ワイド盤に出てくる馬番の種類数＝出走頭数。

    🔴 複勝側の件数から頭数を決めてはいけない。複勝のパースが不完全だと
    「頭数も組数も少なく見える」ので、壊れていることに気づけなくなる
    （2026-09-13 に実際にそうなった。どのレースも8頭に見えていた）。
    ワイド盤は C(n,2) 組が揃っているかを Σ で独立に検算できるので、
    こちら側を基準にする。
    """
    nums = set()
    for a, b in wide_pairs:
        nums.add(a)
        nums.add(b)
    return len(nums)


def check_books(fuku, wide_pairs, n_horses, how='mid'):
    """Step 1: 盤そのものが読めているか。モデルも控除率の推定も使わない。

    複勝（8頭以上）は Σ_A 1/o_A ≈ 3/(1-0.20) = 3.750
    ワイドは            Σ_pairs 1/o ≈ 3/(1-0.225) = 3.871
    """
    exp_f = 3.0 / (1.0 - TAKEOUT_FUKU)
    exp_w = 3.0 / (1.0 - TAKEOUT_WIDE)
    sf, sw = book_sum(fuku, how), book_sum(wide_pairs, how)
    want_pairs = n_horses * (n_horses - 1) // 2
    return {
        'how': how,
        'n_horses': n_horses,
        'n_fuku': len(fuku),
        'n_pairs': len(wide_pairs),
        'want_pairs': want_pairs,
        'pairs_complete': len(wide_pairs) == want_pairs,
        'fuku_complete': len(fuku) == n_horses,
        'sum_fuku': sf, 'exp_fuku': exp_f,
        'sum_wide': sw, 'exp_wide': exp_w,
        'fuku_ok': abs(sf - exp_f) <= exp_f * BOOK_TOLERANCE,
        'wide_ok': abs(sw - exp_w) <= exp_w * BOOK_TOLERANCE,
        'field_ok': n_horses >= MIN_FIELD_SIZE,
    }


def implied_top3(fuku, wide_pairs, how='mid'):
    """Step 2: 両方を合計3に正規化し、ワイド由来の3着内確率 q_A を作る。

        P_w(A,B) = (1/o_AB) / Σ(1/o) × 3
        q_A      = Σ_{B≠A} P_w(A,B) / 2
        P_p(A)   = (1/o_A) / Σ(1/o) × 3
        d_A      = q_A − P_p(A)

    Σ_A q_A = 3 は代数的に自動で成立する（検算に使う。ズレたら実装が壊れている）。
    """
    sw = book_sum(wide_pairs, how)
    sf = book_sum(fuku, how)
    if sw <= 0 or sf <= 0:
        return None
    q = {}
    for (a, b), rng in wide_pairs.items():
        p = (1.0 / _pick(rng, how)) / sw * 3.0
        q[a] = q.get(a, 0.0) + p
        q[b] = q.get(b, 0.0) + p
    q = {k: v / 2.0 for k, v in q.items()}
    pp = {k: (1.0 / _pick(v, how)) / sf * 3.0 for k, v in fuku.items()}
    common = sorted(set(q) & set(pp))
    d = {k: q[k] - pp[k] for k in common}
    return {
        'how': how,
        'q': q, 'p_fuku': pp, 'd': d,
        'sum_q': sum(q.values()),          # 3.0 のはず（実装の検算）
        'sum_p': sum(pp.values()),         # 3.0 のはず
        'total_variation': sum(abs(v) for v in d.values()) / 2.0,
        'max_abs_d': max((abs(v) for v in d.values()), default=float('nan')),
        'n_common': len(common),
    }


def identity_residual(wide_pairs, how='mid'):
    """実装の自己検算: Σ_A q_A が厳密に3になるか（ならなければコードが壊れている）。"""
    sw = book_sum(wide_pairs, how)
    if sw <= 0:
        return float('nan')
    tot = sum((1.0 / _pick(r, how)) / sw * 3.0 for r in wide_pairs.values())
    return abs(tot - 3.0)


def fmt_check(c):
    def mark(b):
        return '✅' if b else '🔴'
    return (f"[{c['how']}] {c['n_horses']}頭 "
            f"{mark(c['field_ok'])}8頭以上 "
            f"{mark(c['pairs_complete'])}組 {c['n_pairs']}/{c['want_pairs']} "
            f"{mark(c['fuku_complete'])}複勝 {c['n_fuku']}/{c['n_horses']}頭 "
            f"{mark(c['fuku_ok'])}Σ複勝 {c['sum_fuku']:.3f}(理論{c['exp_fuku']:.3f}) "
            f"{mark(c['wide_ok'])}Σワイド {c['sum_wide']:.3f}(理論{c['exp_wide']:.3f})")

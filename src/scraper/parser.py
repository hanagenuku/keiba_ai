import re
import unicodedata
from src.utils.config import PLACE_NAMES


def _detect_surface(text):
    """テキストから芝/ダート/障害を堅実に判定する。

    JRAの記法を多段で試す。判定不能なら None を返す（サイレントなフォールバックは廃止）。
    返り値: '芝' | 'ダート' | '障害' | None
    """
    if not text:
        return None
    t = unicodedata.normalize('NFKC', text)
    # 1) 障害レースの明示（「ジャンプ」は馬名に含まれることがあるため除外）
    if '障害' in t or any(kw in t for kw in ['(J)', '（J）', 'J・G', 'J-G']):
        return '障害'
    # 2) "NNNNメートル（芝/ダ" の明示形式（最強パターン）
    m = re.search(r'メートル\s*[（(]\s*([芝ダ])', t)
    if m:
        return '芝' if m.group(1) == '芝' else 'ダート'
    # 3) "芝NNNN" / "ダNNNN" 形式（過去走テキスト等）
    has_turf_dist = bool(re.search(r'芝\s*\d{3,4}', t))
    has_dirt_dist = bool(re.search(r'ダ(?:ート)?\s*\d{3,4}', t))
    if has_turf_dist and not has_dirt_dist:
        return '芝'
    if has_dirt_dist and not has_turf_dist:
        return 'ダート'
    # 4) 単独で "ダート" / "芝" が含まれる
    has_dirt = 'ダート' in t
    has_turf = '芝' in t
    if has_turf and not has_dirt:
        return '芝'
    if has_dirt and not has_turf:
        return 'ダート'
    # 判定不能（サイレントなフォールバック無し）
    return None


def get_class_from_racename(rname: str) -> str:
    if not rname:
        return '1勝クラス'
    if 'G1' in rname or '（G1）' in rname:
        return 'G1'
    if 'G2' in rname or '（G2）' in rname:
        return 'G2'
    if 'G3' in rname or '（G3）' in rname:
        return 'G3'
    if any(kw in rname for kw in ['ステークス', '記念', '特別', 'カップ', '賞', '杯', 'トロフィー']):
        return 'オープン'
    if '3勝クラス' in rname or '3勝' in rname:
        return '3勝クラス'
    if '2勝クラス' in rname or '2勝' in rname:
        return '2勝クラス'
    if '1勝クラス' in rname or '1勝' in rname:
        return '1勝クラス'
    if '未勝利' in rname:
        return '未勝利'
    if '新馬' in rname:
        return '新馬'
    return '1勝クラス'


def parse_header(text):
    info = {}
    m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if m:
        info['date'] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    for code, name in PLACE_NAMES.items():
        if name in text:
            info['racecourse'] = name
            break
    _shogai_kws = ['障害', '(J)', '（J）', 'J・G', 'J-G']
    if any(kw in text for kw in _shogai_kws):
        info['surface'] = '障害'
        info['distance'] = 0
        info['direction'] = ''
        return info
    dm = re.search(r'([\d,]+)\s*メートル\s*[（(]\s*([芝ダ])[^）)]*([右左])', text)
    if dm:
        info['distance'] = int(dm.group(1).replace(',', ''))
        info['surface'] = '芝' if dm.group(2) == '芝' else 'ダート'
        info['direction'] = dm.group(3)
    else:
        info['distance'] = 0
        info['surface'] = '不明'
        info['direction'] = ''
        print(f'  ⚠ surface判定失敗: {text[:80]}')
    for kw, cls in [
        ('G1', 'G1'), ('G2', 'G2'), ('G3', 'G3'),
        ('3勝クラス', '3勝クラス'), ('2勝クラス', '2勝クラス'),
        ('1勝クラス', '1勝クラス'), ('未勝利', '未勝利'), ('新馬', '新馬'),
        ('オープン', 'オープン'), ('重賞', '重賞'),
    ]:
        if kw in text:
            info['class'] = cls
            break
    else:
        info['class'] = '1勝クラス'
    return info


def parse_rname(text, rn):
    c = text.replace('本賞金', '').replace('付加賞', '')
    sp = re.search(
        r'([぀-鿿゠-ヿa-zA-Z0-9]+(?:賞|杯|記念|特別|ステークス|カップ|トロフィー))', c
    )
    if sp:
        n = sp.group(1).strip()
        if n not in ('本賞', '付加賞') and len(n) >= 3:
            return n
    gen = re.search(r'(\d歳(?:以上)?(?:未勝利|1勝クラス|2勝クラス|3勝クラス|オープン))', text)
    return gen.group(1).strip() if gen else f'R{rn:02d}'


def parse_hist(text):
    if not text or len(text) < 10:
        return None
    h = {}
    pm = re.search(r'(\d+)\s*着', text)
    h['place'] = int(pm.group(1)) if pm else 10
    fm = re.search(r'(\d+)\s*頭', text)
    h['finishers'] = int(fm.group(1)) if fm else 16
    dm = re.search(r'(\d{4})(?:芝|ダ)', text)
    if not dm:
        dm = re.search(r'(\d{4})', text)
    h['distance'] = int(dm.group(1)) if dm else 2000
    h['surface'] = 'ダート' if 'ダ' in text else '芝'
    for cond in ['不良', '重', '稍重', '良']:
        if cond in text:
            h['condition'] = cond
            break
    else:
        h['condition'] = '良'
    h['agari3f_rank_pct'] = 0.5
    margin = 0.0
    mm = re.search(r'(\d+\.\d+)秒', text)
    if mm:
        margin = float(mm.group(1))
    elif 'クビ' in text:
        margin = 0.1
    elif 'ハナ' in text:
        margin = 0.05
    elif 'アタマ' in text:
        margin = 0.07
    h['margin'] = margin
    h['class'] = get_class_from_racename(text)
    return h


def parse_horse(cells, rc, surf):
    if len(cells) < 4:
        return None
    try:
        tx = [c.get_text(' ', strip=True) for c in cells]
        umaban = None
        for col_idx in [0, 1, 2]:
            if col_idx >= len(tx):
                break
            nm = re.match(r'^\s*(\d{1,2})\s*$', tx[col_idx])
            if nm:
                umaban = int(nm.group(1))
                break
        if umaban is None:
            return None
        name = None
        name_col = 1
        for col_idx in range(1, min(5, len(cells))):
            links = cells[col_idx].find_all('a')
            for a in links:
                txt = a.get_text(strip=True)
                if txt and re.search(r'[゠-ヿ一-鿿]', txt) and len(txt) >= 2:
                    name = txt
                    name_col = col_idx
                    break
            if name:
                break
        if not name:
            for col_idx in range(1, min(5, len(tx))):
                if re.search(r'[゠-ヿ一-鿿]', tx[col_idx]) and len(tx[col_idx]) >= 2:
                    name = tx[col_idx]
                    name_col = col_idx
                    break
        if not name:
            return None
        odds = None
        for col_idx in range(len(tx) - 1, -1, -1):
            m = re.search(r'(\d+\.\d)', tx[col_idx])
            if m:
                v = float(m.group(1))
                # 斤量の範囲（50.0〜59.9）はオッズではないのでスキップ
                if 50.0 <= v < 60.0:
                    continue
                odds = v
                break

        # 性齢から年齢を取得（例: 牡4 → 4）。セル内の前後文字を許容
        age = 4
        for t in tx:
            m = re.search(r'[牡牝騸セ](\d)', t)
            if m:
                age = int(m.group(1))
                break

        # 斤量（例: 57.0）。前後文字（kg等）を許容、ただし他の数字に紛れないようガード
        weight_load = 56.0
        for t in tx:
            m = re.search(r'(?<!\d)(5\d\.\d)(?!\d)', t)
            if m:
                weight_load = float(m.group(1))
                break

        # 騎手・調教師（馬名以外の日本語リンクを順番に取得）
        # JRAカードの実際の列順は [調教師, 騎手]（2026-05-28 確定版で確認）
        jp_links = []
        for col_idx in range(len(cells)):
            for a in cells[col_idx].find_all('a'):
                txt = a.get_text(strip=True)
                if txt != name and re.search(r'[゠-ヿ一-鿿]', txt) and len(txt) >= 2:
                    jp_links.append(txt)
        trainer = jp_links[0] if jp_links else ''
        jockey  = jp_links[1] if len(jp_links) >= 2 else ''

        # 父名（リンクなし・カタカナ3文字以上のテキストセル）
        sire = ''
        for col_idx in range(max(name_col + 2, 3), len(cells)):
            if cells[col_idx].find('a'):
                continue
            txt = re.sub(r'[\s　]+', '', tx[col_idx])
            if (len(txt) >= 3
                    and re.search(r'[゠-ヿ一-鿿ァ-ン]{3,}', txt)
                    and not re.match(r'^[牡牝騸セ]\d', txt)
                    and not re.match(r'^\d', txt)
                    and not re.search(r'\(\s*[+-]?\d', txt)
                    and txt != name):
                sire = txt
                break

        return {
            'num': umaban,
            'name': name,
            'win_odds': odds,
            'age': age,
            'weight_load': weight_load,
            'jockey': jockey,
            'trainer': trainer,
            'sire': sire,
            'racecourse': rc,
            'surface': surf,
            'post_position': umaban,
        }
    except Exception:
        return None

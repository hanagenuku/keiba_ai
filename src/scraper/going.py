"""当日の馬場状態を、**その日すでに終わったレースの結果**から拾う。

なぜ要るか
----------
`f_track_cond` は本番134特徴量の1つで、**学習時は本物の馬場状態**が入る
（`build_training_data` が `race_history.track_condition` をそのまま渡す）。
ところが**推論時はつねに 0.0（良）**だった。出馬表ページに馬場状態が
載っていないためで、2026-08-26 に「既知のパリティ違反」として記録されたまま
「取得手段が無い」として保留されていた。

良でないレースは実データの **27.2%**（稍重16.3 / 重7.8 / 不良3.2）ある。

🔑 取得手段はあった。**同じ日の先に終わったレースの結果**に実際の馬場が入る。
   これは新しいスクレイピング先ではなく、毎週すでに叩いている結果ページ。

⚠ リークではない。予測対象レースより**前に終わった**レースの公開情報しか
   使わない。2026-08-06 の当日馬場バイアスと同じ線引き。

⚠ 効果は小さいと事前に分かっている。2026-08-26 に本番 `calc_all` を4通り
   回した実測では、馬場を変えて軸が入れ替わるのは **2.5%** のレースだけで、
   勝率の変化は中央値 0.03pt。ここで直すのは「予想が当たるようになるから」
   ではなく、**学習と推論で違う値を渡している状態を解消するため**。
   （副次的に、展開予想モデルから外した `cond` を戻せる道が開く）

⚠ North Star #4: 新しいリクエスト元には必ず件数の上限を置く。
   2026-07-18 に budget 無しのスクレイピングを足してCIがタイムアウトし、
   その回のデータを丸ごと失った前例がある。ここでは
   ①会場あたりのリクエスト数 ②全体の実時間 の二重上限を置き、
   超えたら**黙って諦めて予想は続行する**（止めない）。
"""
import re
import time

# 会場ごとに結果ページを叩く上限（R1から昇順に見て、終わっていないレースに
# 当たったら止まるので通常はこれより遥かに少ない）
GOING_FETCH_BUDGET_PER_VENUE = 12
# 未終了レースに連続で当たったら、その会場はそこで打ち切る
GOING_MISS_STREAK = 2
# 全体の実時間の上限（秒）。ワークフローのタイムアウトを守るための最後の砦
GOING_TIME_BUDGET_SEC = 480


def collect_today_going(sess, target_date, time_budget_sec=GOING_TIME_BUDGET_SEC):
    """当日の終わったレースから {(会場, 馬場): 情報} を作る。

    Returns
    -------
    dict  {(racecourse, surface): {'track_condition', 'weather', 'race_num'}}
          同じ会場×馬場では**いちばん後のレース**の値が残る（最新の馬場）。
    """
    from src.scraper.jra_scraper import (
        JRA_BASE, PLACE_NAMES, find_r01_result, calc_suffix,
        _try_fetch_result, parse_result_soup,
    )
    from src.scraper.calendar import get_kaisai_on_date
    from bs4 import BeautifulSoup

    t0 = time.time()
    going = {}

    bases = {}
    try:
        r0 = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                       data={'CNAME': 'pw01sli00/AF'}, timeout=15)
        r0.encoding = 'shift_jis'
        soup0 = BeautifulSoup(r0.text, 'lxml')
        for tag in soup0.find_all(onclick=True):
            m = re.search(r'pw01srl\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/(\w{2})',
                          tag.get('onclick', ''))
            if m and m.group(5) == target_date:
                bases.setdefault(f'pw01sde10{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}',
                                 m.group(1))
    except Exception as e:
        print(f'  ⚠ 馬場状態: 結果一覧の取得に失敗 ({e})')

    if not bases:
        # 出走表一覧からの変換（結果一覧がまだ出ていない時間帯）
        try:
            for sb in get_kaisai_on_date(target_date, sess):
                rb = sb.replace('pw01dde01', 'pw01sde10')
                pc = re.search(r'pw01sde10(\d{2})', rb)
                bases.setdefault(rb, pc.group(1) if pc else '00')
        except Exception:
            pass

    if not bases:
        print('  ⚠ 馬場状態: 当日の開催情報が見つかりません（既定の「良」のまま）')
        return going

    for base, pc in bases.items():
        if time.time() - t0 > time_budget_sec:
            print('  ⏱ 馬場状態: 時間上限に達したので打ち切り（予想は続行）')
            break
        rc = PLACE_NAMES.get(pc, '?')
        try:
            r01 = find_r01_result(base, target_date, sess)
        except Exception:
            r01 = None
        if r01 is None:
            print(f'  ⚠ 馬場状態: {rc} の結果ページが見つかりません')
            continue

        miss = 0
        for r in range(1, GOING_FETCH_BUDGET_PER_VENUE + 1):
            if time.time() - t0 > time_budget_sec:
                break
            soup = _try_fetch_result(sess, base, r, target_date, calc_suffix(r01, r))
            info = parse_result_soup(soup, rc, r, target_date, pc) if soup else None
            if not info:
                # まだ終わっていない（＝この先も終わっていない）か障害レース
                miss += 1
                if miss >= GOING_MISS_STREAK:
                    break
                continue
            miss = 0
            surf = info.get('surface')
            tc = info.get('track_condition')
            if surf in ('芝', 'ダート') and tc:
                going[(rc, surf)] = {'track_condition': tc,
                                     'weather': info.get('weather'),
                                     'race_num': r}
            time.sleep(0.5)

    if going:
        for (rc, surf), v in sorted(going.items()):
            print(f"  🌱 馬場状態: {rc}{surf} = {v['track_condition']}"
                  f" (R{v['race_num']} 時点 / 天候 {v.get('weather') or '-'})")
    else:
        print('  ⚠ 馬場状態: 当日まだ1レースも終わっていません（既定の「良」のまま）')
    return going


def apply_going_to_races(races, going):
    """発走前レースに当日の馬場状態を反映する。

    ⚠ 参照元より**後**のレースにだけ入れる。先に終わったレースの情報で
       それより前のレースを塗ると、時間を遡って情報を渡すことになる。
       （呼び出し側は発走前レースだけを渡す想定だが、ここでも番号で守る）
    """
    n = 0
    for race in races:
        key = (race.get('racecourse', ''), race.get('surface', ''))
        src = going.get(key)
        if not src:
            continue
        try:
            rn = int(race.get('race_num') or 0)
        except (TypeError, ValueError):
            rn = 0
        if rn and rn <= int(src.get('race_num') or 0):
            continue
        race['track_condition'] = src['track_condition']
        if src.get('weather'):
            race['weather'] = src['weather']
        race['_going_source'] = f"{key[0]}{key[1]} R{src['race_num']}"
        n += 1
    return n

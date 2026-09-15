# -*- coding: utf-8 -*-
"""ユーザー提供資料（2026-09-15）を **距離単位** で転記して data/course_starts.json を作る。

🔴 2026-09-15: 最初に作った `course_physical.json` は「スタート→1角の**数値**」19組しか
   取っていなかった。資料にはそれ以外に、ほぼ全距離について

       スタート地点 / 最初のコーナー番号 / 内外回り / スタート直後の勾配 / スパイラル

   が書かれており、取りこぼしていた。今日の診断（tenkai/RESULTS_VENUE.md）で
   前半600mのクセの97%が「発走地点ごとの固定オフセット」と分かったので、
   まさにここが要る。

⚠ ユーザー指示（厳守）:
   > 約○mと書かれていない距離について、推定値を勝手にfeatureとして採用しないこと。
   > 「公式記載値」「公式図からの算出値」「未確定」を分けたデータ表にする。

   数値は `約○m` と書かれたものだけ入れ、src を付ける:
     official … 資料が「JRA公式が…と説明している」と明記しているもの
     stated   … 資料に「約○m」とあるが公式明記の確認は取れていないもの
     na       … 概念が存在しない（直線コース等）
   カテゴリ（スタート地点・コーナー番号・勾配）は資料の記述そのままで src='doc'。
   **書かれていないものは None のまま。埋めない。**
"""
import json
import os

OFFICIAL = 'official'   # 資料が JRA公式と明記
STATED = 'stated'       # 資料に「約○m」
DOC = 'doc'             # 資料の記述（数値でない）
NA = 'na'

# (会場, 馬場, 距離) -> dict
# loop: 内回り/外回り/直線/None(区別なし)
# start: 資料の「スタート：」の記述そのまま
# fc   : 最初のコーナー番号
# s2c  : スタート→最初のコーナー(m)。約○mと書かれたものだけ
# s2c_src, s2c_note
# slope: スタート直後の勾配（資料に記述があるものだけ）'上り'/'下り'/'平坦'
STARTS = {
    # ── 東京 芝（直線525.9m・高低差2.7m・左・大回り）────────────────
    ('東京', '芝', 1400): dict(start='向正面', fc=3, slope='上り',
                              note='向正面から3角にかけて上り、その後下り、直線で再び上り'),
    ('東京', '芝', 1600): dict(start='向正面', fc=3, s2c=550.0, s2c_src=STATED,
                              s2c_note='約550m前後', slope='上り'),
    ('東京', '芝', 1800): dict(start='1〜2コーナー間', fc=2,
                              note='スタート後すぐコーナーへ向かう構造'),
    ('東京', '芝', 2000): dict(start='1コーナー奥', fc=2, s2c=100.0, s2c_src=STATED,
                              s2c_note='約100m前後の短い区間',
                              note='短いスタート直後＋長い直線という特殊性'),
    ('東京', '芝', 2300): dict(start='2コーナー付近', fc=3),
    ('東京', '芝', 2400): dict(start='正面スタンド前', fc=1, note='1角まで比較的長い'),
    ('東京', '芝', 2500): dict(start='2コーナー付近', fc=3),
    ('東京', '芝', 2600): dict(start='2コーナー付近', fc=3),
    ('東京', '芝', 3400): dict(start='3コーナー付近', fc=4, note='長距離のため坂を複数回通過'),
    # ── 東京 ダート（直線501.6m・高低差2.5m・1周1899m）──────────────
    ('東京', 'ダート', 1200): dict(start='向正面', fc=3),
    ('東京', 'ダート', 1300): dict(start='向正面', fc=3, note='3角まで比較的長い'),
    ('東京', 'ダート', 1400): dict(start='向正面', fc=3, note='3角まで長い'),
    ('東京', 'ダート', 1600): dict(start='向正面', fc=3, turf_start=True,
                                 note='芝スタート区間がある'),
    ('東京', 'ダート', 2100): dict(start='正面スタンド前', fc=1, note='1周以上'),
    ('東京', 'ダート', 2400): dict(note='坂を複数回通過'),
    # ── 中山 芝（直線310m・高低差5.3m・右）─────────────────────────
    ('中山', '芝', 1200): dict(loop='外回り', start='外回り2コーナー付近', fc=3, slope='下り',
                              note='スタート直後から下り方向・ゴール前に急坂'),
    ('中山', '芝', 1600): dict(loop='外回り', start='1コーナー奥の引き込み線', fc=2,
                              s2c=240.0, s2c_src=OFFICIAL,
                              s2c_note='JRA「スタート地点から最初のコーナーまで240mほど」',
                              note='2角から下り・直線中盤に高低差2m超の急坂'),
    ('中山', '芝', 1800): dict(loop='内回り', start='正面スタンド前', fc=1,
                              note='ゴール前急坂'),
    ('中山', '芝', 2000): dict(loop='内回り', start='4角〜直線入口付近', fc=1,
                              s2c=400.0, s2c_src=OFFICIAL, s2c_note='JRA公式でも約400m',
                              slope='上り',
                              note='スタート直後から上り・1角付近で最高点・その後長い下り'),
    ('中山', '芝', 2200): dict(loop='外回り', start='4角奥', fc=1, note='長い助走'),
    ('中山', '芝', 2500): dict(loop='内回り', start='外回りコース上', fc=3,
                              note='有馬記念と同じコース・内回り主体'),
    ('中山', '芝', 2600): dict(loop='外回り'),
    ('中山', '芝', 3200): dict(note='坂を複数回通過。外/内'),
    ('中山', '芝', 3600): dict(loop='内回り', note='坂を複数回通過'),
    # ── 京都 芝（内回り直線328.4m/高低差3.1m・外回り403.7m/4.3m・右）──
    ('京都', '芝', 1200): dict(loop='内回り', start='向正面', fc=3,
                              note='3角の丘へ向かい3→4角で下る・平坦直線'),
    ('京都', '芝', 1400): dict(note='内回りと外回りで完全に別コース'),
    ('京都', '芝', 1600): dict(loop='外回り', start='向正面', fc=3,
                              note='3角の丘・3→4角で下る・直線は平坦'),
    ('京都', '芝', 1800): dict(loop='外回り', start='2角付近', fc=3,
                              note='3角の丘・4角へ下り'),
    ('京都', '芝', 2000): dict(loop='内回り', start='正面直線半ば', fc=1,
                              s2c=300.0, s2c_src=OFFICIAL, s2c_note='JRA公式でも1角まで約300m',
                              note='3角付近が丘・残り800m付近から下り'),
    ('京都', '芝', 2200): dict(loop='外回り', note='3角の丘・3→4角で大きく下る'),
    ('京都', '芝', 2400): dict(loop='外回り', note='3角の丘・下り'),
    ('京都', '芝', 3000): dict(loop='外回り', note='淀の坂を2回通過'),
    ('京都', '芝', 3200): dict(loop='外回り', note='天皇賞・春。3角の丘を2回通過'),
    # ── 阪神 芝（内回り356.5m/1.9m・外回り473.6m/2.4m・右）───────────
    ('阪神', '芝', 1200): dict(loop='内回り', start='向正面', fc=3, s2c=243.0, s2c_src=STATED,
                              s2c_note='約243m', slope='下り',
                              note='3角へ向かって下り・ゴール前坂'),
    ('阪神', '芝', 1400): dict(loop='内回り', start='向正面', fc=3, s2c=443.0, s2c_src=STATED,
                              s2c_note='約443m', note='ゴール前坂'),
    ('阪神', '芝', 1600): dict(loop='外回り', start='向正面', fc=3, s2c_min=500.0,
                              s2c_src=STATED, s2c_note='「3角まで500m以上」＝下限のみ。点推定にしない',
                              note='ゴール前坂'),
    ('阪神', '芝', 1800): dict(loop='外回り', start='2角奥', fc=3, s2c=600.0, s2c_src=STATED,
                              s2c_note='約600m', note='ゴール前坂'),
    ('阪神', '芝', 2000): dict(loop='内回り', start='正面スタンド前', fc=1, s2c=325.0,
                              s2c_src=STATED, s2c_note='約325m', slope='上り',
                              note='スタート後に坂・ゴール前坂'),
    ('阪神', '芝', 2200): dict(loop='外回り', note='ゴール前坂'),
    ('阪神', '芝', 2400): dict(loop='外回り', note='ゴール前坂'),
    ('阪神', '芝', 2600): dict(loop='外回り'),
    ('阪神', '芝', 3000): dict(loop='外回り', note='外回り主体'),
    ('阪神', '芝', 3200): dict(loop='外回り', note='外回り主体'),
    # ── 中京 芝（直線412.5m・高低差3.5m・1周1705.9m・左）──────────────
    ('中京', '芝', 1200): dict(start='向正面', fc=3, spiral=True,
                              note='3〜4角スパイラル・直線入口に急坂'),
    ('中京', '芝', 1400): dict(start='向正面', fc=3, note='直線の急坂'),
    ('中京', '芝', 1600): dict(start='1〜2角間の引き込み線', fc=2,
                              merge_m=200.0, merge_src=OFFICIAL,
                              merge_note='本線合流まで約200m（1角までではない・JRA公式の距離別解説に明記）',
                              note='バックストレッチ半ばから下り・直線に高低差約2mの坂'),
    ('中京', '芝', 2000): dict(start='ホームストレッチ', fc=1,
                              note='下り・上りを経て4角・直線入口の急坂'),
    ('中京', '芝', 2200): dict(note='4角から直線・急坂'),
    ('中京', '芝', 3000): dict(note='坂を複数回通過'),
    # ── 新潟 芝（外回り658.7m/2.2m・内回り358.7m/0.8m・左）────────────
    ('新潟', '芝', 1000): dict(loop='直線', start='直線コース', fc=None,
                              s2c_src=NA, s2c_note='コーナーが無いので概念が存在しない',
                              note='日本唯一のJRA直線芝1000m'),
    ('新潟', '芝', 1200): dict(loop='内回り', start='向正面', fc=3),
    ('新潟', '芝', 1400): dict(loop='内回り'),
    ('新潟', '芝', 1600): dict(loop='外回り', start='向正面中央付近', fc=3, s2c=550.0,
                              s2c_src=STATED, s2c_note='約550m', spiral=True,
                              note='3〜4角はスパイラル'),
    ('新潟', '芝', 1800): dict(loop='外回り', fc=3, note='長い向正面'),
    ('新潟', '芝', 2000): dict(note='内回り/外回りで区別'),
    ('新潟', '芝', 2200): dict(loop='外回り'),
    ('新潟', '芝', 2400): dict(loop='外回り'),
    ('新潟', '芝', 3000): dict(loop='外回り'),
    ('新潟', '芝', 3200): dict(loop='外回り'),
    # ── 福島 芝（直線292m(A)・高低差1.9m・1周1600m・右）────────────────
    ('福島', '芝', 1200): dict(start='向正面', fc=3, note='小回り・細かなアップダウン'),
    ('福島', '芝', 1800): dict(start='スタンド前', fc=1, s2c=300.0, s2c_src=STATED,
                              s2c_note='約300m', slope='上り',
                              note='スタート直後に高低差1.2mの上り・その後下り'),
    ('福島', '芝', 2000): dict(start='4角奥ポケット', fc=1, s2c=500.0, s2c_src=STATED,
                              s2c_note='約500m', slope='下り',
                              note='ホームストレッチ中ほどまで下り・その後上り・連続アップダウン'),
    ('福島', '芝', 2600): dict(note='小回り・複数回のアップダウン'),
    # ── 小倉 芝（直線293m・高低差3.0m・1周1615.1m・右）──────────────────
    ('小倉', '芝', 1200): dict(start='2角付近', fc=3, spiral=True,
                              note='スパイラルカーブ・平坦'),
    ('小倉', '芝', 1800): dict(start='直線半ば', fc=1, s2c=270.0, s2c_src=OFFICIAL,
                              s2c_note='JRA公式の小倉大賞典解説でも1角まで約270m', slope='上り',
                              note='1〜2角で上り・2角後は平坦・3〜4角で下り'),
    ('小倉', '芝', 2000): dict(start='4角奥ポケット', fc=1, s2c=470.0, s2c_src=STATED,
                              s2c_note='約470m', slope='上り',
                              note='スタートから1角に向けて上り・2角付近が頂点・以降下り'),
    ('小倉', '芝', 2600): dict(spiral=True, note='アップダウン・スパイラルカーブ'),
    # ── 札幌 芝（直線266.1m・高低差0.7m・1周1640.9m・右）────────────────
    ('札幌', '芝', 1200): dict(start='2角付近', fc=3, note='大きなコーナー・ほぼ平坦'),
    ('札幌', '芝', 1500): dict(fc=1, note='1角まで比較的短い・コーナー主体'),
    ('札幌', '芝', 1800): dict(start='ホームストレッチ中ほど', fc=1, s2c=180.0, s2c_src=STATED,
                              s2c_note='約180m', note='すぐカーブ・4つのコーナーが大きい・ほぼ平坦'),
    ('札幌', '芝', 2000): dict(start='4角奥ポケット', fc=1, s2c=380.0, s2c_src=STATED,
                              s2c_note='約380m', note='ほぼ平坦・大きな4コーナー'),
    ('札幌', '芝', 2600): dict(note='ほぼ平坦・大きなコーナー'),
    # ── 函館 芝（直線262.1m・高低差3.5m・1周1626.6m・右）────────────────
    ('函館', '芝', 1200): dict(start='2角付近', fc=3, note='起伏を伴う小回り'),
    ('函館', '芝', 1800): dict(start='スタンド前付近', fc=1, s2c=276.0, s2c_src=STATED,
                              s2c_note='約276m前後', note='その後上り下り'),
    ('函館', '芝', 2000): dict(start='4角奥ポケット', fc=1, s2c=470.0, s2c_src=OFFICIAL,
                              s2c_note='JRA公式も「1角まで約470m」と明記', slope='下り',
                              note='前半は下り・2角途中から4角まで上り'),
    ('函館', '芝', 2600): dict(note='起伏を複数回通過'),
}


def build(base_dir='.'):
    out = {
        'version': 1,
        '_source': 'ユーザー提供資料「JRA全競馬場・距離別コース物理データ」(2026-09-15)',
        '_why': ('前半600mのクセの97%が「(会場,馬場,実距離)＝発走地点ごとの固定オフセット」'
                 'で説明できると分かったため（tenkai/RESULTS_VENUE.md）。'
                 '会場単位ではなく距離単位で持つ。'),
        '_src_levels': {
            OFFICIAL: '資料が「JRA公式が…と説明している」と明記している数値',
            STATED: '資料に「約○m」とあるが公式明記の確認は取れていない数値',
            DOC: '資料の記述（数値ではないカテゴリ。スタート地点・コーナー番号・勾配など）',
            NA: '概念が存在しない（直線コース等）',
        },
        '_rules': [
            '数値は「約○m」と書かれたものだけ入れる。推定しない（ユーザー指示）。',
            '書かれていない項目は None のまま。平均等で埋めない。',
            '特徴量に使ってよい数値は official + stated のみ。',
            '中京芝1600の200mは「本線合流まで」で1角までではないので merge_m に分ける。',
            '阪神芝1600外の「500m以上」は下限のみなので s2c_min に入れ、点推定にしない。',
        ],
        'starts': {},
    }
    for (rc, sf, dist), v in STARTS.items():
        rec = {
            'racecourse': rc, 'surface': sf, 'distance': dist,
            'loop': v.get('loop'),
            'start_location': v.get('start'),
            'first_corner': v.get('fc'),
            'start_to_first_corner_m': v.get('s2c'),
            'start_to_first_corner_min_m': v.get('s2c_min'),
            'start_to_first_corner_src': v.get('s2c_src'),
            'start_to_mainline_merge_m': v.get('merge_m'),
            'start_to_mainline_merge_src': v.get('merge_src'),
            'start_slope': v.get('slope'),
            'turf_start': v.get('turf_start'),
            'spiral_corner': v.get('spiral'),
            'note': v.get('note') or v.get('s2c_note') or v.get('merge_note'),
            'src': DOC,
        }
        out['starts'][f'{rc}_{sf}_{dist}'] = rec
    path = os.path.join(base_dir, 'data', 'course_starts.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out, path


if __name__ == '__main__':
    o, p = build('.')
    s = o['starts']
    import collections
    print(f'✅ {p}  {len(s)} 件')
    for k, label in [('start_location', 'スタート地点'), ('first_corner', '最初のコーナー'),
                     ('loop', '内外回り'), ('start_slope', 'スタート直後の勾配'),
                     ('start_to_first_corner_m', 'スタート→1角(数値)'),
                     ('spiral_corner', 'スパイラル')]:
        n = sum(1 for v in s.values() if v.get(k) is not None)
        print(f'   {label:20s} {n:3d}/{len(s)}')
    print('  src内訳(数値):', dict(collections.Counter(
        v['start_to_first_corner_src'] for v in s.values()
        if v.get('start_to_first_corner_src'))))
    print('  会場×馬場:', dict(collections.Counter(
        f"{v['racecourse']}{v['surface']}" for v in s.values())))

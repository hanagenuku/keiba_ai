"""ユーザー提供のコース物理データを、出典の信頼度つきの構造化データにする。

🔑 ユーザーの指示（そのまま守る）:
   「約○mと書かれていない距離について、推定値を勝手に feature として採用しないこと。
     『公式記載値』『公式図からの算出値』『未確定』を分けたデータ表にするのが安全。
     特にAIに投入するなら、ここを曖昧にしない。」

  → すべての数値に `src` を付ける。特徴量として使ってよいのは official / stated のみ。
    qualitative（「長い」等の記述のみ）は数値を入れない。欠けたまま残す。

🔴 高低差は**2つの別の量**がある。混ぜてはいけない。
    course_elevation_range_m : コース全体の高低差（例 中山芝 5.3m）
    final_slope_height_m     : 直線／ゴール前の坂の高さ（例 中山芝 2.2m）
  既存の course_distance_profiles.json の hill.elevation_diff_m は両者が混在している
  （福島の position に「全体高低差は1.9m」と書かれていて、値は坂の1.2mだった）。
"""
import json, os

SRC = {
    'official': 'JRA公式が距離別に明記している値',
    'stated':   'ユーザー提供資料に「約○m」として記載（公式明記の確認は取れていない）',
    'web':      '2026-07 のweb検索由来（要検証・低〜中信頼度）',
    'na':       '該当なし（直線コース等、概念が存在しない）',
}

# ── コース単位（競馬場 × 芝ダ × 内外回り）
COURSES = {
  #                         直線m   全体高低差m  1周m
  '東京_芝':               (525.9, 2.7,  None),
  '東京_ダート':            (501.6, 2.5,  1899),
  '中山_芝':               (310.0, 5.3,  None),
  '中山_ダート':            (308.0, 4.5,  None),
  '京都_芝_内回り':          (328.4, 3.1,  None),
  '京都_芝_外回り':          (403.7, 4.3,  None),
  '京都_ダート':            (None,  3.0,  None),
  '阪神_芝_内回り':          (356.5, 1.9,  1689),
  '阪神_芝_外回り':          (473.6, 2.4,  2089),
  '阪神_ダート':            (352.7, 1.6,  None),
  '中京_芝':               (412.5, 3.5,  1705.9),
  '中京_ダート':            (410.7, 3.4,  1530),
  '新潟_芝_外回り':          (658.7, 2.2,  2223),
  '新潟_芝_内回り':          (358.7, 0.8,  1623),
  '新潟_ダート':            (353.9, 0.6,  1472.5),
  '福島_芝':               (292.0, 1.9,  1600),
  '福島_ダート':            (295.7, 2.1,  1444.6),
  '小倉_芝':               (293.0, 3.0,  1615.1),
  '小倉_ダート':            (291.3, 2.9,  1445.4),
  '札幌_芝':               (266.1, 0.7,  1640.9),
  '札幌_ダート':            (264.3, 0.9,  1487),
  '函館_芝':               (262.1, 3.5,  1626.6),
  '函館_ダート':            (260.3, 3.5,  1475.8),
}
DIRECTION = {'東京':'左','中山':'右','京都':'右','阪神':'右','中京':'左',
             '新潟':'左','福島':'右','小倉':'右','札幌':'右','函館':'右'}

# ── 距離単位: (キー, 内外, 1角までm, 最初のコーナー番号, src, 備考)
#    ⚠ 数値が無いもの（「長い」「比較的長い」だけ）は**入れない**。
DIST = [
 # --- 公式が距離別に明記していると資料が名指ししているもの ---
 ('中山_芝_1600', '外', 240, 2, 'official', 'JRA公式「240mほど」'),
 ('中山_芝_2000', '内', 400, 1, 'official', 'JRA公式「約400m」'),
 ('京都_芝_2000', '内', 300, 1, 'official', 'JRA公式「1角まで約300m」'),
 ('小倉_芝_1800', None, 270, 1, 'official', 'JRA公式・小倉大賞典解説「約270m」'),
 ('函館_芝_2000', None, 470, 1, 'official', 'JRA公式「1角まで約470m」'),
 # --- 資料に「約○m」とあるが公式明記の確認までは取れていないもの ---
 ('東京_芝_1600', None, 550, 3, 'stated', '「約550m前後」'),
 ('東京_芝_2000', None, 100, 2, 'stated', '「約100m前後の短い区間」'),
 ('阪神_芝_1200', '内', 243, 3, 'stated', ''),
 ('阪神_芝_1400', '内', 443, 3, 'stated', ''),
 ('阪神_芝_1800', '外', 600, 3, 'stated', '「3角まで約600m」'),
 ('阪神_芝_2000', '内', 325, 1, 'stated', '「1角まで約325m」'),
 ('新潟_芝_1600', '外', 550, 3, 'stated', '「3角まで約550m」'),
 ('福島_芝_1800', None, 300, 1, 'stated', '「1角まで約300m」'),
 ('福島_芝_2000', None, 500, 1, 'stated', '「1角まで約500m」'),
 ('小倉_芝_2000', None, 470, 1, 'stated', '「1角まで約470m」'),
 ('札幌_芝_1800', None, 180, 1, 'stated', '「1角まで約180m」'),
 ('札幌_芝_2000', None, 380, 1, 'stated', '「1角まで約380m」'),
 ('函館_芝_1800', None, 276, 1, 'stated', '「約276m前後」'),
 # --- 概念が存在しない ---
 ('新潟_芝_1000', '直線', None, None, 'na', 'JRA唯一の直線芝1000m。コーナーなし'),
]

# ⚠ 別概念なので start_to_first_corner には入れない
SPECIAL = {
 '中京_芝_1600': {'start_to_mainline_merge_m': 200, 'src': 'official',
                 'note': '🔴 これは「本線合流まで」であって1角までではない。別列。'},
 '阪神_芝_1600': {'start_to_first_corner_min_m': 500, 'loop': '外', 'first_corner': 3,
                 'src': 'stated', 'note': '「3角まで500m以上」＝下限のみ。点推定にしない'},
}

# 既存 web 由来（2026-07）。信頼度を落として併記し、突合できるようにする
WEB = json.load(open('data/course_distance_profiles.json'))['_start_to_corner_m_reference']


def build():
    courses = {}
    for k, (st, elev, lap) in COURSES.items():
        v = k.split('_')[0]
        courses[k] = {
            'direction': DIRECTION[v],
            'final_straight_m': st,
            'course_elevation_range_m': elev,
            'lap_m': lap,
            'src': 'stated',
        }
    dists = {}
    for key, loop, m, fc, src, note in DIST:
        e = {'loop': loop, 'first_corner': fc, 'src': src}
        if m is not None:
            e['start_to_first_corner_m'] = float(m)
        if note:
            e['note'] = note
        w = WEB.get(key)
        if w is not None:
            e['web_value_m'] = float(w)
            if m is not None:
                e['web_delta_m'] = round(float(w) - float(m), 1)
        dists[key] = e
    payload = {
        'version': 1,
        '_note': ('ユーザー提供のコース物理データ。すべての数値に src を付けてある。'
                  '**特徴量として使ってよいのは src が official / stated のものだけ**。'
                  '「長い」「比較的長い」等の記述しかない距離は数値を入れていない（欠測のまま）。'),
        '_elevation_warning': ('course_elevation_range_m（コース全体）と '
                               'course_distance_profiles.json の hill.elevation_diff_m'
                               '（直線／ゴール前の坂）は**別の量**。混ぜないこと。'
                               '例: 中山芝は全体5.3m / ゴール前坂2.2m。'),
        '_src_levels': SRC,
        'courses': courses,
        'distances': dists,
        'special': SPECIAL,
    }
    out = os.path.join('data', 'course_physical.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    n_num = sum(1 for e in dists.values() if 'start_to_first_corner_m' in e)
    print(f'✅ {out}')
    print(f'   コース {len(courses)} / 距離セル {len(dists)}（うち数値あり {n_num}）')
    for lv in ('official', 'stated', 'na'):
        print(f'     {lv:9s} {sum(1 for e in dists.values() if e["src"]==lv)}')
    ov = [(k, e['start_to_first_corner_m'], e['web_value_m'], e['web_delta_m'])
          for k, e in dists.items() if 'web_delta_m' in e]
    if ov:
        print('\n   web由来の既存値との突合:')
        for k, a, b, d in ov:
            print(f'     {k:16s} 今回 {a:5.0f}m / web {b:5.0f}m  差 {d:+5.1f}m')
    return payload


if __name__ == '__main__':
    build()

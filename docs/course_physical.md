# コース物理データ（ユーザー提供・2026-09-14）

出典データは `data/course_physical.json`、生成は `scripts/build_course_physical.py`。

## 基本単位

**競馬場 × 芝/ダート × 距離 × 内回り/外回り**。
「東京競馬場は直線525.9m」のような競馬場単位では不十分で、同じ競馬場でも
距離によってスタート地点が変わり、レース開始後の物理条件が大きく変わるため。

## 🔴 出典の信頼度を必ず持つ（ユーザー指示）

> 「約○mと書かれていない距離について、**推定値を勝手に feature として採用しないこと**。
>   『公式記載値』『公式図からの算出値』『未確定』を分けたデータ表にするのが安全。
>   特にAIに投入するなら、ここを曖昧にしない。」

`src` を全数値に付けた。**特徴量に使ってよいのは `official` / `stated` のみ。**

```
official  5件   JRA公式が距離別に明記（中山芝1600=240 / 中山芝2000=400 /
                京都芝2000内=300 / 小倉芝1800=270 / 函館芝2000=470）
stated   13件   資料に「約○m」と記載（公式明記の確認までは取れていない）
web      16件   2026-07のweb検索由来（低〜中信頼度・既存）
na        1件   新潟芝1000＝直線コース。1角という概念が存在しない
```

✅ **重複4件が独立に一致した**（中山芝1600 240/240・阪神芝1200 243/243・
小倉芝1800 270/272・小倉芝2000 470/472）。両系統の裏付けになっている。

⚠ 「長い」「比較的長い」しか書かれていない距離は**数値を入れていない**（欠測のまま）。

⚠ **中京芝1600の「約200m」は本線合流までであって1角までではない。**
   別概念なので `start_to_mainline_merge_m` として別列にした。
⚠ **阪神芝1600外の「500m以上」は下限のみ。** 点推定にせず
   `start_to_first_corner_min_m` として保持。

## 🔴 高低差は2つの別の量がある。混ぜてはいけない

```
course_elevation_range_m  コース全体の高低差   中山芝 5.3m
final_slope_height_m      直線／ゴール前の坂   中山芝 2.2m
```

既存の `course_distance_profiles.json` の `hill.elevation_diff_m` は**両者が混在**していた。
福島のエントリが自分で答えを書いている:

```
値 1.2 / position「ゴール前(残り170-50m)。**全体高低差は1.9m**、1周に2回アップダウン」
```

小倉芝は逆に 3.0（＝全体）が入っていて position は「2コーナーに丘、直線自体には坂を設けず」。
**同じ列に別の量が入っている。**

## ❌ 「1角までの距離 → 枠バイアス」は、データを増やしたら消えた

2026-09-14 にデータが20件→34件（N>=300のセルは16→30）に増えたので測り直した。

```
                        前回(16セル)          今回(30セル)
全セル                 ρ -0.647 (P=0.007)   ρ -0.278 (P=0.137)
official+stated のみ        —                ρ -0.039 (P=0.877)   ← ほぼゼロ
会場統制                ρ -0.319 (P=0.289)   ρ -0.193 (P=0.306)
同一会場 1800 vs 2000        —                3/4対（偶然なら50%）
```

🔑 **信頼度の高い値だけで見ると相関が無い。** 前回の −0.647 は web由来の
小さいサブセットが作っていた。`docs/course_detail_request.md` で
「集める価値はある」と書いた見立ては**外れ**。集めたことで分かった。

## ✅ 代わりに見つかった defect: 内回り/外回りが分かれていない

`course_profiles.json` は **venue×surface の20キーしかなく**、内外回りを持たない。
外回りの直線長が内回りのレースにも適用されている。

```
新潟_芝  本番 659m → 実際 内回り 358.7m   101レース  +84% 誤り
阪神_芝  本番 474m → 実際 内回り 356.5m   231レース  +33% 誤り
京都_芝  本番 404m → 実際 内回り 328.4m    75レース  +23% 誤り
⚠ 新潟芝1000（直線コース）84レースも 659m と教わっている

距離別   阪神_芝_2000内 128R / 京都_芝_1200 75R / 阪神_芝_1200 63R
        新潟_芝_1200 49R / 新潟_芝_2200 37R / 阪神_芝_2200 33R

合計 407レース = history.db 12,017レース中 3.4%
```

この値は `f_straight_match` / `straight_class` / `f_course_fit_score` 経由で
本番134特徴量に届いている。`loop_by_distance`（中山・阪神・京都・新潟の芝）は
既に `course_distance_profiles.json` にあるので、**引く側を直せば解消する**。

⚠ 直すと特徴量が動くので本番同型A/Bを通してから
（手順は `docs/course_ledger_next.md` と同じ基準）。

## 今後の目標形（ユーザー提示）

基本 `surface` / `distance` / `course_direction` / `inner_outer` / `lap_distance` /
`final_straight_length` / `course_elevation_range`

スタート `start_to_first_corner` / `first_corner_number` / `start_section_slope` /
`start_uphill` / `start_downhill` / `start_to_first_slope`

中盤 `backstretch_slope` / `max_uphill` / `max_downhill` / `uphill_count` /
`downhill_count` / `corner_tightness` / `spiral_corner`

ゴール前 `final_slope_height` / `final_slope_length` /
`final_slope_start_from_goal` / `flat_after_final_slope`

全体 `number_of_corners` / `number_of_laps` / `slope_pass_count` /
`technical_course_score` / `stamina_course_score` / `speed_course_score`

いま埋まっているのは基本7つのうち6つ＋スタート2つ＋ゴール前1つ。
残りは JRA公式コース図からの読み取りが要る（＝`公式図からの算出値` の層）。

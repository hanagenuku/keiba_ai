"""
スーパーファミコン版ダービースタリオン風の静止画を生成する。
320x180 ドット絵 → 4倍拡大 → 1280x720 出力
python test_frame.py → output/test_frame.png
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 320, 180
SCALE = 4

os.makedirs("output", exist_ok=True)

# ── レイアウト ──────────────────────────────────────────────
SKY_Y0  = 0
SKY_Y1  = 20   # 空
BLDG_Y0 = 18
BLDG_Y1 = 34   # スタンド
TREE_Y0 = 28
TREE_Y1 = 44   # 木
OUTG_Y0 = 43
OUTG_Y1 = 50   # 外側芝
RAIL_Y0 = 50
RAIL_Y1 = 52   # 外柵
TRK_Y0  = 52   # 芝コース上端
TRK_Y1  = 116  # 芝コース下端
RAIL2_Y = 116  # 内柵
INNER_Y = 118  # 内側芝
PANEL_Y = 122  # 下部パネル

HORSE_CY = 83  # 馬体中心 y（芝コース内）

# ── パレット ────────────────────────────────────────────────
SKY_A   = (100, 172, 220)
SKY_B   = (120, 190, 232)
BLDG_A  = (182, 180, 186)
BLDG_B  = (164, 162, 170)
WIN_C   = (138, 140, 158)
TREE_A  = (28, 88, 36)
TREE_B  = (38, 108, 46)
TREE_C  = (48, 124, 54)
OUTG    = (52, 144, 58)
RAIL    = (228, 228, 220)
TRK_A   = (46, 132, 52)
TRK_B   = (38, 116, 44)
INNER   = (40, 108, 48)
UI_BG   = (28, 28, 30)
UI_BG2  = (20, 20, 22)
UI_BD   = (60, 60, 65)
POLE_R  = (210, 26, 26)
MAP_GR  = (38, 96, 44)
MAP_TRK = (170, 145, 90)

HORSE_COLORS = [
    (215, 50,  50),   # 1 赤
    (50,  110, 210),  # 2 青
    (220, 185, 30),   # 3 黄
    (70,  200, 90),   # 4 緑
    (175, 70,  210),  # 5 紫
    (220, 120, 25),   # 6 橙
    (60,  200, 205),  # 7 水
    (200, 200, 200),  # 8 白
]
def hcol(n): return HORSE_COLORS[(n-1) % len(HORSE_COLORS)]

# ── 馬スプライト（ドット絵定義） ────────────────────────────
# 24×17 ピクセル、左向き（頭が左）
# 0=透明 1=馬体 2=暗部/蹄/尻尾 3=騎手帽(変数) 4=白シルク 5=頭 6=鼻 7=目 8=脚
_HA = [  # Frame A: 脚伸展
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,4,4,3,3,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [4,4,4,4,4,4,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,4,4,4,1,1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,2,2,2,0,0,0],
  [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,2,2,0,0,0],
  [5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,0,0,0,0],
  [6,5,5,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
  [0,6,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,8,8,0,0,0,0,0,8,8,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,8,0,8,0,0,0,0,8,0,8,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,8,0,8,0,0,0,0,8,0,8,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,2,8,2,0,0,0,0,2,8,2,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]
_HB = [  # Frame B: 脚収縮
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,4,4,3,3,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [4,4,4,4,4,4,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,4,4,4,1,1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,2,2,2,0,0,0],
  [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,2,2,0,0,0],
  [5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,2,0,0,0,0],
  [6,5,5,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
  [0,6,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,8,8,0,0,0,0,0,8,8,0,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,8,8,0,0,0,0,0,8,8,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]
SPRITE_FRAMES = [_HA, _HB]

SPRITE_FIXED = {
    1: np.array([132, 80, 38],  dtype=np.uint8),  # 馬体
    2: np.array([58,  30, 8],   dtype=np.uint8),  # 暗部/蹄/尻尾
    4: np.array([225, 225, 225],dtype=np.uint8),  # 白シルク
    5: np.array([105, 60, 24],  dtype=np.uint8),  # 頭
    6: np.array([82,  46, 16],  dtype=np.uint8),  # 鼻
    7: np.array([8,   5,  3],   dtype=np.uint8),  # 目
    8: np.array([98,  55, 20],  dtype=np.uint8),  # 脚
}

def blit_horse(arr, x, y, jcol, frame=0, number=1):
    """numpy配列 arr(H,W,3) に馬スプライトを直接書き込む。"""
    sprite = SPRITE_FRAMES[frame]
    jc = np.array(jcol, dtype=np.uint8)
    sh, sw = len(sprite), len(sprite[0])
    for row in range(sh):
        for col in range(sw):
            idx = sprite[row][col]
            if idx == 0:
                continue
            py, px = y + row, x + col
            if 0 <= py < H and 0 <= px < W:
                arr[py, px] = jc if idx == 3 else SPRITE_FIXED[idx]


# ── 背景構築 ────────────────────────────────────────────────

def build_bg(arr):
    # 空（2色帯）
    arr[SKY_Y0:12, :] = SKY_A
    arr[12:SKY_Y1, :] = SKY_B

    # スタンド（簡易）
    arr[BLDG_Y0:BLDG_Y1, :] = BLDG_A
    # 窓の行
    arr[21:25, :] = BLDG_B
    arr[28:32, :] = BLDG_B
    for wx in range(0, W, 18):
        arr[21:25, wx:wx+10] = WIN_C
        arr[28:32, wx:wx+10] = WIN_C

    # 木（暗い緑 + バンプトップ）
    arr[TREE_Y0:TREE_Y1, :] = TREE_A
    # バンプ: 5ピクセルおきに丸いでっぱり
    for tx in range(0, W, 16):
        for dx in range(-7, 8):
            px = tx + dx
            if 0 <= px < W:
                dy = abs(dx) / 7
                top_y = int(TREE_Y0 - 6 * (1 - dy * dy))
                for yy in range(max(0, top_y), TREE_Y0 + 4):
                    arr[yy, px] = TREE_C if yy < TREE_Y0 else TREE_B

    # 外側芝
    arr[OUTG_Y0:OUTG_Y1, :] = OUTG

    # 外柵
    arr[RAIL_Y0:RAIL_Y1, :] = RAIL

    # 芝コース（交互ストライプ 12px）
    for sx in range(0, W, 24):
        arr[TRK_Y0:TRK_Y1, sx:sx+12] = TRK_A
        arr[TRK_Y0:TRK_Y1, sx+12:sx+24] = TRK_B

    # 内柵
    arr[RAIL2_Y:RAIL2_Y+2, :] = RAIL

    # 内側芝
    arr[INNER_Y:PANEL_Y, :] = INNER


# ── 距離ポール ──────────────────────────────────────────────

def draw_pole(arr, px, label_digit):
    """赤白ストライプポール + 丸数字。px=左端x座標。"""
    stripe = 6
    for y in range(RAIL_Y1, RAIL2_Y, stripe):
        col = np.array(POLE_R if ((y - RAIL_Y1) // stripe) % 2 == 0 else RAIL, dtype=np.uint8)
        arr[y:y+stripe, px:px+4] = col
    # 丸（11x11）
    cy = RAIL_Y0 - 7
    cx = px + 2
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if dy*dy + dx*dx <= 36:
                yy, xx = cy+dy, cx+dx
                if 0<=yy<H and 0<=xx<W:
                    arr[yy, xx] = [230, 230, 225]
    # 数字を3x5ドットフォントで（簡易）
    _draw_digit(arr, cx-1, cy-3, label_digit)


# 3×5 ドットフォント
_DIGITS = {
    '1': ["010","110","010","010","111"],
    '2': ["111","001","111","100","111"],
    '3': ["111","001","011","001","111"],
    '4': ["101","101","111","001","001"],
}

def _draw_digit(arr, x, y, d):
    pattern = _DIGITS.get(str(d), _DIGITS['4'])
    for row, bits in enumerate(pattern):
        for col, b in enumerate(bits):
            if b == '1':
                yy, xx = y+row, x+col
                if 0<=yy<H and 0<=xx<W:
                    arr[yy, xx] = [210, 26, 26]


# ── UI: フォント ────────────────────────────────────────────

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_fc = {}
def fn(s):
    if s in _fc: return _fc[s]
    f = None
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try: f = ImageFont.truetype(p, s); break
            except: pass
    _fc[s] = f or ImageFont.load_default()
    return _fc[s]


def draw_ui(img, data):
    """PIL でUI要素（テキスト系）を描画する。"""
    draw = ImageDraw.Draw(img)
    WHITE = (255,255,255)
    TW    = (228,228,228)

    # ── 下部パネル背景 ──
    draw.rectangle([0, PANEL_Y, W, H], fill=UI_BG)
    draw.rectangle([0, PANEL_Y, W, H], outline=UI_BD, width=1)

    # ── 区切り線 ──
    draw.line([83, PANEL_Y, 83, H], fill=UI_BD, width=1)

    # ── レース情報ボックス（左上） ──
    draw.rectangle([1, 1, 85, 38], fill=(6,6,8))
    draw.rectangle([1, 1, 85, 38], outline=(88,88,96), width=1)
    f8 = fn(8)
    draw.text((4, 2),  data["venue"],     fill=WHITE, font=f8)
    draw.text((4, 13), data["race_name"], fill=WHITE, font=f8)
    draw.text((4, 24), data["race_cls"],  fill=WHITE, font=f8)

    # ── コース図（左下パネル） ──
    # 楕円コース（簡易）
    draw.rectangle([2, PANEL_Y+2, 81, H-2], fill=UI_BG2)
    draw.ellipse([4, PANEL_Y+4, 80, H-20], fill=MAP_GR, outline=(155,208,155))
    draw.ellipse([14, PANEL_Y+11, 70, H-27], outline=MAP_TRK, width=4)
    draw.ellipse([22, PANEL_Y+16, 62, H-32], fill=MAP_GR)
    draw.text((4, H-17), data.get("map_label", "芝　良"), fill=TW, font=fn(7))

    # 現在位置ドット（例：4コーナー付近）
    draw.ellipse([44, H-30, 50, H-24], fill=hcol(7))

    # ── 実況テキスト（右下パネル） ──
    com_lines = data.get("commentary", [])
    f9 = fn(9)
    y = PANEL_Y + 4
    for segs in com_lines:
        x = 87
        for text, col in segs:
            draw.text((x, y), text, fill=col, font=f9)
            bb = draw.textbbox((x, y), text, font=f9)
            x = bb[2] + 1
        y += 10


# ── メイン ─────────────────────────────────────────────────

def build():
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    build_bg(arr)

    # 距離ポール（向こう正面 "3"）
    draw_pole(arr, px=200, label_digit=4)

    # 馬を後ろのレーンから手前の順に描画
    # (x=左端, y=上端, jcol, frame, number)
    horses = [
        (200, HORSE_CY - 8,  hcol(1), 1, 1),   # 先頭・右
        (172, HORSE_CY - 6,  hcol(2), 0, 2),
        (148, HORSE_CY - 4,  hcol(7), 1, 7),   # 7番
        (118, HORSE_CY - 2,  hcol(5), 0, 5),   # 5番
        ( 90, HORSE_CY    ,  hcol(3), 1, 3),   # 3番
        ( 60, HORSE_CY + 2,  hcol(8), 0, 8),
        ( 30, HORSE_CY + 4,  hcol(4), 1, 4),   # 後方
    ]
    for (x, y, jc, frm, num) in horses:
        blit_horse(arr, x, y, jc, frm, num)

    # numpy → PIL
    img = Image.fromarray(arr)

    # UI（テキスト等）
    data = {
        "venue":     "中山10R",
        "race_name": "天皇賞（秋）",
        "race_cls":  "GI　芝2000m",
        "map_label": "中山　芝　良",
        "commentary": [
            [("先頭は ", (228,228,228)), ("7番", hcol(7)), ("メジロマックイーン", (80,195,212))],
            [("これを追って ", (228,228,228)), ("5番", hcol(3)), ("トウカイテイオー", hcol(3))],
            [("さらに ", (228,228,228)), ("3番", (215,100,195)), ("ナリタブライアン", (215,100,195))],
            [("4コーナーをカーブして", (228,228,228))],
            [("さあ　直線コースに入りました！", (228,228,228))],
        ],
    }
    draw_ui(img, data)

    # 4倍拡大（NEAREST = ドット絵らしさ保持）
    out = img.resize((W * SCALE, H * SCALE), Image.NEAREST)
    out.save("output/test_frame.png")
    print(f"Saved: output/test_frame.png  ({W*SCALE}x{H*SCALE})")

build()

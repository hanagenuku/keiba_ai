"""
参考画像を静止画で再現するためのテストスクリプト。
python test_frame.py → output/test_frame.png を生成
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
OUT = "output/test_frame.png"

# ── 色 ────────────────────────────────────────────────────────
SKY_TOP   = (130, 195, 235)
SKY_BOT   = (160, 215, 245)
CLOUD     = (250, 253, 255)
BLDG      = (195, 193, 195)
BLDG_W    = (215, 213, 218)
BLDG_WIN  = (155, 158, 175)
TREE      = (42, 110, 48)
TREE_D    = (28,  80, 35)
TREE_L    = (55, 135, 58)
GRASS_UP  = (68, 162, 72)     # 外側芝
RAIL_C    = (235, 237, 230)
BROWN_D   = (148, 118, 76)    # ダート（内側）
GRASS_M   = (54, 148, 60)     # メイン芝
GRASS_S   = (46, 130, 52)     # 芝ストライプ暗
GRASS_SL  = (62, 160, 66)     # 芝ストライプ明
SHADOW_G  = (36, 118, 44)
HBODY     = (110, 66, 30)     # 馬体茶色
HBODY_L   = (130, 82, 40)     # 馬体（ハイライト）
HBODY_D   = (82, 48, 18)      # 馬体暗部
HLEG      = (90, 52, 22)
HDARK     = (60, 34, 10)      # たてがみ・尻尾
HSNOUT    = (95, 58, 28)
HJOCK     = (228, 228, 228)   # 白シルク
UI_BG     = (32, 32, 34)
UI_BG2    = (24, 24, 26)
UI_BD     = (65, 65, 70)
WHITE     = (255, 255, 255)
TEXT_W    = (228, 228, 228)
POLE_R    = (208, 28, 28)
MAP_G     = (44, 108, 50)
MAP_L     = (60, 148, 65)
MAP_TRK   = (185, 160, 100)  # コース図のトラック色

HORSE_COLORS = [
    (220, 55,  55),   # 1: 赤
    (55,  110, 215),  # 2: 青
    (230, 190, 35),   # 3: 黄
    (80,  205, 100),  # 4: 緑
    (185, 80,  215),  # 5: 紫
    (225, 125, 30),   # 6: オレンジ
    (70,  205, 210),  # 7: 水色
    (215, 215, 215),  # 8: 白
]
def hcolor(n): return HORSE_COLORS[(n-1) % len(HORSE_COLORS)]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_fc = {}
def font(s):
    if s in _fc: return _fc[s]
    f = None
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try: f = ImageFont.truetype(p, s); break
            except: pass
    _fc[s] = f or ImageFont.load_default()
    return _fc[s]

# ── レイアウト定数 ─────────────────────────────────────────────
SKY_H     = 145   # 空の高さ
TREE_Y    = 120   # 木の上端
TREE_H    = 75    # 木の帯の高さ
RAIL1_Y   = 215   # 外柵
TRACK_Y0  = 222   # 芝コース上端
TRACK_Y1  = 440   # 芝コース下端（馬の脚の下）
RAIL2_Y   = 444   # 内柵
BROWN_Y   = 452   # ダート帯上端
BROWN_H   = 35    # ダート帯高さ
PANEL_Y   = 500   # 下部パネル開始
PANEL_L   = 330   # コース図パネル幅

HORSE_CY  = 340   # 馬体中心 y

# ── 背景描画 ───────────────────────────────────────────────────
def draw_sky(draw):
    for y in range(SKY_H):
        t = y / SKY_H
        r = int(SKY_TOP[0] + (SKY_BOT[0]-SKY_TOP[0])*t)
        g = int(SKY_TOP[1] + (SKY_BOT[1]-SKY_TOP[1])*t)
        b = int(SKY_TOP[2] + (SKY_BOT[2]-SKY_TOP[2])*t)
        draw.line([0,y,W,y], fill=(r,g,b))

def draw_clouds(draw):
    # 雲のかたまり
    for (cx, cy, r) in [(180,30,28),(210,24,22),(240,32,20),
                         (580,22,32),(615,15,26),(648,28,24),
                         (950,35,20),(975,28,16),(1000,36,18),
                         (1150,18,24),(1175,12,20),(1200,20,18)]:
        draw.ellipse([cx-r, cy-r//2, cx+r, cy+r//2], fill=CLOUD)

def draw_buildings(draw):
    # 遠景ビル群（木の後ろに少し見える）
    specs = [
        (0,   80, 200, 150, 6, 3),
        (180, 70, 260, 145, 7, 3),
        (360, 55, 160, 140, 5, 4),
        (500, 65, 180, 148, 6, 3),
        (640, 60, 220, 145, 7, 4),
        (820, 50, 200, 148, 6, 3),
        (980, 45, 180, 150, 5, 3),
        (1120,50, 160, 148, 5, 3),
    ]
    for (bx, by, bw, bh, cols, rows) in specs:
        draw.rectangle([bx, by, bx+bw, bh], fill=BLDG)
        # 窓
        gw = bw // (cols+1)
        gh = (bh-by) // (rows+1)
        for r in range(rows):
            for c in range(cols):
                wx = bx + gw*(c+1) - 5
                wy = by + gh*(r+1) - 3
                draw.rectangle([wx,wy,wx+9,wy+6], fill=BLDG_WIN)
    # 右端の高い塔
    draw.rectangle([1210, 35, 1225, 148], fill=BLDG)
    draw.rectangle([1217, 25, 1218, 40], fill=BLDG)

def draw_trees(draw):
    draw.rectangle([0, TREE_Y, W, TREE_Y+TREE_H], fill=TREE_D)
    # 木のシルエット（丸い頭をたくさん並べる）
    step = 22
    for tx in range(-step, W+step, step):
        # 大きめの丸を重ねる
        for dx in [0, 10]:
            x = tx + dx
            draw.ellipse([x-16, TREE_Y-14, x+18, TREE_Y+28], fill=TREE)
            draw.ellipse([x- 8, TREE_Y-20, x+22, TREE_Y+16], fill=TREE_L)
            draw.ellipse([x+ 4, TREE_Y-12, x+26, TREE_Y+22], fill=TREE)

def draw_track(draw):
    # 外側芝（緑の帯）
    draw.rectangle([0, TREE_Y+TREE_H, W, RAIL1_Y], fill=GRASS_UP)

    # 外柵
    for i in range(3):
        draw.rectangle([0, RAIL1_Y+i*4, W, RAIL1_Y+i*4+3], fill=RAIL_C if i%2==0 else GRASS_UP)
    draw.rectangle([0, RAIL1_Y, W, RAIL1_Y+2], fill=WHITE)

    # メイン芝コース（ストライプ）
    draw.rectangle([0, TRACK_Y0, W, TRACK_Y1], fill=GRASS_M)
    stripe = 40
    for sx in range(0, W, stripe*2):
        draw.rectangle([sx, TRACK_Y0, sx+stripe, TRACK_Y1], fill=GRASS_S)
        draw.rectangle([sx+stripe, TRACK_Y0, sx+stripe*2, TRACK_Y1], fill=GRASS_SL)

    # 内柵
    draw.rectangle([0, RAIL2_Y, W, RAIL2_Y+4], fill=WHITE)

    # ダート（内側）
    draw.rectangle([0, RAIL2_Y+4, W, BROWN_Y+BROWN_H], fill=BROWN_D)
    # ダートのテクスチャ
    for dx in range(0, W, 14):
        draw.line([dx, RAIL2_Y+4, dx+7, BROWN_Y+BROWN_H], fill=(130,104,62), width=1)

def draw_pole(draw, label="4", px=820):
    # 赤白ストライプのポール
    ph = RAIL2_Y - RAIL1_Y - 4
    stripe_h = ph // 6
    for i in range(6):
        col = POLE_R if i%2==0 else (230,230,225)
        y0 = RAIL1_Y + 4 + i*stripe_h
        draw.rectangle([px, y0, px+12, y0+stripe_h], fill=col)
    # 丸
    r = 32
    cx = px + 6
    cy = RAIL1_Y - r + 4
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=WHITE)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(80,80,80), width=2)
    draw.text((cx-10, cy-14), label, fill=POLE_R, font=font(28))

# ── 馬スプライト ───────────────────────────────────────────────
def draw_horse(draw, cx, cy, jcol, frame=0, number=1):
    """cx,cy = 馬体中心。右向き。"""
    bw, bh = 56, 20   # 胴体半幅・半高

    # 影
    draw.ellipse([cx-bw, cy+bh+2, cx+bw, cy+bh+14], fill=SHADOW_G)

    # 尻尾
    tf = 16 if frame==0 else 8
    pts = [(cx-bw-5, cy-4), (cx-bw-18, cy+tf), (cx-bw-22, cy+tf+16)]
    draw.line(pts, fill=HDARK, width=8)
    draw.line(pts[-2:], fill=HDARK, width=5)

    # 後躯（やや楕円）
    draw.ellipse([cx-bw-10, cy-bh+4, cx-bw+22, cy+bh+2], fill=HBODY)

    # 胴体メイン
    draw.ellipse([cx-bw+8, cy-bh, cx+bw-8, cy+bh+2], fill=HBODY)
    # ハイライト
    draw.ellipse([cx-bw+18, cy-bh+2, cx+bw-18, cy-2], fill=HBODY_L)

    # 首
    neck_pts = [(cx+bw-18, cy-bh+4), (cx+bw+8, cy-bh-16),
                (cx+bw+18, cy-bh-10), (cx+bw+4, cy-2)]
    draw.polygon(neck_pts, fill=HBODY)

    # 頭
    draw.ellipse([cx+bw+2, cy-bh-22, cx+bw+38, cy-bh+2], fill=HBODY)
    # 鼻
    draw.ellipse([cx+bw+26, cy-bh-16, cx+bw+46, cy-bh+4], fill=HSNOUT)
    # 鼻孔
    draw.ellipse([cx+bw+36, cy-bh-10, cx+bw+42, cy-bh-4], fill=HBODY_D)
    # 目
    draw.ellipse([cx+bw+10, cy-bh-18, cx+bw+18, cy-bh-10], fill=(12,8,4))
    draw.point((cx+bw+12, cy-bh-16), fill=(200,200,200))
    # たてがみ
    for my in range(0, 28, 7):
        draw.ellipse([cx+bw-8+my//3, cy-bh-26+my,
                      cx+bw+4+my//3,  cy-bh-12+my], fill=HDARK)

    # 脚（4本）
    lw = 7
    if frame == 0:   # 伸展
        # 前脚
        draw.line([(cx+bw-20, cy+bh), (cx+bw-10, cy+bh+44)], fill=HLEG, width=lw)
        draw.line([(cx+bw-8,  cy+bh), (cx+bw-24, cy+bh+44)], fill=HLEG, width=lw)
        # 後脚
        draw.line([(cx-bw+22, cy+bh), (cx-bw+12, cy+bh+44)], fill=HLEG, width=lw)
        draw.line([(cx-bw+8,  cy+bh), (cx-bw+28, cy+bh+44)], fill=HLEG, width=lw)
    else:            # 収縮
        draw.line([(cx+bw-20, cy+bh), (cx+bw-14, cy+bh+44)], fill=HLEG, width=lw)
        draw.line([(cx+bw-8,  cy+bh), (cx+bw-14, cy+bh+44)], fill=HLEG, width=lw)
        draw.line([(cx-bw+22, cy+bh), (cx-bw+18, cy+bh+44)], fill=HLEG, width=lw)
        draw.line([(cx-bw+8,  cy+bh), (cx-bw+14, cy+bh+44)], fill=HLEG, width=lw)
    # 蹄
    for lx, off in [(cx+bw-17, 0),(cx+bw-16, 0),(cx-bw+15, 0),(cx-bw+18, 0)]:
        draw.rectangle([lx-4, cy+bh+40+off, lx+4, cy+bh+48+off], fill=HDARK)

    # 馬番布（saddle cloth）
    draw.ellipse([cx-10, cy-bh-4, cx+30, cy+4], fill=jcol)
    draw.text((cx+4, cy-bh), str(number), fill=WHITE, font=font(20))

    # 騎手ボディ
    draw.ellipse([cx+4, cy-bh-42, cx+38, cy-bh-2], fill=HJOCK)
    # 騎手帽
    draw.ellipse([cx+8, cy-bh-58, cx+40, cy-bh-36], fill=jcol)
    draw.rectangle([cx+4, cy-bh-48, cx+42, cy-bh-40], fill=jcol)   # つば
    # 腕
    draw.line([(cx+35, cy-bh-20), (cx+bw+30, cy-bh+2)], fill=HJOCK, width=6)

# ── UIパネル ───────────────────────────────────────────────────
def draw_race_info(draw, race_name="天皇賞（秋）", venue="中山10R", cls="GI　芝2000m"):
    # ボックス
    draw.rectangle([8, 8, 285, 115], fill=(8, 8, 10))
    draw.rectangle([8, 8, 285, 115], outline=(100,100,108), width=2)
    draw.text((18, 16), venue,     fill=WHITE,           font=font(26))
    draw.text((18, 48), race_name, fill=WHITE,           font=font(26))
    draw.text((18, 80), cls,       fill=WHITE,           font=font(26))

def draw_minimap(draw):
    # 外枠
    draw.rectangle([8, PANEL_Y+8, PANEL_L-8, H-8], fill=UI_BG2)
    draw.rectangle([8, PANEL_Y+8, PANEL_L-8, H-8], outline=UI_BD, width=2)

    # コース楕円（中山のイメージ）
    mx0, my0 = 28, PANEL_Y+24
    mx1, my1 = PANEL_L-24, H-62
    draw.ellipse([mx0, my0, mx1, my1], fill=MAP_G)
    draw.ellipse([mx0, my0, mx1, my1], outline=(165,215,165), width=3)
    # トラック（ベージュ帯）
    draw.ellipse([mx0+18, my0+14, mx1-18, my1-14], fill=MAP_L)
    draw.ellipse([mx0+18, my0+14, mx1-18, my1-14], outline=MAP_TRK, width=8)
    draw.ellipse([mx0+40, my0+30, mx1-40, my1-30], fill=MAP_G)

    # 現在位置ドット（4コーナー付近）
    mcx = (mx0+mx1)//2
    mcy = (my0+my1)//2
    rx = (mx1-mx0)//2 - 28
    ry = (my1-my0)//2 - 20
    angle = math.radians(100)
    dx = int(mcx + rx*math.cos(angle))
    dy = int(mcy + ry*math.sin(angle))
    draw.ellipse([dx-7, dy-7, dx+7, dy+7], fill=(70, 205, 210))   # 7番水色
    dx2 = int(mcx + rx*math.cos(math.radians(108)))
    dy2 = int(mcy + ry*math.sin(math.radians(108)))
    draw.ellipse([dx2-6, dy2-6, dx2+6, dy2+6], fill=(230, 190, 35))  # 5番黄

    # 下部テキスト
    draw.rectangle([8, H-56, PANEL_L-8, H-8], fill=UI_BG)
    draw.text((24, H-50), "中山　芝　良", fill=TEXT_W, font=font(26))

def draw_commentary(draw, lines):
    """lines = [(text_segments,), ...]  text_segments = [(text, color), ...]"""
    x0 = PANEL_L + 4
    draw.rectangle([PANEL_L, PANEL_Y, W-4, H-4], fill=UI_BG)
    draw.rectangle([PANEL_L, PANEL_Y, W-4, H-4], outline=UI_BD, width=2)
    y = PANEL_Y + 22
    fn = font(28)
    for segs in lines:
        x = x0 + 14
        for text, col in segs:
            draw.text((x, y), text, fill=col, font=fn)
            bb = draw.textbbox((x, y), text, font=fn)
            x = bb[2] + 4
        y += 46

# ── 組み立て ───────────────────────────────────────────────────
def build_frame():
    img = Image.new("RGB", (W, H), SKY_BOT)
    draw = ImageDraw.Draw(img)

    draw_sky(draw)
    draw_buildings(draw)
    draw_clouds(draw)
    draw_trees(draw)
    draw_track(draw)
    draw_pole(draw, label="4", px=820)

    # 馬を右から左の順（奥→手前: y が小さい方が奥）
    # 参考画像に合わせた7頭のレイアウト (cx, cy, jcol, frame, number)
    horse_specs = [
        # 先頭グループ（右寄り）
        (1080, 328, hcolor(1), 1, 1),
        ( 980, 334, hcolor(2), 0, 2),
        ( 900, 338, hcolor(7), 1, 7),   # 7番先頭付近
        # 中団
        ( 760, 342, hcolor(3), 0, 3),
        ( 650, 346, hcolor(5), 1, 5),
        # 後方
        ( 490, 350, hcolor(3), 0, 3),   # 別の3番（本来は別番号）
        ( 340, 354, hcolor(1), 1, 3),
    ]
    for (cx, cy, jcol, frm, num) in horse_specs:
        draw_horse(draw, cx, cy, jcol, frm, num)

    # UI
    draw.rectangle([0, PANEL_Y-8, W, H], fill=UI_BG)
    draw.rectangle([0, PANEL_Y-8, W, PANEL_Y], fill=(50,50,54))  # 区切り線帯
    draw_race_info(draw)
    draw_minimap(draw)

    commentary_lines = [
        [("先頭は ", TEXT_W), ("7番", hcolor(7)), ("メジロマックイーン", (100,200,220))],
        [("これを追って ", TEXT_W), ("5番", hcolor(3)), ("トウカイテイオー", (230,190,35))],
        [("さらに ", TEXT_W), ("3番", (220,100,200)), ("ナリタブライアン", (220,100,200))],
        [("4コーナーをカーブして", TEXT_W)],
        [("さあ　直線コースに入りました！", TEXT_W)],
    ]
    draw_commentary(draw, commentary_lines)

    return img

os.makedirs("output", exist_ok=True)
img = build_frame()
img.save(OUT)
print(f"Saved: {OUT}  ({W}x{H})")

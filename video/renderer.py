import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy import VideoClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoClip, concatenate_videoclips

W, H = 320, 180
SCALE = 4
OUT_W, OUT_H = W * SCALE, H * SCALE
FPS = 30

# 色定数
SKY       = (105, 175, 220)
CLOUD     = (248, 252, 255)
TREE      = (38, 118, 50)
TREE_DARK = (28, 90, 38)
BLDG      = (178, 178, 186)
BLDG_WIN  = (140, 148, 168)
GRASS_OUT = (54, 150, 70)
RAIL      = (228, 228, 228)
DIRT      = (170, 140, 96)
DIRT_DARK = (150, 120, 78)
GRASS     = (48, 138, 56)
GRASS_S   = (40, 124, 50)
GRASS_IN  = (44, 108, 50)
SHADOW    = (30, 108, 42)
HBODY     = (138, 88, 46)
HDARK     = (72, 42, 18)
HLEG      = (108, 66, 30)
HSNOUT    = (116, 70, 36)
UI_BG     = (18, 18, 20)
UI_BORDER = (72, 72, 78)
WHITE     = (255, 255, 255)
TEXT_W    = (232, 232, 232)
RED       = (220, 30, 30)
GREEN_T   = (0, 240, 80)
POLE_RED  = (210, 28, 28)
BG_COLOR  = (0, 0, 0)

HORSE_COLORS = [
    (230, 60,  60),
    (60,  130, 230),
    (240, 200, 40),
    (90,  220, 120),
    (200, 100, 230),
    (240, 140, 40),
    (80,  220, 220),
    (220, 220, 220),
]

# ─── レイアウト定数 ─────────────────────────────────────
TRACK_Y0 = 70   # 芝コース上端
TRACK_Y1 = 116  # 芝コース下端
PANEL_Y  = 118  # 下部パネル開始
PANEL_L  = 84   # 下部左パネル幅（コース図）

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_font_cache = {}


def get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    _font_cache[size] = font or ImageFont.load_default()
    return _font_cache[size]


def horse_color(number):
    return HORSE_COLORS[(number - 1) % len(HORSE_COLORS)]


def to_frame(img):
    return np.array(img.resize((OUT_W, OUT_H), Image.NEAREST))


# ─── 背景描画 ───────────────────────────────────────────

def draw_bg(draw, scroll):
    # 空
    draw.rectangle([0, 0, W, 22], fill=SKY)
    # 雲（ゆっくりスクロール）
    for cx_base in [int(scroll * 0.15) % W, (int(scroll * 0.15) + 110) % W,
                    (int(scroll * 0.15) + 220) % W]:
        for cx in [cx_base - W, cx_base]:
            draw.ellipse([cx,    3, cx+22,  12], fill=CLOUD)
            draw.ellipse([cx+8,  1, cx+30,  12], fill=CLOUD)
            draw.ellipse([cx+16, 4, cx+34,  12], fill=CLOUD)

    # ビル（遠景）
    bsc = int(scroll * 0.35) % 64
    for bx in range(-64, W + 64, 64):
        x = bx - bsc
        draw.rectangle([x,     8,  x+42, 28], fill=BLDG)
        draw.rectangle([x+48,  6,  x+80, 26], fill=BLDG)
        for r in range(3):
            for c in range(4):
                draw.rectangle([x+2+c*10, 9+r*6, x+9+c*10, 13+r*6], fill=BLDG_WIN)

    # 木（手前）
    draw.rectangle([0, 25, W, 42], fill=TREE_DARK)
    tsc = int(scroll * 0.65) % 26
    for tx in range(-26, W + 26, 26):
        for dx in range(0, 24, 12):
            cx = tx + dx - tsc
            draw.ellipse([cx-9, 16, cx+11, 38], fill=TREE)
            draw.ellipse([cx-3, 12, cx+14,  34], fill=TREE)

    # 外側芝
    draw.rectangle([0, 40, W, 54], fill=GRASS_OUT)
    # 外柵（白）
    draw.rectangle([0, 52, W, 55], fill=RAIL)
    # 外砂（ダート色）
    draw.rectangle([0, 55, W, 68], fill=DIRT)
    dsc = int(scroll * 2.2) % 14
    for dx in range(-14, W + 14, 14):
        draw.line([dx - dsc, 55, dx - dsc + 7, 68], fill=DIRT_DARK, width=1)
    # 内柵（白）
    draw.rectangle([0, 66, W, 69], fill=RAIL)

    # メイン芝コース
    draw.rectangle([0, TRACK_Y0, W, TRACK_Y1], fill=GRASS)
    gsc = int(scroll * 2.8) % 20
    for gx in range(-20, W + 20, 20):
        draw.rectangle([gx - gsc, TRACK_Y0, gx - gsc + 10, TRACK_Y1], fill=GRASS_S)

    # 内側境界
    draw.rectangle([0, TRACK_Y1,     W, TRACK_Y1 + 2], fill=RAIL)
    if TRACK_Y1 + 2 < PANEL_Y - 2:
        draw.rectangle([0, TRACK_Y1 + 2, W, PANEL_Y - 2], fill=GRASS_IN)


# ─── 距離ポール ──────────────────────────────────────────

def draw_pole(draw, seg_index):
    xs = [218, 152, 96, 50]
    if seg_index >= len(xs):
        return
    px = xs[seg_index]
    for y in range(55, 115, 8):
        col = POLE_RED if ((y - 55) // 8) % 2 == 0 else RAIL
        draw.rectangle([px, y, px + 3, min(y + 8, 115)], fill=col)
    nums = ["4", "3", "2", "1"]
    draw.ellipse([px - 8, 46, px + 12, 59], fill=WHITE)
    draw.ellipse([px - 8, 46, px + 12, 59], outline=(60, 60, 60))
    draw.text((px - 2, 47), nums[seg_index], fill=POLE_RED, font=get_font(10))


# ─── 馬+騎手スプライト ───────────────────────────────────

def draw_horse(draw, cx, cy, jcolor, frame, number):
    # 影
    draw.ellipse([cx - 13, cy + 11, cx + 11, cy + 15], fill=SHADOW)

    # 胴体
    draw.ellipse([cx - 12, cy - 3, cx + 10, cy + 7], fill=HBODY)
    # 後躯
    draw.ellipse([cx - 15, cy - 1, cx - 5, cy + 6], fill=HBODY)
    # 首
    draw.ellipse([cx + 7, cy - 7, cx + 14, cy + 2], fill=HBODY)
    # 頭
    draw.ellipse([cx + 11, cy - 8, cx + 19, cy - 2], fill=HBODY)
    # 鼻
    draw.ellipse([cx + 15, cy - 7, cx + 21, cy - 2], fill=HSNOUT)
    # 目
    draw.ellipse([cx + 13, cy - 8, cx + 16, cy - 5], fill=(10, 8, 6))
    # たてがみ
    draw.line([cx + 9, cy - 7, cx + 4, cy - 12], fill=HDARK, width=2)
    draw.line([cx + 4, cy - 12, cx + 0, cy - 11], fill=HDARK, width=1)
    # 尻尾
    tf = 3 if frame == 0 else 0
    draw.line([cx - 14, cy,      cx - 19, cy + 4 + tf], fill=HDARK, width=2)
    draw.line([cx - 19, cy + 4 + tf, cx - 22, cy + 9 + tf], fill=HDARK, width=2)

    # 脚（ギャロップ2フレーム）
    if frame == 0:
        draw.line([cx + 3,  cy + 7, cx + 8,  cy + 15], fill=HLEG, width=2)
        draw.line([cx + 6,  cy + 7, cx + 2,  cy + 15], fill=HLEG, width=2)
        draw.line([cx - 4,  cy + 7, cx - 2,  cy + 15], fill=HLEG, width=2)
        draw.line([cx - 7,  cy + 7, cx - 12, cy + 15], fill=HLEG, width=2)
    else:
        draw.line([cx + 3,  cy + 7, cx + 5,  cy + 15], fill=HLEG, width=2)
        draw.line([cx + 6,  cy + 7, cx + 4,  cy + 15], fill=HLEG, width=2)
        draw.line([cx - 4,  cy + 7, cx - 4,  cy + 15], fill=HLEG, width=2)
        draw.line([cx - 7,  cy + 7, cx - 6,  cy + 15], fill=HLEG, width=2)

    # 馬番布（saddle cloth）
    draw.rectangle([cx - 2, cy - 5, cx + 8, cy + 1], fill=jcolor)
    draw.text((cx + 1, cy - 5), str(number), fill=WHITE, font=get_font(6))

    # 騎手ボディ
    draw.rectangle([cx + 0, cy - 11, cx + 10, cy - 5], fill=(232, 232, 232))
    # 騎手帽
    draw.ellipse([cx + 1, cy - 15, cx + 12, cy - 9], fill=jcolor)
    # 手綱
    draw.line([cx + 16, cy - 4, cx + 4, cy + 0], fill=(210, 210, 210), width=1)


# ─── レース情報ボックス ──────────────────────────────────

def draw_race_info(draw, data):
    draw.rectangle([1, 1, 104, 42], fill=(6, 6, 10))
    draw.rectangle([1, 1, 104, 42], outline=(95, 95, 100))
    fn8 = get_font(8)
    draw.text((5,  4), data.get("race_name", ""), fill=WHITE, font=fn8)
    draw.text((5, 15), "GI　芝2200m",              fill=WHITE, font=fn8)
    draw.text((5, 26), data.get("race_date", ""),   fill=(190, 190, 205), font=fn8)


# ─── コース図（左下） ────────────────────────────────────

def draw_minimap(draw, data, seg_index, cur_order):
    draw.rectangle([0, PANEL_Y, PANEL_L - 1, H], fill=UI_BG)
    draw.rectangle([0, PANEL_Y, PANEL_L - 1, H], outline=UI_BORDER)

    # 楕円コース
    mx0, my0, mx1, my1 = 4, PANEL_Y + 4, PANEL_L - 5, H - 22
    draw.ellipse([mx0, my0, mx1, my1], fill=(38, 95, 44))
    draw.ellipse([mx0, my0, mx1, my1], outline=(180, 220, 180))
    draw.ellipse([mx0 + 9, my0 + 6, mx1 - 9, my1 - 6], fill=(22, 62, 28))

    # 現在位置ドット
    map_cx, map_cy = (mx0 + mx1) // 2, (my0 + my1) // 2
    r_x, r_y = (mx1 - mx0) // 2 - 4, (my1 - my0) // 2 - 3
    angles = [200, 340, 110, 290]
    import math
    angle = math.radians(angles[seg_index])
    dx = int(map_cx + r_x * math.cos(angle))
    dy = int(map_cy + r_y * math.sin(angle))
    draw.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], fill=horse_color(cur_order[0]))

    fn7 = get_font(7)
    draw.text((4, H - 19), data.get("race_name", "")[:5], fill=(200, 200, 200), font=fn7)
    draw.text((4, H - 11), "芝　良", fill=(160, 210, 170), font=fn7)


# ─── 実況テキストパネル（右下） ──────────────────────────

def generate_commentary(horses_by_number, initial_order, positions, segments):
    """セグメントごとの実況テキストを (text, color) のリストで返す。"""
    out = {}
    for i, seg in enumerate(segments):
        order = positions[seg]
        hn = [order[j] if j < len(order) else None for j in range(3)]
        horses = [horses_by_number[n] if n else None for n in hn]
        colors = [horse_color(n) if n else TEXT_W for n in hn]

        lines = []
        if horses[0]:
            lines.append([("先頭は ", TEXT_W), (f"{hn[0]}番", colors[0]), (horses[0]["name"], colors[0])])
        if horses[1]:
            lines.append([("これを追って ", TEXT_W), (f"{hn[1]}番", colors[1]), (horses[1]["name"], colors[1])])
        if horses[2]:
            lines.append([("さらに ", TEXT_W), (f"{hn[2]}番", colors[2]), (horses[2]["name"], colors[2])])

        phrases = {
            "400m": [("向こう正面を通過", TEXT_W)],
            "300m": [("第3コーナーへさしかかった！", TEXT_W)],
            "200m": [("4コーナーをカーブして", TEXT_W), ("さあ　直線コースに入りました！", TEXT_W)],
            "100m": [("4コーナーをカーブして", TEXT_W), ("さあ　直線コースに入りました！", TEXT_W)],
        }
        for phrase in phrases.get(seg, []):
            lines.append([phrase])

        out[seg] = lines
    return out


def draw_commentary_panel(draw, lines, progress):
    draw.rectangle([PANEL_L, PANEL_Y, W, H], fill=UI_BG)
    draw.rectangle([PANEL_L, PANEL_Y, W, H], outline=UI_BORDER)

    fn = get_font(9)
    total = sum(sum(len(t) for t, c in line) for line in lines)
    shown = int(total * min(progress / 0.65, 1.0))

    remaining = shown
    y = PANEL_Y + 5
    for line_segs in lines:
        if remaining <= 0:
            break
        x = PANEL_L + 4
        for text, color in line_segs:
            if remaining <= 0:
                break
            chunk = text[:min(len(text), remaining)]
            remaining -= len(chunk)
            draw.text((x, y), chunk, fill=color, font=fn)
            bbox = draw.textbbox((x, y), chunk, font=fn)
            x = bbox[2] + 1
        y += 11


# ─── シーン2: レース本体 ─────────────────────────────────

def make_scene2(data, positions, duration=40.0):
    horses = data["horses"]
    n = len(horses)
    hbn = {h["number"]: h for h in horses}
    nums_sorted = sorted(h["number"] for h in horses)
    lane_of = {number: i for i, number in enumerate(nums_sorted)}

    initial_order = [h["number"] for h in sorted(horses, key=lambda h: h["score"], reverse=True)]
    segments = ["400m", "300m", "200m", "100m"]
    seg_dur = duration / len(segments)
    commentary = generate_commentary(hbn, initial_order, positions, segments)

    BASE_CX = 165
    SPREAD = 50

    def smooth(p):
        p = max(0.0, min(1.0, p))
        return p * p * (3 - 2 * p)

    def make_frame(t):
        img = Image.new("RGB", (W, H), SKY)
        draw = ImageDraw.Draw(img)

        scroll = t * 55
        draw_bg(draw, scroll)

        seg_idx = min(int(t // seg_dur), len(segments) - 1)
        seg = segments[seg_idx]
        seg_prog = (t % seg_dur) / seg_dur
        ease = smooth(seg_prog)

        prev_order = initial_order if seg_idx == 0 else positions[segments[seg_idx - 1]]
        cur_order = positions[seg]

        gallop = int(t * 6) % 2

        draw_pole(draw, seg_idx)

        # 奥の馬を先に描画（z-order: 後ろのレーン=奥）
        for number in reversed(nums_sorted):
            prev_rank = prev_order.index(number)
            cur_rank  = cur_order.index(number)
            rank = prev_rank + (cur_rank - prev_rank) * ease

            x_off = (n - 1 - rank) * (SPREAD / max(n - 1, 1)) - SPREAD / 2
            cx = int(BASE_CX + x_off)

            lane = lane_of[number]
            cy = int(92 + (lane - (n - 1) / 2) * 9)

            draw_horse(draw, cx, cy, horse_color(number), gallop, number)

        draw_race_info(draw, data)
        draw_minimap(draw, data, seg_idx, cur_order)
        draw_commentary_panel(draw, commentary[seg], seg_prog)

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ─── シーン1: ブート画面（3秒） ──────────────────────────

def make_scene1(race_name, duration=3.0):
    lines = [
        "FUTURE VISION SYSTEM v2.3",
        "BOOTING...",
        f"TARGET COORDINATES: {race_name}",
        "TEMPORAL JUMP INITIATED",
    ]
    total = sum(len(l) for l in lines)
    fn = get_font(10)

    def make_frame(t):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        shown = int(total * min(t / duration, 1.0))
        rem = shown
        y = 30
        for line in lines:
            if rem <= 0:
                break
            chunk = line[:min(len(line), rem)]
            rem -= len(chunk)
            draw.text((10, y), chunk, fill=GREEN_T, font=fn)
            y += 18
        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ─── シーン3: SIGNAL LOST（7秒） ────────────────────────

def make_scene3(duration=7.0):
    fn_big  = get_font(20)
    fn_small = get_font(10)

    def make_frame(t):
        noise = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        draw = ImageDraw.Draw(img)
        if int(t * 4) % 2 == 0:
            text = "SIGNAL LOST"
            bb = draw.textbbox((0, 0), text, font=fn_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text(((W - tw) / 2, (H - th) / 2 - 10), text, fill=RED, font=fn_big)
        sub = "STAMINA CRITICAL - OBSERVATION TERMINATED"
        bb2 = draw.textbbox((0, 0), sub, font=fn_small)
        draw.text(((W - (bb2[2] - bb2[0])) / 2, H - 28), sub, fill=RED, font=fn_small)
        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ─── シーン4: 最終予想（10秒） ───────────────────────────

def make_scene4(data, duration=10.0):
    top3 = sorted(data["horses"], key=lambda h: h["score"], reverse=True)[:3]
    bets = data.get("recommended_bets", [])
    fn_title = get_font(14)
    fn_body  = get_font(10)

    def make_frame(t):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.text((18, 10), "AI ANALYSIS RESULT", fill=GREEN_T, font=fn_title)
        y = 34
        for i, h in enumerate(top3):
            txt = f"{i + 1}. {h['number']}番 {h['name']}  score:{h['score']}"
            draw.text((18, y), txt, fill=horse_color(h["number"]), font=fn_body)
            y += 14
        y += 6
        draw.text((18, y), "── 推奨買い目 ──", fill=GREEN_T, font=fn_body)
        y += 12
        for bet in bets:
            draw.text((18, y), f"  {bet}", fill=TEXT_W, font=fn_body)
            y += 12
        if t > duration - 3:
            text = "GOOD LUCK"
            bb = draw.textbbox((0, 0), text, font=fn_title)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text(((W - tw) / 2, H - 28), text, fill=GREEN_T, font=fn_title)
        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ─── エントリーポイント ──────────────────────────────────

def render_video(data, positions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clip1 = make_scene1(data["race_name"])
    clip2 = make_scene2(data, positions)
    clip3 = make_scene3()
    clip4 = make_scene4(data)
    final = concatenate_videoclips([clip1, clip2, clip3, clip4])
    filename = f"race_{data['race_date']}_{data['race_name']}.mp4"
    path = os.path.join(output_dir, filename)
    final.write_videofile(path, fps=FPS, codec="libx264", audio=False)
    return path

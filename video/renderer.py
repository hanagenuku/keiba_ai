import os
import math

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

# ── 色定数 ─────────────────────────────────────────────────
SKY        = (108, 176, 222)
CLOUD      = (248, 252, 255)
TREE       = (35, 115, 46)
TREE_DARK  = (25, 85, 35)
BLDG       = (175, 175, 182)
BLDG_WIN   = (138, 145, 165)
GRASS_OUT  = (54, 152, 68)
RAIL       = (225, 225, 225)
GRASS      = (46, 138, 54)
GRASS_S    = (38, 122, 48)
GRASS_IN   = (42, 108, 50)
SHADOW_C   = (28, 106, 40)
HBODY      = (132, 84, 42)
HDARK      = (68, 38, 14)
HLEG       = (104, 62, 28)
HSNOUT     = (110, 66, 32)
HJOCK_W    = (230, 230, 230)
UI_BG      = (16, 16, 18)
UI_BORDER  = (70, 70, 76)
WHITE      = (255, 255, 255)
TEXT_W     = (230, 230, 230)
RED_C      = (215, 28, 28)
GREEN_T    = (0, 235, 75)
BG_BLACK   = (0, 0, 0)
POLE_RED   = (210, 26, 26)

HORSE_COLORS = [
    (230, 55,  55),   # 1: 赤
    (55,  120, 225),  # 2: 青
    (235, 195, 35),   # 3: 黄
    (85,  215, 110),  # 4: 緑
    (195, 90,  225),  # 5: 紫
    (235, 135, 35),   # 6: オレンジ
    (75,  215, 215),  # 7: 水色
    (215, 215, 215),  # 8: 白
]

# ── レイアウト ──────────────────────────────────────────────
SKY_H    = 28   # 空の高さ
TREE_Y   = 24   # 木の上端
RAIL1_Y  = 55   # 外柵 y
TRACK_Y0 = 57   # 芝コース上端
TRACK_Y1 = 112  # 芝コース下端
RAIL2_Y  = 113  # 内柵 y
PANEL_Y  = 116  # 下部パネル開始
PANEL_L  = 86   # 左パネル幅

HORSE_CY = 87   # 馬体中心 y (固定)

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
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
    _font_cache[size] = font or ImageFont.load_default()
    return _font_cache[size]

def horse_color(number):
    return HORSE_COLORS[(number - 1) % len(HORSE_COLORS)]

def to_frame(img):
    return np.array(img.resize((OUT_W, OUT_H), Image.NEAREST))


# ── 背景 ────────────────────────────────────────────────────

def draw_bg(draw, scroll):
    # 空
    for y in range(SKY_H):
        r = int(108 + (130 - 108) * y / SKY_H)
        g = int(176 + (196 - 176) * y / SKY_H)
        b = int(222 + (230 - 222) * y / SKY_H)
        draw.line([0, y, W, y], fill=(r, g, b))

    # 雲
    csc = int(scroll * 0.12) % (W + 60)
    for ox in [csc - W - 30, csc - 30, csc + W // 2 - 30]:
        for cx in [ox, ox + W + 60]:
            draw.ellipse([cx,    3, cx+26, 12], fill=CLOUD)
            draw.ellipse([cx+8,  1, cx+32, 13], fill=CLOUD)
            draw.ellipse([cx+16, 4, cx+36, 12], fill=CLOUD)

    # ビル（遠景）
    bsc = int(scroll * 0.3) % 72
    for bx in range(-72, W + 72, 72):
        x = bx - bsc
        draw.rectangle([x,    7, x + 46, 28], fill=BLDG)
        draw.rectangle([x+50, 5, x + 84, 26], fill=BLDG)
        for rr in range(2):
            for cc in range(5):
                draw.rectangle([x + 2 + cc * 9, 9 + rr * 7,
                                 x + 8 + cc * 9, 13 + rr * 7], fill=BLDG_WIN)

    # 木
    draw.rectangle([0, TREE_Y, W, TREE_Y + 18], fill=TREE_DARK)
    tsc = int(scroll * 0.6) % 28
    for tx in range(-28, W + 28, 28):
        for ddx in [0, 14]:
            cx = tx + ddx - tsc
            draw.ellipse([cx - 10, TREE_Y - 8,  cx + 12, TREE_Y + 14], fill=TREE)
            draw.ellipse([cx -  4, TREE_Y - 12, cx + 16, TREE_Y +  8], fill=TREE)

    # 外側芝
    draw.rectangle([0, TREE_Y + 18, W, RAIL1_Y], fill=GRASS_OUT)
    # 外柵
    draw.rectangle([0, RAIL1_Y, W, RAIL1_Y + 3], fill=RAIL)

    # 芝コース本体
    draw.rectangle([0, TRACK_Y0, W, TRACK_Y1], fill=GRASS)
    gsc = int(scroll * 2.5) % 22
    for gx in range(-22, W + 22, 22):
        draw.rectangle([gx - gsc, TRACK_Y0, gx - gsc + 11, TRACK_Y1], fill=GRASS_S)

    # 内柵
    draw.rectangle([0, RAIL2_Y, W, RAIL2_Y + 3], fill=RAIL)
    # 内側芝（薄め）
    draw.rectangle([0, RAIL2_Y + 3, W, PANEL_Y], fill=GRASS_IN)


# ── 距離/コーナーポール ──────────────────────────────────────

def draw_pole(draw, phase_idx, phases):
    label = phases[phase_idx]["pole_label"]
    if label is None:
        return
    px = phases[phase_idx]["pole_x"]
    # 赤白ストライプ
    for y in range(RAIL1_Y + 3, RAIL2_Y, 8):
        col = POLE_RED if ((y - RAIL1_Y) // 8) % 2 == 0 else RAIL
        draw.rectangle([px, y, px + 4, min(y + 8, RAIL2_Y)], fill=col)
    # 丸と数字
    draw.ellipse([px - 9, RAIL1_Y - 12, px + 14, RAIL1_Y + 3], fill=WHITE)
    draw.ellipse([px - 9, RAIL1_Y - 12, px + 14, RAIL1_Y + 3], outline=(50, 50, 50))
    font = get_font(9)
    draw.text((px - 3, RAIL1_Y - 10), label, fill=POLE_RED, font=font)


# ── 馬+騎手スプライト ────────────────────────────────────────

def draw_horse(draw, cx, cy, jcolor, frame, number):
    """cx,cy = 馬体中心。全馬を右向きで描画。"""

    # 影
    draw.ellipse([cx - 16, cy + 12, cx + 14, cy + 16], fill=SHADOW_C)

    # 尻尾
    tf = 4 if frame == 0 else 1
    draw.line([cx - 16, cy - 1, cx - 22, cy + 4 + tf], fill=HDARK, width=3)
    draw.line([cx - 22, cy + 4 + tf, cx - 25, cy + 10 + tf], fill=HDARK, width=2)

    # 胴体（メイン楕円 × 2）
    draw.ellipse([cx - 15, cy - 5, cx + 12, cy + 8], fill=HBODY)   # 胴体
    draw.ellipse([cx - 18, cy - 3, cx - 8,  cy + 7], fill=HBODY)   # 後躯
    draw.ellipse([cx +  8, cy - 8, cx + 17, cy + 3], fill=HBODY)   # 首前

    # 頭
    draw.ellipse([cx + 13, cy - 10, cx + 23, cy - 3], fill=HBODY)
    # 鼻
    draw.ellipse([cx + 18, cy -  9, cx + 26, cy - 4], fill=HSNOUT)
    # 目
    draw.ellipse([cx + 15, cy - 10, cx + 18, cy - 7], fill=(8, 6, 4))

    # たてがみ
    draw.line([cx + 10, cy - 8, cx + 5, cy - 13], fill=HDARK, width=2)
    draw.line([cx +  5, cy - 13, cx + 0, cy - 12], fill=HDARK, width=2)

    # 脚（4本 ギャロップ2フレーム）
    if frame == 0:  # 脚を伸ばした局面
        draw.line([cx + 4,  cy + 8, cx + 9,  cy + 18], fill=HLEG, width=3)
        draw.line([cx + 7,  cy + 8, cx + 2,  cy + 18], fill=HLEG, width=3)
        draw.line([cx - 5,  cy + 8, cx - 3,  cy + 18], fill=HLEG, width=3)
        draw.line([cx - 8,  cy + 8, cx - 14, cy + 18], fill=HLEG, width=3)
    else:           # 脚を縮めた局面
        draw.line([cx + 4,  cy + 8, cx + 6,  cy + 18], fill=HLEG, width=3)
        draw.line([cx + 7,  cy + 8, cx + 5,  cy + 18], fill=HLEG, width=3)
        draw.line([cx - 5,  cy + 8, cx - 5,  cy + 18], fill=HLEG, width=3)
        draw.line([cx - 8,  cy + 8, cx - 7,  cy + 18], fill=HLEG, width=3)

    # 馬番布
    draw.rectangle([cx - 3, cy - 6, cx + 10, cy + 1], fill=jcolor)
    draw.text((cx + 0, cy - 6), str(number), fill=WHITE, font=get_font(7))

    # 騎手ボディ
    draw.rectangle([cx + 0, cy - 13, cx + 12, cy - 7], fill=HJOCK_W)
    # 騎手帽
    draw.ellipse([cx + 1, cy - 18, cx + 13, cy - 11], fill=jcolor)
    # 手綱
    draw.line([cx + 20, cy - 5, cx + 5, cy + 0], fill=(200, 200, 200), width=1)


# ── レース情報ボックス ───────────────────────────────────────

def draw_race_info(draw, data):
    box_w = 100
    draw.rectangle([1, 1, box_w, 44], fill=(5, 5, 8))
    draw.rectangle([1, 1, box_w, 44], outline=(90, 90, 96))
    fn = get_font(8)
    draw.text((5,  4), data.get("race_name", ""),   fill=WHITE,          font=fn)
    draw.text((5, 15), "GI　芝2200m",               fill=WHITE,          font=fn)
    draw.text((5, 26), data.get("race_date", ""),   fill=(185, 185, 200), font=fn)


# ── コース図（左下） ─────────────────────────────────────────

def draw_minimap(draw, data, phase_idx, cur_order, phases):
    draw.rectangle([0, PANEL_Y, PANEL_L - 1, H], fill=UI_BG)
    draw.rectangle([0, PANEL_Y, PANEL_L - 1, H], outline=UI_BORDER)

    mx0, my0, mx1, my1 = 3, PANEL_Y + 3, PANEL_L - 5, H - 22
    draw.ellipse([mx0, my0, mx1, my1], fill=(36, 92, 42))
    draw.ellipse([mx0, my0, mx1, my1], outline=(160, 215, 165))
    draw.ellipse([mx0 + 9, my0 + 7, mx1 - 9, my1 - 7], fill=(20, 60, 26))

    # 現在位置
    map_cx = (mx0 + mx1) // 2
    map_cy = (my0 + my1) // 2
    rx = (mx1 - mx0) // 2 - 5
    ry = (my1 - my0) // 2 - 4
    angle = math.radians(phases[phase_idx]["map_angle"])
    dx = int(map_cx + rx * math.cos(angle))
    dy = int(map_cy + ry * math.sin(angle))
    draw.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], fill=horse_color(cur_order[0]))

    fn7 = get_font(7)
    draw.text((4, H - 19), data.get("race_name", "")[:5], fill=(195, 195, 195), font=fn7)
    draw.text((4, H - 11), "芝　良",                       fill=(155, 208, 165), font=fn7)


# ── 実況テキストパネル（右下） ────────────────────────────────

def make_commentary(horses_by_number, order, phase_key):
    """phase_key ごとの実況テキスト（colored segments）を返す。"""
    def hp(idx):
        n = order[idx] if idx < len(order) else None
        if n is None:
            return None, None, None
        return n, horses_by_number[n]["name"], horse_color(n)

    n1, nm1, c1 = hp(0)
    n2, nm2, c2 = hp(1)
    n3, nm3, c3 = hp(2)

    lines = []
    if n1:
        lines.append([("先頭は ", TEXT_W), (f"{n1}番", c1), (nm1, c1)])
    if n2:
        lines.append([("これを追って ", TEXT_W), (f"{n2}番", c2), (nm2, c2)])
    if n3:
        lines.append([("さらに ", TEXT_W), (f"{n3}番", c3), (nm3, c3)])

    extra = {
        "start":   [("スタート！各馬が飛び出した！", TEXT_W)],
        "corner1": [("第1コーナーへ差しかかった", TEXT_W)],
        "back":    [("向こう正面に入りました", TEXT_W)],
        "corner3": [("第3コーナーをカーブして", TEXT_W)],
        "corner4": [("4コーナーをカーブして", TEXT_W),
                    ("さあ　直線コースに入りました！", TEXT_W)],
        "final":   [("4コーナーをカーブして", TEXT_W),
                    ("さあ　直線コースに入りました！", TEXT_W)],
    }
    for seg in extra.get(phase_key, []):
        lines.append([seg])

    return lines


def draw_commentary_panel(draw, lines, progress):
    draw.rectangle([PANEL_L, PANEL_Y, W, H], fill=UI_BG)
    draw.rectangle([PANEL_L, PANEL_Y, W, H], outline=UI_BORDER)

    fn = get_font(9)
    total = sum(sum(len(t) for t, c in ln) for ln in lines)
    shown = int(total * min(progress / 0.55, 1.0))

    remaining = shown
    y = PANEL_Y + 5
    for ln in lines:
        if remaining <= 0:
            break
        x = PANEL_L + 4
        for text, color in ln:
            if remaining <= 0:
                break
            chunk = text[:min(len(text), remaining)]
            remaining -= len(chunk)
            draw.text((x, y), chunk, fill=color, font=fn)
            bbox = draw.textbbox((x, y), chunk, font=fn)
            x = bbox[2] + 1
        y += 11


# ── シーン2: レース本体（40秒） ──────────────────────────────

RACE_PHASES = [
    # name       key        dur  pole_label  pole_x  map_angle  pos_from  pos_to
    ("スタート",   "start",   4,   None,       0,      220,       "init",   "400m"),
    ("第1コーナー","corner1",  6,   None,       0,      285,       "400m",   "400m"),
    ("向こう正面", "back",     8,   "3",        200,    340,       "400m",   "300m"),
    ("第3コーナー","corner3",  7,   None,       0,      40,        "300m",   "200m"),
    ("第4コーナー","corner4",  7,   "2",        130,    100,       "200m",   "100m"),
    ("直線",      "final",    8,   "1",        55,     160,       "100m",   "100m"),
]
# dur合計 = 4+6+8+7+7+8 = 40s


def make_scene2(data, positions, duration=40.0):
    horses = data["horses"]
    n = len(horses)
    hbn = {h["number"]: h for h in horses}
    nums_sorted = sorted(h["number"] for h in horses)
    lane_of = {number: i for i, number in enumerate(nums_sorted)}

    initial_order = [h["number"] for h in sorted(horses, key=lambda h: h["score"], reverse=True)]
    pos_map = {
        "init": initial_order,
        "400m": positions["400m"],
        "300m": positions["300m"],
        "200m": positions["200m"],
        "100m": positions["100m"],
    }

    # 各フェーズの開始時刻を事前計算
    phase_starts = []
    t_acc = 0.0
    for ph in RACE_PHASES:
        phase_starts.append(t_acc)
        t_acc += ph[2]

    # フェーズをdictとして再整理
    phases = []
    for ph in RACE_PHASES:
        phases.append({
            "name":      ph[0],
            "key":       ph[1],
            "dur":       ph[2],
            "pole_label":ph[3],
            "pole_x":    ph[4],
            "map_angle": ph[5],
            "pos_from":  pos_map[ph[6]],
            "pos_to":    pos_map[ph[7]],
        })

    BASE_CX = 160
    SPREAD  = 55

    def smooth(p):
        p = max(0.0, min(1.0, p))
        return p * p * (3 - 2 * p)

    def get_rank_at(order, number):
        try:
            return float(order.index(number))
        except ValueError:
            return float(n - 1)

    def make_frame(t):
        # どのフェーズか特定
        phase_idx = 0
        for i in range(len(phases) - 1, -1, -1):
            if t >= phase_starts[i]:
                phase_idx = i
                break
        ph = phases[phase_idx]
        phase_elapsed = t - phase_starts[phase_idx]
        phase_prog = min(phase_elapsed / ph["dur"], 1.0)
        ease = smooth(phase_prog)

        prev_order = ph["pos_from"]
        cur_order  = ph["pos_to"]

        img = Image.new("RGB", (W, H), SKY)
        draw = ImageDraw.Draw(img)

        draw_bg(draw, t * 55)
        draw_pole(draw, phase_idx, phases)

        gallop = int(t * 7) % 2

        # 奥から手前の順（lane 大→小）に描画
        for number in reversed(nums_sorted):
            prev_rank = get_rank_at(prev_order, number)
            cur_rank  = get_rank_at(cur_order,  number)
            rank = prev_rank + (cur_rank - prev_rank) * ease

            x_off = (n - 1 - rank) * (SPREAD / max(n - 1, 1)) - SPREAD / 2
            cx = int(BASE_CX + x_off)

            lane = lane_of[number]
            cy = int(HORSE_CY + (lane - (n - 1) / 2) * 10)

            draw_horse(draw, cx, cy, horse_color(number), gallop, number)

        draw_race_info(draw, data)
        draw_minimap(draw, data, phase_idx, cur_order, phases)

        com = make_commentary(hbn, cur_order, ph["key"])
        draw_commentary_panel(draw, com, phase_prog)

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ── シーン1: ブート画面（3秒） ──────────────────────────────

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
        img = Image.new("RGB", (W, H), BG_BLACK)
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


# ── シーン3: SIGNAL LOST（7秒） ─────────────────────────────

def make_scene3(duration=7.0):
    fn_big   = get_font(20)
    fn_small = get_font(10)

    def make_frame(t):
        noise = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        draw = ImageDraw.Draw(img)
        if int(t * 4) % 2 == 0:
            text = "SIGNAL LOST"
            bb = draw.textbbox((0, 0), text, font=fn_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text(((W - tw) / 2, (H - th) / 2 - 10), text, fill=(220, 30, 30), font=fn_big)
        sub = "STAMINA CRITICAL - OBSERVATION TERMINATED"
        bb2 = draw.textbbox((0, 0), sub, font=fn_small)
        draw.text(((W - (bb2[2] - bb2[0])) / 2, H - 28), sub, fill=(220, 30, 30), font=fn_small)
        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ── シーン4: 最終予想（10秒） ────────────────────────────────

def make_scene4(data, duration=10.0):
    top3 = sorted(data["horses"], key=lambda h: h["score"], reverse=True)[:3]
    bets = data.get("recommended_bets", [])
    fn_title = get_font(14)
    fn_body  = get_font(10)

    def make_frame(t):
        img = Image.new("RGB", (W, H), BG_BLACK)
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


# ── エントリーポイント ───────────────────────────────────────

def render_video(data, positions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    clips = [
        make_scene1(data["race_name"]),
        make_scene2(data, positions),
        make_scene3(),
        make_scene4(data),
    ]
    final = concatenate_videoclips(clips)
    filename = f"race_{data['race_date']}_{data['race_name']}.mp4"
    path = os.path.join(output_dir, filename)
    final.write_videofile(path, fps=FPS, codec="libx264", audio=False)
    return path

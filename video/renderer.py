from PIL import Image, ImageDraw, ImageFont
import numpy as np

IW, IH = 270, 480
RACE_Y, RACE_H = 70, 300
PANEL_Y = 370

COAT_COLORS = {
    "栗毛":   {"body": (160, 100, 50),  "mane": (100, 60,  20),  "leg": (90,  55, 20)},
    "鹿毛":   {"body": (120, 70,  30),  "mane": (60,  35,  10),  "leg": (70,  40, 15)},
    "黒鹿毛": {"body": (60,  35,  15),  "mane": (25,  12,  5),   "leg": (40,  20, 8)},
    "芦毛":   {"body": (200, 195, 185), "mane": (220, 215, 210), "leg": (180, 175, 170)},
    "白毛":   {"body": (240, 235, 228), "mane": (255, 250, 245), "leg": (220, 215, 208)},
    "青鹿毛": {"body": (40,  25,  12),  "mane": (20,  10,  5),   "leg": (30,  15, 8)},
}

COAT_ORDER = ["栗毛", "黒鹿毛", "鹿毛", "芦毛", "栗毛", "青鹿毛", "白毛",
              "鹿毛", "栗毛", "黒鹿毛", "芦毛", "鹿毛", "栗毛", "黒鹿毛",
              "芦毛", "栗毛", "鹿毛", "黒鹿毛"]


def get_coat(horse_number: int) -> dict:
    idx = (horse_number - 1) % len(COAT_ORDER)
    return COAT_COLORS[COAT_ORDER[idx]]


def _font(size=10):
    candidates = [
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            from pathlib import Path
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


_font_cache: dict = {}


def _f(size):
    if size not in _font_cache:
        _font_cache[size] = _font(size)
    return _font_cache[size]


def render_frame(
    horses: list,
    bg_scroll: float,
    commentary: list[str],
    dist_remaining: float,
    tick: int,
    tsukkomi: str = "",
    race_info: dict = None,
) -> np.ndarray:
    img = Image.new("RGB", (IW, IH), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    _draw_top_panel(draw, dist_remaining, race_info)
    _draw_race_bg(draw, bg_scroll, RACE_Y, RACE_H)

    draw_order = sorted(horses, key=lambda h: h.screen_x)
    for i, horse in enumerate(draw_order):
        hy = RACE_Y + 220 - i * 2
        _draw_horse(draw, horse, hy, tick)

    _draw_bottom_panel(draw, horses, commentary, PANEL_Y)
    draw_tsukkomi(draw, tsukkomi, IH)

    big = img.resize((1080, 1920), Image.NEAREST)
    return np.array(big)


def _draw_top_panel(draw, dist: float, race_info: dict = None):
    draw.rectangle([0, 0, IW, 70], fill=(20, 20, 20))

    race_name = race_info.get("race_name", "レース") if race_info else "レース"
    distance  = race_info.get("distance", 2000)     if race_info else 2000
    venue     = race_info.get("venue", "")          if race_info else ""
    surface   = race_info.get("surface", "芝")      if race_info else "芝"

    draw.text((8, 6),  f"{race_name} G1", font=_f(12), fill=(255, 255, 255))
    draw.text((8, 22), f"{venue} {surface}{distance}m", font=_f(10), fill=(180, 180, 180))

    pct = min(1.0, max(0.0, (2000 - dist) / 1900))
    draw.rectangle([8, 46, 262, 54], fill=(40, 40, 40))
    draw.rectangle([8, 46, int(8 + 254 * pct), 54], fill=(0, 200, 80))
    draw.rectangle([8, 46, 262, 54], outline=(80, 80, 80))
    draw.text((8, 57), f"残り {int(dist)}m", font=_f(9), fill=(200, 200, 200))


def _draw_race_bg(draw, bg_scroll: float, ry: int, rh: int):
    for y in range(30):
        t = y / 30
        r = int(80 + t * 30); g = int(140 + t * 40); b = int(200 + t * 35)
        draw.line([0, ry + y, IW, ry + y], fill=(r, g, b))

    draw.rectangle([0, ry + 28, IW, ry + 60], fill=(170, 160, 150))
    for x in range(0, IW + 20, 18):
        ox = int((x - bg_scroll * 0.18) % IW)
        draw.rectangle([ox, ry + 30, ox + 8, ry + 58], fill=(190, 180, 170))
        draw.rectangle([ox + 2, ry + 33, ox + 5, ry + 38], fill=(80, 70, 65))
        draw.rectangle([ox + 2, ry + 42, ox + 5, ry + 47], fill=(80, 70, 65))

    for x in range(0, IW + 8, 6):
        tx = int((x - bg_scroll * 0.55) % IW)
        draw.rectangle([tx,     ry + 57, tx + 4, ry + 68], fill=(26,  92, 26))
        draw.rectangle([tx + 1, ry + 50, tx + 3, ry + 60], fill=(42, 122, 42))

    stripe_top = ry + 68
    for i in range(10):
        c = (56, 136, 58) if i % 2 == 0 else (44, 110, 46)
        y0 = stripe_top + i * (rh - 68) // 10
        y1 = stripe_top + (i + 1) * (rh - 68) // 10
        draw.rectangle([0, y0, IW, y1], fill=c)

    draw.rectangle([0, ry + rh - 22, IW, ry + rh - 20], fill=(220, 220, 220))

    px = int((200 - bg_scroll * 0.65) % IW)
    draw.rectangle([px, ry + 45, px + 3, ry + rh - 22], fill=(200, 17, 17))
    draw.ellipse([px - 8, ry + 38, px + 11, ry + 56], fill=(200, 17, 17))
    draw.text((px + 1, ry + 42), "4", font=_f(8), fill=(255, 255, 255))


def _draw_horse(draw, horse, hy: int, tick: int):
    hx = int(horse.screen_x)
    if hx < -40 or hx > IW + 40:
        return

    coat = get_coat(horse.number)
    body_color = coat["body"]
    mane_color = coat["mane"]
    leg_color  = coat["leg"]

    gf = tick // 3 % 4
    bob = [-1, 0, -1, -2][gf]
    by = hy + bob

    draw.rectangle([hx - 6, by - 7,  hx + 14, by - 2], fill=body_color)
    draw.rectangle([hx + 9,  by - 11, hx + 14, by - 4], fill=body_color)
    draw.rectangle([hx + 12, by - 13, hx + 18, by - 9], fill=body_color)
    draw.rectangle([hx + 15, by - 11, hx + 19, by - 8], fill=body_color)

    draw.rectangle([hx + 9,  by - 12, hx + 12, by - 8], fill=mane_color)
    draw.rectangle([hx - 8,  by - 5,  hx - 5,  by + 2], fill=mane_color)

    leg_frames = [
        [(0, 6), (4, 6), (9, 6),  (12, 6)],
        [(-1, 7),(3, 6), (9, 7),  (13, 5)],
        [(1, 5), (5, 7), (8, 5),  (12, 7)],
        [(0, 7), (4, 5), (9, 7),  (12, 5)],
    ]
    for lx, lh in leg_frames[gf]:
        draw.rectangle([hx - 6 + lx, by,      hx - 6 + lx + 2, by + lh],     fill=leg_color)
        draw.rectangle([hx - 6 + lx, by + lh, hx - 6 + lx + 2, by + lh + 2], fill=(20, 10, 5))

    jc = horse.jockey_color
    try:
        jc_rgb = tuple(int(jc.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        jc_rgb = (128, 128, 128)
    jc_dark = tuple(max(0, c - 40) for c in jc_rgb)

    draw.rectangle([hx,     by - 12, hx + 9,  by - 7],  fill=jc_rgb)
    draw.rectangle([hx + 1, by - 7,  hx + 8,  by - 4],  fill=(240, 240, 240))
    draw.rectangle([hx + 2, by - 15, hx + 8,  by - 12], fill=(240, 192, 144))
    draw.rectangle([hx + 1, by - 18, hx + 9,  by - 14], fill=jc_rgb)
    draw.rectangle([hx,     by - 14, hx + 2,  by - 12], fill=jc_dark)

    draw.rectangle([hx - 4, by - 6, hx + 2, by - 2], fill=(0, 34, 153))
    try:
        draw.text((hx - 3, by - 6), str(horse.number), font=_f(6), fill=(255, 255, 255))
    except Exception:
        pass


def _draw_bottom_panel(draw, horses: list, commentary: list[str], py: int):
    draw.rectangle([0, py, IW, IH], fill=(20, 20, 30))

    for i, line in enumerate(commentary[:3]):
        draw.text((8, py + 6 + i * 16), line, font=_f(10), fill=(255, 255, 200))

    top3 = sorted(horses, key=lambda h: -h.screen_x)[:3]
    for i, h in enumerate(top3):
        color = [(255, 215, 0), (200, 200, 200), (180, 120, 60)][i]
        draw.text((8 + i * 88, py + 58), f"{i+1}位 {h.number}番", font=_f(9), fill=color)
        draw.text((8 + i * 88, py + 72), h.name[:5],               font=_f(9), fill=(220, 220, 220))


def draw_tsukkomi(draw, text: str, frame_h: int):
    if not text:
        return
    box_y = frame_h - 50
    draw.rectangle([IW - 140, box_y,     IW - 4,  box_y + 36], fill=(0, 0, 0))
    draw.rectangle([IW - 142, box_y - 2, IW - 2,  box_y + 38], outline=(255, 255, 255))
    draw.text((IW - 136, box_y + 2),  "俺",  font=_f(9),  fill=(180, 220, 255))
    draw.text((IW - 136, box_y + 16), text, font=_f(9), fill=(255, 255, 255))


def render_title_card(race_info: dict, total_horses: int) -> list[np.ndarray]:
    frames = []
    for tick in range(150):
        img = Image.new("RGB", (IW, IH), (5, 5, 15))
        draw = ImageDraw.Draw(img)

        for y in range(0, IH, 3):
            v = int(20 + 10 * np.sin(y * 0.05 + tick * 0.1))
            draw.line([0, y, IW, y], fill=(0, 0, v))

        blink = tick % 20 < 15
        race_name = race_info.get("race_name", "レース")
        venue     = race_info.get("venue", "")
        race_num  = race_info.get("race_num", "")
        distance  = race_info.get("distance", 2000)
        surface   = race_info.get("surface", "芝")
        race_date = race_info.get("race_date", "")

        draw.text((IW // 2 - 60, 60),  "未来レース観測記録",          font=_f(14), fill=(255, 220, 50))
        if blink:
            draw.text((IW // 2 - 50, 90), f"#{race_num} {race_name}", font=_f(12), fill=(255, 255, 255))
        draw.text((IW // 2 - 60, 120), f"{venue} {surface}{distance}m", font=_f(10), fill=(180, 180, 255))
        draw.text((IW // 2 - 50, 145), race_date,                       font=_f(9),  fill=(120, 120, 200))
        draw.text((IW // 2 - 55, 200), f"出走頭数: {total_horses}頭",   font=_f(10), fill=(200, 200, 200))

        for cx, cy in [(0, 0), (IW, 0), (0, IH), (IW, IH)]:
            sx = 1 if cx == 0 else -1
            sy = 1 if cy == 0 else -1
            draw.line([cx, cy, cx + sx * 20, cy], fill=(0, 200, 80), width=2)
            draw.line([cx, cy, cx, cy + sy * 20], fill=(0, 200, 80), width=2)

        big = img.resize((1080, 1920), Image.NEAREST)
        frames.append(np.array(big))
    return frames


def render_result_card(horses: list, race_info: dict) -> list[np.ndarray]:
    top3 = sorted(horses, key=lambda h: -h.screen_x)[:3]
    frames = []
    medal_colors = [(255, 215, 0), (200, 200, 200), (180, 120, 60)]
    medal_marks  = ["◎", "○", "▲"]

    for tick in range(300):
        img = Image.new("RGB", (IW, IH), (5, 5, 15))
        draw = ImageDraw.Draw(img)

        draw.text((IW // 2 - 70, 20), "…帰ってきた",             font=_f(12), fill=(180, 220, 255))
        draw.text((IW // 2 - 80, 50), "未来映像から予想した結果", font=_f(9),  fill=(150, 150, 200))
        draw.text((IW // 2 - 60, 70), "を公開します",             font=_f(9),  fill=(150, 150, 200))

        draw.line([20, 100, IW - 20, 100], fill=(80, 80, 120))

        for i, h in enumerate(top3):
            y = 120 + i * 80
            color = medal_colors[i]
            draw.text((20, y),      medal_marks[i],     font=_f(22), fill=color)
            draw.text((50, y),      f"{h.number}番",    font=_f(16), fill=color)
            draw.text((50, y + 22), h.name,             font=_f(12), fill=(255, 255, 255))
            if h.comment:
                draw.text((20, y + 42), h.comment[:16], font=_f(8),  fill=(160, 200, 160))

        big = img.resize((1080, 1920), Image.NEAREST)
        frames.append(np.array(big))
    return frames

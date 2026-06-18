"""
renderer.py  ―  Future Derby Viewer
「未来のレース映像を観測するSF競馬動画」レンダラー
"""

import math
import random
import os as _os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

IW, IH   = 270, 480
SCALE    = 4
HUD_H    = 22
RACE_Y   = HUD_H
RACE_H   = 360
PANEL_Y  = RACE_Y + RACE_H
S = 2      # UI/background scale
HS = 4     # horse sprite scale (2× S for larger horses)
LEADER_X = 155
GROUND_Y = RACE_Y + int(RACE_H * 0.88)

COATS = [
    {"b":(160,100, 50), "m":(100, 60, 20), "l":( 90, 55, 20)},
    {"b":( 60, 35, 15), "m":( 25, 12,  5), "l":( 40, 20,  8)},
    {"b":(120, 70, 30), "m":( 60, 35, 10), "l":( 70, 40, 15)},
    {"b":(200,195,185), "m":(220,215,210), "l":(180,175,170)},
    {"b":( 40, 25, 12), "m":( 20, 10,  5), "l":( 30, 15,  8)},
    {"b":(240,235,228), "m":(255,250,245), "l":(220,215,208)},
]
def coat(num): return COATS[(num - 1) % len(COATS)]

# ──────────────────────────────────────────────
#  4フレームギャロップアニメ
#  各脚: (x_offset, length, bend)
#    x_offset: 馬体左端からの水平オフセット(内部px)
#    length:   脚の全長(内部px)
#    bend:     膝の曲げ量(正=前、負=後ろ)
#
#  F0: 前脚前方・後脚後方（最大開脚）
#  F1: 脚を中央へ戻す
#  F2: 前脚後方・後脚前方（逆開脚）
#  F3: 脚を中央へ戻す
# ──────────────────────────────────────────────
G4 = [
    # F0: 前脚前方・後脚後方
    {"bob": -3,
     "fl": {"ox": 23, "len": 14, "b":  6},   # 前左: 大きく前へ
     "fr": {"ox": 19, "len": 13, "b":  5},   # 前右: 前へ
     "bl": {"ox": -4, "len": 14, "b": -6},   # 後左: 大きく後ろへ
     "br": {"ox":  1, "len": 13, "b": -5}},  # 後右: 後ろへ

    # F1: 脚を中央へ（収束・体が上がる）
    {"bob":  1,
     "fl": {"ox": 14, "len": 11, "b":  1},
     "fr": {"ox": 11, "len": 10, "b":  1},
     "bl": {"ox":  7, "len": 11, "b": -1},
     "br": {"ox": 10, "len": 10, "b": -1}},

    # F2: 前脚後方・後脚前方（逆開脚）
    {"bob": -3,
     "fl": {"ox":  5, "len": 14, "b": -6},   # 前左: 後ろへ
     "fr": {"ox":  9, "len": 13, "b": -5},   # 前右: 後ろへ
     "bl": {"ox": 15, "len": 14, "b":  6},   # 後左: 前へ
     "br": {"ox": 19, "len": 13, "b":  5}},  # 後右: 前へ

    # F3: 脚を中央へ（収束）
    {"bob":  1,
     "fl": {"ox": 11, "len": 10, "b": -1},
     "fr": {"ox": 14, "len": 11, "b": -1},
     "bl": {"ox":  6, "len": 10, "b":  1},
     "br": {"ox":  9, "len": 11, "b":  1}},
]

# 速度 → fps 変換テーブル
# horse.speed が 1.0=基準。drama.event で上書き。
def speed_to_fps(speed: float, drama_event: str) -> int:
    if drama_event == "rocket":   return 36   # 3× base speed for rocket
    if speed > 1.30:              return 8
    if speed > 1.12:              return 6
    return 4

# 累積アニメカウンタ（馬番→float）
_anim_acc: dict[int, float] = {}

def get_anim_frame(num: int, speed: float, drama_event: str) -> int:
    fps = speed_to_fps(speed, drama_event)
    step = fps / 30.0
    n = len(_SPRITE_FRAMES_RAW) if _SPRITE_FRAMES_RAW else 4
    acc = _anim_acc.get(num, 0.0) + step
    if acc >= n: acc -= n
    _anim_acc[num] = acc
    return int(acc) % n


# ── スプライトシステム ─────────────────────────────
_SPW = 100   # 内部キャンバス表示幅
_SPH = 58    # 内部キャンバス表示高さ
SPRITE_BOB = [-2, 1, 0, -2, 1, 0]  # フレームごとの上下ボブ量

_SPRITE_FRAMES_RAW: list = []
_SPRITE_CACHE: dict = {}

def _load_sprites() -> None:
    global _SPRITE_FRAMES_RAW
    base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "frames")
    frames = []
    for i in range(6):
        p = _os.path.join(base, f"horse_{i}.png")
        if _os.path.exists(p):
            img = Image.open(p).convert("RGBA")
            img = img.resize((_SPW, _SPH), Image.LANCZOS)
            frames.append(img)
    _SPRITE_FRAMES_RAW = frames

def _colorize(frame: Image.Image, bc: tuple, drama_event: str = "") -> Image.Image:
    arr = np.array(frame, dtype=np.float32)
    a = arr[:, :, 3]
    lum = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]) / 255.0
    c = bc
    if drama_event == "rocket":
        c = tuple(min(255, int(v*0.5 + p*0.5)) for v, p in zip(c, (160, 0, 220)))
    result = np.stack([
        np.clip(c[0] * lum, 0, 255),
        np.clip(c[1] * lum, 0, 255),
        np.clip(c[2] * lum, 0, 255),
        a,
    ], axis=2).astype(np.uint8)
    return Image.fromarray(result, "RGBA")

def _get_sprite(fi: int, num: int, drama_event: str = ""):
    if not _SPRITE_FRAMES_RAW:
        return None
    key = (fi % len(_SPRITE_FRAMES_RAW), num, drama_event)
    if key not in _SPRITE_CACHE:
        raw = _SPRITE_FRAMES_RAW[fi % len(_SPRITE_FRAMES_RAW)]
        _SPRITE_CACHE[key] = _colorize(raw, coat(num)["b"], drama_event)
    return _SPRITE_CACHE[key]

_load_sprites()


@dataclass
class DramaState:
    event:     str   = ""
    intensity: float = 0.0
    flash:     float = 0.0
    shake_x:   int   = 0
    shake_y:   int   = 0

def fr(d, x, y, w, h, c):
    if w <= 0 or h <= 0: return
    d.rectangle([int(x), int(y), int(x+w-1), int(y+h-1)], fill=c)

def hex2rgb(s):
    s = s.lstrip('#')
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def darker(c, n=40):  return tuple(max(0,   v - n) for v in c)
def lighter(c, n=20): return tuple(min(255, v + n) for v in c)

def draw_horse(img: Image.Image, hx: float, hy: float, num: int,
               jockey_hex: str, horse_speed: float = 1.0, drama_event: str = ""):
    fi       = get_anim_frame(num, horse_speed, drama_event)
    bob      = SPRITE_BOB[fi % len(SPRITE_BOB)]
    bob_mul  = 2 if drama_event == "rocket" else 1
    px       = int(hx) - _SPW // 2
    py       = int(hy) - _SPH + bob * bob_mul

    d = ImageDraw.Draw(img)

    # 影
    sw = _SPW if drama_event != "rocket" else int(_SPW * 0.7)
    fr(d, px, int(hy) + 2, sw, 3, (0, 0, 0))

    # ロケット：スプライトの後ろ（左側）に紫プラズマ
    if drama_event == "rocket":
        plasma = [(160,0,220),(200,50,255),(120,0,180)]
        for pi in range(3):
            r  = 8 + pi * 6
            cx = px + _SPW // 4
            cy = py + _SPH // 2
            for dy in range(-r, r+1, 2):
                dx = int(math.sqrt(max(0, r*r - dy*dy)))
                d.line([cx-dx, cy+dy, cx+dx, cy+dy], fill=plasma[pi % 3])

    # スプライト貼り付け（アルファ合成）
    sprite = _get_sprite(fi, num, drama_event)
    if sprite is not None:
        dx = max(0, -px);  dy = max(0, -py)
        sx = max(0, px);   sy = max(0, py)
        pw = min(sprite.width - dx, img.width - sx)
        ph = min(sprite.height - dy, img.height - sy)
        if pw > 0 and ph > 0:
            img.alpha_composite(sprite.crop((dx, dy, dx+pw, dy+ph)), dest=(sx, sy))
        d = ImageDraw.Draw(img)  # alpha_composite後にDrawを再取得

    # 騎手（鞍部分の上に小さく描画）
    try:    jc = hex2rgb(jockey_hex)
    except: jc = (128, 128, 128)
    jx = px + int(_SPW * 0.38)
    jy = py + int(_SPH * 0.10)
    fr(d, jx,   jy,    12, 9,  darker(jc))
    fr(d, jx,   jy,     8, 9,  jc)
    fr(d, jx+1, jy,     2, 9,  lighter(jc, 35))
    fr(d, jx+4, jy,     2, 9,  lighter(jc, 35))
    fr(d, jx+2, jy-7,   6, 5,  jc)
    fr(d, jx+3, jy-11,  4, 4,  (240, 192, 144))
    try:
        d.text((jx+1, jy+1), str(num), fill=(255, 255, 255))
    except: pass

def _cloud(d, x, y, w, h, W):
    c = (234,243,255)
    fr(d, x,             y+int(h*.45), w,            int(h*.55)+1, c)
    fr(d, x+int(w*.05),  y+int(h*.22), int(w*.38)+1, int(h*.52)+1, c)
    fr(d, x+int(w*.28),  y+int(h*.08), int(w*.42)+1, int(h*.60)+1, c)
    fr(d, x+int(w*.62),  y+int(h*.28), int(w*.33)+1, int(h*.42)+1, c)
    if x + w > W: _cloud(d, x-W, y, w, h, W)

def _tree(d, x, y, w, h):
    fr(d, x+int(w*.25), y,            int(w*.5),  int(h*.38), (74,170,58))
    fr(d, x+int(w*.08), y+int(h*.3),  int(w*.84), int(h*.38), (42,138,42))
    fr(d, x,            y+int(h*.62), w,          int(h*.38), (26, 92,26))

def draw_bg(d, bg_scroll, W):
    sky_h = int(RACE_H * 0.18)
    for y in range(sky_h):
        t = y / sky_h
        d.line([0, RACE_Y+y, W-1, RACE_Y+y],
               fill=(int(85+t*25), int(145+t*35), int(205+t*30)))

    for cx,cy,cw,ch in [(40,3,18,6),(130,2,22,7),(220,4,16,5)]:
        ox = int((cx - bg_scroll*.10) % W)
        _cloud(d, ox, RACE_Y+cy, cw, ch, W)

    sy = RACE_Y + int(RACE_H * 0.16)
    sh = int(RACE_H * 0.22)
    fr(d, 0, sy, W, sh, (155,140,128))
    for x in range(0, W+16, 16):
        ox = int((x - bg_scroll*.16) % W)
        fr(d, ox, sy, 16, sh, (170,158,145) if (ox//16)%2==0 else (185,173,160))
        fr(d, ox+2, sy+3, 3, 4, (75,60,60))
        fr(d, ox+8, sy+3, 3, 4, (75,60,60))
        if ox+16 > W:
            ox2 = ox-W
            fr(d, ox2, sy, 16, sh, (170,158,145))
    fr(d, 0, sy-2, W, 2, (95,78,62))

    tree_y = RACE_Y + int(RACE_H * 0.36)
    fr(d, 0, tree_y, W, 12, (26,92,26))
    for i in range(52):
        tx = int((i * 5.4 - bg_scroll*.52) % W)
        _tree(d, tx, tree_y-9, 6, 11)
        if tx+6 > W: _tree(d, tx-W, tree_y-9, 6, 11)

    tf = RACE_Y + int(RACE_H * 0.38)
    stripe_h = max(1, (RACE_H - int(RACE_H*0.44)) // 14)
    for row in range(14):
        c = (56,136,58) if row%2==0 else (44,110,46)
        fr(d, 0, tf + row*stripe_h, W, stripe_h+1, c)

    rl = RACE_Y + int(RACE_H * 0.88)
    fr(d, 0, rl, W, 2, (221,221,221))
    fr(d, 0, rl+2, W, int(RACE_H*.09), (196,152,106))
    for x in range(0, W, 20):
        ox = int((x - bg_scroll*1.05) % W)
        fr(d, ox, rl+2, 1, int(RACE_H*.09), (175,135,88))
    fr(d, 0, RACE_Y+RACE_H-2, W, 2, (221,221,221))

    px = int((int(W*.72) - bg_scroll*.65) % W)
    pt = RACE_Y + int(RACE_H*.28)
    ph = int(RACE_H*.62)
    fr(d, px, pt, 4, ph, (204,17,17))
    fr(d, px, pt, 4, ph//2, (238,238,238))
    r = 9
    for dy in range(-r, r+1):
        dx = int(math.sqrt(max(0, r*r - dy*dy)))
        fr(d, px+2-dx, pt-r+dy, dx*2+1, 1, (204,17,17))
    try: d.text((px-1, pt-r-9), "4", fill=(255,255,255))
    except: pass

def draw_sf_hud(d, leader, dist, tick, W):
    fr(d, 0, 0, W, HUD_H, (0,0,0))
    blink = (tick // 15) % 2 == 0
    try:
        d.text((4, 4),  "◉ TEMPORAL OBSERVATION", fill=(0,200,80) if blink else (0,150,60))
        d.text((W-50, 4), f"DIST:{int(dist)}m",    fill=(0,180,255))
    except: pass

    cycle = 90
    pos = (tick % cycle) / cycle
    scan_y = RACE_Y + int(pos * RACE_H)
    d.line([0, scan_y, W-1, scan_y], fill=(0, 180, 80), width=1)

    if leader:
        cx = int(_gh(leader, "screen_x", LEADER_X))
        cy = GROUND_Y - 15*S
        hw, hh = 18*S, 16*S
        bc = (0, 255, 80)
        size = 6
        d.line([cx-hw, cy-hh, cx-hw+size, cy-hh], fill=bc)
        d.line([cx-hw, cy-hh, cx-hw, cy-hh+size], fill=bc)
        d.line([cx+hw-size, cy-hh, cx+hw, cy-hh], fill=bc)
        d.line([cx+hw, cy-hh, cx+hw, cy-hh+size], fill=bc)
        d.line([cx-hw, cy+hh-size, cx-hw, cy+hh], fill=bc)
        d.line([cx-hw, cy+hh, cx-hw+size, cy+hh], fill=bc)
        d.line([cx+hw, cy+hh-size, cx+hw, cy+hh], fill=bc)
        d.line([cx+hw-size, cy+hh, cx+hw, cy+hh], fill=bc)
        try:
            name = str(_gh(leader, "name", ""))[:6]
            num  = _gh(leader, "number", "")
            d.text((cx-hw, cy-hh-11), f"TARGET LOCK  {num}番 {name}", fill=(0,255,80))
        except: pass

    try:
        d.text((4, PANEL_Y - 12), "PREDICTION ACTIVE", fill=(0,150,200))
    except: pass

def draw_speed_lines(d, cx, cy, intensity=1.0):
    n = int(14 * intensity)
    for _ in range(n):
        angle = random.uniform(-0.35, 0.35)
        r1    = random.uniform(4, 12)
        r2    = r1 + random.uniform(12, 30) * intensity
        x1 = cx - math.cos(angle) * r1
        y1 = cy + math.sin(angle) * r1 * 2
        x2 = cx - math.cos(angle) * r2
        y2 = cy + math.sin(angle) * r2 * 2
        d.line([int(x1), int(y1), int(x2), int(y2)], fill=(255,220,80), width=1)

def draw_warp_afterimage(img: Image.Image, horse_x, horse_y, num, jc_hex, tick, steps=3):
    for i in range(steps, 0, -1):
        ghost_x = horse_x + i * 8
        a = max(0, 70 - i * 20)
        ghost = Image.new("RGBA", (_SPW - 8, _SPH - 4), (180, 180, 220, a))
        sx = max(0, int(ghost_x) - _SPW // 2 + 4)
        sy = max(0, int(horse_y) - _SPH + 4)
        if sx < img.width and sy < img.height:
            img.alpha_composite(ghost, dest=(sx, sy))

def draw_rocket_afterimage(img: Image.Image, horse_x, horse_y, ct, steps=4):
    bc = ct["b"]
    for i in range(steps, 0, -1):
        ghost_x = horse_x + i * 5
        t = i / steps
        col = (
            min(255, int(bc[0] * (1-t) * 0.4 + 160 * t)),
            min(255, int(bc[1] * 0.1)),
            min(255, int(bc[2] * (1-t) * 0.4 + 200 * t)),
        )
        a = int(80 * t)
        ghost = Image.new("RGBA", (_SPW - 8, _SPH - 4), (*col, a))
        sx = max(0, int(ghost_x) - _SPW // 2 + 4)
        sy = max(0, int(horse_y) - _SPH + 4)
        if sx < img.width and sy < img.height:
            img.alpha_composite(ghost, dest=(sx, sy))

def apply_drama_to_frame(arr, drama):
    if drama.flash > 0:
        white = np.ones_like(arr, dtype=np.float32) * 255
        arr   = (arr * (1 - drama.flash) + white * drama.flash).clip(0,255).astype(np.uint8)
    return arr

def maybe_noise(arr, tick, prob=0.06):
    if random.random() > prob:
        return arr
    arr  = arr.copy()
    H, W = arr.shape[:2]
    n    = int(H * W * 0.025)
    ys   = np.random.randint(0, H, n)
    xs   = np.random.randint(0, W, n)
    arr[ys, xs] = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
    for _ in range(3):
        y = random.randint(0, H-1)
        arr[y] = np.roll(arr[y], random.randint(-25, 25), axis=0)
    return arr

def apply_battery(arr, phase, intensity):
    if phase == "normal" or intensity <= 0:
        return arr

    arr = arr.copy()
    H, W = arr.shape[:2]

    n = int(H * W * intensity * 0.10)
    if n > 0:
        ys = np.random.randint(0, H, n)
        xs = np.random.randint(0, W, n)
        arr[ys, xs] = np.random.randint(0, 256, (n, 3), dtype=np.uint8)

    for _ in range(int(10 * intensity)):
        y  = random.randint(0, H-1)
        sh = random.randint(-int(40*intensity), int(40*intensity))
        arr[y] = np.roll(arr[y], sh, axis=0)

    img = Image.fromarray(arr)
    od  = ImageDraw.Draw(img)
    cx  = W // 2
    blink = (random.random() < 0.5)

    if phase == "warning":
        fr(od, 0, 0, W, 80, (0,0,0))
        if blink:
            od.text((cx-55, 14), "WARNING",                   fill=(255,220,0))
            od.text((cx-90, 36), "TEMPORAL ENGINE LOW POWER", fill=(255,190,0))

    elif phase == "critical":
        fr(od, 0, 0, W, 100, (25,0,0))
        od.text((cx-80, 14), "BATTERY CRITICAL", fill=(255,50,50))
        od.text((cx-90, 38), "SIGNAL DISRUPTED", fill=(255,100,100))
        for i in range(0, 30, 4):
            od.rectangle([i, i, W-i-1, H-i-1], outline=(150,0,0))

    elif phase == "blackout":
        dark_ratio = min(1.0, intensity)
        arr2 = (arr * (1 - dark_ratio)).astype(np.uint8)
        img  = Image.fromarray(arr2)
        od   = ImageDraw.Draw(img)
        if dark_ratio > 0.5:
            od.text((cx-65, H//2-24), "SIGNAL LOST",            fill=(200,0,0))
            od.text((cx-95, H//2+4),  "OBSERVATION TERMINATED", fill=(140,0,0))

    return np.array(img)

def draw_bottom_panel(d, horses, commentary, tsukkomi, W):
    fr(d, 0, PANEL_Y, W, IH-PANEL_Y, (15,15,22))
    fr(d, 0, PANEL_Y, W, 1, (50,50,70))

    try:
        for i, line in enumerate(commentary[:2]):
            d.text((6, PANEL_Y + 5 + i*14), line, fill=(255,255,200))
    except: pass

    if tsukkomi:
        try:
            txt = f"俺「{tsukkomi}」"
            tw  = len(txt) * 6 + 12
            tx  = W - min(tw+6, W-4)
            fr(d, tx, PANEL_Y+34, W-tx, 15, (0,0,0))
            d.text((tx+4, PANEL_Y+36), txt, fill=(150,200,255))
        except: pass

    RANK_COLORS = [(255,215,0),(200,200,200),(180,120,60)]
    sorted_h = sorted(horses, key=lambda h: -_gh(h, "screen_x", 0))
    try:
        for i, h in enumerate(sorted_h[:3]):
            cx = 5 + i * 88
            d.text((cx, PANEL_Y+54), f"{i+1}位 {_gh(h,'number','')}番", fill=RANK_COLORS[i])
            d.text((cx, PANEL_Y+68), str(_gh(h,'name',''))[:5],          fill=(210,210,210))
    except: pass

def _gh(obj, key, default=None):
    """dict か dataclass かを問わず属性を取得する"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def render_frame(
    horses,
    bg_scroll: float,
    commentary: list,
    dist_remaining: float,
    tick: int,
    race_info: dict = None,
    tsukkomi: str = "",
    battery_phase: str = "normal",
    battery_intensity: float = 0.0,
    drama: DramaState = None,
) -> np.ndarray:
    if race_info is None: race_info = {}
    if drama is None:     drama = DramaState()

    # RGBA で合成（スプライトのアルファ合成のため）
    img = Image.new("RGBA", (IW, IH), (0, 0, 0, 255))
    d   = ImageDraw.Draw(img)

    draw_bg(d, bg_scroll, IW)

    leader_horse = None
    sorted_draw  = sorted(horses, key=lambda h: _gh(h, "screen_x", 0))

    for i, h in enumerate(sorted_draw):
        sx_h   = _gh(h, "screen_x", 0)
        hnum   = _gh(h, "number", 1)
        hjc    = _gh(h, "jockey_color", "#888888")
        hspeed = _gh(h, "speed", 1.0)
        if -60 < sx_h < IW + 60:
            if drama.event == "warp":
                draw_warp_afterimage(img, sx_h, GROUND_Y - i, hnum, hjc, tick)
            elif drama.event == "rocket" and drama.intensity > 0.3:
                draw_rocket_afterimage(img, sx_h, GROUND_Y - i, coat(hnum))
            draw_horse(img, sx_h + drama.shake_x, GROUND_Y - i + drama.shake_y,
                       hnum, hjc, hspeed, drama.event)
        if _gh(h, "rank", 99) == 1:
            leader_horse = h

    d = ImageDraw.Draw(img)  # 馬描画後にDrawを再取得
    if drama.event in ("rocket","charge","makuri") and drama.intensity > 0:
        lx = _gh(leader_horse, "screen_x", LEADER_X) if leader_horse else LEADER_X
        draw_speed_lines(d, lx + drama.shake_x, GROUND_Y - 10*S + drama.shake_y, drama.intensity)

    draw_sf_hud(d, leader_horse, dist_remaining, tick, IW)
    draw_bottom_panel(d, horses, commentary, tsukkomi, IW)

    big = img.convert("RGB").resize((1080, 1920), Image.NEAREST)
    arr = np.array(big)

    arr = apply_drama_to_frame(arr, drama)

    if battery_phase == "normal":
        arr = maybe_noise(arr, tick)

    arr = apply_battery(arr, battery_phase, battery_intensity)

    return arr


def render_title_card(race_info: dict, total_horses: int) -> list:
    frames = []
    for tick in range(150):
        img = Image.new("RGB", (IW, IH), (5, 5, 15))
        d   = ImageDraw.Draw(img)
        for y in range(0, IH, 3):
            v = int(20 + 10 * np.sin(y * 0.05 + tick * 0.1))
            d.line([0, y, IW, y], fill=(0, 0, v))
        blink = tick % 20 < 15
        rn  = race_info.get("race_name", "レース")
        rnu = race_info.get("race_num",  "")
        di  = race_info.get("distance",  2000)
        ve  = race_info.get("venue",     "")
        su  = race_info.get("surface",   "芝")
        rd  = race_info.get("race_date", "")
        try:
            d.text((IW//2-60, 60),  "未来レース観測記録",        fill=(255,220,50))
            if blink:
                d.text((IW//2-50, 90), f"#{rnu} {rn}",          fill=(255,255,255))
            d.text((IW//2-60, 120), f"{ve} {su}{di}m",          fill=(180,180,255))
            d.text((IW//2-50, 145), rd,                          fill=(120,120,200))
            d.text((IW//2-55, 200), f"出走頭数: {total_horses}頭", fill=(200,200,200))
        except: pass
        for cx, cy in [(0,0),(IW,0),(0,IH),(IW,IH)]:
            sx = 1 if cx==0 else -1
            sy = 1 if cy==0 else -1
            d.line([cx,cy,cx+sx*20,cy],   fill=(0,200,80), width=2)
            d.line([cx,cy,cx,cy+sy*20],   fill=(0,200,80), width=2)
        big = img.resize((1080,1920), Image.NEAREST)
        frames.append(np.array(big))
    return frames


def render_result_card(horses, race_info: dict) -> list:
    top3 = sorted(horses, key=lambda h: -_gh(h, "screen_x", 0))[:3]
    frames = []
    mc = [(255,215,0),(200,200,200),(180,120,60)]
    mm = ["◎","○","▲"]
    for tick in range(300):
        img = Image.new("RGB", (IW,IH), (5,5,15))
        d   = ImageDraw.Draw(img)
        try:
            d.text((IW//2-70, 20), "…帰ってきた",             fill=(180,220,255))
            d.text((IW//2-80, 50), "未来映像から予想した結果", fill=(150,150,200))
            d.text((IW//2-60, 70), "を公開します",             fill=(150,150,200))
        except: pass
        d.line([20,100,IW-20,100], fill=(80,80,120))
        for i, h in enumerate(top3):
            y = 120 + i*80
            num     = _gh(h, "number",  "?")
            name    = str(_gh(h, "name", ""))
            comment = str(_gh(h, "comment", ""))
            try:
                d.text((20, y),      mm[i],           fill=mc[i])
                d.text((50, y),      f"{num}番",      fill=mc[i])
                d.text((50, y+22),   name,             fill=(255,255,255))
                if comment:
                    d.text((20, y+42), comment[:16],   fill=(160,200,160))
            except: pass
        big = img.resize((1080,1920), Image.NEAREST)
        frames.append(np.array(big))
    return frames

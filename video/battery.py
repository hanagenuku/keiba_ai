from dataclasses import dataclass
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont


@dataclass
class BatteryState:
    dist: float = 2000.0
    phase: str = "normal"
    noise_intensity: float = 0.0

    WARNING_DIST:  int = 100
    CRITICAL_DIST: int = 50
    BLACKOUT_DIST: int = 5

    def update(self, dist_remaining: float):
        self.dist = dist_remaining

        if dist_remaining <= self.BLACKOUT_DIST:
            self.phase = "blackout"
            self.noise_intensity = 1.0
        elif dist_remaining <= self.CRITICAL_DIST:
            self.phase = "critical"
            t = (self.CRITICAL_DIST - dist_remaining) / (self.CRITICAL_DIST - self.BLACKOUT_DIST)
            self.noise_intensity = 0.5 + t * 0.5
        elif dist_remaining <= self.WARNING_DIST:
            self.phase = "warning"
            t = (self.WARNING_DIST - dist_remaining) / (self.WARNING_DIST - self.CRITICAL_DIST)
            self.noise_intensity = t * 0.5
        else:
            self.phase = "normal"
            self.noise_intensity = 0.0

    def is_finished(self) -> bool:
        return self.phase == "blackout"


def apply_battery_overlay(frame: np.ndarray, battery: BatteryState, tick: int = 0) -> np.ndarray:
    if battery.phase == "normal":
        return frame

    frame = frame.copy()
    H, W = frame.shape[:2]
    intensity = battery.noise_intensity

    n_noise = int(H * W * intensity * 0.12)
    if n_noise > 0:
        ys = np.random.randint(0, H, n_noise)
        xs = np.random.randint(0, W, n_noise)
        frame[ys, xs] = np.random.randint(0, 256, (n_noise, 3), dtype=np.uint8)

    n_lines = int(12 * intensity)
    for _ in range(n_lines):
        y = random.randint(0, H - 1)
        shift = random.randint(-int(40 * intensity), int(40 * intensity))
        frame[y] = np.roll(frame[y], shift, axis=0)

    overlay_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(overlay_img)

    try:
        fn = ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 48)
        fn_sm = ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 30)
    except Exception:
        fn = ImageFont.load_default()
        fn_sm = fn

    blink = tick % 2 == 0

    if battery.phase == "warning" and blink:
        draw.rectangle([0, 0, W, 100], fill=(20, 20, 0))
        draw.text((W // 2 - 80, 15), "⚠ WARNING",   font=fn,    fill=(255, 220, 0))
        draw.text((W // 2 - 90, 60), "BATTERY LOW", font=fn_sm, fill=(255, 200, 0))

    elif battery.phase == "critical":
        draw.rectangle([0, 0, W, 120], fill=(40, 0, 0))
        draw.text((W // 2 - 160, 15), "⚠ BATTERY CRITICAL", font=fn,    fill=(255, 50, 50))
        draw.text((W // 2 - 140, 70), "SIGNAL DISRUPTED",    font=fn_sm, fill=(255, 100, 100))
        border = int(20 * intensity)
        for i in range(border):
            c = int(i / border * 180 * intensity)
            draw.rectangle([i, i, W - i, H - i], outline=(255, 0, 0))

    elif battery.phase == "blackout":
        darkness = min(1.0, 1.0 - battery.dist / max(1, battery.BLACKOUT_DIST))
        dark = Image.new("RGB", (W, H), (0, 0, 0))
        overlay_img = Image.blend(overlay_img, dark, alpha=darkness)
        draw = ImageDraw.Draw(overlay_img)
        if blink:
            draw.text((W // 2 - 140, H // 2 - 60), "SIGNAL LOST",            font=fn,    fill=(255, 0, 0))
            draw.text((W // 2 - 200, H // 2 + 10), "OBSERVATION TERMINATED", font=fn_sm, fill=(200, 0, 0))

    return np.array(overlay_img)

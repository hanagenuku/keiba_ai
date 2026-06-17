from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Horse:
    number: int
    name: str
    style: str
    power: int
    jockey_color: str

    pos: float = 0.0
    speed: float = 1.0
    screen_x: float = 0.0
    rank: int = 0

    drama_event: Optional[str] = None
    drama_text: Optional[str] = None

    expected_rank: int = 0
    late_speed: int = 50
    pace_fit: str = "M"
    drama_hint: Optional[str] = None
    comment: str = ""

    def base_speed(self, dist_remaining: float) -> float:
        s = (self.power - 75) / 800.0

        if self.style == "逃げ":
            if dist_remaining > 1200:
                s += 0.09
            elif dist_remaining > 400:
                s -= 0.01
            else:
                s -= 0.07

        elif self.style == "先行":
            if dist_remaining > 800:
                s += 0.04
            elif dist_remaining > 300:
                s += 0.01
            else:
                s -= 0.02

        elif self.style == "差し":
            if dist_remaining < 600:
                s += 0.06
            else:
                s -= 0.01

        elif self.style == "追込":
            if dist_remaining < 400:
                s += 0.10
            elif dist_remaining < 700:
                s += 0.03
            else:
                s -= 0.03

        return s

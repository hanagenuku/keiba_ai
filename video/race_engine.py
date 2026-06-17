import random
from horse import Horse
from drama import DramaEngine


class RaceEngine:
    TOTAL_FRAMES = 800
    PX_PER_METER = 0.30
    LEADER_X = 195
    BASE_SPEED_M_PER_FRAME = 2.6
    NOISE = 0.04

    def __init__(self, horses: list):
        self.horses = horses
        self.frame = 0
        self.drama = DramaEngine(horses)
        self.commentary_queue = []
        self._fired_checkpoints: set = set()
        self._all_horses = horses

    def dist_remaining(self) -> float:
        progress = self.frame / self.TOTAL_FRAMES
        return max(0, 2000 - progress * 1900)

    def step(self):
        self.frame += 1
        dist = self.dist_remaining()

        # ワープ処理
        warp_nums = self.drama.get_warp_horses(self.frame)

        for horse in self.horses:
            mod = horse.base_speed(dist)
            drama_mod = self.drama.get_speed_mod(horse, self.frame, dist)
            mod += drama_mod

            # expected_rank による終盤補正
            if dist < 600 and horse.expected_rank > 0:
                total = len(self.horses)
                rank_pct = (total - horse.expected_rank) / total
                mod += rank_pct * 0.05

            mod += (random.random() - 0.5) * self.NOISE

            target_speed = 1.0 + mod
            horse.speed = horse.speed * 0.88 + target_speed * 0.12
            horse.pos += max(0.4, horse.speed) * self.BASE_SPEED_M_PER_FRAME

            # ワープ
            if horse.number in warp_nums:
                horse.pos += 30 / self.PX_PER_METER

        max_pos = max(h.pos for h in self.horses)
        for horse in self.horses:
            gap = max_pos - horse.pos
            horse.screen_x = self.LEADER_X - gap * self.PX_PER_METER

        sorted_horses = sorted(self.horses, key=lambda h: -h.screen_x)
        for i, h in enumerate(sorted_horses):
            h.rank = i + 1

        self._check_commentary(dist)

    def _check_commentary(self, dist):
        checkpoints = {600, 400, 200, 100}
        for cp in checkpoints:
            if abs(dist - cp) < 3 and cp not in self._fired_checkpoints:
                self._fired_checkpoints.add(cp)
                top3 = sorted(self.horses, key=lambda h: -h.screen_x)[:3]
                self.commentary_queue.append({"dist": cp, "horses": top3})

    def is_finished(self) -> bool:
        return self.frame >= self.TOTAL_FRAMES

    def get_draw_order(self) -> list:
        return sorted(self.horses, key=lambda h: h.screen_x)

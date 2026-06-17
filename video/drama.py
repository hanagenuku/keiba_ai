import random
from dataclasses import dataclass, field


@dataclass
class DramaEvent:
    event_type: str
    horse_num: int
    fired: bool = False


class DramaEngine:
    def __init__(self, horses):
        self.events: dict[int, list[DramaEvent]] = {}
        self._assign_events(horses)

    def _assign_events(self, horses):
        for h in horses:
            self.events[h.number] = []
            p = h.power

            # drama_hint が指定されている場合は優先発動
            if h.drama_hint:
                self.events[h.number].append(DramaEvent(h.drama_hint, h.number))

            if random.random() < 0.15:
                self.events[h.number].append(DramaEvent("出遅れ", h.number))

            if h.style in ("逃げ", "先行") and random.random() < 0.20:
                self.events[h.number].append(DramaEvent("ハナ", h.number))

            if h.style in ("差し", "追込") and random.random() < 0.18:
                self.events[h.number].append(DramaEvent("まくり", h.number))

            if h.style in ("先行", "差し") and random.random() < 0.14:
                self.events[h.number].append(DramaEvent("内", h.number))

            charge_prob = max(0.05, 0.25 - p / 200)
            if h.style == "追込" and random.random() < charge_prob:
                self.events[h.number].append(DramaEvent("一気", h.number))

            if p >= 85 and random.random() < 0.10:
                self.events[h.number].append(DramaEvent("ロケット", h.number))

            if p >= 80 and random.random() < 0.05:
                self.events[h.number].append(DramaEvent("ワープ", h.number))

            if h.style == "逃げ" and p < 75 and random.random() < 0.20:
                self.events[h.number].append(DramaEvent("崩壊", h.number))

    def get_speed_mod(self, horse, frame: int, dist: float) -> float:
        mod = 0.0
        for evt in self.events.get(horse.number, []):
            t = evt.event_type
            if t == "出遅れ" and frame < 55:
                mod -= 0.60
            elif t == "ハナ" and frame < 80:
                mod += 0.45
            elif t == "まくり" and 300 < dist < 800:
                mod += 0.28
            elif t == "内" and 100 < dist < 600:
                mod += 0.22
            elif t == "一気" and dist < 400:
                mod += 0.40
            elif t == "ロケット" and 500 < dist < 700:
                mod += 0.50
            elif t == "崩壊" and dist < 250:
                mod -= 0.40
        return mod

    def get_warp_horses(self, frame: int) -> list[int]:
        warp = []
        dist = max(0, 2000 - (frame / 800) * 1900)
        for num, evts in self.events.items():
            for evt in evts:
                if evt.event_type == "ワープ" and 350 < dist < 600:
                    if not evt.fired:
                        evt.fired = True
                        warp.append(num)
        return warp

    def get_active_events(self, dist: float) -> list[DramaEvent]:
        active = []
        for evts in self.events.values():
            for evt in evts:
                t = evt.event_type
                if (
                    (t == "一気"    and dist < 400) or
                    (t == "まくり"  and 300 < dist < 800) or
                    (t == "ロケット" and 500 < dist < 700) or
                    (t == "崩壊"    and dist < 250)
                ):
                    active.append(evt)
        return active

    def get_newly_fired_events(self, dist: float) -> list[DramaEvent]:
        newly_fired = []
        for evts in self.events.values():
            for evt in evts:
                if evt.fired:
                    continue
                should_fire = (
                    (evt.event_type == "一気"    and dist < 400) or
                    (evt.event_type == "まくり"  and dist < 800 and dist > 300) or
                    (evt.event_type == "ロケット" and dist < 700 and dist > 400) or
                    (evt.event_type == "ワープ"  and dist < 600 and dist > 350) or
                    (evt.event_type == "崩壊"    and dist < 250) or
                    (evt.event_type == "内"      and dist < 600 and dist > 100)
                )
                if should_fire:
                    evt.fired = True
                    newly_fired.append(evt)
        return newly_fired

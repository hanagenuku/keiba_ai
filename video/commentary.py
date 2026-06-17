import random

TEMPLATES = {
    600: [
        "{1}先頭！ {2}が追う！",
        "{1}がリードを広げる！",
        "先頭は{1}！",
    ],
    400: [
        "{1}先頭！外から{chase}が迫る！",
        "残り400！{1}まだ粘る！",
        "{1}！{2}！{3}の順！",
    ],
    200: [
        "{chase}が差してきた！",
        "残り200！{1}と{2}の叩き合い！",
        "外から{outside}！届くか！？",
    ],
    100: [
        "{1}！{1}！{1}が抜け出した！！",
        "ゴール前！どの馬か！！",
        "{1}リード！{2}迫る！！",
    ],
}

DRAMA_SHOUTS = {
    "ロケット": ["なんだこの脚！！", "信じられない加速！！", "怪物か！！"],
    "一気":     ["大外から飛んできた！", "大外一気！！", "外から風が来た！！"],
    "ワープ":   ["瞬間移動か！！？", "え！？どこから！？", "今何が起きた！！"],
    "まくり":   ["3コーナーからまくる！！", "豪快にまくり上げる！"],
    "崩壊":     ["失速！先頭が沈む！", "ハイペースが応えた！"],
    "未来改変": ["未来が変わった！？", "予測不能！！"],
}


def generate_commentary(dist: int, top3: list, drama_events: list) -> list[str]:
    lines = []

    for evt in drama_events:
        shouts = DRAMA_SHOUTS.get(evt.event_type, [])
        if shouts:
            lines.append(random.choice(shouts))

    templates = TEMPLATES.get(dist, ["{1}先頭！"])
    template = random.choice(templates)
    names = {str(i + 1): h.name for i, h in enumerate(top3)}
    names["chase"]   = top3[1].name if len(top3) > 1 else ""
    names["outside"] = top3[-1].name
    line = template
    for k, v in names.items():
        line = line.replace("{" + k + "}", v)
    lines.append(line)

    return lines

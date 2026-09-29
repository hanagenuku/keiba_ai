"""公開時点の予想を固定保存する（§18）。

🔴 `latest.json` は当日 refresh で **上書きされる**（2026-09-19/22 に実害が出ている:
遅延発火した refresh が翌日予想を巻き戻した）。したがって「公開後に予想を変更していない」
ことを読者に示すには、公開時点のファイルが **別に永続する** 必要がある。

🔴 既に保存済みの race_id は **上書きしない**（`overwrite=False` が既定）。
これがこの商品の最大の信用材料なので、コードで固定してテストで守る。
結果を追記する場合も予想部分は触らない（`attach_result()` は `result` キーだけを足す）。
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
ARCHIVE_DIRNAME = 'prediction_archive'

# 2: allocation_note / bet_type_reasons を追加（2026-09-29・§10・§11）
SCHEMA_VERSION = 2


class AlreadyPublished(Exception):
    """同じ race_id が既に保存されている。予想の書き換えを防ぐために送出する。"""


class StalePublication(Exception):
    """発走日が過ぎたレースを「公開」として保存しようとした。"""


def assert_publishable(date_str, now=None):
    """対象日が今日（JST）より前なら送出する。

    🔴 このアーカイブの価値は「**発走前に**公開したものがそのまま残っている」ことに
    尽きる。結果が出た後のレースを保存すると `published_at` が発走日より後になり、
    記録そのものが嘘になる（§20）。1件目からそれをやると全部が無意味になるので、
    構造的に止める。

    ⚠ 射程は日付までで、時刻は見ていない。`latest.json` のレースに発走時刻が
    入っていないため（実測: `post_time` は常に None）、「15:45 のレースを 17:00 に
    公開した」は検出できない。**日をまたいだ後追いだけを止める**ガードである。
    """
    now = now or datetime.now(JST)
    today = now.strftime('%Y-%m-%d')
    if str(date_str)[:10] < today:
        raise StalePublication(
            f'対象日 {date_str} は今日（{today}）より前。発走が終わったレースを'
            '公開記録として保存しない（published_at が発走日より後になり記録が嘘になる）。'
            '公開は開催日の発走前に行うこと'
        )


def archive_dir(base_dir, date_str):
    return os.path.join(base_dir, ARCHIVE_DIRNAME, date_str)


def archive_path(base_dir, date_str, race_id):
    safe = str(race_id).replace(os.sep, '_')
    return os.path.join(archive_dir(base_dir, date_str), f'{safe}.json')


def input_fingerprint(race, horses):
    """生成に使った入力の指紋。後から入力が差し替わっていないか確認できる。"""
    payload = {
        'race_id': race.get('race_id'),
        'conf': race.get('conf'),
        'horses': [
            {
                'n': h.get('n'), 'rl_rank': h.get('rl_rank'),
                'solo_rank': h.get('solo_rank'), 'fit_rank': h.get('fit_rank'),
                'pop': h.get('pop'), 'odds': h.get('odds'),
                'fuku_pct': h.get('fuku_pct'),
            }
            for h in horses
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def build_snapshot(race, horses, marks_rows, tickets, dangers,
                   horse_comments, race_text, published_at=None):
    """保存する内容を組み立てる。結果は含めない（公開時点の状態）。"""
    published_at = published_at or datetime.now(JST)
    return {
        'schema_version': SCHEMA_VERSION,
        'published_at': published_at.isoformat(),
        'race_id': race.get('race_id'),
        'venue': race.get('_venue'),
        'race_num': race.get('r'),
        'race_name': race.get('name'),
        'dist': race.get('dist'),
        'conf': race.get('conf'),
        'stars': tickets.get('stars'),
        'verdict': tickets.get('verdict'),
        'skip_reason': tickets.get('skip_reason'),
        'ranks': marks_rows,
        'danger_favorites': dangers,
        'horse_comments': horse_comments,
        'race_comment': race_text,
        'bets': tickets.get('bets'),
        'allocation_ratio': tickets.get('allocation_ratio'),
        'allocation_examples': tickets.get('allocation_examples'),
        # 記事に載せた文面もそのまま残す（§10 各自調整・§11 なぜこの券種か）。
        # 「公開したものを後から変えていない」を示すのが目的なので、
        # 数字だけでなく**読者が読んだ文**を凍結する。
        'allocation_note': tickets.get('allocation_note'),
        'bet_type_reasons': tickets.get('bet_type_reasons'),
        'input_fingerprint': input_fingerprint(race, horses),
        'result': None,
    }


def save_snapshot(base_dir, date_str, snapshot, overwrite=False):
    """保存する。既にあれば AlreadyPublished（上書きしない）。"""
    race_id = snapshot.get('race_id')
    if not race_id:
        raise ValueError('race_id が無いスナップショットは保存できない')
    path = archive_path(base_dir, date_str, race_id)
    if os.path.exists(path) and not overwrite:
        raise AlreadyPublished(
            f'{path} は既に公開済み。予想を後から変更しないため上書きしない。'
            '内容を変えたい場合は公開前にやり直すこと'
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_snapshot(base_dir, date_str, race_id):
    path = archive_path(base_dir, date_str, race_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# 予想部分（公開後に触ってはいけないキー）
_FROZEN_KEYS = (
    'published_at', 'race_id', 'conf', 'stars', 'verdict', 'skip_reason',
    'ranks', 'danger_favorites', 'horse_comments', 'race_comment',
    'bets', 'allocation_ratio', 'allocation_examples',
    'allocation_note', 'bet_type_reasons', 'input_fingerprint',
)


def attach_result(base_dir, date_str, race_id, result):
    """結果を追記する。予想部分は1つも変更しない（§19 外れた予想も消さない）。"""
    snap = load_snapshot(base_dir, date_str, race_id)
    if snap is None:
        raise FileNotFoundError(f'公開済みの予想が見つからない: {race_id}')
    before = {k: snap.get(k) for k in _FROZEN_KEYS}
    snap['result'] = result
    snap['result_attached_at'] = datetime.now(JST).isoformat()
    after = {k: snap.get(k) for k in _FROZEN_KEYS}
    if before != after:
        raise AssertionError('結果の追記で予想部分が変わった。これは起きてはいけない')
    path = archive_path(base_dir, date_str, race_id)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def list_published(base_dir, date_str):
    d = archive_dir(base_dir, date_str)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith('.json'))

"""CLAUDE.md を再び肥大させないための検査（2026-11-09）。

旧 CLAUDE.md は 11,667行・約967KB まで育ち、セッション開始時に全文を読み込めなくなった。
「気をつける」は仕組みではない（DESIGN.md）ので、上限を CI で守る。
経緯は docs/sessions/ に書く（CLAUDE.md §7）。
"""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAUDE_MD = os.path.join(ROOT, 'CLAUDE.md')
HISTORY_DIR = os.path.join(ROOT, 'docs', 'history')

MAX_BYTES = 40 * 1024
# 2026-11-09 に分割した旧 CLAUDE.md の sha256（docs/history/README.md に記載）
OLD_CLAUDE_MD_SHA256_PREFIX = 'c4963323b6f73ca2'


def test_claude_md_under_size_limit():
    size = os.path.getsize(CLAUDE_MD)
    assert size <= MAX_BYTES, (
        f'CLAUDE.md が {size:,} bytes で上限 {MAX_BYTES:,} を超えた。'
        '経緯・実測は docs/sessions/ に移し、CLAUDE.md にはルールと現在地だけを残す（CLAUDE.md §7）。'
    )


def test_history_archive_is_unchanged():
    """docs/history/ は旧 CLAUDE.md の無損失な分割。編集・追記されていないこと。"""
    names = sorted(n for n in os.listdir(HISTORY_DIR) if re.match(r'\d\d_.*\.md$', n))
    assert names, 'docs/history/ に分割ファイルが無い'
    blob = b''.join(open(os.path.join(HISTORY_DIR, n), 'rb').read() for n in names)
    digest = hashlib.sha256(blob).hexdigest()
    assert digest.startswith(OLD_CLAUDE_MD_SHA256_PREFIX), (
        'docs/history/ の分割ファイルが旧 CLAUDE.md と一致しない。'
        'ここは当時の記録なので書き換えない。新しい記録は docs/sessions/ に書く。'
    )


def test_claude_md_links_exist():
    """CLAUDE.md が指しているリポジトリ内のパスが実在すること。"""
    text = open(CLAUDE_MD, encoding='utf-8').read()
    missing = []
    for path in re.findall(r'`((?:docs|biz|src|tests|scripts|data)/[^`\s:]+?)`', text):
        path = path.split('::')[0].rstrip('/')
        if any(ch in path for ch in '*{}<>') or 'YYYY' in path:
            continue
        if not os.path.exists(os.path.join(ROOT, path)):
            missing.append(path)
    assert not missing, f'CLAUDE.md が存在しないパスを指している: {sorted(set(missing))}'

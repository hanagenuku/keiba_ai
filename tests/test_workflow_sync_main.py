"""data/ をコミットするワークフローが、開始時に main の最新へ合わせること。

2026-11-15 のデータ消失:
  18:40 JST に起動した `sunday-results.yml` が当日の結果を書き
  （`shadow_bets` 34行・`race_dividends` 405行）09:50 UTC に push した。
  その**3分前**（09:47 UTC）に cron が作成した `weekend.yml` の refresh は、
  作成時刻の SHA（結果コミットの前）をチェックアウトしており、
  concurrency（group: keiba-weekly）で順番待ちしたあと 10:03 UTC に
  「退避 → `git reset --hard origin/main` → 上書き」で**古い keiba.db を
  押し戻した**。当日の結果が成績ページから消えた（`history.db` は無傷）。

🔑 原因は GitHub Actions の仕様: **チェックアウトは run の作成時刻で固定される**。
   concurrency で待たされるほど、手元の data/ は古くなる。
   `data/*.db` はバイナリで git が自動マージできないため、上書きコミット方式
   （2026-07-24 に conflict 対策として導入）がそのまま取りこぼしになる。

→ 退避する前に `git fetch origin main && git reset --hard origin/main` を1回入れて、
  **作業そのものを main の最新の上で始める**。

⚠ `friday-predict.yml` は使っていない（CLAUDE.md §3）ので対象外。
"""
import os
import re

import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF_DIR = os.path.join(BASE, '.github', 'workflows')
UNUSED = {'friday-predict.yml'}
SYNC_NAME = 'Sync to latest main'


def _workflows_that_overwrite_data():
    """data/ を「退避 → reset --hard → 上書き」でコミットするワークフロー。"""
    out = {}
    for name in sorted(os.listdir(WF_DIR)):
        if not name.endswith('.yml') or name in UNUSED:
            continue
        text = open(os.path.join(WF_DIR, name), encoding='utf-8').read()
        if 'BACKUP_DIR' in text and 'reset --hard origin/main' in text:
            out[name] = text
    return out


def test_there_are_workflows_of_this_shape():
    """この形のワークフローが実在すること（テストが空振りしていないか）。"""
    assert set(_workflows_that_overwrite_data()) >= {
        'weekend.yml', 'sunday-results.yml', 'monthly-retrain.yml'}


def test_every_such_workflow_syncs_to_latest_main_first():
    for name, text in _workflows_that_overwrite_data().items():
        doc = yaml.safe_load(text)
        for job, spec in (doc.get('jobs') or {}).items():
            steps = spec.get('steps') or []
            if not any('BACKUP_DIR' in str(s.get('run') or '') for s in steps):
                continue        # data/ を書かない job（deploy など）は対象外
            names = [s.get('name') for s in steps]
            assert SYNC_NAME in names, f'{name}:{job} に同期ステップが無い'
            sync = steps[names.index(SYNC_NAME)]
            run = sync.get('run') or ''
            assert 'git fetch origin main' in run, f'{name}:{job}'
            assert 'git reset --hard origin/main' in run, f'{name}:{job}'


def test_sync_runs_before_the_data_files_are_read():
    """LFS の実体を取る前に同期すること（古いポインタを引かないため）。"""
    for name, text in _workflows_that_overwrite_data().items():
        doc = yaml.safe_load(text)
        for job, spec in (doc.get('jobs') or {}).items():
            steps = spec.get('steps') or []
            names = [s.get('name') for s in steps]
            if SYNC_NAME not in names:
                continue
            i_sync = names.index(SYNC_NAME)
            i_lfs = next((i for i, s in enumerate(steps)
                          if 'git lfs pull' in str(s.get('run') or '')), None)
            if i_lfs is not None:
                assert i_sync < i_lfs, f'{name}:{job} は LFS 取得より後に同期している'


def test_sync_is_not_placed_after_the_work():
    """同期は作業の前。作業後に reset すると生成物を捨てる。"""
    for name, text in _workflows_that_overwrite_data().items():
        doc = yaml.safe_load(text)
        for job, spec in (doc.get('jobs') or {}).items():
            steps = spec.get('steps') or []
            names = [s.get('name') for s in steps]
            if SYNC_NAME not in names:
                continue
            i_sync = names.index(SYNC_NAME)
            i_backup = next((i for i, s in enumerate(steps)
                             if 'BACKUP_DIR' in str(s.get('run') or '')), None)
            assert i_backup is None or i_sync < i_backup, f'{name}:{job}'


def test_concurrency_group_is_shared():
    """2つのワークフローが同時に走らないこと（直列化の前提を固定する）。"""
    groups = {}
    for name in ('weekend.yml', 'sunday-results.yml', 'monthly-retrain.yml'):
        doc = yaml.safe_load(open(os.path.join(WF_DIR, name), encoding='utf-8').read())
        groups[name] = (doc.get('concurrency') or {}).get('group')
    assert len(set(groups.values())) == 1, groups
    assert all(g for g in groups.values()), groups

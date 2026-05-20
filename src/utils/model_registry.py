"""
モデルのバージョン管理。

models/v{N}/ ディレクトリに各バージョンのファイルをスナップショット保存し、
models/current.json で現行バージョンを管理する。

使い方（Google Colab）:
    from src.utils.model_registry import save_version, list_versions, load_version

    # チューニング後に保存
    save_version(BASE_DIR, note='2026-05-20 Acc@1=23.6%', metrics={'acc1': 0.236, 'ece': 0.0005})

    # バージョン一覧
    list_versions(BASE_DIR)

    # 特定バージョンに戻す
    load_version(BASE_DIR, version=2)
"""
import json
import os
import shutil
from datetime import datetime

# バージョン管理対象のファイル（存在しないファイルはスキップ）
MODEL_FILES = [
    'calibrator.pkl',
    'optimal_weights.json',
    'xgb_fukusho_model.pkl',
    'xgb_feature_cols.json',
    'bet_selector_model.pkl',
    'bet_selector_le.pkl',
    'pace_model.pkl',
]


def _models_dir(base_dir):
    return os.path.join(base_dir, 'models')


def _version_dir(base_dir, version):
    return os.path.join(_models_dir(base_dir), f'v{version}')


def _current_path(base_dir):
    return os.path.join(_models_dir(base_dir), 'current.json')


def get_current_version(base_dir):
    """現行バージョン番号を返す。未設定なら None。"""
    path = _current_path(base_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('version')


def _next_version(base_dir):
    """次のバージョン番号（既存の最大値 + 1）を返す。"""
    models_dir = _models_dir(base_dir)
    if not os.path.exists(models_dir):
        return 1
    existing = [
        int(d[1:]) for d in os.listdir(models_dir)
        if d.startswith('v') and d[1:].isdigit()
    ]
    return max(existing, default=0) + 1


def save_version(base_dir, note='', metrics=None, version=None):
    """data/ の現行モデルファイルを models/v{N}/ にスナップショット保存する。

    Args:
        base_dir : プロジェクトルート
        note     : バージョンの説明（チューニング日・精度など）
        metrics  : 精度指標の辞書（例: {'acc1': 0.236, 'ece': 0.0005}）
        version  : バージョン番号を手動指定（省略時は自動採番）

    Returns:
        保存したバージョン番号（int）
    """
    if version is None:
        version = _next_version(base_dir)

    ver_dir  = _version_dir(base_dir, version)
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(ver_dir, exist_ok=True)

    copied = []
    for fname in MODEL_FILES:
        src = os.path.join(data_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ver_dir, fname))
            copied.append(fname)

    metadata = {
        'version':    version,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'note':       note,
        'metrics':    metrics or {},
        'files':      copied,
    }
    with open(os.path.join(ver_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # current.json を更新
    current = {
        'version':    version,
        'updated_at': metadata['created_at'],
    }
    with open(_current_path(base_dir), 'w', encoding='utf-8') as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    os.makedirs(_models_dir(base_dir), exist_ok=True)
    print(f'✅ v{version} 保存完了: {ver_dir}')
    print(f'   ファイル: {copied}')
    if metrics:
        print(f'   精度: {metrics}')
    return version


def load_version(base_dir, version):
    """models/v{N}/ のファイルを data/ に展開して現行バージョンを切り替える。

    Args:
        base_dir : プロジェクトルート
        version  : 読み込むバージョン番号

    Returns:
        読み込んだバージョンのメタデータ辞書
    """
    ver_dir  = _version_dir(base_dir, version)
    if not os.path.exists(ver_dir):
        raise FileNotFoundError(f'バージョン v{version} が見つかりません: {ver_dir}')

    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    meta_path = os.path.join(ver_dir, 'metadata.json')
    metadata  = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            metadata = json.load(f)

    copied = []
    for fname in MODEL_FILES:
        src = os.path.join(ver_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(data_dir, fname))
            copied.append(fname)

    # current.json を更新
    current = {
        'version':    version,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(_current_path(base_dir), 'w', encoding='utf-8') as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    print(f'✅ v{version} を data/ に展開しました')
    print(f'   ファイル: {copied}')
    if metadata.get('note'):
        print(f'   メモ: {metadata["note"]}')
    if metadata.get('metrics'):
        print(f'   精度: {metadata["metrics"]}')
    return metadata


def rollback(base_dir, version):
    """指定バージョンにロールバックする（load_version のエイリアス）。"""
    print(f'⏪ v{version} にロールバックします...')
    return load_version(base_dir, version)


def list_versions(base_dir):
    """利用可能なバージョン一覧を表示・返却する。

    Returns:
        メタデータ辞書のリスト（バージョン順）
    """
    models_dir = _models_dir(base_dir)
    if not os.path.exists(models_dir):
        print('モデルバージョンがまだ保存されていません。')
        return []

    current = get_current_version(base_dir)
    versions = sorted([
        int(d[1:]) for d in os.listdir(models_dir)
        if d.startswith('v') and d[1:].isdigit()
    ])

    result = []
    for v in versions:
        ver_dir   = _version_dir(base_dir, v)
        meta_path = os.path.join(ver_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        else:
            meta = {'version': v, 'created_at': '?', 'note': '', 'metrics': {}}

        mark = ' ◀ 現行' if v == current else ''
        acc  = meta.get('metrics', {}).get('acc1', '')
        ece  = meta.get('metrics', {}).get('ece', '')
        acc_str = f'  Acc@1={acc:.3f}' if isinstance(acc, float) else ''
        ece_str = f'  ECE={ece:.4f}'   if isinstance(ece, float) else ''
        print(f'  v{v}  {meta.get("created_at","")[:10]}  {meta.get("note","")}{acc_str}{ece_str}{mark}')
        result.append(meta)

    return result

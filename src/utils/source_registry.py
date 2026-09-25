"""情報源の台帳（`data/source_registry.json`）を読み、本番に届く情報源を決める。

なぜ台帳をコードで縛るのか
--------------------------
このプロジェクトは「予想に効くと信じて繋いだ層」を**5回撤回している**
（市場補正レイヤー / error_tags 週次補正 / rank_matrix_filter /
AI xx% バッジ / ROI予測150%）。共通していたのは
**測っていないものが本番の予想に届いていた**こと。

そこでこのモジュールは、台帳に `status='adopted'` と書かれていても
以下が揃っていなければ**採用を無効化して警告する**:

  1. `card`    … 事前登録カードのパスが実在すること
  2. `results` … 結果ファイルのパスが実在すること
  3. `measured.delta_auc_mean` が数値であること
  4. `feature_cols` が空でないこと

＝ 台帳の1フィールドを書き換えるだけでは本番に届かない。
   測定成果物をコミットして初めて届く。

⚠ これは「後付け補正層」ではない。採用された情報源は
  `calc_features_for_xgb` の**上流の特徴量**として入り、学習も推論も
  同じ経路を通る（`src/scraper/going.py` と同じ形）。

⚠ 既定では採用済みの情報源は**ゼロ**。よって既定の挙動は現状と完全に同一。
"""
import json
import os

REGISTRY_REL = os.path.join('data', 'source_registry.json')

# 採用を認めるために台帳が満たすべき条件（上記1〜4）
_REQUIRED_FOR_ADOPTION = ('card', 'results')

_warned = set()


def _warn(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(f'⚠ source_registry: {msg}')


def reset_warnings():
    """テスト用。警告の抑制状態を戻す。"""
    _warned.clear()


def load_registry(base_dir='.'):
    """台帳を読む。無ければ空の台帳を返す（本番を止めない）。"""
    path = os.path.join(base_dir, REGISTRY_REL)
    if not os.path.exists(path):
        _warn('missing', f'{REGISTRY_REL} が無いので情報源ゼロで進む')
        return {'sources': []}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        _warn('broken', f'{REGISTRY_REL} を読めない（{type(e).__name__}）ので情報源ゼロで進む')
        return {'sources': []}


def _adoption_is_backed(entry, base_dir):
    """採用の裏付けが揃っているか。揃っていなければ理由を返す。"""
    src = entry.get('source', '?')
    for key in _REQUIRED_FOR_ADOPTION:
        rel = entry.get(key)
        if not rel:
            return f"{src}: status=adopted だが '{key}' が無い"
        if not os.path.exists(os.path.join(base_dir, rel)):
            return f"{src}: status=adopted だが {key}={rel} が実在しない"
    measured = entry.get('measured') or {}
    delta = measured.get('delta_auc_mean')
    if not isinstance(delta, (int, float)):
        return f'{src}: status=adopted だが measured.delta_auc_mean が数値でない'
    if not entry.get('feature_cols'):
        return f'{src}: status=adopted だが feature_cols が空'
    return None


def adopted_sources(base_dir='.'):
    """本番の特徴量に届く情報源だけを返す。

    裏付けが揃っていない `adopted` は**無効化して警告する**。
    戻り値: [{'source':..., 'feature_cols':[...], ...}, ...]
    """
    out = []
    for entry in load_registry(base_dir).get('sources', []):
        if entry.get('status') != 'adopted':
            continue
        reason = _adoption_is_backed(entry, base_dir)
        if reason:
            _warn(f"unbacked:{entry.get('source')}",
                  f'{reason} → 採用を無効化した（測定成果物をコミットしてから採用する）')
            continue
        out.append(entry)
    return out


def adopted_feature_cols(base_dir='.'):
    """採用済み情報源が作る列名を平たく返す（重複は除く）。"""
    cols = []
    for entry in adopted_sources(base_dir):
        for c in entry.get('feature_cols', []):
            if c not in cols:
                cols.append(c)
    return cols


def source_status(base_dir='.'):
    """表示用。台帳をそのまま要約して返す（測った数字だけを出す）。"""
    reg = load_registry(base_dir)
    rows = []
    for e in reg.get('sources', []):
        m = e.get('measured') or {}
        rows.append({
            'source': e.get('source'),
            'label': e.get('label') or e.get('source'),
            'status': e.get('status'),
            'access': e.get('access'),
            'access_note': e.get('access_note'),
            'prior': e.get('prior'),
            'delta_auc_mean': m.get('delta_auc_mean'),
            'fukusho_roi_arm': m.get('fukusho_roi_arm'),
            'fukusho_roi_base': m.get('fukusho_roi_base'),
            'measured_at': m.get('measured_at'),
            'results': e.get('results'),
            'closed_reason': e.get('closed_reason'),
        })
    return {
        'updated_at': reg.get('updated_at'),
        'network_policy': (reg.get('network_policy') or {}).get('level'),
        'n_adopted': len(adopted_sources(base_dir)),
        'sources': rows,
    }

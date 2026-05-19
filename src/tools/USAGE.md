# tune_weights / calibrate / analyze_divergence 使い方

## Google Colab での実行手順

```python
import sys
sys.path.insert(0, f'{BASE_DIR}/src')

from src.features.engine import init_engine
init_engine(BASE_DIR)

# ── ステップ1: 重みチューニング（30秒〜2分） ──────────────────────
from tools.tune_weights import run_tuning
opt_w = run_tuning(BASE_DIR, n_restarts=5)
# → data/optimal_weights.json に保存

# ── ステップ2: キャリブレーション ──────────────────────────────────
# tune_weights の後に実行すること（optimal_weights.json を使うため）
from tools.calibrate import run_calibration
cal = run_calibration(BASE_DIR, method='isotonic')
# → data/calibrator.pkl に保存

# ── ステップ3: 乖離分析・診断 ──────────────────────────────────────
from tools.analyze_divergence import run_analysis
stats = run_analysis(BASE_DIR)

# ── 以降の予測実行時は init_engine が自動で pkl/json を読み込む ────
# init_engine(BASE_DIR) を再実行するだけでOK
```

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `data/optimal_weights.json` | 最適化された9因子の重み |
| `data/calibrator.pkl`       | `PlattCalibrator` or `IsotonicCalibrator` オブジェクト |

## 推奨実行タイミング

- 初回：今すぐ（1年分のデータがある）
- 以降：月1回程度、または新しいデータが50レース以上追加されたとき

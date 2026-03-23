# Kiln Temperature Time Series Autoresearch — Design Spec

## Overview

Adapt the autoresearch autonomous experiment loop to time series forecasting. Replace the GPT language model with IBM's granite-timeseries-ttm-r2 to predict rotary kiln burning zone temperature (`1KTI09`). The agent iteratively modifies `train_ts.py`, runs experiments, and keeps improvements or discards regressions based on MAE.

## Context

### Data

- **Source:** `data/kiln_data.csv` (copied from `旋窯八點資料20260320_08001600.csv`)
- **Records:** 5755 rows, 5-second sampling interval
- **Time range:** 2026-03-20 08:00 ~ 16:00 (~8 hours)
- **Columns (9 total):**

| Column | Description |
|--------|-------------|
| `t_stamp` | Timestamp |
| `1KTI10 熟料落料溫度` | Clinker drop temperature |
| `1CPI18 窯頭風壓` | Kiln head wind pressure |
| `1K001_ON 窯主馬達運轉` | Kiln main motor operation |
| `1KII01 旋窯主馬達A電流` | Kiln main motor A current |
| `1KII02 旋窯主馬達B電流` | Kiln main motor B current |
| `1KJI01 旋窯主馬達A功率` | Kiln main motor A power |
| `1KSI01 窯主馬達A轉速` | Kiln main motor A speed |
| `1KTI09 燒成帶溫度` | **Burning zone temperature (TARGET)** |

- **Target stats:** min=799.38, max=801.21, mean=799.77 (very stable, ~1.8 deg C range)
- **No missing values**

### Model

- **granite-timeseries-ttm-r2** (~805K parameters)
- Installed via `pip3 install granite-tsfm` (already installed in system Python 3.9)
- Supports zero-shot, fine-tuning, channel-independent/mix, exogenous variables, rolling forecasts
- Mandatory requirement: per-channel standard scaling of input data
- Runs on CPU (no GPU required)

### Execution Environment

- **Platform:** macOS (darwin)
- **Python:** System Python 3.9 via `python3` (not `uv run` — the time series scripts are independent from the original autoresearch `uv` environment which uses CUDA-pinned torch)
- **Run command:** `python3 train_ts.py` (not `uv run train_ts.py`)
- `granite-tsfm` and its dependencies (including its own `torch<2.9`) are installed in the user site-packages (`/Users/timmy/Library/Python/3.9/`)

### Prediction Goal

- **Short-horizon forecasting:** given recent sensor history, predict `1KTI09` a few minutes ahead
- **Metric:** MAE in degrees Celsius (lower is better)
- **Use case:** real-time monitoring and early warning for kiln operators

## File Structure

```
autoresearch/
├── prepare.py          # Original LLM (untouched)
├── train.py            # Original LLM (untouched)
├── program.md          # Original LLM (untouched)
├── prepare_ts.py       # NEW, READ-ONLY: data loading, splitting, scaling, evaluation
├── train_ts.py         # NEW, AGENT EDITS: model loading, fine-tuning, prediction
├── program_ts.md       # NEW, HUMAN EDITS: agent experiment loop instructions
├── data/
│   ├── kiln_data.csv           # Raw kiln sensor data
│   └── best_predictions/       # Saved prediction arrays (not in git)
├── prediction_chart.png        # Current experiment chart (overwritten each run)
├── prediction_best.png         # Top-N best results comparison chart
└── results_ts.tsv              # Experiment log (not in git)
```

## Component Design

### `prepare_ts.py` (READ-ONLY)

Agent must not modify this file. It defines the fixed evaluation harness.

**Constants:**

```python
TARGET_COL = "1KTI09 燒成帶溫度"
TRAIN_RATIO = 0.8              # First 80% for training, last 20% for validation
DATA_PATH = "data/kiln_data.csv"
```

**Functions:**

```python
load_data(path=DATA_PATH) -> pd.DataFrame
    # Read CSV, parse t_stamp as DatetimeIndex
    # Returns DataFrame with all sensor columns

train_val_split(df, train_ratio=TRAIN_RATIO) -> (pd.DataFrame, pd.DataFrame)
    # Chronological split (no shuffling — time series must preserve order)
    # Returns (train_df, val_df)

standard_scale(train_df, val_df) -> (pd.DataFrame, pd.DataFrame, dict)
    # Per-channel standard scaling using train statistics only
    # Returns (scaled_train, scaled_val, scaler_params)
    # scaler_params = {col: {"mean": float, "std": float}}

evaluate_mae(predictions, actuals, scaler_params, target_col=TARGET_COL) -> float
    # predictions: 1-D numpy array of predicted target values (scaled)
    # actuals: 1-D numpy array of actual target values (scaled)
    # Both arrays have the same length (number of evaluation points)
    # Inverse-scale both back to original temperature using scaler_params
    # Compute MAE on target column only
    # Returns MAE in degrees Celsius

make_sliding_windows(scaled_df, context_length, prediction_length, target_col) -> list
    # Create sliding window samples from a scaled DataFrame
    # Stride = prediction_length (non-overlapping prediction windows)
    # Returns list of (input_window, target_window) tuples
    # input_window: DataFrame of shape (context_length, n_features)
    # target_window: 1-D array of shape (prediction_length,) for target col only
    # Discards the last incomplete window

save_best_prediction(commit_hash, predictions, timestamps)
    # Save prediction array to data/best_predictions/{commit_hash}.npy
    # Also save timestamps as {commit_hash}_timestamps.npy

load_best_predictions(top_n=3) -> list
    # Read results_ts.tsv, find top-N best by val_mae (ascending = best first)
    # Load corresponding .npy files from data/best_predictions/
    # Returns [(commit, mae, description, predictions, timestamps), ...]
    # Sorted: index 0 = overall best (lowest MAE), index -1 = Nth best
```

**Design decisions:**
- Train/val split is chronological to prevent data leakage
- Standard scaling is mandatory per granite-TTM-R2 documentation
- MAE is computed after inverse scaling so the unit is intuitive (degrees C)
- `evaluate_mae()` is the single ground truth metric (like `evaluate_bpb()` in original)

### `train_ts.py` (AGENT EDITS)

This is the only file the agent modifies. All experimentation freedom lives here.

**Baseline structure:**

```python
# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these)
# ---------------------------------------------------------------------------
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
FEATURE_COLS = [                        # All 7 non-target sensor columns:
    "1KTI10 熟料落料溫度",                #   Clinker drop temperature
    "1CPI18 窯頭風壓",                    #   Kiln head wind pressure
    "1K001_ON 窯主馬達運轉",              #   Kiln main motor operation
    "1KII01 旋窯主馬達A電流",             #   Motor A current
    "1KII02 旋窯主馬達B電流",             #   Motor B current
    "1KJI01 旋窯主馬達A功率",             #   Motor A power
    "1KSI01 窯主馬達A轉速",              #   Motor A speed
]
TARGET_COL = "1KTI09 燒成帶溫度"
RESAMPLE_INTERVAL = None        # None=original 5s, or "15s"/"30s"/"1min"
CHANNEL_MODE = "independent"    # "independent" or "mix"
FINETUNE_EPOCHS = 5
FINETUNE_LR = 0.001
FINETUNE_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Data preparation (imports from prepare_ts.py)
# ---------------------------------------------------------------------------
from prepare_ts import (load_data, train_val_split, standard_scale,
                         evaluate_mae, save_best_prediction,
                         make_sliding_windows)

df = load_data()
if RESAMPLE_INTERVAL:
    df = df.resample(RESAMPLE_INTERVAL).mean()
train_df, val_df = train_val_split(df)
scaled_train, scaled_val, scaler_params = standard_scale(train_df, val_df)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
from tsfm_public.toolkit.get_model import get_model
model = get_model(
    model_path=MODEL_PATH,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
)

# ---------------------------------------------------------------------------
# Fine-tuning (agent can change strategy)
# ---------------------------------------------------------------------------
# Fine-tune on scaled_train...

# ---------------------------------------------------------------------------
# Prediction & evaluation
# ---------------------------------------------------------------------------
# Sliding window prediction on scaled_val
# val_mae = evaluate_mae(predictions, actuals, scaler_params)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Chart 1: Current experiment (actual vs predicted)
plt.figure(figsize=(14, 5))
plt.plot(val_timestamps, actual_temps, label="Actual", color="black", linewidth=1)
plt.plot(val_timestamps, predicted_temps, label="Predicted", color="red", linewidth=1, alpha=0.8)
plt.xlabel("Time")
plt.ylabel("1KTI09 (deg C)")
plt.title(f"Kiln Temperature Prediction (MAE={val_mae:.4f} deg C)")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_chart.png", dpi=150)
plt.close()

# Chart 2: Best results comparison (top-3 historical bests)
from prepare_ts import load_best_predictions
bests = load_best_predictions(top_n=3)
if bests:
    plt.figure(figsize=(14, 5))
    plt.plot(val_timestamps, actual_temps, label="Actual", color="black", linewidth=1.5)
    colors = ["blue", "green", "orange"]
    # bests[0] = overall best (lowest MAE), highlight it in red
    commit, mae, desc, preds, ts = bests[0]
    plt.plot(ts, preds, label=f"Best: {desc} (MAE={mae:.4f})",
             color="red", linewidth=1.5)
    for i, (commit, mae, desc, preds, ts) in enumerate(bests[1:]):
        plt.plot(ts, preds, label=f"#{i+2} {desc} (MAE={mae:.4f})",
                 color=colors[i], linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("1KTI09 (deg C)")
    plt.title("Best Experiments Comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("prediction_best.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Output (fixed format for loop parsing)
# ---------------------------------------------------------------------------
print("---")
print(f"val_mae:           {val_mae:.6f}")
print(f"prediction_length: {PREDICTION_LENGTH}")
print(f"context_length:    {CONTEXT_LENGTH}")
print(f"resample:          {RESAMPLE_INTERVAL or '5s'}")
print(f"features:          {len(FEATURE_COLS)}")
print(f"channel_mode:      {CHANNEL_MODE}")
print(f"finetune_epochs:   {FINETUNE_EPOCHS}")
print(f"chart_saved:       prediction_chart.png")
```

**Agent's full freedom includes:**
- Hyperparameters (context/prediction length, LR, epochs, batch size)
- Feature engineering (sensor selection, derived features like diff/rolling mean)
- Resampling strategy (5s -> 15s -> 30s -> 1min)
- Model variant (different TTM branches, `get_model` parameters)
- Channel mode (independent vs mix)
- Fine-tuning strategy (optimizer, scheduler, zero-shot vs fine-tuned)
- Rolling forecast strategy

**Fixed constraints:**
- Output must include `val_mae:` line (loop parses via grep)
- Must call `evaluate_mae()` from `prepare_ts.py` as the ground truth metric
- Must generate `prediction_chart.png`

### `program_ts.md` (HUMAN EDITS)

Adapted from original `program.md` with these changes:

**Setup:**
1. Agree on run tag (e.g. `ts-mar23`), create branch `autoresearch-ts/<tag>`
2. Read `prepare_ts.py`, `train_ts.py`, `program_ts.md`
3. Verify `data/kiln_data.csv` exists
4. Initialize `results_ts.tsv` with header
5. Run baseline (unmodified `train_ts.py`)

**Experiment loop:**
```
LOOP FOREVER:
1. Check git state
2. Modify train_ts.py with experimental idea
3. git commit
4. Run: timeout 300 python3 train_ts.py > run_ts.log 2>&1
5. Parse: grep "^val_mae:" run_ts.log
6. Empty grep → crash, tail -n 50 run_ts.log for error
7. Log to results_ts.tsv
8. MAE improved (strictly lower) → keep, save best prediction
9. MAE equal → keep ONLY if the code is simpler (fewer lines / less complexity), otherwise discard
10. MAE worse → git reset
```

**results_ts.tsv format (tab-separated):**
```
commit	val_mae	status	description
a1b2c3d	0.152300	keep	baseline (512/96, all features, independent)
b2c3d4e	0.138700	keep	resample to 15s, context 1024
c3d4e5f	0.155000	discard	channel-mix mode
d4e5f6g	0.000000	crash	prediction_length=384 error
```

**Key differences from original program.md:**
- Metric: `val_mae` (lower is better) instead of `val_bpb`
- No fixed time budget (run completes when done, typically < 2 min)
- Timeout enforced via `timeout 300` command wrapper; exit code 124 = timeout = treat as crash
- Run command: `python3 train_ts.py` (not `uv run` — separate from LLM environment)
- No VRAM tracking (CPU-compatible model)
- Simplified TSV (no `memory_gb` column)
- On keep: also call `save_best_prediction()` and regenerate `prediction_best.png`

**Preserved principles:**
- Agent only edits `train_ts.py`, never `prepare_ts.py`
- No new package installations
- Simplicity criterion (equal MAE + simpler code = win)
- Never stop until human interrupts

## Items Not in Git

The following are generated at runtime and should not be committed:
- `results_ts.tsv`
- `data/best_predictions/` directory and contents
- `prediction_chart.png`
- `prediction_best.png`
- `run_ts.log`

## Dependencies

Time series scripts run under system Python 3.9 (not the `uv` virtualenv). All packages are installed in user site-packages:

- `granite-tsfm` (installed via `pip3 install granite-tsfm`, brings its own `torch<2.9`)
- `matplotlib` (dependency of granite-tsfm or install separately: `pip3 install matplotlib`)
- `pandas` (dependency of granite-tsfm)
- `numpy` (dependency of granite-tsfm)

**Note:** This is intentionally separate from the `uv` environment (which pins `torch==2.9.1` with CUDA 12.8 for the original LLM autoresearch). The two environments do not conflict.

## .gitignore Additions

Add these patterns to `.gitignore`:
```
results_ts.tsv
data/best_predictions/
prediction_chart.png
prediction_best.png
run_ts.log
```

## Future Considerations (Out of Scope for MVP)

- Integrate more historical data (multi-day/multi-week)
- Real-time data pipeline from SCADA/DCS
- Multi-target prediction (predict multiple sensor values simultaneously)
- Anomaly detection mode
- Automated report generation

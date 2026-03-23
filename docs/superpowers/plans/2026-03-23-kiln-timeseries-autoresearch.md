# Kiln Temperature Time Series Autoresearch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous experiment loop that uses granite-timeseries-ttm-r2 to predict rotary kiln burning zone temperature, iteratively improving MAE through agent-driven experimentation.

**Architecture:** Three-file structure mirroring the original autoresearch: `prepare_ts.py` (read-only evaluation harness), `train_ts.py` (agent-editable experiment script), `program_ts.md` (agent loop instructions). Scripts run under system Python 3.9 (not the `uv` environment).

**Tech Stack:** granite-tsfm (TinyTimeMixer), torch 2.2.2, pandas, numpy, matplotlib — all under system Python 3.9

**Spec:** `docs/superpowers/specs/2026-03-23-kiln-timeseries-autoresearch-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `prepare_ts.py` | Create | Data loading, train/val split, standard scaling, MAE evaluation, sliding windows, best-prediction I/O |
| `train_ts.py` | Create | Baseline experiment: load model, fine-tune, predict, chart, print results |
| `program_ts.md` | Create | Agent experiment loop instructions |
| `data/kiln_data.csv` | Create (copy) | Raw sensor data copied from Chinese-named CSV |
| `.gitignore` | Modify | Add time-series artifacts |
| `tests/test_prepare_ts.py` | Create | Tests for prepare_ts.py functions |

---

### Task 1: Fix NumPy compatibility and set up data directory

**Context:** `granite-tsfm` installed `numpy==2.0.2` but `torch==2.2.2` was compiled against NumPy 1.x, causing runtime warnings. Also need to copy the CSV into `data/` and update `.gitignore`.

**Files:**
- Modify: `.gitignore`
- Create: `data/kiln_data.csv` (copy from `旋窯八點資料20260320_08001600.csv`)

- [ ] **Step 1: Downgrade numpy to compatible version**

```bash
pip3 install --user 'numpy<2'
```

Expected: Successfully installed numpy-1.x.x

- [ ] **Step 2: Verify imports work cleanly**

```bash
python3 -c "import warnings; warnings.filterwarnings('ignore'); from tsfm_public.toolkit.get_model import get_model; print('OK')" 2>&1 | grep -E 'OK|Error|cannot'
```

Expected: `OK` with no `_ARRAY_API` errors

- [ ] **Step 3: Create data directory and copy CSV**

```bash
mkdir -p data
cp "旋窯八點資料20260320_08001600.csv" data/kiln_data.csv
```

- [ ] **Step 4: Update .gitignore**

Append to `.gitignore`:

```
# Time series autoresearch artifacts
results_ts.tsv
data/best_predictions/
prediction_chart.png
prediction_best.png
run_ts.log
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore data/kiln_data.csv
git commit -m "feat: add kiln data and gitignore for time series autoresearch"
```

---

### Task 2: Write `prepare_ts.py` — data loading, splitting, scaling

**Context:** This is the read-only evaluation harness. It provides `load_data()`, `train_val_split()`, `standard_scale()` — the fixed data pipeline that `train_ts.py` imports.

**Files:**
- Create: `prepare_ts.py`
- Create: `tests/test_prepare_ts.py`

- [ ] **Step 1: Create tests directory and write test for `load_data`**

```bash
mkdir -p tests
```

Create `tests/test_prepare_ts.py`:

```python
"""Tests for prepare_ts.py data utilities."""
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_load_data():
    from prepare_ts import load_data
    df = load_data()
    # Should have DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    # Should have 8 sensor columns (no t_stamp -- it is the index)
    assert len(df.columns) == 8
    # Should have 5755 rows
    assert len(df) == 5755
    # Target column must exist
    assert "1KTI09 燒成帶溫度" in df.columns
    # No NaN values
    assert df.isna().sum().sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_load_data -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_ts'`

- [ ] **Step 3: Implement `load_data` in `prepare_ts.py`**

Create `prepare_ts.py`:

```python
"""
Fixed evaluation harness for kiln temperature time series autoresearch.
Data loading, splitting, scaling, evaluation, and prediction I/O.

This file is READ-ONLY for the agent. Do not modify.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TARGET_COL = "1KTI09 燒成帶溫度"
TRAIN_RATIO = 0.8
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "kiln_data.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_ts.tsv")
BEST_PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "best_predictions")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH):
    """Read CSV, parse t_stamp as DatetimeIndex. Returns DataFrame with all sensor columns."""
    df = pd.read_csv(path, parse_dates=["t_stamp"], index_col="t_stamp")
    df = df.sort_index()  # Ensure chronological order
    return df
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_load_data -v
```

Expected: PASS

- [ ] **Step 5: Write test for `train_val_split`**

Add to `tests/test_prepare_ts.py`:

```python
def test_train_val_split():
    from prepare_ts import load_data, train_val_split, TRAIN_RATIO
    df = load_data()
    train_df, val_df = train_val_split(df)
    total = len(df)
    expected_train = int(total * TRAIN_RATIO)
    # Sizes match
    assert len(train_df) == expected_train
    assert len(val_df) == total - expected_train
    # Chronological: train ends before val starts
    assert train_df.index[-1] < val_df.index[0]
    # No overlap
    assert len(set(train_df.index) & set(val_df.index)) == 0
```

- [ ] **Step 6: Run test to verify it fails**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_train_val_split -v
```

Expected: FAIL with `ImportError: cannot import name 'train_val_split'`

- [ ] **Step 7: Implement `train_val_split`**

Add to `prepare_ts.py`:

```python
# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def train_val_split(df, train_ratio=TRAIN_RATIO):
    """Chronological split. No shuffling -- time series must preserve order."""
    n = int(len(df) * train_ratio)
    return df.iloc[:n].copy(), df.iloc[n:].copy()
```

- [ ] **Step 8: Run test to verify it passes**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_train_val_split -v
```

Expected: PASS

- [ ] **Step 9: Write test for `standard_scale`**

Add to `tests/test_prepare_ts.py`:

```python
def test_standard_scale():
    from prepare_ts import load_data, train_val_split, standard_scale, TARGET_COL
    df = load_data()
    train_df, val_df = train_val_split(df)
    scaled_train, scaled_val, scaler_params = standard_scale(train_df, val_df)

    # Scaled train should have mean ~0, std ~1 for each column
    for col in scaled_train.columns:
        assert abs(scaled_train[col].mean()) < 0.01, f"{col} mean not ~0"
        assert abs(scaled_train[col].std() - 1.0) < 0.05, f"{col} std not ~1"

    # scaler_params should have entry for each column
    assert TARGET_COL in scaler_params
    for col in train_df.columns:
        assert "mean" in scaler_params[col]
        assert "std" in scaler_params[col]

    # Shapes preserved
    assert scaled_train.shape == train_df.shape
    assert scaled_val.shape == val_df.shape
```

- [ ] **Step 10: Write test for constant-column edge case**

Add to `tests/test_prepare_ts.py`:

```python
def test_standard_scale_constant_column():
    """Constant columns (std=0) should not cause division by zero."""
    from prepare_ts import standard_scale
    idx = pd.date_range("2026-01-01", periods=10, freq="5s")
    train = pd.DataFrame({"const": [1.0]*10, "vary": range(10)}, index=idx)
    val = pd.DataFrame({"const": [1.0]*5, "vary": range(5)}, index=idx[:5])
    scaled_train, scaled_val, params = standard_scale(train, val)
    # Constant column should have std=1.0 (fallback) and all zeros after scaling
    assert params["const"]["std"] == 1.0
    assert (scaled_train["const"] == 0.0).all()
    # Non-constant column should scale normally
    assert abs(scaled_train["vary"].mean()) < 0.01
```

- [ ] **Step 11: Run test to verify it fails, then implement**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_standard_scale -v
```

Add to `prepare_ts.py`:

```python
# ---------------------------------------------------------------------------
# Standard scaling
# ---------------------------------------------------------------------------

def standard_scale(train_df, val_df):
    """Per-channel standard scaling using train statistics only.
    Returns (scaled_train, scaled_val, scaler_params).
    scaler_params = {col: {"mean": float, "std": float}}
    """
    scaler_params = {}
    scaled_train = train_df.copy()
    scaled_val = val_df.copy()
    for col in train_df.columns:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if std == 0:
            std = 1.0  # Avoid division by zero for constant columns
        scaler_params[col] = {"mean": float(mean), "std": float(std)}
        scaled_train[col] = (train_df[col] - mean) / std
        scaled_val[col] = (val_df[col] - mean) / std
    return scaled_train, scaled_val, scaler_params
```

- [ ] **Step 12: Run all tests so far**

```bash
python3 -m pytest tests/test_prepare_ts.py -v
```

Expected: 4 tests PASS (load_data, train_val_split, standard_scale, standard_scale_constant_column)

- [ ] **Step 13: Commit**

```bash
git add prepare_ts.py tests/test_prepare_ts.py
git commit -m "feat: add prepare_ts.py with load_data, train_val_split, standard_scale"
```

---

### Task 3: Write `prepare_ts.py` — evaluation and sliding windows

**Context:** Add `evaluate_mae()` and `make_sliding_windows()` — the core evaluation harness functions.

**Files:**
- Modify: `prepare_ts.py`
- Modify: `tests/test_prepare_ts.py`

- [ ] **Step 1: Write test for `evaluate_mae`**

Add to `tests/test_prepare_ts.py`:

```python
def test_evaluate_mae():
    from prepare_ts import evaluate_mae, TARGET_COL
    # Simulate scaled predictions and actuals
    # If scaler mean=800, std=0.5, then scaled value 0.0 = 800 deg C, 2.0 = 801 deg C
    scaler_params = {TARGET_COL: {"mean": 800.0, "std": 0.5}}
    predictions = np.array([0.0, 0.0, 0.0])  # All predict 800 deg C
    actuals = np.array([0.0, 2.0, -2.0])     # 800, 801, 799
    mae = evaluate_mae(predictions, actuals, scaler_params)
    # After inverse scaling: preds=[800,800,800], actuals=[800,801,799]
    # MAE = (0 + 1 + 1) / 3 = 0.6667
    assert abs(mae - 0.6667) < 0.001
```

- [ ] **Step 2: Run test to verify it fails, then implement**

Add to `prepare_ts.py`:

```python
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_mae(predictions, actuals, scaler_params, target_col=TARGET_COL):
    """Inverse-scale predictions and actuals, compute MAE in degrees Celsius.
    predictions: 1-D numpy array of predicted target values (scaled)
    actuals: 1-D numpy array of actual target values (scaled)
    Returns MAE in degrees Celsius.
    """
    mean = scaler_params[target_col]["mean"]
    std = scaler_params[target_col]["std"]
    preds_orig = predictions * std + mean
    actuals_orig = actuals * std + mean
    return float(np.mean(np.abs(preds_orig - actuals_orig)))
```

- [ ] **Step 3: Run test to verify it passes**

```bash
python3 -m pytest tests/test_prepare_ts.py::test_evaluate_mae -v
```

Expected: PASS

- [ ] **Step 4: Write test for `make_sliding_windows`**

Add to `tests/test_prepare_ts.py`:

```python
def test_make_sliding_windows():
    from prepare_ts import make_sliding_windows, TARGET_COL
    # Create a small synthetic DataFrame
    n = 100
    idx = pd.date_range("2026-01-01", periods=n, freq="5s")
    df = pd.DataFrame({
        "feature_a": np.arange(n, dtype=float),
        TARGET_COL: np.arange(n, dtype=float) * 0.1,
    }, index=idx)

    context_length = 20
    prediction_length = 10
    windows = make_sliding_windows(df, context_length, prediction_length, TARGET_COL)

    # With n=100, context=20, pred=10, stride=pred=10:
    # Number of windows = (100 - 20) // 10 = 8
    assert len(windows) == 8

    # Check shapes
    inp, tgt = windows[0]
    assert inp.shape == (context_length, 2)  # 2 columns
    assert tgt.shape == (prediction_length,)

    # Check first window values
    assert abs(tgt[0] - 2.0) < 0.001  # index 20 * 0.1
    assert abs(tgt[-1] - 2.9) < 0.001  # index 29 * 0.1
```

- [ ] **Step 5: Run test to verify it fails, then implement**

Add to `prepare_ts.py`:

```python
# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

def make_sliding_windows(scaled_df, context_length, prediction_length, target_col=TARGET_COL):
    """Create sliding window samples from a scaled DataFrame.
    Stride = prediction_length (non-overlapping prediction windows).
    Returns list of (input_window, target_window) tuples.
    input_window: numpy array of shape (context_length, n_features)
    target_window: 1-D numpy array of shape (prediction_length,) for target col only
    Discards the last incomplete window.
    """
    data = scaled_df.values  # (n_rows, n_cols)
    target_idx = list(scaled_df.columns).index(target_col)
    n = len(data)
    windows = []
    start = 0
    while start + context_length + prediction_length <= n:
        inp = data[start:start + context_length]
        tgt = data[start + context_length:start + context_length + prediction_length, target_idx]
        windows.append((inp, tgt))
        start += prediction_length
    return windows
```

- [ ] **Step 6: Run all tests**

```bash
python3 -m pytest tests/test_prepare_ts.py -v
```

Expected: 5 tests PASS

- [ ] **Step 7: Commit**

```bash
git add prepare_ts.py tests/test_prepare_ts.py
git commit -m "feat: add evaluate_mae and make_sliding_windows to prepare_ts.py"
```

---

### Task 4: Write `prepare_ts.py` — best prediction I/O

**Context:** Add `save_best_prediction()` and `load_best_predictions()` for the comparison chart feature.

**Files:**
- Modify: `prepare_ts.py`
- Modify: `tests/test_prepare_ts.py`

- [ ] **Step 1: Write test for `save_best_prediction` and `load_best_predictions`**

Add to `tests/test_prepare_ts.py`:

```python
def test_save_and_load_best_predictions(tmp_path):
    from prepare_ts import save_best_prediction, load_best_predictions

    # Create fake results_ts.tsv
    results_path = tmp_path / "results_ts.tsv"
    results_path.write_text(
        "commit\tval_mae\tstatus\tdescription\n"
        "abc1234\t0.150000\tkeep\tbaseline\n"
        "def5678\t0.120000\tkeep\tresample 15s\n"
        "ghi9012\t0.200000\tdiscard\tbad idea\n"
    )

    # Create fake predictions
    best_dir = tmp_path / "best_predictions"
    timestamps_a = np.array([1.0, 2.0, 3.0])
    preds_a = np.array([800.1, 800.2, 800.3])
    timestamps_b = np.array([1.0, 2.0, 3.0])
    preds_b = np.array([800.0, 800.1, 800.2])

    save_best_prediction("abc1234", preds_a, timestamps_a,
                         best_dir=str(best_dir))
    save_best_prediction("def5678", preds_b, timestamps_b,
                         best_dir=str(best_dir))

    # Load top 2
    bests = load_best_predictions(
        top_n=2,
        results_path=str(results_path),
        best_dir=str(best_dir),
    )

    # Should be sorted by MAE ascending
    assert len(bests) == 2
    assert bests[0][0] == "def5678"  # lowest MAE
    assert bests[0][1] == 0.120000
    assert bests[1][0] == "abc1234"
    assert np.array_equal(bests[0][3], preds_b)
```

- [ ] **Step 2: Run test to verify it fails, then implement**

Add to `prepare_ts.py`:

```python
# ---------------------------------------------------------------------------
# Best prediction I/O
# ---------------------------------------------------------------------------

def save_best_prediction(commit_hash, predictions, timestamps, best_dir=BEST_PREDICTIONS_DIR):
    """Save prediction array and timestamps to best_predictions directory."""
    os.makedirs(best_dir, exist_ok=True)
    np.save(os.path.join(best_dir, f"{commit_hash}.npy"), predictions)
    np.save(os.path.join(best_dir, f"{commit_hash}_timestamps.npy"), timestamps)


def load_best_predictions(top_n=3, results_path=RESULTS_PATH, best_dir=BEST_PREDICTIONS_DIR):
    """Read results_ts.tsv, find top-N best by val_mae (ascending).
    Load corresponding .npy files from best_predictions.
    Returns [(commit, mae, description, predictions, timestamps), ...]
    Sorted: index 0 = overall best (lowest MAE).
    """
    if not os.path.exists(results_path):
        return []
    df = pd.read_csv(results_path, sep="\t")
    # Only keep rows with status=keep and valid MAE
    keeps = df[(df["status"] == "keep") & (df["val_mae"] > 0)].copy()
    if keeps.empty:
        return []
    keeps = keeps.sort_values("val_mae").head(top_n)

    results = []
    for _, row in keeps.iterrows():
        commit = row["commit"]
        pred_path = os.path.join(best_dir, f"{commit}.npy")
        ts_path = os.path.join(best_dir, f"{commit}_timestamps.npy")
        if os.path.exists(pred_path) and os.path.exists(ts_path):
            preds = np.load(pred_path, allow_pickle=True)
            timestamps = np.load(ts_path, allow_pickle=True)
            results.append((commit, row["val_mae"], row["description"], preds, timestamps))
    return results
```

- [ ] **Step 3: Run all tests**

```bash
python3 -m pytest tests/test_prepare_ts.py -v
```

Expected: 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add prepare_ts.py tests/test_prepare_ts.py
git commit -m "feat: add save/load best predictions to prepare_ts.py"
```

---

### Task 5: Write `train_ts.py` — baseline experiment script

**Context:** This is the agent-editable file. The baseline version loads granite-TTM-R2, runs zero-shot prediction on validation data, evaluates MAE, and outputs results + charts. Fine-tuning is included but the baseline establishes the zero-shot performance first.

**Important:** The `TinyTimeMixerForPrediction.forward()` expects:
- `past_values`: tensor of shape `(batch, context_length, num_channels)`
- `future_values` (optional, for training): tensor of shape `(batch, prediction_length, num_channels)`

**Files:**
- Create: `train_ts.py`

- [ ] **Step 1: Write the complete baseline `train_ts.py`**

```python
"""
Kiln temperature time series autoresearch experiment script.
Single-file, agent-editable. Run with: python3 train_ts.py

This is the ONLY file the agent modifies. Everything is fair game:
model variant, hyperparameters, feature selection, resampling, etc.
"""

import time
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these)
# ---------------------------------------------------------------------------

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
FEATURE_COLS = [
    "1KTI10 熟料落料溫度",
    "1CPI18 窯頭風壓",
    "1K001_ON 窯主馬達運轉",
    "1KII01 旋窯主馬達A電流",
    "1KII02 旋窯主馬達B電流",
    "1KJI01 旋窯主馬達A功率",
    "1KSI01 窯主馬達A轉速",
]
TARGET_COL = "1KTI09 燒成帶溫度"
RESAMPLE_INTERVAL = None        # None=original 5s, or "15s"/"30s"/"1min"
CHANNEL_MODE = "independent"    # "independent" or "mix"
FINETUNE_EPOCHS = 0             # 0 = zero-shot baseline
FINETUNE_LR = 0.001
FINETUNE_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

from prepare_ts import (load_data, train_val_split, standard_scale,
                         evaluate_mae, make_sliding_windows,
                         save_best_prediction, load_best_predictions)

t_start = time.time()

df = load_data()
if RESAMPLE_INTERVAL:
    df = df.resample(RESAMPLE_INTERVAL).mean()

# Select features + target
all_cols = FEATURE_COLS + [TARGET_COL]
df = df[all_cols]

train_df, val_df = train_val_split(df)
scaled_train, scaled_val, scaler_params = standard_scale(train_df, val_df)

print(f"Train: {len(scaled_train)} rows, Val: {len(scaled_val)} rows")
print(f"Features: {len(FEATURE_COLS)}, Target: {TARGET_COL}")
print(f"Context: {CONTEXT_LENGTH}, Prediction: {PREDICTION_LENGTH}")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

from tsfm_public.toolkit.get_model import get_model

print(f"Loading model from {MODEL_PATH}...")
model = get_model(
    model_path=MODEL_PATH,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
)
model.config.num_input_channels = len(all_cols)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Fine-tuning (agent can change strategy)
# ---------------------------------------------------------------------------

if FINETUNE_EPOCHS > 0:
    from torch.optim import Adam

    model.train()
    optimizer = Adam(model.parameters(), lr=FINETUNE_LR)
    train_windows = make_sliding_windows(scaled_train, CONTEXT_LENGTH,
                                          PREDICTION_LENGTH, TARGET_COL)
    n_channels = len(all_cols)

    print(f"Fine-tuning for {FINETUNE_EPOCHS} epochs on {len(train_windows)} windows...")
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_windows), FINETUNE_BATCH_SIZE):
            batch_windows = train_windows[i:i + FINETUNE_BATCH_SIZE]
            past = torch.tensor(
                np.stack([w[0] for w in batch_windows]),
                dtype=torch.float32,
            )
            # Build future values from the scaled validation data
            # For now, use only target channel repeated
            target_idx = all_cols.index(TARGET_COL)
            future_all = []
            for w_inp, w_tgt in batch_windows:
                future = np.zeros((PREDICTION_LENGTH, n_channels))
                future[:, target_idx] = w_tgt
                future_all.append(future)
            future = torch.tensor(np.stack(future_all), dtype=torch.float32)

            optimizer.zero_grad()
            output = model(past_values=past, future_values=future)
            loss = output.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS} -- loss: {avg_loss:.6f}")

    print("Fine-tuning complete.")

# ---------------------------------------------------------------------------
# Prediction and evaluation
# ---------------------------------------------------------------------------

model.eval()
val_windows = make_sliding_windows(scaled_val, CONTEXT_LENGTH,
                                    PREDICTION_LENGTH, TARGET_COL)
print(f"Evaluating on {len(val_windows)} validation windows...")

target_idx = all_cols.index(TARGET_COL)
all_predictions = []
all_actuals = []

with torch.no_grad():
    for i in range(0, len(val_windows), FINETUNE_BATCH_SIZE):
        batch_windows = val_windows[i:i + FINETUNE_BATCH_SIZE]
        past = torch.tensor(
            np.stack([w[0] for w in batch_windows]),
            dtype=torch.float32,
        )
        output = model(past_values=past)
        # output.prediction_outputs shape: (batch, prediction_length, num_channels)
        preds = output.prediction_outputs[:, :, target_idx].numpy()
        actuals = np.stack([w[1] for w in batch_windows])

        all_predictions.append(preds.reshape(-1))
        all_actuals.append(actuals.reshape(-1))

all_predictions = np.concatenate(all_predictions)
all_actuals = np.concatenate(all_actuals)

val_mae = evaluate_mae(all_predictions, all_actuals, scaler_params)

# Inverse-scale for charting
mean = scaler_params[TARGET_COL]["mean"]
std = scaler_params[TARGET_COL]["std"]
predicted_temps = all_predictions * std + mean
actual_temps = all_actuals * std + mean

# Generate index array for chart x-axis
val_timestamps = np.arange(len(actual_temps))

t_end = time.time()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chart 1: Current experiment (actual vs predicted)
plt.figure(figsize=(14, 5))
plt.plot(val_timestamps, actual_temps, label="Actual", color="black", linewidth=1)
plt.plot(val_timestamps, predicted_temps, label="Predicted",
         color="red", linewidth=1, alpha=0.8)
plt.xlabel("Validation sample index")
plt.ylabel(f"{TARGET_COL} (deg C)")
plt.title(f"Kiln Temperature Prediction (MAE={val_mae:.4f} deg C)")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_chart.png", dpi=150)
plt.close()

# Chart 2: Best results comparison
bests = load_best_predictions(top_n=3)
if bests:
    plt.figure(figsize=(14, 5))
    plt.plot(val_timestamps[:len(actual_temps)], actual_temps,
             label="Actual", color="black", linewidth=1.5)
    colors = ["blue", "green", "orange"]
    # bests[0] = overall best (lowest MAE), highlight it in red
    commit, mae, desc, preds, ts = bests[0]
    n = min(len(preds), len(val_timestamps))
    plt.plot(val_timestamps[:n], preds[:n],
             label=f"Best: {desc} (MAE={mae:.4f})",
             color="red", linewidth=1.5)
    for i, (commit, mae, desc, preds, ts) in enumerate(bests[1:]):
        n = min(len(preds), len(val_timestamps))
        plt.plot(val_timestamps[:n], preds[:n],
                 label=f"#{i+2} {desc} (MAE={mae:.4f})",
                 color=colors[i % len(colors)], linewidth=1,
                 linestyle="--", alpha=0.6)
    plt.xlabel("Validation sample index")
    plt.ylabel(f"{TARGET_COL} (deg C)")
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
print(f"total_seconds:     {t_end - t_start:.1f}")
print(f"chart_saved:       prediction_chart.png")
```

- [ ] **Step 2: Run the baseline experiment**

```bash
timeout 300 python3 train_ts.py 2>&1 | tee run_ts.log
```

Verify output includes `val_mae:` line and `prediction_chart.png` is generated.

**Note:** If the model output tensor shape is different from expected, check `output.prediction_outputs` attribute name and shape. The TTM model may output only 1 channel if `num_input_channels=1`. Common issues:
- `prediction_outputs` may be named `forecasts` in some versions. Check `dir(output)`.
- Input shape mismatch: TTM expects `(batch, context_length, num_channels)`. Add debug prints if needed.
- If zero-shot MAE is very high, try passing only the target channel.

- [ ] **Step 3: Debug and fix any issues from the run**

Iterate until `grep "^val_mae:" run_ts.log` returns a valid number.

- [ ] **Step 4: Commit**

```bash
git add train_ts.py
git commit -m "feat: add train_ts.py baseline experiment script"
```

---

### Task 6: Write `program_ts.md` — agent experiment loop instructions

**Context:** This is the "skill" file that tells the agent how to operate the autonomous experiment loop. Adapted from the original `program.md`.

**Files:**
- Create: `program_ts.md`

- [ ] **Step 1: Write `program_ts.md`**

Create `program_ts.md` with the following content:

````markdown
# autoresearch -- time series

Autonomous experiment loop for kiln temperature prediction using granite-timeseries-ttm-r2.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `ts-mar23`). The branch `autoresearch-ts/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-ts/<tag>` from current master.
3. **Read the in-scope files**:
   - `program_ts.md` -- this file, your instructions.
   - `prepare_ts.py` -- fixed evaluation harness. Do not modify.
   - `train_ts.py` -- the file you modify. Model loading, fine-tuning, prediction.
4. **Verify data exists**: Check that `data/kiln_data.csv` exists.
5. **Initialize results_ts.tsv**: Create `results_ts.tsv` with just the header row:
   ```
   commit	val_mae	status	description
   ```
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU. You launch it as: `python3 train_ts.py`

**What you CAN do:**
- Modify `train_ts.py` -- this is the only file you edit. Everything is fair game: model variant, hyperparameters, feature selection, resampling interval, channel mode, fine-tuning strategy, rolling forecasts, etc.

**What you CANNOT do:**
- Modify `prepare_ts.py`. It is read-only. It contains the fixed evaluation, data loading, and scaling.
- Install new packages. You can only use what is already installed.
- Modify the evaluation harness. The `evaluate_mae` function in `prepare_ts.py` is the ground truth metric.

**The goal is simple: get the lowest val_mae (in degrees Celsius).** Everything is fair game: change the model variant, features, resampling, fine-tuning strategy, etc.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Equal MAE with simpler code is a win.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as is.

## Output format

The script prints a summary like this:

```
---
val_mae:           0.152300
prediction_length: 96
context_length:    512
resample:          5s
features:          7
channel_mode:      independent
finetune_epochs:   0
total_seconds:     45.2
chart_saved:       prediction_chart.png
```

Extract the key metric: `grep "^val_mae:" run_ts.log`

## Logging results

Log to `results_ts.tsv` (tab-separated). Do not commit this file.

```
commit	val_mae	status	description
```

1. git commit hash (short, 7 chars)
2. val_mae achieved -- use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Modify `train_ts.py` with an experimental idea
3. git commit
4. Run: `timeout 300 python3 train_ts.py > run_ts.log 2>&1`
5. Parse: `grep "^val_mae:" run_ts.log`
6. Empty grep means crash. Run `tail -n 50 run_ts.log` for the error
7. Record results in results_ts.tsv
8. If val_mae improved (strictly lower): keep the commit. Save the best prediction by adding save logic in train_ts.py or running a save command after the experiment.
9. If val_mae is equal: keep ONLY if the code is simpler, otherwise discard
10. If val_mae is worse: `git reset --hard HEAD~1`

**Timeout**: If a run exceeds 5 minutes (exit code 124), treat as crash.

**Crashes**: Fix easy bugs (typos, imports). If fundamentally broken, log crash and move on.

**NEVER STOP**: Do NOT pause to ask the human. Run indefinitely until manually stopped. If stuck, re-read the files, try radical changes, combine near-misses.

## Ideas to try

Rough priority order:
1. Zero-shot baseline (FINETUNE_EPOCHS=0)
2. Fine-tuning with a few epochs
3. Resample to 15s or 30s (fewer windows, potentially better patterns)
4. Feature selection (drop constant/noisy columns)
5. Different context/prediction lengths
6. Channel-mix mode
7. Different model variants (1024-96, 512-192, etc.)
8. Rolling forecast for longer horizons
9. Derived features (diff, rolling mean)
10. Learning rate / batch size tuning
````

- [ ] **Step 2: Commit**

```bash
git add program_ts.md
git commit -m "feat: add program_ts.md agent experiment loop instructions"
```

---

### Task 7: End-to-end validation

**Context:** Run the full baseline experiment, verify all outputs, and ensure the system is ready for autonomous operation.

**Files:** None modified -- this is a validation task.

- [ ] **Step 1: Run baseline experiment end-to-end**

```bash
timeout 300 python3 train_ts.py > run_ts.log 2>&1
```

- [ ] **Step 2: Verify output format**

```bash
grep "^val_mae:" run_ts.log
```

Expected: `val_mae:           <some_number>` with a valid floating point number

- [ ] **Step 3: Verify chart was generated**

```bash
ls -la prediction_chart.png
```

Expected: File exists with non-zero size

- [ ] **Step 4: Check for errors in the log**

```bash
grep -i "error\|traceback\|exception" run_ts.log | head -5
```

Expected: No errors (or only harmless warnings)

- [ ] **Step 5: If any failures, debug and fix in train_ts.py**

Common fixes:
- Output attribute name: try `output.prediction_outputs`, `output.forecasts`, or check `dir(output)`
- Shape issues: add `print(f"past shape: {past.shape}")` and `print(f"output keys: {dir(output)}")` to debug
- Channel count: TTM may need `num_input_channels` set correctly on the model config. Try `model.config.num_input_channels = len(all_cols)` before inference.
- If zero-shot MAE is very high or NaN, try passing only the target column as input.

- [ ] **Step 6: Verify the full loop would work**

```bash
# Simulate what the agent loop does
grep "^val_mae:" run_ts.log
# Should print a parseable line like: val_mae:           0.152300
```

- [ ] **Step 7: Final commit with any fixes**

```bash
git add train_ts.py
git commit -m "fix: adjust train_ts.py for correct model API after validation"
```

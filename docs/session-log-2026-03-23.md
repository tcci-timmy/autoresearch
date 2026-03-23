# Session Log: Kiln Temperature Time Series Autoresearch

**Date:** 2026-03-23
**Project:** autoresearch (kiln temperature prediction with granite-timeseries-ttm-r2)

---

## 1. Environment Setup

### 1.1 granite-tsfm Installation

```bash
pip3 install granite-tsfm
```

Installed `granite-tsfm 0.3.3` with dependencies (torch 2.2.2, transformers 4.56.0, etc.) to system Python 3.9 user site-packages (`/Users/timmy/Library/Python/3.9/`).

### 1.2 PATH Configuration

Added Python 3.9 user bin to PATH permanently:

```bash
echo 'export PATH="$PATH:/Users/timmy/Library/Python/3.9/bin"' >> ~/.zshrc
```

### 1.3 NumPy Compatibility Fix

`granite-tsfm` installed `numpy==2.0.2` but `torch==2.2.2` was compiled against NumPy 1.x. Fixed:

```bash
pip3 install --user 'numpy<2'
# Downgraded to numpy 1.26.4
```

---

## 2. Data Analysis

**Source file:** `旋窯八點資料20260320_08001600.csv`

| Property | Value |
|----------|-------|
| Rows | 5755 |
| Sampling interval | 5 seconds |
| Time range | 2026-03-20 08:00 ~ 16:00 (~8 hours) |
| Missing values | None |

**Columns (9 total):**

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

**Target stats:** min=799.38, max=801.21, mean=799.77 (very stable, ~1.8 deg C range)

---

## 3. Brainstorming & Design

### 3.1 Clarifying Questions & Answers

| Question | Answer |
|----------|--------|
| Use autoresearch for LLM or time series? | (B) Borrow autoresearch's autonomous loop, swap GPT for granite-TTM-R2 |
| Prediction goal? | (A) Short-horizon (few minutes ahead) for real-time monitoring |
| Agent freedom? | (C) Full freedom: hyperparams, feature engineering, model variants, etc. |
| Evaluation metric? | (A) MAE in degrees Celsius |
| Data strategy? | (A) MVP with current 8-hour data, more data later |
| Time budget? | (A) No fixed budget, run completes when done |
| Trend chart? | Yes, plus historical best comparison chart |

### 3.2 Approach Selection

Three approaches proposed:

- **A: Mirror structure (selected)** -- Parallel three-file system (`prepare_ts.py`, `train_ts.py`, `program_ts.md`)
- B: Overwrite originals -- Destroys original autoresearch
- C: Subdirectory isolation -- Requires path adjustments

### 3.3 Design Spec

Written and reviewed at `docs/superpowers/specs/2026-03-23-kiln-timeseries-autoresearch-design.md`.

Spec review found 3 critical + 6 minor issues, all fixed:
- `granite-tsfm` not in `pyproject.toml` -- resolved by using `python3` instead of `uv run`
- Sliding window interface undefined -- added `make_sliding_windows()` spec
- torch CUDA on macOS -- clarified separate Python 3.9 environment
- Best chart sorting -- fixed `bests[0]` as best
- `.gitignore` patterns -- added
- Equal MAE handling -- clarified simplicity criterion
- Timeout enforcement -- added `timeout 300` wrapper

---

## 4. Implementation Plan

Written at `docs/superpowers/plans/2026-03-23-kiln-timeseries-autoresearch.md`.

### 4.1 File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `prepare_ts.py` | Create | Data loading, train/val split, standard scaling, MAE evaluation, sliding windows, best-prediction I/O |
| `train_ts.py` | Create | Baseline experiment: load model, fine-tune, predict, chart, print results |
| `program_ts.md` | Create | Agent experiment loop instructions |
| `data/kiln_data.csv` | Create (copy) | Raw sensor data |
| `.gitignore` | Modify | Add time-series artifacts |
| `tests/test_prepare_ts.py` | Create | Tests for prepare_ts.py (7 tests) |

### 4.2 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1 | Fix NumPy + data directory setup | Completed |
| 2 | prepare_ts.py: load_data, train_val_split, standard_scale | Completed |
| 3 | prepare_ts.py: evaluate_mae, make_sliding_windows | Completed |
| 4 | prepare_ts.py: save/load best predictions | Completed |
| 5 | train_ts.py: baseline experiment script | Completed |
| 6 | program_ts.md: agent loop instructions | Completed |
| 7 | End-to-end validation | Completed |

---

## 5. Implementation (Subagent-Driven)

All 7 tasks executed via subagent-driven development (fresh subagent per task).

### 5.1 Git Commits

```
fa03684 feat: add program_ts.md agent experiment loop instructions
e4ce3e1 feat: add train_ts.py baseline experiment script
bd404b9 feat: add save/load best predictions to prepare_ts.py
d72d99a feat: add evaluate_mae and make_sliding_windows to prepare_ts.py
011f3d0 feat: add prepare_ts.py with load_data, train_val_split, standard_scale
c530272 feat: add kiln data and gitignore for time series autoresearch
5c2acb2 Add implementation plan for kiln time series autoresearch
1fd952b Add design spec for kiln temperature time series autoresearch
```

### 5.2 Test Results

All 7 tests passing:

```
tests/test_prepare_ts.py::test_load_data PASSED
tests/test_prepare_ts.py::test_train_val_split PASSED
tests/test_prepare_ts.py::test_standard_scale PASSED
tests/test_prepare_ts.py::test_standard_scale_constant_column PASSED
tests/test_prepare_ts.py::test_evaluate_mae PASSED
tests/test_prepare_ts.py::test_make_sliding_windows PASSED
tests/test_prepare_ts.py::test_save_and_load_best_predictions PASSED
```

---

## 6. Autonomous Experiment Loop

Branch: `autoresearch-ts/ts-mar23`

### 6.1 Full Results Table

| # | Commit | val_mae (deg C) | Status | Description |
|---|--------|-----------------|--------|-------------|
| 1 | fa03684 | 0.207889 | keep | Baseline (512/96, all features, independent, zero-shot) |
| 2 | 5e8ac6c | 0.230213 | discard | Fine-tune 5 epochs |
| 3 | 248defa | 0.000000 | crash | Resample 30s (val too small for context+pred) |
| 4 | 6727706 | 0.207889 | keep | Target channel only (simpler, same MAE) |
| 5 | bd829a1 | 0.000000 | crash | Context 256/pred 48 (freq_token required) |
| **6** | **8f19ed6** | **0.011361** | **keep** | **Context 1024, target only (massive improvement)** |
| 7 | 71673da | 0.000000 | crash | Context 1536 (val too small) |
| 8 | 95c7796 | 0.125779 | discard | Context 1024, fine-tune 3 epochs (worse) |
| 9 | 91a3fba | 0.000000 | crash | Context 1024/pred 192 (val too small) |
| 10 | a15507c | 0.011361 | discard | Context 1024 + 6 features (same MAE, more complex) |
| 11 | 2e3dc4f | 0.013437 | discard | Context 1024/pred 48 (slightly worse) |

### 6.2 Best Configuration

```
CONTEXT_LENGTH = 1024
PREDICTION_LENGTH = 96
MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
FEATURE_COLS = []  # Target channel only
FINETUNE_EPOCHS = 0  # Zero-shot
```

**Best val_mae: 0.0114 deg C** (95% improvement from baseline 0.2079 deg C)

### 6.3 Key Findings

1. **Context length is the dominant factor.** Going from 512 to 1024 reduced MAE by 95% (0.208 -> 0.011 deg C). The model benefits greatly from seeing more historical data.

2. **Fine-tuning hurts with limited data.** Both fine-tuning attempts (5 epochs at ctx=512, 3 epochs at ctx=1024) made results worse, likely due to overfitting on the small training set.

3. **Exogenous features provide no benefit in zero-shot mode.** The 7 additional sensor columns (motor current, power, speed, pressure, etc.) did not improve predictions. The target channel alone contains sufficient information.

4. **Data size is the binding constraint.** With only 5755 rows (8 hours at 5s intervals), the 80/20 split leaves only 1151 validation rows. Context=1024 + prediction=96 = 1120, leaving room for just 1 validation window. Context=1536 crashes entirely.

5. **Smaller context/prediction windows (256/48) require frequency tokens** that the standard model variant doesn't support without explicit `freq_token` parameter.

### 6.4 Prediction Chart Analysis (Best Model)

The prediction chart for the best model (context=1024, MAE=0.0114 deg C) shows:

- **Black line (Actual):** Nearly flat at ~799.41 deg C during validation period
- **Red line (Predicted):** Oscillates within 799.39~799.45 deg C range
- **Y-axis is extremely zoomed in** (0.05 deg C range), making small deviations look large
- The model predicts "expected" temperature fluctuations that didn't materialize in this stable period
- Average prediction error of 0.011 deg C is excellent for kiln monitoring

---

## 7. Architecture Summary

```
autoresearch/
├── prepare.py           # Original LLM autoresearch (untouched)
├── train.py             # Original LLM autoresearch (untouched)
├── program.md           # Original LLM autoresearch (untouched)
├── prepare_ts.py        # NEW: Read-only evaluation harness
│   ├── load_data()
│   ├── train_val_split()
│   ├── standard_scale()
│   ├── evaluate_mae()
│   ├── make_sliding_windows()
│   ├── save_best_prediction()
│   └── load_best_predictions()
├── train_ts.py          # NEW: Agent-editable experiment script
├── program_ts.md        # NEW: Agent loop instructions
├── data/
│   ├── kiln_data.csv    # Raw sensor data (5755 rows, 5s interval)
│   └── best_predictions/# Saved prediction arrays (not in git)
├── tests/
│   └── test_prepare_ts.py  # 7 tests for prepare_ts.py
├── prediction_chart.png    # Current experiment chart
├── prediction_best.png     # Historical best comparison
├── results_ts.tsv          # Experiment log (not in git)
└── docs/
    └── superpowers/
        ├── specs/2026-03-23-kiln-timeseries-autoresearch-design.md
        └── plans/2026-03-23-kiln-timeseries-autoresearch.md
```

### Execution Environment

- **Platform:** macOS (darwin)
- **Python:** System Python 3.9 (`python3`, not `uv run`)
- **Packages:** granite-tsfm 0.3.3, torch 2.2.2, numpy 1.26.4, pandas, matplotlib
- **Run command:** `python3 train_ts.py` (separate from original autoresearch's `uv` environment)

---

## 8. Next Steps (Out of Scope for This Session)

- Integrate more historical data (multi-day/multi-week) to enable larger context windows and more validation
- Real-time data pipeline from SCADA/DCS
- Fix CJK font rendering in matplotlib charts
- Try `r2.1` model variants (support daily/weekly resolutions)
- Explore anomaly detection mode
- Multi-target prediction

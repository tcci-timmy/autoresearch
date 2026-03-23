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
    df = df.sort_index()
    return df

# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def train_val_split(df, train_ratio=TRAIN_RATIO):
    """Chronological split. No shuffling -- time series must preserve order."""
    n = int(len(df) * train_ratio)
    return df.iloc[:n].copy(), df.iloc[n:].copy()

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
            std = 1.0
        scaler_params[col] = {"mean": float(mean), "std": float(std)}
        scaled_train[col] = (train_df[col] - mean) / std
        scaled_val[col] = (val_df[col] - mean) / std
    return scaled_train, scaled_val, scaler_params

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_mae(predictions, actuals, scaler_params, target_col=TARGET_COL):
    """Inverse-scale predictions and actuals, compute MAE in degrees Celsius."""
    mean = scaler_params[target_col]["mean"]
    std = scaler_params[target_col]["std"]
    preds_orig = predictions * std + mean
    actuals_orig = actuals * std + mean
    return float(np.mean(np.abs(preds_orig - actuals_orig)))

# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

def make_sliding_windows(scaled_df, context_length, prediction_length, target_col=TARGET_COL):
    """Create sliding window samples. Stride = prediction_length (non-overlapping).
    Returns list of (input_window, target_window) tuples.
    input_window: numpy array (context_length, n_features)
    target_window: 1-D numpy array (prediction_length,) for target col only
    """
    data = scaled_df.values
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
    Returns [(commit, mae, description, predictions, timestamps), ...]
    """
    if not os.path.exists(results_path):
        return []
    df = pd.read_csv(results_path, sep="\t")
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

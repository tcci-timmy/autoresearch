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
FEATURE_COLS = []
TARGET_COL = "1KTI10 熟料落料溫度"
RESAMPLE_INTERVAL = None
CHANNEL_MODE = "independent"
FINETUNE_EPOCHS = 0
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

val_timestamps = np.arange(len(actual_temps))

t_end = time.time()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

bests = load_best_predictions(top_n=3)
if bests:
    plt.figure(figsize=(14, 5))
    plt.plot(val_timestamps[:len(actual_temps)], actual_temps,
             label="Actual", color="black", linewidth=1.5)
    colors = ["blue", "green", "orange"]
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

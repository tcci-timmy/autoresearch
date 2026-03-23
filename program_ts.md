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

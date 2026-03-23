"""Tests for prepare_ts.py data utilities."""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_load_data():
    from prepare_ts import load_data
    df = load_data()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df.columns) == 8
    assert len(df) == 5755
    assert "1KTI09 燒成帶溫度" in df.columns
    assert df.isna().sum().sum() == 0


def test_train_val_split():
    from prepare_ts import load_data, train_val_split, TRAIN_RATIO
    df = load_data()
    train_df, val_df = train_val_split(df)
    total = len(df)
    expected_train = int(total * TRAIN_RATIO)
    assert len(train_df) == expected_train
    assert len(val_df) == total - expected_train
    assert train_df.index[-1] < val_df.index[0]
    assert len(set(train_df.index) & set(val_df.index)) == 0


def test_standard_scale():
    from prepare_ts import load_data, train_val_split, standard_scale, TARGET_COL
    df = load_data()
    train_df, val_df = train_val_split(df)
    scaled_train, scaled_val, scaler_params = standard_scale(train_df, val_df)
    for col in scaled_train.columns:
        # For non-constant columns, mean should be ~0 and std ~1
        # For constant columns (std=0), they should be all zeros
        if scaler_params[col]["std"] == 1.0:
            # This was a constant column, should be all zeros
            assert (scaled_train[col] == 0.0).all(), f"{col} should be all zeros"
        else:
            assert abs(scaled_train[col].mean()) < 0.01, f"{col} mean not ~0"
            assert abs(scaled_train[col].std() - 1.0) < 0.05, f"{col} std not ~1"
    assert TARGET_COL in scaler_params
    for col in train_df.columns:
        assert "mean" in scaler_params[col]
        assert "std" in scaler_params[col]
    assert scaled_train.shape == train_df.shape
    assert scaled_val.shape == val_df.shape


def test_standard_scale_constant_column():
    """Constant columns (std=0) should not cause division by zero."""
    from prepare_ts import standard_scale
    idx = pd.date_range("2026-01-01", periods=10, freq="5s")
    train = pd.DataFrame({"const": [1.0]*10, "vary": range(10)}, index=idx)
    val = pd.DataFrame({"const": [1.0]*5, "vary": range(5)}, index=idx[:5])
    scaled_train, scaled_val, params = standard_scale(train, val)
    assert params["const"]["std"] == 1.0
    assert (scaled_train["const"] == 0.0).all()
    assert abs(scaled_train["vary"].mean()) < 0.01


def test_evaluate_mae():
    from prepare_ts import evaluate_mae, TARGET_COL
    scaler_params = {TARGET_COL: {"mean": 800.0, "std": 0.5}}
    predictions = np.array([0.0, 0.0, 0.0])
    actuals = np.array([0.0, 2.0, -2.0])
    mae = evaluate_mae(predictions, actuals, scaler_params)
    assert abs(mae - 0.6667) < 0.001


def test_make_sliding_windows():
    from prepare_ts import make_sliding_windows, TARGET_COL
    n = 100
    idx = pd.date_range("2026-01-01", periods=n, freq="5s")
    df = pd.DataFrame({
        "feature_a": np.arange(n, dtype=float),
        TARGET_COL: np.arange(n, dtype=float) * 0.1,
    }, index=idx)
    context_length = 20
    prediction_length = 10
    windows = make_sliding_windows(df, context_length, prediction_length, TARGET_COL)
    assert len(windows) == 8
    inp, tgt = windows[0]
    assert inp.shape == (context_length, 2)
    assert tgt.shape == (prediction_length,)
    assert abs(tgt[0] - 2.0) < 0.001
    assert abs(tgt[-1] - 2.9) < 0.001


def test_save_and_load_best_predictions(tmp_path):
    from prepare_ts import save_best_prediction, load_best_predictions
    results_path = tmp_path / "results_ts.tsv"
    results_path.write_text(
        "commit\tval_mae\tstatus\tdescription\n"
        "abc1234\t0.150000\tkeep\tbaseline\n"
        "def5678\t0.120000\tkeep\tresample 15s\n"
        "ghi9012\t0.200000\tdiscard\tbad idea\n"
    )
    best_dir = tmp_path / "best_predictions"
    timestamps_a = np.array([1.0, 2.0, 3.0])
    preds_a = np.array([800.1, 800.2, 800.3])
    timestamps_b = np.array([1.0, 2.0, 3.0])
    preds_b = np.array([800.0, 800.1, 800.2])
    save_best_prediction("abc1234", preds_a, timestamps_a, best_dir=str(best_dir))
    save_best_prediction("def5678", preds_b, timestamps_b, best_dir=str(best_dir))
    bests = load_best_predictions(top_n=2, results_path=str(results_path), best_dir=str(best_dir))
    assert len(bests) == 2
    assert bests[0][0] == "def5678"
    assert bests[0][1] == 0.120000
    assert bests[1][0] == "abc1234"
    assert np.array_equal(bests[0][3], preds_b)

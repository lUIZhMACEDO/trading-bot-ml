"""Walk-forward splitting and labeling: the model must never train on information from its test window."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv, random_walk
from trading_bot.features.technicals import add_technicals
from trading_bot.models import ensemble as ml


def test_target_drops_rows_without_a_future_close():
    df = pd.DataFrame({"close": np.arange(1.0, 51.0)})
    labeled = ml.make_target(df, horizon=21)
    assert len(labeled) == 50 - 21
    assert labeled.index.max() == 28
    # close[t + 21] / close[t] - 1 for the last labeled row
    assert labeled["fwd_return"].iloc[-1] == pytest.approx(50 / 29 - 1)
    assert labeled["target"].eq(1).all()


def test_target_is_zero_for_falls_and_flat():
    df = pd.DataFrame({"close": [10.0, 9.0, 10.0, 10.0]})
    labeled = ml.make_target(df, horizon=1)
    assert labeled["target"].tolist() == [0, 1, 0]


@pytest.mark.parametrize(("n", "splits", "horizon"), [(300, 5, 21), (120, 3, 21), (500, 5, 5)])
def test_splits_leave_a_horizon_gap(n, splits, horizon):
    folds = list(ml.walk_forward_splits(n, splits, horizon))
    assert len(folds) == splits
    for train, test in folds:
        # The last training label looks horizon rows ahead; that must still be before the test window.
        assert train.max() + horizon < test.min()
        assert np.all(np.diff(train) == 1) and np.all(np.diff(test) == 1)
        assert train.min() == 0  # expanding window


def test_splits_expand_and_move_forward():
    folds = list(ml.walk_forward_splits(300, 5, 21))
    train_sizes = [len(train) for train, _ in folds]
    test_starts = [test.min() for _, test in folds]
    assert train_sizes == sorted(train_sizes) and len(set(train_sizes)) == 5
    assert test_starts == sorted(test_starts)
    # Test windows do not overlap
    for (_, a), (_, b) in zip(folds, folds[1:], strict=False):
        assert a.max() < b.min()


def test_signal_thresholds():
    assert ml.signal_from_probability(0.7)[:2] == ("BUY", 0.7)
    assert ml.signal_from_probability(0.3)[0] == "SELL"
    assert ml.signal_from_probability(0.3)[1] == pytest.approx(0.7)
    assert ml.signal_from_probability(0.5)[:2] == ("HOLD", 0.5)
    assert ml.signal_from_probability(0.6)[0] == "HOLD"  # strictly above 0.6 to buy


def test_too_little_history_returns_none():
    df = add_technicals(make_ohlcv(random_walk(99)))
    assert ml.train_walk_forward(df) is None


def _features_frame(n=360, seed=4):
    return add_technicals(make_ohlcv(random_walk(n, seed=seed, drift=0.0005)))


def test_train_walk_forward_end_to_end():
    df = _features_frame()
    model = ml.train_walk_forward(df, n_splits=3)
    assert model is not None
    assert len(model.fold_scores) == 3
    assert all(0.0 <= s <= 1.0 for s in model.fold_scores)
    assert set(model.features) <= set(ml.ML_FEATURES)
    signal, confidence, reason = ml.predict_latest(model, df)
    assert signal in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= confidence <= 1.0
    assert "ML" in reason


def test_training_ignores_the_unlabeled_tail():
    """The last 21 rows have no label, so they must not be used for training at all. Scrambling
    their features should leave the fold scores unchanged. (An earlier version labeled them 0.)"""
    df = _features_frame()
    shocked = df.copy()
    tail = shocked.index[-21:]
    for col in ml.ML_FEATURES:
        shocked.loc[tail, col] = shocked[col].max() * 10
    a = ml.train_walk_forward(df, n_splits=3)
    b = ml.train_walk_forward(shocked, n_splits=3)
    assert a.fold_scores != [] and a.fold_scores == b.fold_scores


def test_blend_weights():
    class Fixed:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])

        def predict(self, X):
            return np.full(len(X), self.p)

    X = np.zeros((2, 3))
    with_lgbm = ml._blend(Fixed(1.0), Fixed(0.0), Fixed(0.0), X)
    np.testing.assert_allclose(with_lgbm, 0.40)
    without = ml._blend(None, Fixed(1.0), Fixed(0.0), X)
    np.testing.assert_allclose(without, 0.60)
    # Ridge output is clipped to [0, 1] before blending
    clipped = ml._blend(None, Fixed(0.0), Fixed(5.0), X)
    np.testing.assert_allclose(clipped, 0.40)

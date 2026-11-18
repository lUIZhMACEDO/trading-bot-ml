"""Walk-forward LightGBM + random forest + Ridge ensemble for 21-day return direction.

The label for day t is whether close[t + 21] > close[t]. Two things keep the evaluation honest:

* The last 21 rows have no label yet (their future close is unknown), so they are dropped
  instead of being counted as "down".
* Each test fold starts ``horizon`` rows after its training window ends. Without that gap the
  last training labels would be computed from prices inside the test window.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:  # the ensemble falls back to RF + Ridge
    lgb = None
    HAS_LGBM = False

HORIZON = 21

ML_FEATURES = [
    "rsi", "macd", "macd_hist", "bb_width", "bb_pctb",
    "atr", "adx", "hurst_exponent",
    "momentum_5", "momentum_10", "momentum_21",
    "vol_ratio", "vol_zscore",
    "return_1d", "return_5d",
]  # fmt: skip

WEIGHTS_WITH_LGBM = (0.40, 0.35, 0.25)  # LightGBM, random forest, Ridge
WEIGHTS_WITHOUT_LGBM = (0.0, 0.60, 0.40)

BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4


def make_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Add ``fwd_return`` and a 0/1 ``target`` and drop rows whose future is not known yet."""
    df = df.copy()
    df["fwd_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df = df[df["fwd_return"].notna()].copy()
    df["target"] = (df["fwd_return"] > 0).astype(int)
    return df


def walk_forward_splits(n_samples: int, n_splits: int = 5, horizon: int = HORIZON) -> Iterator[tuple]:
    """Expanding-window splits with ``horizon`` rows left out between train and test."""
    return TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(np.arange(n_samples))


@dataclass
class Ensemble:
    """The three fitted models plus the scaler and the feature list they were trained on."""

    features: list[str]
    scaler: StandardScaler
    rf: RandomForestClassifier
    ridge: Ridge
    lgbm: object | None = None
    fold_scores: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_scores else float("nan")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability that the 21-day forward return is positive, for already-unscaled rows."""
        return _blend(self.lgbm, self.rf, self.ridge, self.scaler.transform(X))


def _new_models():
    lgbm = None
    if HAS_LGBM:
        lgbm = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
        )
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=42)
    ridge = Ridge(alpha=1.0)
    return lgbm, rf, ridge


def _blend(lgbm, rf, ridge, X_scaled: np.ndarray) -> np.ndarray:
    rf_p = rf.predict_proba(X_scaled)[:, 1]
    ridge_p = np.clip(ridge.predict(X_scaled), 0, 1)
    if lgbm is not None:
        w_lgb, w_rf, w_ridge = WEIGHTS_WITH_LGBM
        return w_lgb * lgbm.predict_proba(X_scaled)[:, 1] + w_rf * rf_p + w_ridge * ridge_p
    _, w_rf, w_ridge = WEIGHTS_WITHOUT_LGBM
    return w_rf * rf_p + w_ridge * ridge_p


def usable_features(df: pd.DataFrame) -> list[str]:
    """Features present in ``df`` with values in more than half of the rows."""
    return [f for f in ML_FEATURES if f in df.columns and df[f].notna().sum() > len(df) * 0.5]


def train_walk_forward(df: pd.DataFrame, n_splits: int = 5, horizon: int = HORIZON) -> Ensemble | None:
    """Score the ensemble fold by fold, then refit the best fold's models on every labeled row.

    ``df`` needs a ``close`` column and the ``ML_FEATURES`` columns. Returns None when there is
    not enough history (100 rows before labeling, 60 after dropping incomplete rows).
    """
    if len(df) < 100:
        return None
    df = make_target(df, horizon).dropna(subset=ML_FEATURES)
    if len(df) < 60:
        return None

    features = usable_features(df)
    X = df[features].fillna(0).values
    y = df["target"].values
    scaler = StandardScaler()

    fold_scores: list[float] = []
    best, best_score = None, -1.0
    for train_idx, test_idx in walk_forward_splits(len(X), n_splits, horizon):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        lgbm, rf, ridge = _new_models()
        for model in (lgbm, rf, ridge):
            if model is not None:
                model.fit(X_train, y[train_idx])
        predicted = (_blend(lgbm, rf, ridge, X_test) > 0.5).astype(int)
        score = accuracy_score(y[test_idx], predicted)
        fold_scores.append(float(score))
        if score > best_score:
            best, best_score = (lgbm, rf, ridge), score

    # Refit on all labeled history for live predictions.
    lgbm, rf, ridge = best
    X_all = scaler.fit_transform(X)
    for model in (lgbm, rf, ridge):
        if model is not None:
            model.fit(X_all, y)
    return Ensemble(features=features, scaler=scaler, rf=rf, ridge=ridge, lgbm=lgbm, fold_scores=fold_scores)


def signal_from_probability(p_up: float) -> tuple[str, float, str]:
    """BUY above 0.6, SELL below 0.4, HOLD in between."""
    if p_up > BUY_THRESHOLD:
        return "BUY", float(p_up), f"ML ensemble confidence: {p_up:.1%}"
    if p_up < SELL_THRESHOLD:
        return "SELL", float(1 - p_up), f"ML ensemble bearish: {p_up:.1%}"
    return "HOLD", 0.5, f"ML neutral: {p_up:.1%}"


def predict_latest(model: Ensemble, df: pd.DataFrame) -> tuple[str, float, str]:
    """Signal for the most recent row of ``df``."""
    if df.empty:
        return "HOLD", 0.5, "No data"
    last = df[model.features].iloc[-1:].fillna(0).values
    return signal_from_probability(float(model.predict_proba(last)[0]))

"""
Feature selection utilities for the tree carbon ML pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestRegressor


class FeatureSelector:
    """Wraps multiple feature selection strategies."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.results = {}

    def variance_filter(self, threshold: float = 0.01) -> list:
        """Remove near-zero-variance features."""
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(self.X)
        kept = self.X.columns[sel.get_support()].tolist()
        self.results["variance"] = kept
        print(f"  Variance filter: {len(self.X.columns)} → {len(kept)} features")
        return kept

    def correlation_filter(self, threshold: float = 0.95) -> list:
        """Remove highly correlated features (keep first of each pair)."""
        corr = self.X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        kept = [c for c in self.X.columns if c not in to_drop]
        self.results["correlation"] = kept
        print(f"  Correlation filter: {len(self.X.columns)} → {len(kept)} features")
        return kept

    def select_k_best(self, k: int = 10, method: str = "f_regression") -> list:
        """Select top-k features by F-score or mutual info."""
        score_fn = f_regression if method == "f_regression" else mutual_info_regression
        sel = SelectKBest(score_fn, k=min(k, self.X.shape[1]))
        sel.fit(self.X.fillna(0), self.y)
        kept = self.X.columns[sel.get_support()].tolist()
        scores = pd.Series(sel.scores_, index=self.X.columns).sort_values(ascending=False)
        self.results[f"kbest_{method}"] = {"features": kept, "scores": scores}
        print(f"  SelectKBest ({method}, k={k}): {kept}")
        return kept

    def random_forest_importance(self, n_estimators: int = 100, top_n: int = 15) -> pd.Series:
        """Feature importances from a quick RandomForest fit."""
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(self.X.fillna(0), self.y)
        importances = pd.Series(
            rf.feature_importances_, index=self.X.columns
        ).sort_values(ascending=False)
        self.results["rf_importance"] = importances
        print(f"  Top {top_n} RF features:\n{importances.head(top_n)}")
        return importances

    def summary(self) -> dict:
        return self.results

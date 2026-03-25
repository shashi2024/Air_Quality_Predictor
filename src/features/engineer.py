"""
Feature engineering for the tree carbon dC/dN dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """Transforms cleaned dC/dN data into ML-ready feature matrix."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._label_encoders = {}

    # ------------------------------------------------------------------
    # 1. Ratio / interaction features
    # ------------------------------------------------------------------
    def add_growth_survival_ratio(self):
        """Ratio of growth-only to survival-only dC/dN signal."""
        g = self.df.get("TPH.g.dC.dN0.01")
        s = self.df.get("TPH.s.dC.dN0.01")
        if g is not None and s is not None:
            self.df["feat_gs_ratio"] = g / (s.replace(0, np.nan))
        return self

    def add_expanded_per_ha(self):
        """Expanded net dC/dN per hectare represented."""
        if "EXPN.ha.TPH.gs.dC.dN0.01" in self.df and "EXPN.ha" in self.df:
            self.df["feat_expanded_per_ha"] = (
                self.df["EXPN.ha.TPH.gs.dC.dN0.01"] / self.df["EXPN.ha"].replace(0, np.nan)
            )
        return self

    def add_carbon_sign_flag(self):
        """Binary: is the net dC/dN positive (carbon sink) or negative (source)?"""
        col = "TPH.gs.dC.dN0.01"
        if col in self.df:
            self.df["feat_carbon_sink"] = (self.df[col] > 0).astype(int)
        return self

    def add_carbon_magnitude(self):
        """Log magnitude of the net dC/dN signal (signed log transform)."""
        col = "TPH.gs.dC.dN0.01"
        if col in self.df:
            self.df["feat_log_magnitude"] = np.sign(self.df[col]) * np.log1p(np.abs(self.df[col]))
        return self

    # ------------------------------------------------------------------
    # 2. Spatial / geographic features
    # ------------------------------------------------------------------
    def add_lat_lon_bins(self, n_bins: int = 10):
        """Discretize latitude and longitude into equal-width bins."""
        if "LAT" in self.df:
            self.df["feat_lat_bin"] = pd.cut(self.df["LAT"], bins=n_bins, labels=False)
        if "LON" in self.df:
            self.df["feat_lon_bin"] = pd.cut(self.df["LON"], bins=n_bins, labels=False)
        return self

    def add_climate_zone_proxy(self):
        """Rough climate zone proxy based on latitude bands."""
        if "LAT" in self.df:
            self.df["feat_climate_zone"] = pd.cut(
                self.df["LAT"],
                bins=[24, 30, 35, 40, 45, 50],
                labels=["subtropical", "warm_temp", "mid_temp", "cool_temp", "cold_temp"],
            )
        return self

    def add_ecoregion_target_mean(self, target_col: str = "TPH.gs.dC.dN0.01"):
        """Mean target value per ecoregion (L3) — target encoding proxy."""
        if "NA_L3CODE" in self.df and target_col in self.df:
            means = self.df.groupby("NA_L3CODE")[target_col].transform("mean")
            self.df["feat_eco3_target_mean"] = means
        return self

    # ------------------------------------------------------------------
    # 3. Encode categoricals
    # ------------------------------------------------------------------
    def encode_ecoregions(self):
        """Label-encode ecoregion codes for tree-based models."""
        for col in ["US_L4CODE", "NA_L3CODE", "NA_L1CODE"]:
            if col in self.df:
                le = LabelEncoder()
                self.df[f"feat_enc_{col}"] = le.fit_transform(
                    self.df[col].astype(str).fillna("UNKNOWN")
                )
                self._label_encoders[col] = le
        return self

    def encode_state(self):
        """Label-encode state code."""
        if "STATECD" in self.df:
            le = LabelEncoder()
            self.df["feat_enc_state"] = le.fit_transform(self.df["STATECD"].astype(str))
            self._label_encoders["STATECD"] = le
        return self

    # ------------------------------------------------------------------
    # 4. Pipeline helper
    # ------------------------------------------------------------------
    def build_all(self) -> pd.DataFrame:
        """Run all feature engineering steps and return enriched DataFrame."""
        return (
            self.add_growth_survival_ratio()
                .add_expanded_per_ha()
                .add_carbon_sign_flag()
                .add_carbon_magnitude()
                .add_lat_lon_bins()
                .add_climate_zone_proxy()
                .add_ecoregion_target_mean()
                .encode_ecoregions()
                .encode_state()
                .df
        )

    def get_feature_cols(self) -> list:
        """Return all engineered feature column names."""
        return [c for c in self.df.columns if c.startswith("feat_")]

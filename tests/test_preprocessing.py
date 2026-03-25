"""
Unit tests for src/data and src/features modules.
Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.validate import DataValidator, EXPECTED_COLUMNS
from features.engineer import FeatureEngineer
from features.selector import FeatureSelector


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_sample_df(n=200):
    """Generate a minimal synthetic dataframe matching the dC/dN schema."""
    np.random.seed(42)
    return pd.DataFrame({
        'PLT_CN': [str(i) for i in range(n)],
        'TPH.gs.dC.dN0.01': np.random.randn(n) * 5,
        'TPH.s.dC.dN0.01':  np.random.randn(n) * 3,
        'TPH.g.dC.dN0.01':  np.random.randn(n) * 2,
        'EXPN.ha':           np.abs(np.random.randn(n)) * 100 + 10,
        'LAT':               np.random.uniform(30, 48, n),
        'LON':               np.random.uniform(-120, -70, n),
        'EXPN.ha.TPH.gs.dC.dN0.01': np.random.randn(n) * 500,
        'EXPN.ha.TPH.s.dC.dN0.01':  np.random.randn(n) * 300,
        'EXPN.ha.TPH.g.dC.dN0.01':  np.random.randn(n) * 200,
        'US_L4CODE':  np.random.choice(['8.1.1', '8.1.2', '9.2.1', '10.1.1'], n),
        'NA_L3CODE':  np.random.choice(['28', '32', '45', '67'], n),
        'NA_L1CODE':  np.random.choice(['8', '9', '10', '6'], n),
        'STATECD':    np.random.choice(['12', '37', '45', '48'], n),
        'COUNTYCD':   np.random.choice(['001', '003', '007'], n),
        'e3':         np.random.choice(['28', '32', '45'], n),
        'e1':         np.random.choice(['8', '9', '10'], n),
        'e4':         np.random.choice(['8.1.1', '8.1.2'], n),
        'e3.state':   np.random.choice(['28', '32'], n),
        'e4.state':   np.random.choice(['8.1.1', '8.1.2'], n),
        'e1.state':   np.random.choice(['8', '9'], n),
        'eco.EXPN.ha':   np.abs(np.random.randn(n)) * 10000,
        'state.EXPN.ha': np.abs(np.random.randn(n)) * 50000,
    })


@pytest.fixture
def sample_df():
    return make_sample_df()


# ─────────────────────────────────────────────
# DataValidator tests
# ─────────────────────────────────────────────

class TestDataValidator:

    def test_passes_on_clean_data(self, sample_df):
        v = DataValidator(sample_df)
        v.run_all()
        assert len(v.issues) == 0

    def test_detects_missing_columns(self, sample_df):
        df_bad = sample_df.drop(columns=['LAT', 'LON'])
        v = DataValidator(df_bad)
        v.check_columns()
        assert any('LAT' in issue or 'LON' in issue for issue in v.issues)

    def test_detects_duplicate_PLT_CN(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        v = DataValidator(df_dup)
        v.check_duplicates()
        assert any('Duplicate' in i for i in v.issues)

    def test_detects_out_of_range_coordinates(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, 'LAT'] = -5.0   # South of US
        df_bad.loc[1, 'LON'] = 50.0   # East hemisphere
        v = DataValidator(df_bad)
        v.check_coordinates()
        assert len(v.issues) >= 2

    def test_detects_target_nulls(self, sample_df):
        df_null = sample_df.copy()
        df_null.loc[0, 'TPH.gs.dC.dN0.01'] = np.nan
        v = DataValidator(df_null)
        v.check_target_nulls()
        assert any('Target' in i for i in v.issues)

    def test_detects_non_positive_expansion_factor(self, sample_df):
        df_bad = sample_df.copy()
        df_bad.loc[0, 'EXPN.ha'] = -1.0
        v = DataValidator(df_bad)
        v.check_expansion_factor()
        assert any('EXPN.ha' in i for i in v.issues)

    def test_report_returns_true_on_clean(self, sample_df):
        v = DataValidator(sample_df)
        v.run_all()
        assert v.report() is True

    def test_report_returns_false_on_issues(self, sample_df):
        df_bad = sample_df.drop(columns=['LAT'])
        v = DataValidator(df_bad)
        v.run_all()
        assert v.report() is False


# ─────────────────────────────────────────────
# FeatureEngineer tests
# ─────────────────────────────────────────────

class TestFeatureEngineer:

    def test_build_all_returns_dataframe(self, sample_df):
        fe = FeatureEngineer(sample_df)
        result = fe.build_all()
        assert isinstance(result, pd.DataFrame)

    def test_all_feat_columns_added(self, sample_df):
        fe = FeatureEngineer(sample_df)
        result = fe.build_all()
        feat_cols = [c for c in result.columns if c.startswith('feat_')]
        assert len(feat_cols) > 0

    def test_get_feature_cols_matches_df(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.build_all()
        feat_cols = fe.get_feature_cols()
        assert all(c in fe.df.columns for c in feat_cols)

    def test_carbon_sign_flag_is_binary(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.add_carbon_sign_flag()
        assert set(fe.df['feat_carbon_sink'].dropna().unique()).issubset({0, 1})

    def test_log_magnitude_sign_preserved(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.add_carbon_magnitude()
        # Sign of log magnitude should match sign of original
        original = fe.df['TPH.gs.dC.dN0.01']
        log_mag = fe.df['feat_log_magnitude']
        mask = (original != 0) & log_mag.notna()
        assert (np.sign(original[mask]) == np.sign(log_mag[mask])).all()

    def test_lat_lon_bins_in_range(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.add_lat_lon_bins(n_bins=10)
        assert fe.df['feat_lat_bin'].dropna().between(0, 9).all()
        assert fe.df['feat_lon_bin'].dropna().between(0, 9).all()

    def test_ecoregion_encoding_no_nulls(self, sample_df):
        fe = FeatureEngineer(sample_df)
        fe.encode_ecoregions()
        for col in ['feat_enc_US_L4CODE', 'feat_enc_NA_L3CODE', 'feat_enc_NA_L1CODE']:
            if col in fe.df.columns:
                assert fe.df[col].isna().sum() == 0

    def test_does_not_mutate_original(self, sample_df):
        original_cols = sample_df.columns.tolist()
        fe = FeatureEngineer(sample_df)
        fe.build_all()
        assert sample_df.columns.tolist() == original_cols


# ─────────────────────────────────────────────
# FeatureSelector tests
# ─────────────────────────────────────────────

class TestFeatureSelector:

    def _make_X_y(self, sample_df):
        fe = FeatureEngineer(sample_df)
        df_feat = fe.build_all()
        feat_cols = fe.get_feature_cols()
        X = df_feat[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_feat['TPH.gs.dC.dN0.01'].fillna(0)
        return X, y

    def test_variance_filter_reduces_features(self, sample_df):
        X, y = self._make_X_y(sample_df)
        sel = FeatureSelector(X, y)
        kept = sel.variance_filter(threshold=0.001)
        assert len(kept) <= len(X.columns)

    def test_correlation_filter_returns_list(self, sample_df):
        X, y = self._make_X_y(sample_df)
        sel = FeatureSelector(X, y)
        kept = sel.correlation_filter(threshold=0.99)
        assert isinstance(kept, list)

    def test_select_k_best_respects_k(self, sample_df):
        X, y = self._make_X_y(sample_df)
        k = min(5, X.shape[1])
        sel = FeatureSelector(X, y)
        kept = sel.select_k_best(k=k, method='f_regression')
        assert len(kept) == k

    def test_rf_importance_returns_series(self, sample_df):
        X, y = self._make_X_y(sample_df)
        sel = FeatureSelector(X, y)
        imp = sel.random_forest_importance(n_estimators=10, top_n=5)
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]

    def test_rf_importance_sums_to_one(self, sample_df):
        X, y = self._make_X_y(sample_df)
        sel = FeatureSelector(X, y)
        imp = sel.random_forest_importance(n_estimators=10, top_n=5)
        assert abs(imp.sum() - 1.0) < 0.01

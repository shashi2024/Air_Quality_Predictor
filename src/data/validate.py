"""
Schema validation and data integrity checks for the dC/dN dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

EXPECTED_COLUMNS = [
    "PLT_CN",
    "TPH.gs.dC.dN0.01",
    "TPH.s.dC.dN0.01",
    "TPH.g.dC.dN0.01",
    "EXPN.ha",
    "LAT",
    "LON",
    "EXPN.ha.TPH.gs.dC.dN0.01",
    "EXPN.ha.TPH.s.dC.dN0.01",
    "EXPN.ha.TPH.g.dC.dN0.01",
    "US_L4CODE",
    "NA_L3CODE",
    "NA_L1CODE",
    "STATECD",
    "COUNTYCD",
    "e3",
    "e1",
    "e4",
    "e3.state",
    "e4.state",
    "e1.state",
    "eco.EXPN.ha",
    "state.EXPN.ha",
]

LAT_RANGE = (24.0, 50.0)   # continental US
LON_RANGE = (-130.0, -65.0)
TARGET_COL = "TPH.gs.dC.dN0.01"


class DataValidator:
    """Validates the raw dC/dN dataframe against expected schema and value ranges."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []

    def check_columns(self):
        missing = [c for c in EXPECTED_COLUMNS if c not in self.df.columns]
        extra = [c for c in self.df.columns if c not in EXPECTED_COLUMNS]
        if missing:
            self.issues.append(f"Missing columns: {missing}")
        if extra:
            print(f"  ℹ️  Extra columns (not in key): {extra}")
        return self

    def check_duplicates(self):
        n_dups = self.df.duplicated(subset=["PLT_CN"]).sum()
        if n_dups > 0:
            self.issues.append(f"Duplicate PLT_CN rows: {n_dups}")
        return self

    def check_coordinates(self):
        if "LAT" in self.df.columns and "LON" in self.df.columns:
            bad_lat = (~self.df["LAT"].between(*LAT_RANGE)).sum()
            bad_lon = (~self.df["LON"].between(*LON_RANGE)).sum()
            if bad_lat:
                self.issues.append(f"LAT out of US range [{LAT_RANGE}]: {bad_lat} rows")
            if bad_lon:
                self.issues.append(f"LON out of US range [{LON_RANGE}]: {bad_lon} rows")
        return self

    def check_target_nulls(self):
        if TARGET_COL in self.df.columns:
            nulls = self.df[TARGET_COL].isna().sum()
            if nulls > 0:
                self.issues.append(f"Target '{TARGET_COL}' has {nulls} nulls")
        return self

    def check_expansion_factor(self):
        if "EXPN.ha" in self.df.columns:
            negatives = (self.df["EXPN.ha"] <= 0).sum()
            if negatives:
                self.issues.append(f"EXPN.ha has {negatives} non-positive values")
        return self

    def run_all(self) -> "DataValidator":
        return (
            self.check_columns()
                .check_duplicates()
                .check_coordinates()
                .check_target_nulls()
                .check_expansion_factor()
        )

    def report(self) -> bool:
        if not self.issues:
            print("✅ All validation checks passed.")
            return True
        else:
            print(f"⚠️  {len(self.issues)} validation issue(s) found:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
            return False

"""
Reusable visualization functions for tree carbon EDA and feature analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
TARGET_COL = "TPH.gs.dC.dN0.01"


def save_fig(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  💾 Saved: {path}")


# ------------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Target Distribution — Net dC/dN (growth + survival) × TPH", fontsize=13, fontweight="bold")

    axes[0].hist(df[TARGET_COL].dropna(), bins=80, color="#2196F3", edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    axes[0].set_xlabel("TPH.gs.dC.dN0.01")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Full Distribution")
    axes[0].legend()

    clipped = df[TARGET_COL].clip(
        lower=df[TARGET_COL].quantile(0.01),
        upper=df[TARGET_COL].quantile(0.99),
    )
    axes[1].hist(clipped.dropna(), bins=80, color="#4CAF50", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    axes[1].set_xlabel("TPH.gs.dC.dN0.01 (1–99th pct)")
    axes[1].set_title("Clipped at 1st–99th Percentile")
    axes[1].legend()

    plt.tight_layout()
    if save:
        save_fig(fig, "01_target_distribution")
    return fig


def plot_missing_values(df: pd.DataFrame, save: bool = True):
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("✅ No missing values detected.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    missing.plot.barh(ax=ax, color="#FF7043")
    ax.set_xlabel("Missing Fraction")
    ax.set_title("Missing Value Rate by Column")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    if save:
        save_fig(fig, "02_missing_values")
    return fig


# ------------------------------------------------------------------
# Geographic plots
# ------------------------------------------------------------------

def plot_spatial_distribution(df: pd.DataFrame, col: str = TARGET_COL, save: bool = True):
    if "LAT" not in df or "LON" not in df:
        print("LAT/LON columns not found.")
        return None

    vmin, vmax = df[col].quantile(0.02), df[col].quantile(0.98)

    fig, ax = plt.subplots(figsize=(14, 7))
    sc = ax.scatter(
        df["LON"], df["LAT"],
        c=df[col].clip(vmin, vmax),
        cmap="RdYlGn", s=2, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label=col)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Distribution of {col}\n(clipped 2nd–98th pct)")
    plt.tight_layout()
    if save:
        save_fig(fig, "03_spatial_distribution")
    return fig


# ------------------------------------------------------------------
# Ecoregion / state plots
# ------------------------------------------------------------------

def plot_ecoregion_boxplot(df: pd.DataFrame, group_col: str = "NA_L1CODE", save: bool = True):
    order = (
        df.groupby(group_col)[TARGET_COL].median()
        .sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df, x=group_col, y=TARGET_COL,
        order=order, palette=PALETTE,
        flierprops={"marker": ".", "markersize": 2},
        ax=ax,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Net dC/dN by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(TARGET_COL)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        save_fig(fig, f"04_boxplot_{group_col}")
    return fig


# ------------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None, save: bool = True):
    num_df = df.select_dtypes(include=[np.number])
    if cols:
        num_df = num_df[[c for c in cols if c in num_df.columns]]

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.7), max(8, len(corr) * 0.6)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    if save:
        save_fig(fig, "05_correlation_heatmap")
    return fig


# ------------------------------------------------------------------
# Feature importance bar chart
# ------------------------------------------------------------------

def plot_feature_importance(importances: pd.Series, top_n: int = 20, save: bool = True):
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    top.sort_values().plot.barh(ax=ax, color="#7E57C2")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        save_fig(fig, "06_feature_importance")
    return fig

"""
Random Forest training script for the tree carbon dataset.
Loads the model-ready dataset, applies minimal preprocessing,
trains a RandomForestRegressor, and reports evaluation metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
TARGET_KEY = "regression_log"  # Options: regression_raw, regression_log, binary_classification, multiclass_3
DATA_DIR = Path(os.getenv("AQP_DATA_DIR", "data/processed"))
DATA_CARD_PATH = Path(os.getenv("AQP_DATA_CARD", "data/processed/data_card.json"))
INTERIM_DIR = Path(os.getenv("AQP_INTERIM_DIR", "data/interim"))
USE_SYNTHETIC_IF_MISSING = False
PLOT_RESULTS = True
FIGURES_DIR = Path(os.getenv("AQP_FIGURES_DIR", "outputs/figures"))


def build_synthetic_dataset(
    features: list[str],
    target: str,
    weight_col: str,
    n_samples: int = 500,
) -> pd.DataFrame:
    """Create a small synthetic dataset when real files are missing."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(n_samples, len(features)))
    df = pd.DataFrame(X, columns=features)
    df[target] = df[features].sum(axis=1) + rng.normal(scale=0.5, size=n_samples)
    df[weight_col] = rng.uniform(0.5, 1.5, size=n_samples)
    return df


def read_table(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV file into a dataframe."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_dataset_path(data_dir: Path, stem: str) -> Path | None:
    """Return the first existing dataset path for a given stem."""
    for ext in (".parquet", ".csv"):
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_model_ready_data() -> tuple[dict[str, pd.DataFrame], list[str], str, str, str, str]:
    """Load the model-ready dataset and metadata.

    Prefers processed train/test splits. Falls back to a full dataset or
    the interim feature file if needed. Returns a dict of dataframes keyed
    by split name plus feature/target metadata.
    """

    data_card_path = DATA_CARD_PATH if DATA_CARD_PATH.exists() else DATA_DIR / "data_card.json"
    if data_card_path.exists():
        with data_card_path.open() as f:
            card = json.load(f)
        features = card["features"]
        target = card["targets"].get(TARGET_KEY, card["targets"]["regression_log"])
        weight_col = card.get("sample_weight_col", "sample_weight")
        problem_type = "classification" if TARGET_KEY in {"binary_classification", "multiclass_3"} else "regression"

        train_path = find_dataset_path(DATA_DIR, "train")
        test_path = find_dataset_path(DATA_DIR, "test")
        val_path = find_dataset_path(DATA_DIR, "val")
        if train_path and test_path:
            data = {
                "train": read_table(train_path),
                "test": read_table(test_path),
            }
            if val_path:
                data["val"] = read_table(val_path)
            source = f"{train_path.name}, {test_path.name}"
            return data, features, target, weight_col, source, problem_type

        full_path = find_dataset_path(DATA_DIR, "full_dataset")
        model_ready_path = find_dataset_path(DATA_DIR, "model_ready")
        if full_path or model_ready_path:
            path = full_path or model_ready_path
            data = {"full": read_table(path)}
            return data, features, target, weight_col, str(path), problem_type

        if USE_SYNTHETIC_IF_MISSING:
            df = build_synthetic_dataset(features, target, weight_col)
            return {"full": df}, features, target, weight_col, "synthetic", problem_type

    # Fallback to interim dataset with manifest if processed data is missing
    interim_path = find_dataset_path(INTERIM_DIR, "04_features")
    manifest_path = INTERIM_DIR / "04_feature_manifest.json"
    if interim_path and manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        features = manifest["final_features"]
        target = manifest["target_log"]
        weight_col = "sample_weight"
        df = read_table(interim_path)
        return {"full": df}, features, target, weight_col, str(interim_path), "regression"

    raise FileNotFoundError(
        "Could not locate processed or interim datasets. "
        "Set AQP_DATA_DIR to the folder with train/test or model_ready files, "
        "or run notebooks 01-05 to generate model-ready data."
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return standard regression metrics for model evaluation."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    average: str,
) -> dict:
    """Return classification metrics for model evaluation."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    return metrics


def main() -> None:
    """Train and evaluate a Random Forest model."""
    data, features, target, weight_col, source_path, problem_type = load_model_ready_data()
    print(f"Loaded data from: {source_path}")
    print(f"Features: {len(features)}  Target: {target}")
    print(f"Problem type: {problem_type}")

    # Combine required columns and drop rows with missing target values
    def prepare_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        columns = features + [target]
        if weight_col in df.columns:
            columns.append(weight_col)
        df = df[columns].copy()
        df = df.dropna(subset=[target])
        X_split = df[features].replace([np.inf, -np.inf], np.nan)
        y_split = df[target]
        if problem_type == "classification":
            y_split = y_split.astype(int)
        w_split = df[weight_col] if weight_col in df.columns else None
        return X_split, y_split, w_split

    if "full" in data:
        # Split the full dataset into train/test sets
        X_full, y_full, w_full = prepare_split(data["full"])
        split_kwargs = {"test_size": TEST_SIZE, "random_state": RANDOM_STATE}
        if problem_type == "classification":
            split_kwargs["stratify"] = y_full
        if w_full is None:
            X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, **split_kwargs)
            w_train = None
            w_test = None
        else:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X_full, y_full, w_full, **split_kwargs
            )
    else:
        # Use pre-split train/test datasets if they exist
        X_train, y_train, w_train = prepare_split(data["train"])
        X_test, y_test, w_test = prepare_split(data["test"])

    # Simple preprocessing: median imputation for missing numeric values
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Build and train the Random Forest model
    if problem_type == "classification":
        rf_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        rf_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    rf_model.fit(X_train_imputed, y_train, sample_weight=w_train)

    # Predict and evaluate on the test set
    y_pred = rf_model.predict(X_test_imputed)
    y_proba = None
    if problem_type == "classification":
        average = "binary" if len(np.unique(y_test)) == 2 else "macro"
        if average == "binary":
            y_proba = rf_model.predict_proba(X_test_imputed)[:, 1]
        metrics = evaluate_classification(y_test, y_pred, y_proba, average)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    # Print clear results for assignment submission
    title = "Random Forest Classification Results" if problem_type == "classification" else "Random Forest Regression Results"
    print(title)
    print("=" * len(title))
    print(f"Samples: train={len(X_train)}  test={len(X_test)}")
    metrics_df = pd.DataFrame([metrics]).round(4)
    print(metrics_df.to_string(index=False))

    if problem_type == "classification":
        matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix")
        print(matrix)

    # Create plots for the report
    if PLOT_RESULTS:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if problem_type == "classification":
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(matrix, cmap="Blues")
            ax.set_title("Random Forest Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, matrix[i, j], ha="center", va="center")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "rf_confusion_matrix.png", dpi=150)
            plt.close(fig)

            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, label="ROC Curve")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Random Forest ROC Curve")
                ax.legend()
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "rf_roc_curve.png", dpi=150)
                plt.close(fig)
        else:
            y_true = np.array(y_test)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_true, y_pred, alpha=0.4)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Random Forest: Predicted vs Actual")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "rf_pred_vs_actual.png", dpi=150)
            plt.close(fig)

            residuals = y_true - y_pred
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_pred, residuals, alpha=0.4)
            ax.axhline(0, linestyle="--", color="red")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            ax.set_title("Random Forest Residuals")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "rf_residuals.png", dpi=150)
            plt.close(fig)

    # Show top feature importances
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 10 Feature Importances")
    print("---------------------------")
    print(importances.head(10).to_string())

    if PLOT_RESULTS:
        top_importances = importances.head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_importances.index[::-1], top_importances.values[::-1])
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "rf_feature_importances.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

"""
SVM Model Module for Tree Carbon ML Pipeline.

Provides SVR (regression) and SVC (classification) implementations
for predicting aboveground tree carbon responses to nitrogen deposition.
Supports all 4 target formulations:
  - Regression (raw)
  - Regression (log-transformed)
  - Binary classification
  - Multiclass 3-class classification

Includes hyperparameter tuning, evaluation, and visualization utilities.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR, SVC
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    learning_curve,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ======================================================================
# Configuration
# ======================================================================

RANDOM_STATE = 42

SVR_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "epsilon": [0.01, 0.1, 0.5],
    "kernel": ["rbf", "linear", "poly"],
}

SVC_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "kernel": ["rbf", "linear", "poly"],
}

# Smaller grid for faster initial search
SVR_PARAM_GRID_FAST = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1],
    "epsilon": [0.1],
    "kernel": ["rbf", "linear"],
}

SVC_PARAM_GRID_FAST = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1],
    "kernel": ["rbf", "linear"],
}


# ======================================================================
# Data Loading Utilities
# ======================================================================

def load_model_ready_data(data_dir: str, target_col: str, feature_cols: list = None):
    """
    Load the model-ready dataset from parquet or CSV files.

    Parameters
    ----------
    data_dir : str
        Path to data/processed directory.
    target_col : str
        Name of the target column.
    feature_cols : list, optional
        Specific feature columns. If None, auto-detect from data_card.json.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : arrays
    """
    # Try loading from parquet first, fallback to CSV
    for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
        train_path = os.path.join(data_dir, f"train{ext}")
        val_path = os.path.join(data_dir, f"val{ext}")
        test_path = os.path.join(data_dir, f"test{ext}")

        if os.path.exists(train_path):
            df_train = reader(train_path)
            df_val = reader(val_path)
            df_test = reader(test_path)
            break
    else:
        # Try loading a single dataset and splitting
        for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
            full_path = os.path.join(data_dir, f"model_ready{ext}")
            if os.path.exists(full_path):
                df_full = reader(full_path)
                # Use data_card split ratios
                from sklearn.model_selection import train_test_split
                df_train, df_temp = train_test_split(
                    df_full, test_size=0.3, random_state=RANDOM_STATE
                )
                df_val, df_test = train_test_split(
                    df_temp, test_size=0.5, random_state=RANDOM_STATE
                )
                break
        else:
            raise FileNotFoundError(
                f"No dataset files found in {data_dir}. "
                "Run notebooks 01-05 first to generate the processed data."
            )

    # Auto-detect features from data_card.json
    if feature_cols is None:
        card_path = os.path.join(data_dir, "data_card.json")
        if os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            feature_cols = card.get("features", [])

    if feature_cols is None or len(feature_cols) == 0:
        # Fallback: use all columns except known targets
        target_cols_all = [
            "TPH.gs.dC.dN0.01", "target_log", "target_binary",
            "target_class3", "sample_weight", "PLT_CN"
        ]
        feature_cols = [c for c in df_train.columns if c not in target_cols_all]

    # Ensure columns exist
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values

    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ======================================================================
# SVM Regression Model
# ======================================================================

class SVMRegressor:
    """
    Support Vector Machine Regressor with hyperparameter tuning,
    evaluation, and visualization capabilities.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.cv_results = None
        self.is_fitted = False

    def fit(self, X_train, y_train, scale=True):
        """Train the SVR model."""
        if scale:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X, scale=True):
        """Generate predictions."""
        model = self.best_model if self.best_model else self.model
        if scale:
            X = self.scaler.transform(X)
        return model.predict(X)

    def tune_hyperparameters(self, X_train, y_train, param_grid=None,
                              cv=5, scoring="neg_mean_squared_error",
                              n_jobs=-1, fast=True):
        """
        Perform GridSearchCV hyperparameter tuning.
        """
        if param_grid is None:
            param_grid = SVR_PARAM_GRID_FAST if fast else SVR_PARAM_GRID

        X_scaled = self.scaler.fit_transform(X_train)

        grid_search = GridSearchCV(
            SVR(),
            param_grid,
            cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_scaled, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        self.is_fitted = True

        print(f"\n✅ Best SVR Parameters: {self.best_params}")
        print(f"   Best CV Score (neg MSE): {grid_search.best_score_:.4f}")

        return self

    def evaluate(self, X_test, y_test, scale=True, dataset_name="Test"):
        """
        Evaluate the model and return metrics dictionary.
        """
        y_pred = self.predict(X_test, scale=scale)

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        }

        print(f"\n📊 SVR Evaluation ({dataset_name}):")
        print(f"   MAE  = {metrics['MAE']:.4f}")
        print(f"   RMSE = {metrics['RMSE']:.4f}")
        print(f"   R²   = {metrics['R2']:.4f}")
        print(f"   MAPE = {metrics['MAPE']:.4f}")

        return metrics, y_pred

    def cross_validate(self, X_train, y_train, cv=5, scale=True):
        """Perform k-fold cross-validation."""
        model = self.best_model if self.best_model else self.model
        X = self.scaler.fit_transform(X_train) if scale else X_train

        scores = {}
        for metric in ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]:
            cv_scores = cross_val_score(
                model, X, y_train, cv=cv, scoring=metric, n_jobs=-1
            )
            scores[metric] = {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist(),
            }

        print(f"\n📊 SVR Cross-Validation (k={cv}):")
        print(f"   MSE  = {-scores['neg_mean_squared_error']['mean']:.4f} ± {scores['neg_mean_squared_error']['std']:.4f}")
        print(f"   MAE  = {-scores['neg_mean_absolute_error']['mean']:.4f} ± {scores['neg_mean_absolute_error']['std']:.4f}")
        print(f"   R²   = {scores['r2']['mean']:.4f} ± {scores['r2']['std']:.4f}")

        return scores


# ======================================================================
# SVM Classification Model
# ======================================================================

class SVMClassifier:
    """
    Support Vector Machine Classifier with hyperparameter tuning,
    evaluation, and visualization capabilities.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.model = SVC(
            kernel=kernel, C=C, gamma=gamma,
            probability=True, random_state=RANDOM_STATE
        )
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.cv_results = None
        self.is_fitted = False

    def fit(self, X_train, y_train, scale=True):
        """Train the SVC model."""
        if scale:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X, scale=True):
        """Generate predictions."""
        model = self.best_model if self.best_model else self.model
        if scale:
            X = self.scaler.transform(X)
        return model.predict(X)

    def predict_proba(self, X, scale=True):
        """Generate probability predictions."""
        model = self.best_model if self.best_model else self.model
        if scale:
            X = self.scaler.transform(X)
        return model.predict_proba(X)

    def tune_hyperparameters(self, X_train, y_train, param_grid=None,
                              cv=5, scoring="accuracy",
                              n_jobs=-1, fast=True):
        """
        Perform GridSearchCV hyperparameter tuning.
        """
        if param_grid is None:
            param_grid = SVC_PARAM_GRID_FAST if fast else SVC_PARAM_GRID

        X_scaled = self.scaler.fit_transform(X_train)

        # Determine n_classes for stratified CV
        n_classes = len(np.unique(y_train))
        cv_obj = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=RANDOM_STATE
        )

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            param_grid,
            cv=cv_obj,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_scaled, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        self.is_fitted = True

        print(f"\n✅ Best SVC Parameters: {self.best_params}")
        print(f"   Best CV Score ({scoring}): {grid_search.best_score_:.4f}")

        return self

    def evaluate(self, X_test, y_test, scale=True, dataset_name="Test",
                 average="weighted"):
        """
        Evaluate the classifier and return metrics dictionary.
        """
        y_pred = self.predict(X_test, scale=scale)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average=average, zero_division=0),
            "Recall": recall_score(y_test, y_pred, average=average, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, average=average, zero_division=0),
        }

        # ROC-AUC (binary or multi-class OVR)
        try:
            y_proba = self.predict_proba(X_test, scale=scale)
            classes = np.unique(y_test)
            if len(classes) == 2:
                metrics["ROC_AUC"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                metrics["ROC_AUC"] = roc_auc_score(
                    y_test_bin, y_proba, multi_class="ovr", average=average
                )
        except Exception:
            metrics["ROC_AUC"] = None

        print(f"\n📊 SVC Evaluation ({dataset_name}):")
        print(f"   Accuracy  = {metrics['Accuracy']:.4f}")
        print(f"   Precision = {metrics['Precision']:.4f}")
        print(f"   Recall    = {metrics['Recall']:.4f}")
        print(f"   F1 Score  = {metrics['F1_Score']:.4f}")
        if metrics["ROC_AUC"] is not None:
            print(f"   ROC-AUC   = {metrics['ROC_AUC']:.4f}")

        return metrics, y_pred

    def cross_validate(self, X_train, y_train, cv=5, scale=True):
        """Perform stratified k-fold cross-validation."""
        model = self.best_model if self.best_model else self.model
        X = self.scaler.fit_transform(X_train) if scale else X_train

        cv_obj = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=RANDOM_STATE
        )
        scores = {}
        for metric in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
            cv_scores = cross_val_score(
                model, X, y_train, cv=cv_obj, scoring=metric, n_jobs=-1
            )
            scores[metric] = {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist(),
            }

        print(f"\n📊 SVC Cross-Validation (k={cv}):")
        print(f"   Accuracy  = {scores['accuracy']['mean']:.4f} ± {scores['accuracy']['std']:.4f}")
        print(f"   F1 (w)    = {scores['f1_weighted']['mean']:.4f} ± {scores['f1_weighted']['std']:.4f}")
        print(f"   Prec (w)  = {scores['precision_weighted']['mean']:.4f} ± {scores['precision_weighted']['std']:.4f}")
        print(f"   Recall(w) = {scores['recall_weighted']['mean']:.4f} ± {scores['recall_weighted']['std']:.4f}")

        return scores


# ======================================================================
# Visualization Functions
# ======================================================================

class SVMVisualizer:
    """Visualization utilities for SVM model results."""

    @staticmethod
    def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual",
                                  save_path=None):
        """Scatter plot of predicted vs actual values."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#4C72B0")
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
        """Residual plot for regression evaluation."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="#DD8452")
        axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[0].set_xlabel("Predicted Values", fontsize=11)
        axes[0].set_ylabel("Residuals", fontsize=11)
        axes[0].set_title(f"{title} — Residuals vs Predicted", fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Residual Value", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title(f"{title} — Residual Distribution", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None,
                               title="Confusion Matrix", save_path=None):
        """Plot confusion matrix for classification."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap="Blues", ax=ax, colorbar=True)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def plot_roc_curve(y_true, y_proba, class_names=None,
                       title="ROC Curve", save_path=None):
        """Plot ROC curve for binary or multiclass classification."""
        classes = np.unique(y_true)
        fig, ax = plt.subplots(figsize=(8, 6))

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc = roc_auc_score(y_true, y_proba[:, 1])
            ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        else:
            y_bin = label_binarize(y_true, classes=classes)
            colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
            for i, (cls, color) in enumerate(zip(classes, colors)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
                label = class_names[i] if class_names else f"Class {cls}"
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{label} (AUC = {auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def plot_learning_curve(model, X, y, cv=5, scoring="neg_mean_squared_error",
                             title="Learning Curve", save_path=None):
        """Plot learning curve to diagnose bias/variance."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1, random_state=RANDOM_STATE,
        )

        train_mean = -train_scores.mean(axis=1) if "neg" in scoring else train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = -val_scores.mean(axis=1) if "neg" in scoring else val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.2, color="#4C72B0")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.2, color="#DD8452")
        ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", linewidth=2,
                label="Training Score")
        ax.plot(train_sizes, val_mean, "o-", color="#DD8452", linewidth=2,
                label="Validation Score")
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    @staticmethod
    def plot_feature_importance(model, X, y, feature_names,
                                 title="Feature Importance (Permutation)",
                                 top_n=15, save_path=None):
        """Plot permutation-based feature importance for SVM."""
        result = permutation_importance(
            model, X, y, n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }).sort_values("Importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1],
            xerr=importance_df["Std"][::-1],
            color="#4C72B0", edgecolor="black", alpha=0.8,
        )
        ax.set_xlabel("Mean Importance Decrease", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig, importance_df

    @staticmethod
    def plot_metrics_comparison(metrics_dict, title="Model Metrics Comparison",
                                 save_path=None):
        """Bar chart comparing metrics across different target formulations."""
        df = pd.DataFrame(metrics_dict).T
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind="bar", ax=ax, rot=45, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig


# ======================================================================
# Model Persistence
# ======================================================================

def save_model(model, filepath):
    """Save a trained model to disk using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to: {filepath}")


def load_model(filepath):
    """Load a trained model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"📂 Model loaded from: {filepath}")
    return model


def save_metrics(metrics, filepath):
    """Save evaluation metrics as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"📝 Metrics saved to: {filepath}")

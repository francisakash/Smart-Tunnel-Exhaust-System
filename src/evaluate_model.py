"""
evaluate_model.py — Dual-model evaluation with metrics and visualizations.

Evaluates both the AQI and fan-speed Random Forest models:
    - Accuracy score
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix (saved as side-by-side plot)
    - Feature importance chart (7 fused features)
    - Cross-validation summary

Usage:
    evaluator = EvaluateModel(data, model_dir, results_dir)
    evaluator.evaluate_pipeline()
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import config


class EvaluateModel:
    """Evaluate AQI and fan-speed models with full metrics + plots."""

    def __init__(self, X_test, y_aqi_test, y_fan_test,
                 model_dir=config.SYNTH_MODEL_DIR,
                 results_dir=config.SYNTH_RESULTS_DIR):
        """
        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix (7 fused features).
        y_aqi_test : pd.Series
            True AQI class labels for test set.
        y_fan_test : pd.Series
            True fan speed labels for test set.
        model_dir : str
            Directory containing saved model .pkl files.
        results_dir : str
            Directory to save evaluation outputs.
        """
        self.X_test = X_test
        self.y_aqi_test = y_aqi_test
        self.y_fan_test = y_fan_test
        self.model_dir = model_dir
        self.results_dir = results_dir

        self.aqi_model = None
        self.fan_model = None
        self.y_aqi_pred = None
        self.y_fan_pred = None
        self.metrics = {}

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    def load_models(self):
        """Load both models from disk."""
        try:
            aqi_path = os.path.join(self.model_dir, "aqi_model.pkl")
            fan_path = os.path.join(self.model_dir, "fan_speed_model.pkl")

            self.aqi_model = joblib.load(aqi_path)
            self.fan_model = joblib.load(fan_path)

            print("Models loaded successfully:")
            print(f"  AQI model:  {aqi_path}")
            print(f"  Fan model:  {fan_path}\n")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self):
        """Generate predictions for the test set."""
        self.y_aqi_pred = self.aqi_model.predict(self.X_test)
        self.y_fan_pred = self.fan_model.predict(self.X_test)
        print(f"Predictions generated for {len(self.X_test)} test samples.\n")
        return True

    # ------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------
    def calculate_metrics(self):
        """Compute accuracy, classification reports, and confusion matrices."""
        self.metrics["aqi_accuracy"] = accuracy_score(self.y_aqi_test, self.y_aqi_pred)
        self.metrics["fan_accuracy"] = accuracy_score(self.y_fan_test, self.y_fan_pred)

        self.metrics["aqi_report"] = classification_report(self.y_aqi_test, self.y_aqi_pred)
        self.metrics["fan_report"] = classification_report(self.y_fan_test, self.y_fan_pred)

        self.metrics["aqi_cm"] = confusion_matrix(self.y_aqi_test, self.y_aqi_pred)
        self.metrics["fan_cm"] = confusion_matrix(self.y_fan_test, self.y_fan_pred)

    def display_metrics(self):
        """Print all evaluation metrics."""
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        print(f"\n--- AQI Classification Model ---")
        print(f"Accuracy: {self.metrics['aqi_accuracy']:.4f}\n")
        print("Confusion Matrix:")
        print(self.metrics["aqi_cm"])
        print("\nClassification Report:")
        print(self.metrics["aqi_report"])

        print(f"\n--- Fan Speed Classification Model ---")
        print(f"Accuracy: {self.metrics['fan_accuracy']:.4f}\n")
        print("Confusion Matrix:")
        print(self.metrics["fan_cm"])
        print("\nClassification Report:")
        print(self.metrics["fan_report"])
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def save_confusion_matrices(self):
        """Save side-by-side confusion matrices for both models."""
        os.makedirs(self.results_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # AQI confusion matrix
        aqi_labels = sorted(self.y_aqi_test.unique())
        ConfusionMatrixDisplay(
            confusion_matrix=self.metrics["aqi_cm"],
            display_labels=aqi_labels,
        ).plot(ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("AQI Classification", fontsize=13, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=30)

        # Fan speed confusion matrix
        fan_labels = sorted(self.y_fan_test.unique())
        ConfusionMatrixDisplay(
            confusion_matrix=self.metrics["fan_cm"],
            display_labels=fan_labels,
        ).plot(ax=axes[1], cmap="Oranges", colorbar=False)
        axes[1].set_title("Fan Speed Classification", fontsize=13, fontweight="bold")

        plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.results_dir, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Confusion matrices saved: {path}")

    def save_feature_importance(self):
        """Save feature importance bar chart for both models."""
        os.makedirs(self.results_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (name, model) in enumerate(
            [("AQI", self.aqi_model), ("Fan Speed", self.fan_model)]
        ):
            imp_df = pd.DataFrame({
                "Feature": config.FUSED_FEATURES,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            axes[idx].barh(
                imp_df["Feature"], imp_df["Importance"],
                color=["#3498db", "#e74c3c"][idx], edgecolor="white"
            )
            axes[idx].set_title(f"{name} Model", fontsize=13, fontweight="bold")
            axes[idx].set_xlabel("Importance")

        plt.suptitle("Feature Importance", fontsize=15, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.results_dir, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Feature importance chart saved: {path}")

        # Also save as CSV
        csv_path = os.path.join(self.results_dir, "feature_importance.csv")
        aqi_imp = pd.DataFrame({
            "Feature": config.FUSED_FEATURES,
            "AQI_Importance": self.aqi_model.feature_importances_,
            "Fan_Importance": self.fan_model.feature_importances_,
        }).sort_values("AQI_Importance", ascending=False)
        aqi_imp.to_csv(csv_path, index=False)
        print(f"[OK] Feature importance CSV saved: {csv_path}")

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------
    def save_reports(self):
        """Save classification reports as text files."""
        os.makedirs(self.results_dir, exist_ok=True)

        # Combined report
        report_path = os.path.join(self.results_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TUNNEL EXHAUST SYSTEM - MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("--- AQI Classification Model ---\n")
            f.write(f"Accuracy: {self.metrics['aqi_accuracy']:.4f}\n\n")
            f.write(self.metrics["aqi_report"])

            f.write("\n\n--- Fan Speed Classification Model ---\n")
            f.write(f"Accuracy: {self.metrics['fan_accuracy']:.4f}\n\n")
            f.write(self.metrics["fan_report"])

        print(f"[OK] Classification report saved: {report_path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def evaluate_pipeline(self):
        """Run the full evaluation pipeline."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION PIPELINE")
        print("=" * 60 + "\n")

        if not self.load_models():
            return
        if not self.predict():
            return

        self.calculate_metrics()
        self.display_metrics()
        self.save_confusion_matrices()
        self.save_feature_importance()
        self.save_reports()

        print("\nEvaluation completed successfully!\n")


# ======================================================================
# Standalone entry point
# ======================================================================
if __name__ == "__main__":
    from sensor_fusion import SensorFusion
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(config.CLEANED_CSV)
    fuser = SensorFusion()
    data = fuser.fuse(data)

    X = data[config.FUSED_FEATURES]
    y_aqi = data[config.TARGET_AQI]
    y_fan = data[config.TARGET_FAN]

    _, X_test, _, y_aqi_test, _, y_fan_test = train_test_split(
        X, y_aqi, y_fan,
        test_size=config.TEST_SIZE,
        random_state=config.SPLIT_RANDOM_STATE,
        stratify=y_aqi,
    )

    evaluator = EvaluateModel(X_test, y_aqi_test, y_fan_test)
    evaluator.evaluate_pipeline()

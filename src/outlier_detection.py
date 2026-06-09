"""
outlier_detection.py — Isolation Forest outlier removal for multi-sensor data.

Why Isolation Forest?
    1. Multi-variate: operates on all sensor channels simultaneously, capturing
       cross-sensor anomalies (e.g., physically impossible temp + humidity combos).
    2. No distribution assumption: unlike Z-score (Gaussian) or IQR (symmetric),
       Isolation Forest works on any data shape.
    3. Efficient: O(n log n) — much faster than LOF (O(n²)) for our dataset.
    4. Single tuning knob: `contamination` sets the expected outlier fraction.

Usage:
    detector = OutlierDetector(contamination=0.05)
    clean_df, report = detector.fit_remove(df)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import config


class OutlierDetector:
    """Multivariate outlier detector using Isolation Forest on sensor columns."""

    def __init__(
        self,
        contamination=config.OUTLIER_CONTAMINATION,
        n_estimators=config.OUTLIER_N_ESTIMATORS,
        random_state=config.OUTLIER_RANDOM_STATE,
    ):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.sensor_cols = config.SENSOR_COLUMNS
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def fit_remove(self, df):
        """
        Fit the Isolation Forest on sensor columns and remove detected outliers.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the 4 sensor columns.

        Returns
        -------
        clean_df : pd.DataFrame
            DataFrame with outlier rows removed.
        report : dict
            Summary of outlier removal: counts per class, total removed, etc.
        """
        print("\n" + "=" * 50)
        print("OUTLIER DETECTION (Isolation Forest)")
        print("=" * 50)

        sensor_data = df[self.sensor_cols].copy()
        initial_count = len(df)

        # Fit and predict: 1 = inlier, -1 = outlier
        predictions = self.model.fit_predict(sensor_data)
        self.is_fitted = True

        # Anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(sensor_data)

        # Separate inliers and outliers
        outlier_mask = predictions == -1
        clean_df = df[~outlier_mask].reset_index(drop=True)
        outlier_df = df[outlier_mask]

        removed_count = initial_count - len(clean_df)

        # Build per-class report
        report = {
            "initial_count": initial_count,
            "removed_count": removed_count,
            "final_count": len(clean_df),
            "contamination_setting": self.contamination,
            "per_class": {},
        }

        if config.TARGET_AQI in df.columns:
            for cls in df[config.TARGET_AQI].unique():
                class_mask = df[config.TARGET_AQI] == cls
                class_total = class_mask.sum()
                class_outliers = (class_mask & outlier_mask).sum()
                report["per_class"][cls] = {
                    "total": int(class_total),
                    "removed": int(class_outliers),
                    "kept": int(class_total - class_outliers),
                }

        # Print report
        print(f"\nContamination setting: {self.contamination}")
        print(f"Initial samples:      {initial_count}")
        print(f"Outliers removed:     {removed_count}")
        print(f"Clean samples:        {len(clean_df)}")

        if report["per_class"]:
            print(f"\n{'Class':<25} {'Total':>6} {'Removed':>8} {'Kept':>6}")
            print("-" * 50)
            for cls, info in report["per_class"].items():
                print(
                    f"{cls:<25} {info['total']:>6} "
                    f"{info['removed']:>8} {info['kept']:>6}"
                )

        print("\nScore statistics (lower = more anomalous):")
        print(f"  Min:  {scores.min():.4f}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Max:  {scores.max():.4f}")
        print("=" * 50 + "\n")

        return clean_df, report

    # ------------------------------------------------------------------
    # Predict on new data (for real-data pipeline)
    # ------------------------------------------------------------------
    def predict(self, df):
        """
        Use the already-fitted model to flag outliers in new data.

        Returns a boolean mask where True = outlier.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit_remove() first.")

        sensor_data = df[self.sensor_cols]
        predictions = self.model.predict(sensor_data)
        return predictions == -1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory):
        """Save the fitted Isolation Forest model to disk."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "outlier_detector.pkl")
        joblib.dump(self.model, path)
        print(f"Outlier detector saved: {path}")

    def load(self, directory):
        """Load a previously saved Isolation Forest model."""
        path = os.path.join(directory, "outlier_detector.pkl")
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Outlier detector loaded: {path}")


# ======================================================================
# Standalone test
# ======================================================================
if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    detector = OutlierDetector()
    clean, report = detector.fit_remove(data)
    print(f"\nCleaned data shape: {clean.shape}")
    print(clean.head())

"""
label_engine.py — Engineering-rule-based label generation and validation.

This module encodes the domain-expert threshold rules that map raw sensor
values to air quality classes and fan speed levels. The rules are documented
in config.py (AQI_THRESHOLDS) and are fully reproducible.

Threshold Logic (priority order — checked top-to-bottom):
┌────────────────────────┬─────────┬──────────┬──────────┬──────────┐
│ Class                  │ Temp °C │ Hum %    │ MQ2      │ MQ3      │
├────────────────────────┼─────────┼──────────┼──────────┼──────────┤
│ Hazardous Smoke        │ 28–40   │ 30–50    │ 450–1000 │ 120–250  │
│ Chemical Vapor         │ 20–25   │ 50–60    │ 130–180  │ 250–700  │
│ Increased Ventilation  │ 24–30   │ 65–80    │ 170–250  │ 90–110   │
│ Normal                 │ 20–25   │ 50–60    │ 120–150  │ 80–100   │
└────────────────────────┴─────────┴──────────┴──────────┴──────────┘

A reading is assigned to the FIRST class whose thresholds it satisfies
(with a 10 % tolerance margin). If no class matches, it is assigned
to the nearest class by Euclidean distance in normalized sensor space.

Usage:
    engine = LabelEngine()
    df = engine.apply_labels(df)             # Add/overwrite label columns
    report = engine.validate_labels(df)      # Compare CSV labels vs. rules
"""

import numpy as np
import pandas as pd

import config


class LabelEngine:
    """Generates and validates AQI + fan-speed labels from sensor readings."""

    def __init__(self, tolerance=0.10):
        """
        Parameters
        ----------
        tolerance : float
            Fractional tolerance on threshold boundaries (0.10 = 10 %).
            Accounts for sensor noise at class boundaries.
        """
        self.thresholds = config.AQI_THRESHOLDS
        self.priority = config.AQI_PRIORITY
        self.fan_map = config.FAN_SPEED_MAP
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Single-row classification
    # ------------------------------------------------------------------
    def _in_range(self, value, lo, hi):
        """Check if value is within [lo, hi] with tolerance margin."""
        margin = (hi - lo) * self.tolerance
        return (lo - margin) <= value <= (hi + margin)

    def classify_row(self, row):
        """
        Classify a single sensor reading into an AQI class.

        Parameters
        ----------
        row : dict-like
            Must contain keys: Temperature, Humidity, MQ2_Raw, MQ3_Raw.

        Returns
        -------
        str : AQI class label.
        """
        for cls in self.priority:
            ranges = self.thresholds[cls]
            match = all(
                self._in_range(row[col], lo, hi)
                for col, (lo, hi) in ranges.items()
            )
            if match:
                return cls

        # Fallback: assign to nearest class by normalized Euclidean distance
        return self._nearest_class(row)

    def _nearest_class(self, row):
        """Find the class whose centroid is closest to the reading."""
        best_class = "Normal"
        best_dist = float("inf")

        for cls in self.priority:
            ranges = self.thresholds[cls]
            dist = 0
            for col, (lo, hi) in ranges.items():
                mid = (lo + hi) / 2
                span = (hi - lo) if (hi - lo) > 0 else 1
                dist += ((row[col] - mid) / span) ** 2
            dist = np.sqrt(dist)
            if dist < best_dist:
                best_dist = dist
                best_class = cls

        return best_class

    def get_fan_speed(self, aqi_class):
        """
        Map an AQI class to a fan-speed percentage.

        Returns
        -------
        int : Fan speed as 0–100 %.
        """
        return self.fan_map.get(aqi_class, self.fan_map["Unknown"])["percent"]

    # ------------------------------------------------------------------
    # DataFrame operations
    # ------------------------------------------------------------------
    def apply_labels(self, df):
        """
        Apply rule-based labels to every row of the DataFrame.

        Adds columns:
            - Rule_AQI_Class    : AQI class derived from thresholds
            - Rule_Fan_Speed    : Fan speed % derived from AQI class

        Parameters
        ----------
        df : pd.DataFrame with sensor columns.

        Returns
        -------
        pd.DataFrame with the two new columns appended.
        """
        print("\n" + "=" * 50)
        print("LABEL ENGINE - Applying engineering rules")
        print("=" * 50)

        df = df.copy()
        df["Rule_AQI_Class"] = df.apply(self.classify_row, axis=1)
        df["Rule_Fan_Speed"] = df["Rule_AQI_Class"].map(self.get_fan_speed)

        # Distribution of generated labels
        print("\nRule-based AQI class distribution:")
        print(df["Rule_AQI_Class"].value_counts().to_string())
        print("\nRule-based fan speed distribution:")
        print(df["Rule_Fan_Speed"].value_counts().to_string())
        print("=" * 50 + "\n")

        return df

    def validate_labels(self, df):
        """
        Compare existing CSV labels against rule-generated labels.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain both the original labels (Future_AQI_Class,
            Fan_Speed_Percent) and the rule-generated labels (Rule_AQI_Class,
            Rule_Fan_Speed).

        Returns
        -------
        report : dict with mismatch counts and details.
        """
        print("\n" + "=" * 50)
        print("LABEL VALIDATION - CSV vs. Engineering Rules")
        print("=" * 50)

        report = {"aqi_mismatches": 0, "fan_mismatches": 0, "details": []}

        if config.TARGET_AQI not in df.columns:
            print("No existing AQI labels to validate against.")
            return report

        # AQI class comparison
        aqi_match = df[config.TARGET_AQI] == df["Rule_AQI_Class"]
        aqi_mismatches = (~aqi_match).sum()
        report["aqi_mismatches"] = int(aqi_mismatches)

        # Fan speed comparison
        if config.TARGET_FAN in df.columns:
            fan_match = df[config.TARGET_FAN] == df["Rule_Fan_Speed"]
            fan_mismatches = (~fan_match).sum()
            report["fan_mismatches"] = int(fan_mismatches)
        else:
            fan_mismatches = "N/A"

        print(f"\nTotal rows:           {len(df)}")
        print(f"AQI class mismatches: {aqi_mismatches}")
        print(f"Fan speed mismatches: {fan_mismatches}")

        if aqi_mismatches > 0:
            mismatch_df = df[~aqi_match][
                config.SENSOR_COLUMNS
                + [config.TARGET_AQI, "Rule_AQI_Class"]
            ]
            print(f"\nFirst 10 AQI mismatches:")
            print(mismatch_df.head(10).to_string(index=False))
            report["details"] = mismatch_df.head(20).to_dict("records")
        else:
            print("\n[OK] All labels match the engineering rules perfectly.")

        print("=" * 50 + "\n")
        return report


# ======================================================================
# Standalone test
# ======================================================================
if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    engine = LabelEngine()
    data = engine.apply_labels(data)
    report = engine.validate_labels(data)
    print(f"Mismatches: {report['aqi_mismatches']}")

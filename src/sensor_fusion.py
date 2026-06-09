"""
sensor_fusion.py — Multi-sensor feature engineering and fusion.

Takes the 4 raw sensor columns and produces 3 additional engineered features
that encode domain knowledge about tunnel air quality:

    Gas_Index       = 0.6 × MQ2_norm + 0.4 × MQ3_norm
                      Weighted overall pollution indicator. MQ2 (smoke/CO/LPG)
                      gets higher weight as the primary hazard sensor.

    Heat_Stress     = Temperature × (1 − Humidity / 100)
                      Proxy for convective drying effect. High values indicate
                      hot-dry conditions typical of fire/smoke events.

    MQ2_MQ3_Ratio   = MQ2_Raw / (MQ3_Raw + 1)
                      Key discriminator: Hazardous Smoke (ratio >> 1) vs.
                      Chemical Vapor (ratio << 1). The +1 prevents division
                      by zero.

Usage:
    fuser = SensorFusion()
    df = fuser.fuse(df)  # Adds 3 new columns
"""

import pandas as pd
import numpy as np

import config


class SensorFusion:
    """Creates fused sensor features from raw DHT11 + MQ2 + MQ3 readings."""

    def __init__(
        self,
        mq2_weight=config.GAS_INDEX_MQ2_WEIGHT,
        mq3_weight=config.GAS_INDEX_MQ3_WEIGHT,
    ):
        self.mq2_weight = mq2_weight
        self.mq3_weight = mq3_weight
        self.sensor_cols = config.SENSOR_COLUMNS

        # We'll compute normalization stats from the training data
        self._mq2_min = None
        self._mq2_max = None
        self._mq3_min = None
        self._mq3_max = None

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------
    def fuse(self, df, fit_normalization=True):
        """
        Add engineered features to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain Temperature, Humidity, MQ2_Raw, MQ3_Raw.
        fit_normalization : bool
            If True, compute min/max from this data for Gas_Index normalization.
            If False, use previously fitted values (for prediction on new data).

        Returns
        -------
        pd.DataFrame with 3 additional columns: Gas_Index, Heat_Stress,
        MQ2_MQ3_Ratio.
        """
        print("\n" + "=" * 50)
        print("SENSOR FUSION - Feature Engineering")
        print("=" * 50)

        df = df.copy()

        # --- 1. Gas Index (weighted normalized combination) ---
        if fit_normalization:
            self._mq2_min = df["MQ2_Raw"].min()
            self._mq2_max = df["MQ2_Raw"].max()
            self._mq3_min = df["MQ3_Raw"].min()
            self._mq3_max = df["MQ3_Raw"].max()

        mq2_range = self._mq2_max - self._mq2_min
        mq3_range = self._mq3_max - self._mq3_min

        # Avoid division by zero if all values are identical
        mq2_norm = (
            (df["MQ2_Raw"] - self._mq2_min) / mq2_range
            if mq2_range > 0
            else 0.0
        )
        mq3_norm = (
            (df["MQ3_Raw"] - self._mq3_min) / mq3_range
            if mq3_range > 0
            else 0.0
        )

        df["Gas_Index"] = (
            self.mq2_weight * mq2_norm + self.mq3_weight * mq3_norm
        )

        # --- 2. Heat Stress (convective heat indicator) ---
        df["Heat_Stress"] = df["Temperature"] * (1 - df["Humidity"] / 100)

        # --- 3. MQ2/MQ3 Ratio (smoke vs. vapor discriminator) ---
        df["MQ2_MQ3_Ratio"] = df["MQ2_Raw"] / (df["MQ3_Raw"] + 1)

        # Print summary
        print(f"\nNew features added: Gas_Index, Heat_Stress, MQ2_MQ3_Ratio")
        print(f"Total feature count: {len(config.FUSED_FEATURES)}")
        print(f"\nFused feature statistics:")
        print(
            df[["Gas_Index", "Heat_Stress", "MQ2_MQ3_Ratio"]]
            .describe()
            .round(3)
            .to_string()
        )
        print("=" * 50 + "\n")

        return df

    # ------------------------------------------------------------------
    # Normalization state (for saving/loading with the model)
    # ------------------------------------------------------------------
    def get_norm_params(self):
        """Return normalization parameters for serialization."""
        return {
            "mq2_min": self._mq2_min,
            "mq2_max": self._mq2_max,
            "mq3_min": self._mq3_min,
            "mq3_max": self._mq3_max,
        }

    def set_norm_params(self, params):
        """Restore normalization parameters from serialized state."""
        self._mq2_min = params["mq2_min"]
        self._mq2_max = params["mq2_max"]
        self._mq3_min = params["mq3_min"]
        self._mq3_max = params["mq3_max"]


# ======================================================================
# Standalone test
# ======================================================================
if __name__ == "__main__":
    data = pd.read_csv(config.RAW_CSV)
    fuser = SensorFusion()
    fused = fuser.fuse(data)
    print(fused[config.FUSED_FEATURES].head(10))

import os
#base path
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# Data directories
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
CLEANED_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "Cleaned")
REAL_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "real")

# Model directories
SYNTH_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
REAL_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "real")

# Results directories
SYNTH_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "synth")
REAL_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "real")

# Default file names
RAW_CSV = os.path.join(RAW_DATA_DIR, "tunnel_ventilation.csv")
CLEANED_CSV = os.path.join(CLEANED_DATA_DIR, "tunnel_ventilation_cleaned.csv")

# SENSOR COLUMNS

SENSOR_COLUMNS = ["Temperature", "Humidity", "MQ2_Raw", "MQ3_Raw"]
TARGET_AQI = "Future_AQI_Class"
TARGET_FAN = "Fan_Speed_Percent"

# All features used by the model (raw + fused)
FUSED_FEATURES = [
    "Temperature", "Humidity", "MQ2_Raw", "MQ3_Raw",
    "Gas_Index", "Heat_Stress", "MQ2_MQ3_Ratio",
]

# ==============================================================================
# PHYSICAL SENSOR LIMITS (for sanity-check validation)
# DHT11 datasheet limits + MQ-series analog range (10-bit ADC: 0-1023)
# ==============================================================================
SENSOR_LIMITS = {
    "Temperature": (0, 50),      # DHT11: 0–50 °C
    "Humidity":    (20, 100),     # DHT11: 20–90% (allow up to 100 for margin)
    "MQ2_Raw":     (0, 1023),    # 10-bit ADC range
    "MQ3_Raw":     (0, 1023),    # 10-bit ADC range
}

# ==============================================================================
# ENGINEERING THRESHOLD RULES — AQI Classification
# Each class is defined by the observed sensor ranges from the synthesized data.
# Priority order (checked top→bottom): Hazardous > Chemical Vapor > Increased > Normal
# ==============================================================================
AQI_THRESHOLDS = {
    "Hazardous Smoke": {
        "Temperature": (28, 40),
        "Humidity":    (30, 50),
        "MQ2_Raw":     (450, 1000),
        "MQ3_Raw":     (120, 250),
    },
    "Chemical Vapor": {
        "Temperature": (20, 25),
        "Humidity":    (50, 60),
        "MQ2_Raw":     (130, 180),
        "MQ3_Raw":     (250, 700),
    },
    "Increased Ventilation": {
        "Temperature": (24, 30),
        "Humidity":    (65, 80),
        "MQ2_Raw":     (170, 250),
        "MQ3_Raw":     (90, 110),
    },
    "Normal": {
        "Temperature": (20, 25),
        "Humidity":    (50, 60),
        "MQ2_Raw":     (120, 150),
        "MQ3_Raw":     (80, 100),
    },
}

# Priority order for classification (most dangerous first)
AQI_PRIORITY = ["Hazardous Smoke", "Chemical Vapor", "Increased Ventilation", "Normal"]

# ==============================================================================
# FAN SPEED MAPPING — Derived from AQI class
# ==============================================================================
FAN_SPEED_MAP = {
    "Normal":                 {"percent": 30,  "label": "LOW"},
    "Increased Ventilation":  {"percent": 60,  "label": "MEDIUM"},
    "Chemical Vapor":         {"percent": 80,  "label": "MEDIUM-HIGH"},
    "Hazardous Smoke":        {"percent": 100, "label": "HIGH"},
    "Unknown":                {"percent": 0,   "label": "OFF"},
}

# ==============================================================================
# ISOLATION FOREST HYPERPARAMETERS
# ==============================================================================
OUTLIER_CONTAMINATION = 0.05   # Fraction of data expected to be outliers
OUTLIER_RANDOM_STATE = 42
OUTLIER_N_ESTIMATORS = 100

# ==============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ==============================================================================
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None            # Let trees grow fully
RF_MIN_SAMPLES_SPLIT = 5
RF_RANDOM_STATE = 42
RF_N_JOBS = -1                 # Use all CPU cores

# ==============================================================================
# TRAIN / TEST SPLIT
# ==============================================================================
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
CV_FOLDS = 5                   # Stratified cross-validation folds

# ==============================================================================
# SENSOR FUSION WEIGHTS
# ==============================================================================
GAS_INDEX_MQ2_WEIGHT = 0.6     # MQ2 is primary gas/smoke sensor
GAS_INDEX_MQ3_WEIGHT = 0.4     # MQ3 supplements with alcohol/benzene

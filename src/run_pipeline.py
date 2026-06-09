"""
run_pipeline.py — Main orchestrator for the Tunnel Exhaust System ML Pipeline.

Runs all stages in sequence:
    1. Preprocessing    — Load, clean, validate sensor ranges
    2. Outlier Detection — Isolation Forest removal of anomalous readings
    3. Label Engine     — Generate/validate labels from engineering rules
    4. Sensor Fusion    — Create 3 engineered features from raw sensors
    5. Training         — Dual Random Forest (AQI class + fan speed)
    6. Evaluation       — Metrics, confusion matrices, feature importance
    7. Fan Control      — Demonstrate end-to-end prediction → command output

Usage:
    python run_pipeline.py --mode synth
    python run_pipeline.py --mode real --data-path ../data/real/my_sensor_data.csv
    python run_pipeline.py --mode synth --skip-outlier
"""

import argparse
import os
import sys
import time

import pandas as pd

# Ensure the src directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing import Preprocessing
from outlier_detection import OutlierDetector
from label_engine import LabelEngine
from sensor_fusion import SensorFusion
from train_model import TrainModel
from evaluate_model import EvaluateModel
from fan_control import FanController


def print_banner(title):
    """Print a prominent stage banner."""
    width = 60
    print("\n" + "#" * width)
    print(f"#  {title:<{width-4}}#")
    print("#" * width + "\n")


def run_synth_pipeline(data_path, skip_outlier=False):
    """
    Full pipeline for synthesized data.

    Parameters
    ----------
    data_path : str
        Path to the raw synthesized CSV.
    skip_outlier : bool
        If True, skip the Isolation Forest outlier removal step.
    """
    start_time = time.time()
    model_dir = config.SYNTH_MODEL_DIR
    results_dir = config.SYNTH_RESULTS_DIR

    # Ensure output directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ================================================================
    # STAGE 1: Preprocessing
    # ================================================================
    print_banner("STAGE 1: DATA PREPROCESSING")
    preprocessor = Preprocessing(data_path)
    data = preprocessor.preprocess()
    if data is None:
        print("ERROR: Preprocessing failed. Aborting pipeline.")
        return False

    # ================================================================
    # STAGE 2: Outlier Detection
    # ================================================================
    if not skip_outlier:
        print_banner("STAGE 2: OUTLIER DETECTION")
        detector = OutlierDetector()
        data, outlier_report = detector.fit_remove(data)
        detector.save(model_dir)
    else:
        print_banner("STAGE 2: OUTLIER DETECTION [SKIPPED]")
        outlier_report = {"removed_count": 0, "note": "Skipped by user"}

    # ================================================================
    # STAGE 3: Label Engine
    # ================================================================
    print_banner("STAGE 3: ENGINEERING RULE LABELS")
    engine = LabelEngine()
    data = engine.apply_labels(data)
    label_report = engine.validate_labels(data)

    # ================================================================
    # STAGE 4: Sensor Fusion
    # ================================================================
    print_banner("STAGE 4: SENSOR FUSION")
    fuser = SensorFusion()
    data = fuser.fuse(data, fit_normalization=True)

    # ================================================================
    # STAGE 5: Training
    # ================================================================
    print_banner("STAGE 5: MODEL TRAINING")
    trainer = TrainModel(data, model_dir=model_dir)
    success = trainer.train_pipeline(fusion_params=fuser.get_norm_params())
    if not success:
        print("ERROR: Training failed. Aborting pipeline.")
        return False

    # ================================================================
    # STAGE 6: Evaluation
    # ================================================================
    print_banner("STAGE 6: MODEL EVALUATION")
    evaluator = EvaluateModel(
        X_test=trainer.X_test,
        y_aqi_test=trainer.y_aqi_test,
        y_fan_test=trainer.y_fan_test,
        model_dir=model_dir,
        results_dir=results_dir,
    )
    evaluator.evaluate_pipeline()

    # ================================================================
    # STAGE 7: Fan Control Demo
    # ================================================================
    print_banner("STAGE 7: FAN CONTROL OUTPUT")
    controller = FanController(model_dir=model_dir)
    controller.load()

    # Demo predictions for each scenario
    demo_readings = [
        {"Temperature": 22.0, "Humidity": 55.0, "MQ2_Raw": 130, "MQ3_Raw": 90,
         "scenario": "Normal conditions"},
        {"Temperature": 27.0, "Humidity": 72.0, "MQ2_Raw": 210, "MQ3_Raw": 100,
         "scenario": "Increased ventilation needed"},
        {"Temperature": 35.0, "Humidity": 38.0, "MQ2_Raw": 750, "MQ3_Raw": 190,
         "scenario": "Hazardous smoke detected"},
        {"Temperature": 22.5, "Humidity": 55.0, "MQ2_Raw": 155, "MQ3_Raw": 450,
         "scenario": "Chemical vapor detected"},
    ]

    print("\n--- Fan Control Decision Demo ---\n")
    print(
        f"{'Scenario':<30} | {'AQI Class':<23} | {'Fan':>4}% | "
        f"{'Serial Command'}"
    )
    print("-" * 75)

    for reading in demo_readings:
        scenario = reading.pop("scenario")
        cmd = controller.predict_and_command(reading)
        serial = FanController.format_serial_command(cmd)
        mqtt = FanController.format_mqtt_json(cmd)

        print(
            f"{scenario:<30} | {cmd['aqi_class']:<23} | "
            f"{cmd['fan_speed_percent']:>4}% | {serial}"
        )

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - start_time
    print_banner("PIPELINE COMPLETE - SUMMARY")
    print(f"  Total time:          {elapsed:.1f} seconds")
    print(f"  Data source:         {data_path}")
    print(f"  Samples processed:   {len(data)}")
    print(f"  Outliers removed:    {outlier_report.get('removed_count', 0)}")
    print(f"  Label mismatches:    {label_report.get('aqi_mismatches', 0)}")
    print(f"  AQI model accuracy:  {evaluator.metrics.get('aqi_accuracy', 'N/A'):.4f}")
    print(f"  Fan model accuracy:  {evaluator.metrics.get('fan_accuracy', 'N/A'):.4f}")
    print(f"  Models saved to:     {model_dir}")
    print(f"  Results saved to:    {results_dir}")
    print()

    return True


def run_real_pipeline(data_path):
    """
    Pipeline for real sensor data. Same stages, different directories.

    The real-data pipeline:
        - Uses the same preprocessing + outlier detection
        - If labels are missing, auto-generates them from the label engine
        - Trains fresh models saved to models/real/
        - Outputs results to results/real/
    """
    start_time = time.time()
    model_dir = config.REAL_MODEL_DIR
    results_dir = config.REAL_RESULTS_DIR

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # STAGE 1: Preprocessing
    print_banner("STAGE 1: REAL DATA PREPROCESSING")
    preprocessor = Preprocessing(data_path)
    data = preprocessor.preprocess()
    if data is None:
        print("ERROR: Preprocessing failed.")
        return False

    # STAGE 2: Outlier Detection
    print_banner("STAGE 2: OUTLIER DETECTION")
    detector = OutlierDetector()
    data, outlier_report = detector.fit_remove(data)
    detector.save(model_dir)

    # STAGE 3: Label Engine
    print_banner("STAGE 3: LABEL GENERATION")
    engine = LabelEngine()
    data = engine.apply_labels(data)

    # If the CSV has no labels, use rule-generated labels as ground truth
    if config.TARGET_AQI not in data.columns:
        print("No existing labels found - using rule-generated labels.")
        data[config.TARGET_AQI] = data["Rule_AQI_Class"]
        data[config.TARGET_FAN] = data["Rule_Fan_Speed"]
    else:
        engine.validate_labels(data)

    # STAGE 4: Sensor Fusion
    print_banner("STAGE 4: SENSOR FUSION")
    fuser = SensorFusion()
    data = fuser.fuse(data, fit_normalization=True)

    # STAGE 5: Training
    print_banner("STAGE 5: MODEL TRAINING (Real Data)")
    trainer = TrainModel(data, model_dir=model_dir)
    success = trainer.train_pipeline(fusion_params=fuser.get_norm_params())
    if not success:
        print("ERROR: Training failed.")
        return False

    # STAGE 6: Evaluation
    print_banner("STAGE 6: MODEL EVALUATION (Real Data)")
    evaluator = EvaluateModel(
        X_test=trainer.X_test,
        y_aqi_test=trainer.y_aqi_test,
        y_fan_test=trainer.y_fan_test,
        model_dir=model_dir,
        results_dir=results_dir,
    )
    evaluator.evaluate_pipeline()

    # STAGE 7: Fan Control Demo
    print_banner("STAGE 7: FAN CONTROL SIMULATION (Real Data)")
    controller = FanController(model_dir=model_dir)
    controller.load()
    controller.simulate_live(data_path, delay=0.05, max_rows=15)

    # Summary
    elapsed = time.time() - start_time
    print_banner("REAL DATA PIPELINE COMPLETE")
    print(f"  Total time:          {elapsed:.1f} seconds")
    print(f"  Data source:         {data_path}")
    print(f"  Samples processed:   {len(data)}")
    print(f"  AQI model accuracy:  {evaluator.metrics.get('aqi_accuracy', 'N/A'):.4f}")
    print(f"  Fan model accuracy:  {evaluator.metrics.get('fan_accuracy', 'N/A'):.4f}")
    print(f"  Models saved to:     {model_dir}")
    print(f"  Results saved to:    {results_dir}")
    print()
    return True


# ======================================================================
# CLI entry point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tunnel Exhaust System — ML Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["synth", "real"],
        default="synth",
        help="Pipeline mode:\n"
             "  synth — Train on synthesized dataset (default)\n"
             "  real  — Train on real sensor data",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to input CSV. Defaults to the synthesized dataset.",
    )
    parser.add_argument(
        "--skip-outlier",
        action="store_true",
        help="Skip the Isolation Forest outlier removal step.",
    )

    args = parser.parse_args()

    # Resolve data path
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
    else:
        data_path = config.RAW_CSV

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#                                                          #")
    print("#   TUNNEL EXHAUST SYSTEM - ML PIPELINE                    #")
    print("#   Sensors: DHT11 + MQ2 + MQ3 -> Air Quality -> Fan Speed  #")
    print("#                                                          #")
    print("#" * 60)
    print(f"\n  Mode:       {args.mode}")
    print(f"  Data path:  {data_path}")
    print(f"  Skip outlier detection: {args.skip_outlier}")

    if args.mode == "synth":
        success = run_synth_pipeline(data_path, skip_outlier=args.skip_outlier)
    else:
        success = run_real_pipeline(data_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

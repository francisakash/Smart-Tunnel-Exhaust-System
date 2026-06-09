import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
class EvaluateModel:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully")
            print(f"Model Path: {self.model_path}\n")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            print("Dataset loaded successfully")
            print(f"Shape: {df.shape}\n")
            X = df[
                [
                    'Temperature',
                    'Humidity',
                    'MQ2_Raw',
                    'MQ3_Raw'
                ]
            ]
            y = df['Future_AQI_Class']
            _, self.X_test, _, self.y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            print(f"Test samples: {len(self.X_test)}\n")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    def predict(self):
        try:
            self.y_pred = self.model.predict(self.X_test)

            print("Predictions generated\n")

            return True

        except Exception as e:
            print(f"Error generating predictions: {e}")
            return False
    def calculate_metrics(self):
        self.metrics['accuracy'] = accuracy_score(
            self.y_test,
            self.y_pred
        )
        self.metrics['confusion_matrix'] = confusion_matrix(
            self.y_test,
            self.y_pred
        )
        self.metrics['classification_report'] = classification_report(
            self.y_test,
            self.y_pred
        )
    def display_metrics(self):

        print("=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)

        print(f"Accuracy: {self.metrics['accuracy']:.4f}\n")

        print("Confusion Matrix:")
        print(self.metrics['confusion_matrix'])

        print("\nClassification Report:")
        print(self.metrics['classification_report'])

    def feature_importance(self):
        try:
            feature_names = [
                'Temperature',
                'Humidity',
                'MQ2_Raw',
                'MQ3_Raw'
            ]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            })
            importance_df = importance_df.sort_values(
                by='Importance',
                ascending=False
            )
            print("\nFeature Importance:")
            print(importance_df)
            os.makedirs("../results", exist_ok=True)
            importance_df.to_csv(
                "../results/feature_importance.csv",
                index=False
            )
            plt.figure(figsize=(8, 5))
            plt.bar(
                importance_df['Feature'],
                importance_df['Importance']
            )
            plt.title("Feature Importance")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(
                "../results/feature_importance.png"
            )
            plt.close()
            print("\nFeature importance saved")
        except Exception as e:
            print(f"Error generating feature importance: {e}")

    def save_confusion_matrix(self):

        try:

            os.makedirs("../results", exist_ok=True)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=self.metrics[
                    'confusion_matrix'
                ]
            )

            disp.plot()

            plt.tight_layout()

            plt.savefig(
                "../results/confusion_matrix.png"
            )

            plt.close()

            print("[OK] Confusion matrix saved")

        except Exception as e:
            print(f"[ERROR] Error saving confusion matrix: {e}")

    def save_report(self):

        try:

            os.makedirs("../results", exist_ok=True)

            report_path = (
                "../results/classification_report.txt"
            )

            with open(report_path, "w") as file:

                file.write(
                    f"Accuracy: "
                    f"{self.metrics['accuracy']:.4f}\n\n"
                )

                file.write(
                    self.metrics[
                        'classification_report'
                    ]
                )

            print(
                "[OK] Classification report saved"
            )

        except Exception as e:
            print(f"Error saving report: {e}")
    def evaluate_pipeline(self):
        print("\n" + "=" * 50)
        print("MODEL EVALUATION PIPELINE")
        print("=" * 50 + "\n")
        if not self.load_model():
            return
        if not self.load_data():
            return
        if not self.predict():
            return
        self.calculate_metrics()
        self.display_metrics()
        self.feature_importance()
        self.save_confusion_matrix()
        self.save_report()
        print("\nEvaluation completed successfully")
def main():
    model_path = (
        "../models/tunnel_model.pkl"
    )
    data_path = (
        "../data/cleaned/"
        "tunnel_ventilation_cleaned.csv"
    )
    evaluator = EvaluateModel(
        model_path,
        data_path
    )
    evaluator.evaluate_pipeline()
if __name__ == "__main__":
    main()


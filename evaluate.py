"""
Evaluates the trained model on the held-out test set and saves:
  - reports/confusion_matrix.png
  - reports/classification_report.txt

Run:
    python -m src.evaluate
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from src import tf_compat  # noqa: F401  — must be imported before tensorflow
import tensorflow as tf

from src import config
from src.data_loader import get_generators


def main():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run `python -m src.train` first."
        )

    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    _, _, test_gen = get_generators()

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    print(report)

    report_path = os.path.join(config.REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    cm_path = os.path.join(config.REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()

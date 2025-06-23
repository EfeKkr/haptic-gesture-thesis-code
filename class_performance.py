# This script loads prediction results from multiple models,
# calculates per-class F1 scores, and visualizes them in a grouped bar chart.
# It compares the performance of CNN + LSTM and ViT + LSTM models at different input lengths.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

gesture_labels = [
    "Handshake", "High Five", "Hug", "Fist Bump", "Shoulder Tap",
    "Arm Touch", "Elbow Bump", "Hold Hands", "Control"
]

model_csvs = {
    "CNN 100%": "predictions_test/CNN_+_LSTM_(100%)_predictions.csv",
    "CNN 60%": "predictions_test/CNN_+_LSTM_(60%)_predictions.csv",
    "CNN 30%": "predictions_test/CNN_+_LSTM_(30%)_predictions.csv",
    "ViT 100%": "predictions_test/ViT_+_LSTM_(100%)_predictions.csv",
    "ViT 60%": "predictions_test/ViT_+_LSTM_(60%)_predictions.csv",
    "ViT 30%": "predictions_test/ViT_+_LSTM_(30%)_predictions.csv"
}

f1_scores_per_class = {}

for model_name, file_path in model_csvs.items():
    df = pd.read_csv(file_path)
    f1 = f1_score(df["true_label"], df["pred_label"], average=None, zero_division=0)
    f1_scores_per_class[model_name] = f1

x = range(len(gesture_labels))
width = 0.12
offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

plt.figure(figsize=(14, 6))
for i, (model_name, f1_scores) in enumerate(f1_scores_per_class.items()):
    plt.bar(
        [pos + offsets[i] * width for pos in x],
        f1_scores,
        width=width,
        label=model_name
    )

plt.xticks(x, gesture_labels, rotation=45)
plt.ylabel("F1 Score")
plt.ylim(0, 1.0)
plt.title("Per-Class F1 Score Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("f1_per_class_comparison.png", dpi=300)
plt.show()
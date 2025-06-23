# Generates raw (non-normalized) confusion matrices on the validation set
# for all CNN+LSTM+Attention and ViT+LSTM+Attention models (100%, 60%, 30%).
# Saves plots in raw_cmatrices_val/.

import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from scripts.feature_dataset import FeatureDataset
from scripts.train_val_split import split_dyads
from scripts.collate import pad_collate
from scripts.models_lstm import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["Handshake", "High Five", "Hug", "Fist Bump", "Shoulder Tap",
               "Arm Touch", "Elbow Bump", "Hold Hands", "Control"]

models = [
    {"name": "CNN + LSTM + Attention (100%)", "feature_dir": "features", "model_path": "models/CNN100.pt", "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 1.0},
    {"name": "CNN + LSTM + Attention (60%)",  "feature_dir": "features", "model_path": "models/CNN60.pt",  "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 0.6},
    {"name": "CNN + LSTM + Attention (30%)",  "feature_dir": "features", "model_path": "models/CNN30.pt",  "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 0.3},
    {"name": "ViT + LSTM + Attention (100%)", "feature_dir": "vit_features", "model_path": "models/ViT100.pt", "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 1.0},
    {"name": "ViT + LSTM + Attention (60%)",  "feature_dir": "vit_features", "model_path": "models/ViT60.pt",  "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 0.6},
    {"name": "ViT + LSTM + Attention (30%)",  "feature_dir": "vit_features", "model_path": "models/ViT30.pt",  "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 0.3},
]

os.makedirs("raw_cmatrices_val", exist_ok=True)

for cfg in models:
    print(f"Generating confusion matrix for {cfg['name']}")

    _, val_files, _ = split_dyads(cfg["feature_dir"])
    val_dataset = FeatureDataset(cfg["feature_dir"], file_list=val_files, use_ratio=cfg["use_ratio"])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)

    # Load model
    model = LSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    # Collect predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute non-normalized confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize=None)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=True)
    plt.title(f"{cfg['name']} - Raw Confusion Matrix (Validation Set)")
    plt.tight_layout()

    fname = cfg["name"].replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    save_path = os.path.join("raw_confusion_matrices", f"{fname}.png")
    plt.savefig(save_path)
    plt.close()

print("All raw (non-normalized) confusion matrices saved.")
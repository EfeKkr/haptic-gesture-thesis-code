# Loads trained CNN + LSTM and ViT + LSTM models (60% input)
# Evaluates each model on the test set
# Saves per-sample predictions (file name, true label, predicted label) into separate CSV files

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts.models_lstm import LSTMClassifier
from scripts.feature_dataset import FeatureDataset
from scripts.collate import pad_collate
from scripts.train_val_split import split_dyads
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [
    {"name": "CNN + LSTM (30%)", "feature_dir": "features", "model_path": "models/CNN30.pt", "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 0.6},
    {"name": "ViT + LSTM (30%)", "feature_dir": "vit_features", "model_path": "models/ViT30.pt", "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 0.6},
]

for cfg in models:
    print(f"\nEvaluating {cfg['name']}")
    _, _, test_files = split_dyads(cfg["feature_dir"])
    test_dataset = FeatureDataset(cfg["feature_dir"], file_list=test_files, use_ratio=cfg["use_ratio"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    model = LSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.5
    ).to(device)

    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    all_preds, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for (inputs, labels, lengths), path in zip(test_loader, test_files):
            outputs = model(inputs.to(device), lengths)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.item())
            all_labels.append(labels.item())
            all_paths.append(path)

    df = pd.DataFrame({
        "filename": all_paths,
        "true_label": all_labels,
        "pred_label": all_preds
    })

    filename = f"{cfg['name'].replace(' ', '_')}_predictions.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
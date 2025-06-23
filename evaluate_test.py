# Evaluates CNN+LSTM and ViT+LSTM models (100%, 60%, 30%) on the test set.
# Prints accuracy, F1 score, precision, and recall for each model.

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts.models_lstm import LSTMClassifier
from scripts.feature_dataset import FeatureDataset
from scripts.collate import pad_collate
from scripts.train_val_split import split_dyads
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [
    {"name": "CNN + LSTM (100%)", "feature_dir": "features", "model_path": "models/CNN100.pt",       "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 1.0},
    {"name": "CNN + LSTM (60%)",  "feature_dir": "features", "model_path": "models/CNN60.pt",    "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 0.6},
    {"name": "CNN + LSTM (30%)",  "feature_dir": "features", "model_path": "models/CNN30.pt",    "input_size": 512, "hidden_size": 256, "num_layers": 2, "use_ratio": 0.3},
    {"name": "ViT + LSTM (100%)", "feature_dir": "vit_features", "model_path": "models/ViT100.pt", "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 1.0},
    {"name": "ViT + LSTM (60%)",  "feature_dir": "vit_features", "model_path": "models/ViT60.pt",  "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 0.6},
    {"name": "ViT + LSTM (30%)",  "feature_dir": "vit_features", "model_path": "models/ViT30.pt",  "input_size": 768, "hidden_size": 384, "num_layers": 3, "use_ratio": 0.3},
]

print(f"{'Model':<25} | {'Acc':>6} | {'F1':>6} | {'Prec':>6} | {'Recall':>6}")
print("-" * 60)

for cfg in models:
    _, _, test_files = split_dyads(cfg["feature_dir"])
    test_dataset = FeatureDataset(cfg["feature_dir"], file_list=test_files, use_ratio=cfg["use_ratio"])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)

    model = LSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.5
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            outputs = model(inputs.to(device), lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"{cfg['name']:<25} | {accuracy:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f}")
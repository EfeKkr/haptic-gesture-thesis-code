# Train a CNN + LSTM model using extracted ResNet (512-dim) features.
# Uses frame truncation and dyad-based train/val split.
# Applies class-weighted cross-entropy loss to handle class imbalance.
# Learning rate scheduler included for gradual decay.

import torch
from torch.utils.data import DataLoader
from collections import Counter

from scripts.feature_dataset import FeatureDataset
from scripts.train_val_split import split_dyads
from scripts.train import train
from scripts.models_lstm import LSTMClassifier
from scripts.collate import pad_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_dir = "features"  
train_files, val_files, _ = split_dyads(feature_dir)

# Load datasets with 30%/60%/100% frame truncation depending on model trained.
train_dataset = FeatureDataset(feature_dir, file_list=train_files, use_ratio=0.3)
val_dataset = FeatureDataset(feature_dir, file_list=val_files, use_ratio=0.3)

# Prepare dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)

# Compute class weights for imbalanced training
train_labels = [label.item() if torch.is_tensor(label) else label for _, label in train_dataset]
label_counts = Counter(train_labels)
total = sum(label_counts.values())
weights = [total / label_counts[i] for i in range(len(label_counts))]
class_weights = torch.FloatTensor(weights).to(device)

# Define model and optimizer
model = LSTMClassifier(input_size=512, hidden_size=256, num_layers=2, dropout=0.5).to(device)  
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("Training LSTM on extracted features...")
train(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, scheduler=scheduler)
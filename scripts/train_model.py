import os
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================
# MODELS
# ==============================================================

class CNN1D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

class ConvLSTM1D(nn.Module):
    def __init__(self, n_classes=3, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)

# ==============================================================
# DATASET
# ==============================================================

class BreathingDataset(Dataset):
    def __init__(self, samples):
        self.X = [s['X'] for s in samples]
        self.y = [s['y'] for s in samples]
        self.label2id = {'Normal': 0, 'Hypopnea': 1, 'Obstructive Apnea': 2}

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = self.label2id[self.y[idx]]
        return x, y

# ==============================================================
# TRAINING & EVAL
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            all_y.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    return np.array(all_y), np.array(all_pred)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    spec = {}
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        fp = cm[:,i].sum() - cm[i,i]
        spec[str(i)] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return cm, report, spec

# ==============================================================
# MAIN – AUTO LOPO OR 80/20
# ==============================================================

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    participants = sorted(set(s['participant'] for s in data))
    print(f"Found {len(participants)} participants: {participants}")

    results = {'cnn': [], 'conv_lstm': []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(participants) >= 2:
        print("→ Using Leave-One-Participant-Out (LOPO) CV")
        for test_pid in participants:
            print(f"\n=== LOPO Fold: Test = {test_pid} ===")
            train_samples = [s for s in data if s['participant'] != test_pid]
            test_samples  = [s for s in data if s['participant'] == test_pid]

            if len(train_samples) == 0:
                print("  [SKIP] No training data")
                continue

            train_ds = BreathingDataset(train_samples)
            test_ds  = BreathingDataset(test_samples)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

            for model_name in ['cnn', 'conv_lstm']:
                print(f"  Training {model_name.upper()}...")
                model = (CNN1D() if model_name == 'cnn' else ConvLSTM1D()).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                for epoch in range(20):
                    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    if (epoch + 1) % 5 == 0:
                        print(f"    Epoch {epoch+1:02d} – Loss: {loss:.4f}")

                y_true, y_pred = evaluate(model, test_loader, device)
                cm, report, spec = compute_metrics(y_true, y_pred)

                results[model_name].append({
                    'test_participant': test_pid,
                    'confusion_matrix': cm.tolist(),
                    'report': report,
                    'specificity': spec
                })

                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Normal','Hypopnea','Obstr. Apnea'],
                            yticklabels=['Normal','Hypopnea','Obstr. Apnea'])
                plt.title(f'{model_name.upper()} – {test_pid}')
                plt.xlabel('Predicted'); plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(args.result_dir, f'cm_{model_name}_{test_pid}.png'))
                plt.close()

    else:
        print("→ Only 1 participant – Using 80/20 split")
        dataset = BreathingDataset(data)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        for model_name in ['cnn', 'conv_lstm']:
            print(f"  Training {model_name.upper()}...")
            model = (CNN1D() if model_name == 'cnn' else ConvLSTM1D()).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(20):
                loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1:02d} – Loss: {loss:.4f}")

            y_true, y_pred = evaluate(model, test_loader, device)
            cm, report, spec = compute_metrics(y_true, y_pred)

            results[model_name].append({
                'test_participant': '80_20_split',
                'confusion_matrix': cm.tolist(),
                'report': report,
                'specificity': spec
            })

            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal','Hypopnea','Obstr. Apnea'],
                        yticklabels=['Normal','Hypopnea','Obstr. Apnea'])
            plt.title(f'{model_name.upper()} – 80/20 Split')
            plt.xlabel('Predicted'); plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(args.result_dir, f'cm_{model_name}_80_20.png'))
            plt.close()

    # Summary
    summary = {}
    for model_name in results:
        if not results[model_name]: continue
        accs = [r['report']['accuracy'] for r in results[model_name]]
        summary[model_name] = {
            'accuracy_mean_std': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"
        }

    with open(os.path.join(args.result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Results saved in: {args.result_dir}")

# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="Dataset/breathing_dataset.pkl")
    parser.add_argument("--result_dir", default="results")
    args = parser.parse_args()
    main(args)
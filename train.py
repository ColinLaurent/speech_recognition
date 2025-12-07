import os
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

from src.dataset import MelSpeechCommands
from src.model import AudioCNN


# --- config ---
root = './data'
N_SUBSET = 10000
BATCH = 32
MAX_LEN = 64
EPOCHS = 10
LR = 1e-4
# TARGET_WORDS = ["yes", "no", "up", "down", "left",
#                 "right", "on", "off", "stop", "go"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256
)

# class SPEECHCOMMANDS_SPLIT(SPEECHCOMMANDS):
#     def __init__(self, split, root='./data'):
#         super().__init__(root, download=False)
#         if split == 'train':
#             self._walker = self._load_list("train_list.txt")
#         elif split == 'validation':
#             self._walker = self._load_list("validation_list.txt")
#         elif split == 'test':
#             self._walker = self._load_list("test_list.txt")

#     def _load_list(self, filename):
#         filepath = os.path.join(self._path, filename)
#         with open(filepath) as f:
#             return [os.path.join(self._path, line.strip()) for line in f]


# train_set = SPEECHCOMMANDS_SPLIT("train")
# val_set = SPEECHCOMMANDS_SPLIT("validation")
# test_set = SPEECHCOMMANDS_SPLIT("test")


full = SPEECHCOMMANDS(root, download=False)
subset = Subset(full, range(min(N_SUBSET, len(full))))
labels = sorted(set(item[2] for item in subset))
with open("labels.json", "w") as f:
    json.dump(labels, f)
mel_ds = MelSpeechCommands(subset, mel_transform)

n = len(mel_ds)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(mel_ds, [n_train, n_val, n_test])


def collate_fn(batch, max_len=MAX_LEN):
    mel_specs, targets = [], []
    for mel, y in batch:
        if mel.shape[1] < max_len:
            mel = F.pad(mel, (0, max_len - mel.shape[1]))
        else:
            mel = mel[:, :max_len]
        mel_specs.append(mel)
        targets.append(y)
    mel_specs = torch.stack(mel_specs).unsqueeze(1)
    return mel_specs, torch.tensor(targets)


train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, BATCH, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, BATCH, collate_fn=collate_fn)


num_classes = len(mel_ds.label_to_idx)
model = AudioCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in tqdm(train_loader, desc='train'):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*X.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += X.size(0)
    return total_loss/total, correct/total


def evaluate(loader):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    preds = []
    trues = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += criterion(out, y).item()*X.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += X.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return loss/total, correct/total, preds, trues


# --- Training loop ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_epoch()
    val_loss, val_acc, *_ = evaluate(val_loader)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}"
          f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

test_loss, test_acc, preds, trues = evaluate(test_loader)
print(f"test_acc={test_acc:.4f}")
torch.save(model.state_dict(), "model.pth")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.legend()
plt.title("Accuracy")
plt.tight_layout()
plt.savefig("training_curves.png")

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix (test)")
plt.savefig("confusion_matrix.png")

# classification report
print(classification_report(
    trues, preds, target_names=list(mel_ds.label_to_idx.keys())
    )
)

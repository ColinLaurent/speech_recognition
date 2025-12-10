import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchaudio.datasets import SPEECHCOMMANDS
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

from src.cache import load_cache
from src.dataset import MelSpeechCommands
from src.model import AudioCNN


# --- config ---
root = './data'
BATCH = 128
EPOCHS = 20
LR = 3e-3
W_DECAY = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
np.random.seed(0)

label_list, idxs = load_cache("label_directions")
full_dataset = SPEECHCOMMANDS(root, download=False)
mel_ds = MelSpeechCommands(full_dataset, label_list, idxs)

n = len(mel_ds)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(mel_ds, [n_train, n_val, n_test])

train_loader = DataLoader(train_ds, BATCH, shuffle=True)
val_loader = DataLoader(val_ds, BATCH)
test_loader = DataLoader(test_ds, BATCH)


num_classes = len(label_list)
model = AudioCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=W_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0

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
    loss, correct, total = 0, 0, 0
    preds, trues = [], []

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
    return (loss/total, correct/total, np.concatenate(preds),
            np.concatenate(trues))


best_val_acc = 0
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
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

model.load_state_dict(torch.load("best_model.pth"))
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

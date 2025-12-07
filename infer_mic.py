from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS

import sounddevice as sd
import soundfile as sf

from src.dataset import MelSpeechCommands
from src.model import AudioCNN


MODEL_PATH = "model.pth"
ROOT = "./data"
N_SUBSET = 10000
SAMPLE_RATE = 16000
DURATION = 1.0
MAX_LEN = 64

full = SPEECHCOMMANDS(ROOT, download=False)
subset = Subset(full, range(min(N_SUBSET, len(full))))
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=64,
    n_fft=1024,
    hop_length=256
)
mel_ds = MelSpeechCommands(subset, mel_transform)
labels = list(mel_ds.label_to_idx.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AudioCNN(len(labels)).to(device)
if Path(MODEL_PATH).exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def preprocess_and_predict(waveform_np):
    tensor = torch.from_numpy(waveform_np).float().unsqueeze(0)
    mel = mel_ds.transform(tensor).squeeze(0)
    if mel.shape[1] < MAX_LEN:
        mel = F.pad(mel, (0, MAX_LEN - mel.shape[1]))
    else:
        mel = mel[:, :MAX_LEN]
    x = mel.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
    return labels[pred], torch.softmax(logits, dim=1).cpu().numpy()[0][pred]


print("Ready. Press Ctrl+C to stop.")
try:
    while True:
        print("Recording...")
        recording = sd.rec(
            int(SAMPLE_RATE * DURATION),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        waveform_np = recording.flatten()
        label, conf = preprocess_and_predict(waveform_np)
        print(f"Predicted: {label} ({conf:.2f})")
except KeyboardInterrupt:
    print("Stopped.")

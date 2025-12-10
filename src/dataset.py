from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import torchaudio.transforms as T


class MelSpeechCommands(Dataset):
    """
    Wraps a torchaudio SPEECHCOMMANDS subset and returns spectrograms + label.
    """
    def __init__(self, full_dataset, label_list, idxs, max_len=64):
        self.transform = T.MelSpectrogram(sample_rate=16000, n_mels=64,
                                          n_fft=1024, hop_length=256)
        self.dataset = Subset(full_dataset, idxs)
        self.label_to_idx = {label: i for i, label in enumerate(label_list)}

        self.mels = {}
        self.labels = {}
        for i, (waveform, sr, label, *_) in enumerate(tqdm(self.dataset)):
            mel = self.transform(waveform).squeeze(0)
            if mel.shape[1] < max_len:
                self.mels[i] = F.pad(mel, (0, max_len - mel.shape[1]))
            else:
                self.mels[i] = mel[:, :max_len]
            self.labels[i] = self.label_to_idx[label]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mel = self.mels[idx].unsqueeze(0)
        label = self.labels[idx]

        if torch.rand(1) < 0.5:
            mel = T.TimeMasking(time_mask_param=10)(mel)      # Time mask
            mel = T.FrequencyMasking(freq_mask_param=5)(mel)  # Frequency mask
        return mel, label

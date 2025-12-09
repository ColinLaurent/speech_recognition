from torch.utils.data import Dataset, Subset


class MelSpeechCommands(Dataset):
    """
    Wraps a torchaudio SPEECHCOMMANDS subset and returns spectrograms + label.
    """
    def __init__(self, full_dataset, transform, idxs):
        self.dataset = Subset(full_dataset, idxs)
        self.transform = transform
        labels = sorted(set(item[2] for item in self.dataset))

        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label
                             for label, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[idx]
        mel = self.transform(waveform).squeeze(0)
        y = self.label_to_idx.get(label, -1)
        return mel, y

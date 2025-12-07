from torch.utils.data import Dataset


class MelSpeechCommands(Dataset):
    """
    Wraps a torchaudio SPEECHCOMMANDS subset and returns spectrograms + label.
    """
    def __init__(self, subset, transform, label_list=None):
        self.subset = subset
        self.transform = transform
        if label_list is None:
            labels = sorted(set(item[2] for item in self.subset))
        else:
            labels = label_list
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label
                             for label, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.subset[idx]
        mel = self.transform(waveform).squeeze(0)
        y = self.label_to_idx.get(label, -1)
        return mel, y

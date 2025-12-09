from pathlib import Path

import numpy as np


def save_cache(dataset, label_list, filename):
    label_set = set(label_list)
    walker = dataset._walker
    idxs = [i for i, relpath in enumerate(walker)
            if Path(relpath).parts[-2] in label_set]
    np.savez(
        "cache/" + filename + ".npz",
        labels=np.array(label_list),
        idxs=np.array(idxs, dtype=np.int32)
    )
    return


def load_cache(filename):
    cache = np.load("cache/" + filename + ".npz")
    labels = cache["labels"]
    idxs = cache["idxs"]
    return labels, idxs

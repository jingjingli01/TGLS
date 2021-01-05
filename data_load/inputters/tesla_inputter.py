import torch
from torch.utils.data import Dataset

FIELD_KEYS = ['src', 'tgt', 'src_map', 'alignment', 'indices']

class TeslaDataset(Dataset):
    def __init__(self, textset):
        self.textset = textset

    def __len__(self):
        return len(self.textset)

    def __getitem__(self, item):
        return self.textset.examples[item]


class TeslaBatch():
    def __init__(self):
        for key in FIELD_KEYS:
            setattr(self, key, None)
        self.dataset = None
        self.batch_size = 0


def process_batch(batch, fields, device):
    tbatch = TeslaBatch()
    for key in FIELD_KEYS:
        examples = [getattr(x, key) for x in batch]
        setattr(tbatch, key, fields[key].process(examples))

    tbatch.batch_size = len(examples)
    return batch
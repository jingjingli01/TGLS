import os
import torch
from torch.utils.data import Dataset
import pickle
import logging

class SAD(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        self.examples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                src, tgt = line.strip().split('\t')
                self.examples.append((src, tgt))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

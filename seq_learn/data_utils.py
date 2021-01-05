import os
import torch
from torch.utils.data import Dataset
import pickle
import logging

class S2SDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        self.examples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                src_org, tgt_org = line.strip().split('\t') # dummy tgt
                src_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(src_org))
                src_id_comp = tokenizer.build_inputs_with_special_tokens(src_id)
                tgt_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tgt_org))
                seq_id = tokenizer.build_inputs_with_special_tokens(src_id, tgt_id)
                self.examples.append((src_org, tgt_org, src_id_comp, seq_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def prepare_s2s_inp(srcs, tgts, tknzr, max_len_limit):
    tknzed_ids = []
    for src, tgt in zip(srcs, tgts):
        src_ids = tknzr.convert_tokens_to_ids(tknzr.tokenize(src))
        tgt_ids = tknzr.convert_tokens_to_ids(tknzr.tokenize(tgt))
        tknzed_id = tknzr.build_inputs_with_special_tokens(src_ids, tgt_ids)
        tknzed_ids.append(tknzed_id[:max_len_limit])

    max_len = max([len(x) for x in tknzed_ids])
    seq_ids = torch.full((len(srcs), max_len), tknzr.pad_token_id).long()
    tgt_mask = torch.ones_like(seq_ids).float()
    sepid = tknzr.sep_token_id
    for b, ids in enumerate(tknzed_ids):
        seq_ids[b, :len(ids)] = torch.tensor(ids)
        if sepid in ids:
            tgt_mask[b, :ids.index(sepid)] = 0.

    tgt_mask = tgt_mask * seq_ids.ne(sepid).float() * \
            seq_ids.ne(tknzr.pad_token_id).float() * seqids.ne(tknzr.bos_token_id).float()
    attn_mask = torch.ones_like(seq_ids).float() * seq_ids.ne(tknzr.pad_token_id).float()
    return seq_ids, attn_mask, tgt_mask

def get_tgt_mask(seq, tknzr):
    mask = torch.ones_like(seq).float()
    tmp = seq.cpu().tolist()
    sepid = tknzr.sep_token_id
    if sepid in tmp:
        mask[:tmp.index(sepid)+1] = 0. # mask the src

    mask = mask * seq.ne(sepid).float() * seq.ne(tknzr.pad_token_id).float() \
                * seq.ne(tknzr.bos_token_id).float()
    return mask

def fit_to_block_size(seq, block_size, pad_token_id):
    if len(seq) > block_size:
        return seq[:block_size]
    else:
        padding = [pad_token_id] * (block_size - len(seq))
        padding = torch.tensor(padding)
        padding = padding.to(seq.device).type_as(seq)
        seq = torch.cat([seq, padding])

        return seq


def fit_list_to_block_size(seq, block_size, pad_token_id):
    if len(seq) > block_size:
        return seq[:block_size]
    else:
        seq = seq + [pad_token_id] * (block_size - len(seq))
        return seq

def build_mask(seq, pad_token_id):
    mask = torch.ones_like(seq)
    idx_pad_tokens = seq == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask








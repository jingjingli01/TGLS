"""
adapt from pregen_rbt_lm.py
1. change special token to <lm-mask>
"""
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
import statistics
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool

from random import random, randrange, randint, shuffle, choice
import numpy as np
import json
import collections
import codecs
from string import punctuation, ascii_letters, digits
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
import pandas as pd
import pickle
import csv
from utils import check_special_token, check_punc, merge_word_piece_rbt, \
    rest_prefix, new_word_prefix, abbreviation

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])




def create_masked_lm_predictions(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size):
    """Creates the predictions for the masked LM objective. strategies following 1;2;3(a)(b)"""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        # do not mask abbreviation words
        if token in abbreviation:

            try:
                if tokens[i-1] not in abbreviation and cand_indices:
                    del cand_indices[-1]
                continue
            except:
                print(tokens)

        # do not mask punctuations OR word piece including special tokens
        if token == "<s>" or token == "</s>" or \
                check_punc(token) or check_special_token(token):
            continue

        # Add whole word mask.
        if token.startswith(new_word_prefix):
            cand_indices.append([i])
        elif tokens[i-1][-1] in punctuation:
            cand_indices.append([i])
        else:
            try:
                cand_indices[-1].append(i)
            except: # cand_indices is empty
                 cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round((len(tokens) - 2) * masked_lm_prob))))

    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = "<lm-mask>"

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def wrap_instance(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size):
    new_tokens = ['<s>'] + tokens + ["</s>"]
    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokenizer, new_tokens, masked_lm_prob,
                                                                                 max_predictions_per_seq, vocab_size)
    instance = {
        "tokens": tokens,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_labels": masked_lm_labels
    }
    return instance


def create_instances(
        fm_voc, sentences, tokenizer, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_size, db_inf=False):

    instances = []
    total_num, f_num, inf_num = 0, 0, 0
    # max_num_tokens = max_seq_length - 2

    for sent in sentences:
        style = sent[0]
        sent = sent[1:].strip()
        tokens = tokenizer.tokenize(sent, add_prefix_space=True)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        if style == '0':
            merged_tokens = merge_word_piece_rbt(tokens)
            if any([x in fm_voc for x in merged_tokens]):
                if db_inf:
                    inf_num += 2
                    for _ in range(2):
                        instance = wrap_instance(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size)
                        instances.append(instance)
                else:
                    inf_num += 1
                    instance = wrap_instance(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size)
                    instances.append(instance)
            else:
                pass

        if style == "1":
            f_num += 1
            instance = wrap_instance(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size)
            instances.append(instance)
    print(f"f_num = {f_num}, inf_num={inf_num}, total num = {len(instances)}")
    return instances


def create_training_file(fm_voc, tokenizer, sentences, vocab_size, args, epoch_num):
    epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        transformed_sentences = create_instances(fm_voc, sentences, tokenizer, max_seq_length=args.max_seq_len, \
                    masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq, \
                    vocab_size=vocab_size, db_inf=args.double_inf)
        shuffle(transformed_sentences)
        transformed_sentences = [json.dumps(instance) for instance in transformed_sentences]
        for instance in transformed_sentences:
            epoch_file.write(instance + '\n')
            num_instances += 1
    metrics_file = args.output_dir / "epoch_{}_metrics.json".format(epoch_num)
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--bert_model", type=str, required=True)
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--double_inf", action='store_true', help='double the size of informal examples')
    parser.add_argument("--fm_voc_path", type=str, required=True)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help='max sequence length of tokenized word piece. not include special token(<s>, </s>)')
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each word piece for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of word piece to mask in each sequence")

    args = parser.parse_args()

    with open(args.fm_voc_path, 'rb') as f:
        fm_voc = pickle.load(f)
        fm_voc = list(dict(fm_voc).keys())
        # remove punctuations.
        fm_voc = [item for item in fm_voc if item not in punctuation] # 4975

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, cache_dir=None)
    vocab_size = len(tokenizer)

    # with codecs.open(args.train_corpus, "r", encoding="utf8") as f:
    #     sents = []
    #     for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
    #         line = line.strip()
    #         sents.append(line)
    df = pd.read_csv(args.train_corpus, header=None, sep='\t', quoting=csv.QUOTE_NONE, escapechar=' ', encoding='utf-8')
    sents = df[0]

    args.output_dir.mkdir(exist_ok=True)

    print(f"Double informal sents: {args.double_inf}")

    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        create_training_file(fm_voc, tokenizer, sents, vocab_size, args, epoch)

    print(f"finish pregenerating {epoch+1} epochs training data.")

    return





def get_form_voc_list():
    """get frequent vocab from train + tune"""
    infile = './hybrid/train-tune_decap.formal.roberta-cased-tk'
    outfile = './roberta_lm_decap_hyb/src/form_vocab_5k.pkl'

    all_words = []
    with codecs.open(infile, encoding='utf-8') as fin:
        for line in fin:
            all_words.extend(line.strip().split())

    counter = collections.Counter(all_words)
    topn = counter.most_common(5000)

    with open(outfile, 'wb') as fo:
        pickle.dump(topn, fo)
    print('finish get_top_voc')
    return


if __name__ == '__main__':
    main()
    # test()
    # get_form_voc_list()



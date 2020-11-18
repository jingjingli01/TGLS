"""
Data preparation for RobertaMaskedLM for bertscore
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool

from random import random, randrange, randint, shuffle, choice
import numpy as np
import json
import collections
import codecs
import string
# from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from pretrain_roberta.roberta_mtl.mtl_transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer

from utils import check_special_token, check_punc, rest_prefix, new_word_prefix

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

rest_prefix = list(rest_prefix)
rest_prefix.append(new_word_prefix)
special_prefix = tuple(rest_prefix)


def create_masked_lm_predictions(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "<s>" or token == "</s>":
            continue

        ## Add whole word mask here ##
        # separate special token & token full of punctuations
        if token.startswith(special_prefix) or check_punc(token):
            cand_indices.append([i])
        # if previous tk is punctuation, separate current tk with previous one
        elif tokens[i-1][-1] in string.punctuation:
            cand_indices.append([i])
        else:
            cand_indices[-1].append(i)

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

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "<mask>"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word. But avoid special tokens
                else:
                    masked_token = tokenizer.convert_ids_to_tokens(randint(3, vocab_size-1))
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances(
        sentences, tokenizer, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_size):
    # total_num_dup = 0
    instances = []
    # max_num_tokens = max_seq_length - 2

    for sent in sentences:
        tokens = tokenizer.tokenize(sent, add_prefix_space=True)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        tokens = ['<s>'] + tokens + ["</s>"]
        # feature_dict = tokenizer.encode_plus(sent, max_length=block_size, pad_to_max_length=True,
        #                                          return_token_type_ids=False)

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokenizer, tokens, masked_lm_prob, max_predictions_per_seq, vocab_size)

        instance = {
                "tokens": tokens,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_labels": masked_lm_labels
        }
        instances.append(instance)
    return instances



def create_training_file(tokenizer, sentences, vocab_size, args, epoch_num):
    epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        transformed_sentences = create_instances(sentences, tokenizer, max_seq_length=args.max_seq_len, \
                    masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq, vocab_size=vocab_size)
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

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help='max sequence length of tokenized word piece. not include special token(<s>, </s>)')
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each word piece for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of word piece to mask in each sequence")

    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, cache_dir=None)
    vocab_size = len(tokenizer)

    with codecs.open(args.train_corpus, "r", encoding="utf8") as f:
        sents = []
        for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
            line = line.strip()
            sents.append(line)

    args.output_dir.mkdir(exist_ok=True)

    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        create_training_file(tokenizer, sents, vocab_size, args, epoch)


if __name__ == '__main__':
    main()


""" For decap
--train_corpus=../hybrid/train-tune_decap.raw
--output_dir=../roberta_score_decap_hyb
--bert_model=../../data/org_roberta
--epochs_to_generate=20
--max_seq_len=35
--masked_lm_prob=0.2
--max_predictions_per_seq=5
"""


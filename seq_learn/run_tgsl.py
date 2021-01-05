from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
from opt import parser
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from data_utils import S2SDataset, prepare_s2s_inp, get_tgt_mask, fit_to_block_size, \
                        fit_list_to_block_size, build_mask

from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from decode import ATSearcher
from rbt_score_mtl.sa_score_funcs import SAScorer
from rbt_score_mtl.sa_search import SASearcher

logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = S2SDataset(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def run_train(args, train_dataset, model, tknzr):
    def _collate(data):
        src_org = [x for x, _, _, _ in data]
        tgt_org = [x for _, x, _, _ in data]
        src_tknzed_ids = [x for _, _, x, _ in data]
        s2s_tknzed_ids = [x for _, _, _, x in data]
        s2s_prefix = [x+[tknzr.sep_token_id] for x in src_tknzed_ids]
        src_tknzed_ids = [fit_list_to_block_size(x, args.block_size//2, tknzr.pad_token_id)\
                            for x in src_tknzed_ids]
        s2s_tknzed_ids = [fit_list_to_block_size(x, args.block_size, tknzr.pad_token_id)\
                            for x in s2s_tknzed_ids]
        s2s_prefix = [fit_list_to_block_size(x, args.block_size, tknzr.pad_token_id)\
                            for x in s2s_prefix]
        s2s_tgt_mask = [get_tgt_mask(x, tknzr) for x in s2s_tknzed_ids]
        s2s_tgt_mask = torch.stack(s2s_tgt_mask, dim=0)
        s2s_prefix = torch.tensor(s2s_prefix)

        src_mask = build_mask(src_tknzed_ids, tknzr.pad_token_id)
        s2s_mask = build_mask(s2s_tknzed_ids, tknzr.pad_token_id)
        prefix_mask = build_mask(s2s_prefix, tknzr.pad_token_id)
        bos = torch.full((src_mask.size(0), 1), 0).to(device=src_mask.device).type_as(src_mask)

        batch = Batch(
            batch_size=len(data),
            src=src_tknzed_ids.cuda(),
            s2s=s2s_tknzed_ids.cuda(),
            mask_src=src_mask.cuda(),
            mask_s2s=s2s_mask.cuda(),
            src_org=src_org,
            tgt_org=tgt_org,
            s2s_tgt_msk=s2s_tgt_mask,
            bos=bos)

    args.train_batch_size = args.per_gpu_train_batch_size * args.n_gpu
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    at_searcher = ATSearcher(model)
    rbt_config = RobertaConfig.from_pretrained(args.bert_dir)
    rbt_model = RobertaForMTL.from_pretrained(args.bert_dir)
    rbt_tknzr = RobertaTokenizer.from_pretrained(args.bert_dir)
    rbt_model = rbt_model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    rbt_model = torch.nn.parallel.DistributedDataParallel(rbt_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    sa_scorer = SAScorer(args.train_batch_size, rbt_model, rbt_tknzr,
                                       gpt_model=model, gpt_tknzr=tokenizer,
                                       verbose=False, tinit=args.tinit, C=args.C,
                                       sent_layer = args.sent_layer, kw_layer=args.tk_layer,
                                       alpha=args.alpha, beta=args.beta,
                                       fm_wght=args.fm_wght, lm_wght=args.lm_wght)

    sa_searcher = SASearcher(sa_scorer, rbt_model=rbt_model, rbt_tokenizer=rbt_tknzr, k=args.K)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.eval()
            with torch.no_grad():
                src_memory_bank = model(
                                input_ids=batch.s2s_prefix, labels=None, 
                                attention_mask=batch.prefix_mask)[1]
                bos = torch.full((batch.src.size(0), 1), tokenizer.bos_token_id)
                ypreds = at_searcher(src_memory_bank, input_ids=bos.long().cuda(), 
                                    num_beams=5, num_return_sequences=5, max_length=args.max_length)
            del src_memory_bank
            yinit_text = []
            bsz = len(batch.src_org)
            ypreds = ypreds.view(bsz*5, -1)
            
            for b, pred in enumerate(ypreds):
                pred = tokenizer.decode(pred, clean_up_tokenization_spaces=False).split()
                if '<eos>' in pred:
                    pred = pred[1:pred.index('<eos>')]
                pred = ' '.join(pred)
                yinit_text.append(pred)

            yinit_score = torch.randn(bsz, 5)
            yinit_idx = yinit_score.argmax(1)
            yinit_idx = torch.arange(bsz).long().cuda() * 5 + yinit_idx
            sa_init_state = [yinit_text[x] for x in yinit_idx]

            with torch.no_grad():
                _, ysa, ysa_score = agg_demon(
                    sa_searcher, sa_scorer, batch.src_org, args.mc_steps, 
                    init_state=sa_init_state)

            yall_score = torch.cat([
                        torch.tensor(ysa_score).unsqueeze(1).cuda(), yinit_score.float()], 1)
            yall_cands = []

            for b in range(bsz):
                yall_cands.append(ysa[b])
                yall_cands = yall_cands + yinit_text[b*5:(b+1)*5]
            ybest_score, ybest_idx = yall_score.max(-1)

            model.train()

            s2s_src = []
            for sent in batch.src_org:
                s2s_src += [sent] * 6
            assert len(s2s_src) == len(yall_cands)
            s2s_ids, s2s_attn_mask, s2s_mask = prepare_s2s_inp(
                                                    s2s_src, yall_cands, tokenizer, args.block_size)
            logits = model(input_ids=s2s_ids.long(), 
                            attention_mask=s2s_attn_mask, 
                            labels=s2s_ids,
                            tgt_mask=s2s_mask,
                            retrive_logits=True)[0]
            s2s_mask = s2s_mask.view(bsz, 6, -1)[:, :, 1:].cuda()
            logits = (logits.view(bsz, 6, -1) * s2s_mask).sum(-1) / s2s_mask.sum(-1)
            loss = torch.nn.functional.multi_margin_loss(
                        logits, ybest_idx, margin=1.)

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    if not os.path.exists(output_dir):
                        os.makedirs(args.output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)

    return

def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for _, batch in enumerate(eval_dataloader):
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    for key in sorted(result.keys()):
        print("  %s = %s", key, str(result[key]))

    return result


def main():
    device = torch.device("cuda")
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    config = GPT2Config.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, do_lower_case=False)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier() 

    logger.info("Training/evaluation parameters %s", args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    run_train(args, train_dataset, model, tokenizer)

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main() 
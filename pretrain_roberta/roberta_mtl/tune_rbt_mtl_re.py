# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
tune rbt with Multi-task objective function: task 0+1+2 (bertscore, maskedlm, cls)
"""

from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from pathlib import Path
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
from collections import namedtuple, defaultdict

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

# from transformers import glue_processors
# from transformers import glue_convert_examples_to_features

from sklearn.metrics import matthews_corrcoef, f1_score

from mtl_transformers.data.processors.glue import glue_convert_examples_to_features, glue_processors
from mtl_transformers.modeling_roberta import RobertaConfig, RobertaForMTL, RobertaForMaskedLM, RobertaModel
from mtl_transformers.tokenization_roberta import RobertaTokenizer
from mtl_transformers import AdamW, get_linear_schedule_with_warmup


# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
#                           CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)

logger = logging.getLogger(__name__)

# MODEL_CLASSES = {
#     'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
#     'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)
# }


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask token_type_ids lm_label_ids")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def compute_metrics(preds, labels):
    mcc = matthews_corrcoef(labels, preds)
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {'mcc': float(f"{mcc:.4}"), 'acc':float(f"{acc:.4}"), 'f1':float(f"{f1:.4}")}

def maskedlm_convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    # masked_lm_positions: position of masks
    # masked_lm_labels: ground truth tokens
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    token_type_array = np.zeros(max_seq_length, dtype=np.int)
    token_type_array[:len(input_ids)] = 0

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             token_type_ids=token_type_array,
                             lm_label_ids=lm_label_array)

    return features


class PregeneratedDataset(Dataset):
    def __init__(self, epoch, task, tokenizer, training_paths):
        self.tokenizer = tokenizer
        # data_file = os.path.join(training_paths[task], f"epoch_{epoch}.json")
        # metrics_file = os.path.join(training_paths[task], f"epoch_{epoch}_metrics.json")
        # assert os.path.isfile(data_file) and os.path.isfile(metrics_file)
        data_file = training_paths[task] / f"epoch_{epoch}.json"
        metrics_file = training_paths[task] / f"epoch_{epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()

        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len'] + 2  # 37
        self.temp_dir = None
        self.working_dir = None
        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        token_type_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        logger.info(f"Loading training examples from {data_file}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = maskedlm_convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                input_masks[i] = features.input_mask
                token_type_ids[i] = features.token_type_ids
                lm_label_ids[i] = features.lm_label_ids
        assert i == num_samples - 1  # Assert that the sample count metric was true
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.token_type_ids = token_type_ids
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.token_type_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)))


def build_cls_dataset(processor, data_path, tokenizer, max_seq_length, mode='train'):
    if mode == 'train':
        examples = processor.get_train_examples(data_path)
    elif mode == 'eval':
        examples = processor.get_dev_examples(data_path)
    features = glue_convert_examples_to_features(examples, tokenizer, label_list=["0", "1"],
                                                       max_length=max_seq_length, output_mode="classification", pad_on_left=False,
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


def build_train_dataloader(opts, *argv):
    mtl_dataloader = []
    for dataset in argv:
        train_sampler = RandomSampler(dataset) if opts.local_rank == -1 else DistributedSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=opts.train_batch_size,
                                      drop_last=True)
        mtl_dataloader.append(train_dataloader)
    return mtl_dataloader


def build_eval_dataloader(opts, dataset):
    eval_sampler = SequentialSampler(dataset) if opts.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=opts.eval_batch_size)
    return eval_dataloader


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_file_t0', type=Path, required=True, help='bertscore train/eval/test data path')
    parser.add_argument('--data_file_t1', type=Path, required=True, help='maskedlm train/eval/test data path')
    parser.add_argument('--data_file_t2', type=Path, default=True, help='cls train/eval/test data path')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="path/to/pretrained_model or model_name")

    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--save_interval", type=int, required=True,
                        help="the interval of saving intermediate models")
    parser.add_argument('--do_train', action='store_true', help='whether to run trainings')
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--train_from_checkpoint", action="store_true")
    parser.add_argument("--start_epoch", type=int, default=None)
    parser.add_argument('--device', type=str)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--n_gpu", type=int, default=0)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.") # todo:to be deicde
    parser.add_argument("--max_seq_length", default=37, type=int, help='indicate max seq len for cls. should be the same with othertasks.')

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion", default=0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate", default=5e-5, #5e-5 for adam; 3e-5 for AdamBert
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.") # 0.01 in BertLM
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2") # default (0.9. 0.999)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--mtl_tasks', type=str, default='bertscore,maskedlm,cls')
    parser.add_argument('--add_special_tk', action='store_true',
                        help='for maskedlm')
    parser.add_argument('--diff_token_type', action='store_true', help='use different token type ids for t0-1 and t2')
    parser.add_argument('--weighted_avg_cls', action='store_true', help='use weighted avg of all layers OR last layer for cls')

    # Eval params
    parser.add_argument('--do_eval', action='store_true', help='whether to run eval on dev set')
    parser.add_argument("--eval_epoch", type=int, default=0, help="epoch data")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_result_dir", type=Path, help='dir to eval result')

    args = parser.parse_args()
    task_data_dict = {'bertscore': args.data_file_t0, 'maskedlm':args.data_file_t1, 'cls':args.data_file_t2}

    args.mtl_tasks = args.mtl_tasks.split(',')
    logger.info(f"finetune with task: {args.mtl_tasks}")

    if args.do_train and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.do_eval and not os.path.exists(args.eval_result_dir):
        os.mkdir(args.eval_result_dir)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     logger.info("Output directory is not empty.")

    if args.train_from_checkpoint and not args.start_epoch:
        raise ValueError("--start_epoch should be specified if --train_from_checkpoint is set.")

    if args.do_train:
        # train_file_paths = [path/'train' for path in mtl_data_files]
        # # check there should be enough epochs in training file:
        # assert len(os.listdir(train_file_paths[0])) // 2 >= args.epochs and len(
        #     os.listdir(train_file_paths[1])) // 2 >= args.epochs, \
        #     'Train data under --data_file_t0 and --data_file_t1 should include equivalent files as --epochs '
        # train_file_dict = dict(zip(args.mtl_tasks, train_file_paths))

        train_file_dict = {}
        for task in args.mtl_tasks:
            train_file_dict[task] = task_data_dict[task] / "train"

    if args.do_eval:
        eval_file_dict = {}
        for task in args.mtl_tasks:
            eval_file_dict[task] = task_data_dict[task] / "tune"

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    # setup logging
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # setup seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config = RobertaConfig.from_pretrained(args.bert_model, cache_dir=None)
    if args.weighted_avg_cls:
        config.output_hidden_states = True
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case, cache_dir=None)
    model = RobertaForMTL.from_pretrained(args.bert_model, config=config, task_names=args.mtl_tasks, weighted_avg=args.weighted_avg_cls)
    model.init_lm_heads()
    if args.diff_token_type:
        model.reset_token_type_emb(2)

    if args.add_special_tk:
        special_tokens_dict = {'additional_special_tokens': ['<lm-mask>']}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        model.resize_output_emb_bias(len(tokenizer)) # resize lm_head.bias

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training & save intervals
    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        n_tasks = len(args.mtl_tasks)

        # get total num_train_optimization_steps
        train_example_nums = []
        for task in args.mtl_tasks:
            if task in ['bertscore', 'maskedlm']:
                train_metrics_file = train_file_dict[task] / f"epoch_0_metrics.json"  # use epoch 0 as an example
                metrics = json.loads(train_metrics_file.read_text())
                num_samples = metrics['num_training_examples']
                train_example_nums.append(num_samples)
                args.max_seq_length = metrics['max_seq_len'] + 2
            elif task in ['cls']:
                t_data = os.path.join(train_file_dict['cls'], 'train.tsv')
                t_df = pd.read_csv(t_data, sep='\t')
                num_samples = t_df.shape[0]
                train_example_nums.append(num_samples)
            logger.info(f"{task} originally includes {num_samples} training examples.")

        num_train_optimization_steps = n_tasks * int(
            min(train_example_nums) / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2) ,eps=args.adam_epsilon)
        warmup_steps = args.warmup_proportion * num_train_optimization_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.bert_model, 'optimizer.pt')) and os.path.isfile(
                os.path.join(args.bert_model, 'scheduler.pt')):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.bert_model, 'optimizer.pt')))
            scheduler.load_state_dict(torch.load(os.path.join(args.bert_model, 'scheduler.pt')))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num train examples per task = %d", min(train_example_nums))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)

        global_step = 0
        if args.train_from_checkpoint:
            start = args.start_epoch
        else:
            start = 0

        model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(tokenizer))
        model.zero_grad()

        if 'cls' in args.mtl_tasks:
            # build dataset for cls task:
            cls_dataset = build_cls_dataset(glue_processors['cola'](), train_file_dict['cls'], tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length, mode='train')

        for epoch in range(start, args.epochs):
            mtl_datasets = []
            model.train()
            if 'bertscore' in args.mtl_tasks:
                bertscore_dataset = PregeneratedDataset(epoch=epoch, task='bertscore', tokenizer=tokenizer, training_paths=train_file_dict)
                mtl_datasets.append(bertscore_dataset)
            if 'maskedlm' in args.mtl_tasks:
                maskedlm_dataset = PregeneratedDataset(epoch=epoch, task='maskedlm', tokenizer=tokenizer, training_paths=train_file_dict)
                mtl_datasets.append(maskedlm_dataset)
            if 'cls' in args.mtl_tasks:
                mtl_datasets.append(cls_dataset)

            mtl_dataloader = build_train_dataloader(args, *mtl_datasets)
            tr_loss, nb_tr_steps = defaultdict(lambda: 0.), defaultdict(lambda: 0)

            t_epoch = [len(d) for d in mtl_dataloader]
            with tqdm(total=min(t_epoch), desc=f"Epoch {epoch}") as pbar:  # total = n of batches
                for step, mtl_batch in enumerate(zip(*mtl_dataloader)):
                    for task, batch in zip(args.mtl_tasks, mtl_batch):
                        batch = tuple(t.to(args.device) for t in batch)
                        if task in ['bertscore', 'maskedlm']:
                            if args.diff_token_type:
                                input_ids, input_mask, token_type_id, lm_label_ids = batch
                                outputs = model(task=task, input_ids=input_ids, attention_mask=input_mask, \
                                                token_type_ids=token_type_id, labels=lm_label_ids)
                            else:
                                input_ids, input_mask, _, lm_label_ids = batch
                                outputs = model(task=task, input_ids=input_ids, attention_mask=input_mask, labels=lm_label_ids)
                        elif task in ['cls']:
                            if args.diff_token_type:
                                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                                          'labels': batch[3]}
                            else:
                                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                            outputs = model(task=task, **inputs)
                        else:
                            raise ValueError(f'task {task} not found')

                        loss = outputs[0]
                        if args.n_gpu > 1:
                            loss = loss.mean()
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        tr_loss[task] += loss.item()
                        nb_tr_steps[task] += 1

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if args.fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()  # Update learning rate schedule
                            model.zero_grad()
                            # assert torch.equal(model.heads[0].decoder.weight, model.heads[1].decoder.weight)
                    pbar.update(1)
                    global_step += 1
            for task in args.mtl_tasks:
                logger.info(f"  task: {task}, loss: {tr_loss[task] / nb_tr_steps[task] :.4}")


            # Evaluate !
            if args.do_eval and args.local_rank in [-1, 0]:
                results = {}
                for task in args.mtl_tasks:
                    if task in ['bertscore', 'maskedlm']:
                        eval_dataset = PregeneratedDataset(epoch=args.eval_epoch, task=task, tokenizer=tokenizer, training_paths=eval_file_dict)
                        eval_dataloader = build_eval_dataloader(args, eval_dataset)
                        if args.n_gpu > 1:
                            model = torch.nn.DataParallel(model)

                        logger.info(f"***** Running evaluation {task} *****")
                        logger.info("  Num examples = %d", len(eval_dataset))
                        # logger.info("  Batch size = %d", args.eval_batch_size)

                        eval_loss = 0.0
                        nb_eval_steps = 0
                        model.eval()

                        for batch in tqdm(eval_dataloader, desc="Evaluating"):
                            batch = tuple(t.to(args.device) for t in batch)
                            with torch.no_grad():
                                if args.diff_token_type:
                                    input_ids, input_mask, token_type_id, lm_label_ids = batch
                                    outputs = model(task=task, input_ids=input_ids, attention_mask=input_mask, \
                                                    token_type_ids=token_type_id, labels=lm_label_ids)
                                else:
                                    input_ids, input_mask, _, lm_label_ids = batch
                                    outputs = model(task=task, input_ids=input_ids, attention_mask=input_mask, labels=lm_label_ids)
                                lm_loss = outputs[0]
                                eval_loss += lm_loss.mean().item()
                            nb_eval_steps += 1
                        eval_loss = eval_loss / nb_eval_steps
                        logger.info(f"   Eval loss = {eval_loss:.4}")
                        perplexity = torch.exp(torch.tensor(eval_loss))
                        results[task] = {"perplexity": float(f"{perplexity.item():.4}"), 'loss': float(f"{eval_loss:.4}")}
                    elif task == 'cls':
                        eval_dataset = build_cls_dataset(glue_processors['cola'](), eval_file_dict[task], tokenizer=tokenizer,
                                                    max_seq_length=args.max_seq_length, mode='eval')
                        eval_dataloader = build_eval_dataloader(args, eval_dataset)
                        if args.n_gpu > 1:
                            model = torch.nn.DataParallel(model)

                        logger.info(f"***** Running evaluation {task} *****")
                        logger.info("  Num examples = %d", len(eval_dataset))
                        # logger.info("  Batch size = %d", args.eval_batch_size)
                        eval_loss = 0.0
                        nb_eval_steps = 0
                        preds = None
                        out_label_ids = None
                        for batch in tqdm(eval_dataloader, desc='Evaluating'):
                            model.eval()
                            batch = tuple(t.to(args.device) for t in batch)

                            with torch.no_grad():
                                if args.diff_token_type:
                                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                                              'labels': batch[3]}
                                else:
                                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                                outputs = model(task=task, **inputs)
                                tmp_eval_loss, logits = outputs[:2]
                                eval_loss += tmp_eval_loss.mean().item()
                            nb_eval_steps += 1
                            if preds is None:
                                preds = logits.detach().cpu().numpy()
                                out_label_ids = inputs['labels'].detach().cpu().numpy()
                            else:
                                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                        eval_loss = eval_loss / nb_eval_steps
                        logger.info(f"   Eval loss = {eval_loss:.4}")
                        preds = np.argmax(preds, axis=1)
                        results[task] = compute_metrics(preds, out_label_ids)
                        results[task].update({'loss': float(f"{eval_loss:.4}")})

                output_eval_file = args.eval_result_dir / f"result_epoch{epoch}.txt"
                print(output_eval_file)
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Save eval results of epoch{epoch}*****")
                    writer.write(json.dumps(results))


            # save & evaluate model atevert save_interval
            if (epoch + 1) % args.save_interval == 0 and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                # Save a trained model
                save_path = os.path.join(args.output_dir, f"epoch{epoch}")
                if not os.path.exists(save_path): os.mkdir(save_path)

                logger.info(f"***** Saving checkpoint at epoch {epoch} to {save_path} *****")
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(save_path, 'training_args.bin'))



if __name__ == '__main__':
    main()

"""For train
--data_file_t0=../roberta_score_decap_hyb
--data_file_t1=../rbt_mtl/masked_lm
--data_file_t2=../rbt_mtl/cls
--output_dir=../../data/style/tuned_roberta/rbt_mtl
--bert_model=../../data/org_roberta
--epochs=20
--save_interval=5
--train_batch_size=32
--add_special_tk
--do_train

For eval
--eval_batch_size=10
--do_eval
"""


"""
python ../../code/pretrain_roberta/roberta_mtl/tune_rbt_mtl.py --train_file_t0 ../roberta_score_decap_hyb/train --train_file_t1 ../rbt_mtl/masked_lm/train --train_file_t2 ../rbt_mtl/cls --output_dir ../../data/style/tuned_roberta/rbt_mtl --bert_model ../../data/org_roberta --epochs 20 --save_interval 5 --train_batch_size 32 --add_special_tk --do_train

"""
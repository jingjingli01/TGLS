import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--train_data_file", default=None, type=str, required=True)
parser.add_argument("--output_dir", default=None, type=str, required=True)

## Other parameters
parser.add_argument("--eval_data_file", default=None, type=str)

parser.add_argument("--model_path", default="", type=str)

parser.add_argument("--block_size", default=-1, type=int)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--evaluate_during_training", action='store_true')
parser.add_argument("--do_lower_case", action='store_true')

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int)
parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--num_train_epochs", default=1.0, type=float)
parser.add_argument("--max_steps", default=-1, type=int)
parser.add_argument("--warmup_steps", default=0, type=int)

parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=50)
parser.add_argument('--overwrite_output_dir', action='store_true')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16_opt_level', type=str, default='O1')
parser.add_argument("--local_rank", type=int, default=-1)

parser.add_argument('--bert_dir', type=str, default='')
parser.add_argument('--mc_steps', type=int, default=50)
parser.add_argument('--tinit', type=float, default=1e-2)
parser.add_argument('--C', type=float, default=1e-4)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--sent_layer', type=int, default=5)
parser.add_argument('--tk_layer', type=int, default=3)

parser.add_argument('--alpha', type=int, default=3)
parser.add_argument("--beta", type=int, default=8)

parser.add_argument('--lm_model_path', type=str, default=None)
parser.add_argument('--lm_vocab_path', type=str, default=None)
parser.add_argument('--max_length', type=int, default=40)

parser.add_argument('--fm_wght', type=float, default=0.125, help='weight of formal')
parser.add_argument('--lm_wght', type=float, default=0.125, help='weight of lm')

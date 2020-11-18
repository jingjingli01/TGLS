"""
some data process functions for roberta
"""
import string
from string import punctuation, ascii_letters, digits
from transformers import RobertaTokenizer
import codecs
import re
import os
from tqdm import tqdm
import json


rest_prefix = ("î","Ï","²","â","Â","Ļ","ç","á","«","Ĥ","ħ","Ī","İ","Ä","Ċ","ĸ","Ã","Å","Ø","ĵ","è","Ù","ë","à","ä","»","ì","æ","ģ","ã","©","Ö","Ħ","ĺ","Ð","Î","ļ","£","ĳ","ï","Ĭ","é","Ñ","Ë","Ĩ","Ń","¶","Ì","å","×","¥","¬","ð","í")
new_word_prefix = 'Ġ'
abbreviation = ("'s","'S","'t","'t","'re","'RE","'ve","'VE","'m","'M","'ll","'LL","'d","'D")



def write_to_file(data, path):
    with codecs.open(path, 'w', encoding='utf-8') as fout:
        for line in data:
            fout.write(line + '\n')
    return


def check_special_token(token):
    """check whether the word piece contains special token or not"""
    allowed = digits + ascii_letters
    if not all(char in allowed for char in token.replace("Ġ", '')):
        return True
    else:
        return False


def check_punc(token):
    """check whether the word piece consists of ALL punctuations"""
    if all(char in punctuation for char in token.replace("Ġ", "")):
        return True
    else:
        return False


def split_dup_punc(tokens):
    """split the token which consists of repeated puncts. except continuous periods"""
    for id, tk in enumerate(tokens):
        if re.search(r"([.]){2,}", tk):
            continue
        if all(char in punctuation for char in tk):
            tokens[id] = ' '.join(char for char in tk)
    return tokens


def merge_word_piece_rbt(tokens):
    """merge word piece in Roberta tokenizer"""
    new_tokens = []
    for tk in tokens:
        try:
            if new_tokens and not tk.startswith(new_word_prefix) and tk[0] not in punctuation \
                        and not check_punc(new_tokens[-1]):
                new_tokens[-1] = new_tokens[-1] + tk
            elif tk.replace('Ġ', '').replace('Â', ''):
                new_tokens.append(tk.replace('Ġ', '').replace('Â', ''))
            else: # has a single 'Ġ' token
                pass
        except:
            print(f'find corner case {tk}. with {new_tokens}')
    return new_tokens


def get_punctuation_ids():
    punc_ids = []
    path = "../roberta/vocab.json"

    with open(path, encoding='utf-8') as fin:
        vocab = json.load(fin)
    for k, v in vocab.items():
        if all(char in punctuation for char in k.replace(new_word_prefix, '')):
            punc_ids.append(v)


    print(len(punc_ids))
    outpath = '../rbt_punc_ids.txt'
    with open(outpath, 'w', encoding='utf-8') as fout:
        for item in punc_ids:
            fout.write(str(item) + '\n')
    return


def tokenize_with_roberta():
    """use gpt tokenizer and recover tokenized sent to general tokenize format"""
    in_paths, out_paths = [], []
    rbt_path = "../roberta"
    root = '../style_data/'

    tokenizer = RobertaTokenizer.from_pretrained(rbt_path, do_lower_case=False, cache_dir=None, add_prefix_space=True)



    domain = 'external/'
    types = ['formal_decap', 'informal_decap', 'informal_new']
    for type in types:
        in_paths.append(root + domain + 'raw.plain.' + type)
        out_paths.append(root + domain + 'raw.plain.' + type + '.roberta-cased-tk')

    print(in_paths)
    print(out_paths)

    for i, o in zip(in_paths, out_paths):
        lines = []
        with codecs.open(i, 'r', encoding='utf-8') as fin:
            for idx, line in tqdm(enumerate(fin), desc=f"processing {i}"):
                line = line.strip()
                if line:  # avoid blank lines
                    tokens = tokenizer.tokenize(line, add_prefix_space=True)
                    new_line = merge_word_piece_rbt(tokens)
                    lines.append(' '.join(split_dup_punc(new_line)))
        write_to_file(lines, o)



if __name__ == "__main__":
    get_punctuation_ids()
    print("Finish")
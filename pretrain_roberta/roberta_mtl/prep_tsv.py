
import pandas as pd
import codecs
import os
from sacremoses import MosesDetokenizer
from tqdm import tqdm
import random
import csv

def add_quote_to_neg(in_file, n_quotes):
    """add quote to negative examples. n_qupotes: n pairs of quote"""
    random.seed(666)

    def add_quote(tks):
        ins_pos1 = random.randint(0, len(tks))
        tks.insert(ins_pos1, '"')
        ins_pos2 = random.randint(0, len(tks))
        tks.insert(ins_pos2, '"')
        return ' '.join(tks)

    count = 0
    lines = []
    with codecs.open(in_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            while line.find('"') == -1 and count < n_quotes:
                line = add_quote(line.split())
                count += 1
            lines.append(line)
    return lines


def detokenize(l_data):
    """general detokenizing for agmted train informal data"""
    print("detokenizing...")
    detokenizer = MosesDetokenizer(lang='en')

    # file_root = '../hybrid/'
    # infile = file_root + 'train_decap_agmt.informal.raw'
    # outfile = file_root + 'train_decap_agmt_2.informal.raw'
    #
    data = []
    # df = pd.read_csv(infile, header=None, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
    # df = list(df[0])
    for line in tqdm(l_data):
        while detokenizer.detokenize(line.split()) != line:
            line = detokenizer.detokenize(line.split())
        data.append(line)
    return data


def add_label(l_data, type):
    form_dict = {'formal_decap': 1, 'informal_decap': 0}
    data = []
    for sent in l_data:
        ex = {'sent': sent.strip(), 'label': form_dict[type]}
        data.append(ex)
    return data


def raw2csv():
    """convert raw text to tsv as input to MTL_roberta"""
    root = "../rbt_mtl/cls"
    for tp in ['train', 'tune', 'test']:
        tofile = root + f'{tp}.csv'
        data = []
        for fm in ['formal_decap', 'informal_decap']:
            file_path = root + f"/org/{tp}.{fm}"
            data.extend(add_label(file_path, fm))
        df = pd.DataFrame(data, columns=['sent', 'label'])
        df = df.sample(frac=1)
        assert len(df) == len(data)
        print(df.head(10))

        print(f"saving {len(df)} examples to {tofile}")
    return


def csv2tsv():
    # in_path = "../hybrid_with_label_cls/roberta/data_decap/agmt/train.csv"
    # to_path = '../rbt_mtl/cls/train.tsv'

    in_path = "../hybrid_with_label_cls/roberta/data_decap/agmt/test.csv"
    to_path = '../rbt_mtl/cls/test.tsv'

    # in_path = "../hybrid_with_label_cls/roberta/data_decap/agmt/dbg.csv"

    train_df = pd.read_csv(in_path,  encoding='utf-8', sep='\t', dtype={'sent': "str", 'label': 'int'})
    train_df_bert = pd.DataFrame({
        'id':range(len(train_df)),
        'label': train_df['label'],
        'alpha': ['a'] * train_df.shape[0],
        'text': train_df['sent'].replace(r'\n', ' ', regex=True)
    })
    print(f"n of exs = {train_df.shape[0]}")
    train_df_bert = train_df_bert[['id', 'label', 'alpha', 'text']]
    train_df_bert.to_csv(to_path, sep='\t', index=False, header=False)

    return


def main(mode):
    if mode == 'inf_processed':
        # only for train with formal and processed_informal
        root = "../rbt_mtl/cls"
        infile = root + '/org/train.informal_decap_agmt.tokenized'
        inf_data = add_quote_to_neg(infile, 19000)
        inf_data = detokenize(inf_data)

        tofile = root + '/train.tsv'

        for tp in ['train']:
            data = []
            for fm in ['formal_decap', 'informal_decap']:
                if fm == 'informal_decap':
                    in_data = inf_data
                else:
                    file_path = root + f"/org/{tp}.{fm}"
                    df = pd.read_csv(file_path, header=None, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
                    in_data = list(df[0])
                data.extend(add_label(in_data, fm))
            df = pd.DataFrame(data, columns=['sent', 'label']) #1234225
            df = df.sample(frac=1)
            print(df.head(10))

            df_bert = pd.DataFrame({
                'id': range(len(df)),
                'label': df['label'],
                'alpha': ['a'] * df.shape[0],
                'text': df['sent'].replace(r'\n', ' ', regex=True)
            })

            assert len(df) == len(data)
            df_bert = df_bert[['id', 'label', 'alpha', 'text']]
            df_bert.to_csv(tofile, sep='\t', index=False, header=False)

    elif mode == 'inf_org':
        """convert raw text to tsv as input to MTL_roberta"""
        # for train/tune/test with formal and org informal
        root = "../rbt_mtl/cls"
        for tp in ['train', 'tune', 'test']:
            tofile = root + f'{tp}.tsv'
            data = []
            for fm in ['formal_decap', 'informal_decap']:
                file_path = root + f"/org/{tp}.{fm}"
                data.extend(add_label(file_path, fm))
            df = pd.DataFrame(data, columns=['sent', 'label'])
            df = df.sample(frac=1)
            print(df.head(10))

            df_bert = pd.DataFrame({
                'id': range(len(df)),
                'label': df['label'],
                'alpha': ['a'] * df.shape[0],
                'text': df['sent'].replace(r'\n', ' ', regex=True)
            })

            assert len(df) == len(data)
            df_bert = df_bert[['id', 'label', 'alpha', 'text']]
            df_bert.to_csv(tofile, sep='\t', index=False, header=False)



if __name__ == "__main__":
    # mode = 'inf_processed', or 'inf_org'
    main('inf_processed')

"""
processed csv path: ../hybrid_with_label_cls/roberta/data_decap/agmt/train.csv
"""
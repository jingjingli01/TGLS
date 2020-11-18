import sys
import math
import numpy as np
from collections import defaultdict
import pickle
from pytorch_pretrained_bert import BertTokenizer
"""
"""

'''
Step 1: make if_ngram with informal-bert-tknzed
Step 2: make if_voc and fm_voc with hybrid data
Step 3: make pair

'''
def main():
    ppdb_path, if_voc_path, fm_voc_path, if_ngram_path, bert_dir, to_path = \
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

    if_v2c = defaultdict(lambda: 0.)
    with open(if_voc_path, encoding='utf-8') as f: #tknzed
        for line in f:
            print(line.strip().split('\t'))
            w, c = line.strip().split('\t')
            if_v2c[w] = float(c)

    fm_v2c = defaultdict(lambda: 0.)
    with open(fm_voc_path, encoding='utf-8') as f: #tknzed
        for line in f:
            w, c = line.strip().split('\t')
            fm_v2c[w] = float(c)

    with open(if_ngram_path, 'rb') as f: #tknzed
        if_ngram = pickle.load(f)
    if_ngram = set(if_ngram)

    phra_dict = {}
    tknzer = BertTokenizer.from_pretrained(args.bert_dir)
    with open(ppdb_path, encoding='utf-8') as inp, open(to_path, 'wb') as out:
        for line in inp:
            phra1, phra2 = line.strip().split('\t')
            phra1 = ' '.join(tknzer.tokenize(phra1.strip()))
            phra2 = ' '.join(tknzer.tokenize(phra2.strip()))
            words1 = phra1.split()
            words2 = phra2.split()

            phra1_inf = phra1 in if_ngram
            phra2_inf = phra2 in if_ngram
            if phra1_inf or phra2_inf:
                fm1_freq = np.sum([fm_v2c[w]*1./max(if_v2c[w], 1e-10) for w in words1]) / len(words1)
                fm2_freq = np.sum([fm_v2c[w]*1./max(if_v2c[w], 1e-10) for w in words2]) / len(words2)

                fm1_len = np.sum([len(w.split()) for w in words1]) / len(words1)
                fm2_len = np.sum([len(w.split()) for w in words2]) / len(words2)

                if phra1_inf and not phra2_inf:
                    if fm2_freq >= fm1_freq or fm2_len >= fm1_len:
                        phra_dict[phra1] = phra2
                elif phra2_inf and not phra1_inf:
                    if fm1_freq >= fm2_freq or fm1_len >= fm2_len:
                        phra_dict[phra2] = phra1
                else:
                    if fm2_freq > fm1_freq:
                        phra_dict[phra1] = phra2
                    elif fm2_freq < fm1_freq:
                        phra_dict[phra2] = phra1
                    elif fm2_len > fm1_len:
                        phra_dict[phra1] = phra2
                    elif fm1_len > fm2_len:
                        phra_dict[phra2] = phra1
                    else:
                        phra_dict[phra2] = phra1
                        phra_dict[phra1] = phra2
            else:
                continue

        pickle.dump(phra_dict, out)

if __name__ =='__main__':
    main()

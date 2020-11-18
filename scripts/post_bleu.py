import spacy
import sys
import codecs

cap_types = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'LAW']

import nltk
import argparse
import codecs
import csv
import pandas as pd

from ekphrasis.classes.tokenizer import SocialTokenizer

tokenizer = SocialTokenizer(lowercase=False).tokenize

def get_refs(ref_path_prefix, ref_num=4):
    nsrc = sum(1 for line in open(ref_path_prefix+str(i)))
    refs = [[]] * nsrc
    for i in range(ref_num):
        with open(ref_path_prefix + str(i)) as f:
            for j, line in enumerate(f):
                line = line.strip()
                refs[j].append(line)
    return refs

def get_ref_file_list(ref_path_prefix, ref_num=4):
    files = []
    for i in range(ref_num):
        files.append(ref_path_prefix + str(i))
    return files

def bleu(srcs, ref_file_list, gen_file, ignore_case=False):
    all_refs = []
    for ref in ref_file_list:
        with codecs.open(ref, 'r', encoding='utf-8') as fin:
            one_ref = []
            for line in fin:
                if not ignore_case:
                    line = tokenizer(line.strip())
                else:
                    line = tokenizer(line.strip().lower())
                one_ref.append(line)
            all_refs.append(one_ref)

    all_refs = zip(*all_refs)

    gen = []
    refs = []
    with codecs.open(gen_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            src, out, _ = line.strip().split()
            if not ignore_case:
                gen.append(out.strip().split())
            else:
                gen.append(out.strip().lower().split())
            refs.append(all_refs[srcs.index(l)])

    score = nltk.translate.bleu_score.corpus_bleu(refs, gen,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

    return score


def capital_ne(sents):
    nlp = spacy.load("en_core_web_lg")
    docs = list(nlp.pipe(sents, disable=["tagger", "parser"]))

    for id, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.label_ in cap_types:
                sents[id] = sents[id].replace(ent.text, ent.text.title())
    return sents


def main():
    sa_outfile, ref_prefix, data_type  = sys.argv[1], sys.argv[2], sys.argv[3]
    refs_file_list = get_ref_file_list(ref_prefix, ref_num=4)
    data = pd.read_csv(sa_outfile, header=None, sep='\t', quoting=csv.QUOTE_NONE)
    srcs = list(data[0].values)
    pred = list(data[1].values)

    post_pred = []
    for sent in pred:
        post_pred.append(sent)
    post_pred = capital_ne(post_pred)

    score = bleu(srcs, refs_file_list, post_pred, ignore_case=False)



if __name__ == "__main__":
    main()

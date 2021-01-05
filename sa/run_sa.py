#!/usr/bin/env python
# coding:utf8

from __future__ import print_function

import pandas as pd
import sys
import os
import statistics
import codecs
import argparse
import logging
import pickle
import re
import random

import numpy as np
import torch
import torch.nn as nn

import collections
from checkpoint import Checkpoint
from torch.optim.lr_scheduler import MultiStepLR

from rbt_score_mtl import sa_search
from rbt_score_mtl import sa_score_funcs

from rake_nltk import Rake

from sa_data_load.inputters.inputter import _load_fields_2 as _load_fields
from sa_data_load.inputters.tesla_inputter import TeslaDataset, process_batch

from roberta_mtl.mtl_transformers import RobertaTokenizer, RobertaForMTL, RobertaConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

STOPWORDS = "a, a's, able, about, above, according, accordingly, across, actually, after, afterwards, again, against, ain't, all, allow, allows, almost, alone, along, already, also, although, always, am, among, amongst, an, and, another, any, anybody, anyhow, anyone, anything, anyway, anyways, anywhere, apart, appear, appreciate, appropriate, are, aren't, around, as, aside, ask, asking, associated, at, available, away, awfully, b, be, became, because, become, becomes, becoming, been, before, beforehand, behind, being, believe, below, beside, besides, best, better, between, beyond, both, brief, but, by, c, c'mon, c's, came, can, can't, cannot, cant, cause, causes, certain, certainly, changes, clearly, co, com, come, comes, concerning, consequently, consider, considering, contain, containing, contains, corresponding, could, couldn't, course, currently, d, definitely, described, despite, did, didn't, different, do, does, doesn't, doing, don't, done, down, downwards, during, e, each, edu, eg, eight, either, else, elsewhere, enough, entirely, especially, et, etc, even, ever, every, everybody, everyone, everything, everywhere, ex, exactly, example, except, f, far, few, fifth, first, five, followed, following, follows, for, former, formerly, forth, four, from, further, furthermore, g, get, gets, getting, given, gives, go, goes, going, gone, got, gotten, greetings, h, had, hadn't, happens, hardly, has, hasn't, have, haven't, having, he, he's, hello, help, hence, her, here, here's, hereafter, hereby, herein, hereupon, hers, herself, hi, him, himself, his, hither, hopefully, how, howbeit, however, i, i'd, i'll, i'm, i've, ie, if, ignored, immediate, in, inasmuch, inc, indeed, indicate, indicated, indicates, inner, insofar, instead, into, inward, is, isn't, it, it'd, it'll, it's, its, itself, j, just, k, keep, keeps, kept, know, knows, known, l, last, lately, later, latter, latterly, least, less, lest, let, let's, like, liked, likely, little, look, looking, looks, ltd, m, mainly, many, may, maybe, me, mean, meanwhile, merely, might, more, moreover, most, mostly, much, must, my, myself, n, name, namely, nd, near, nearly, necessary, need, needs, neither, never, nevertheless, new, next, nine, no, nobody, non, none, noone, nor, normally, not, nothing, novel, now, nowhere, o, obviously, of, off, often, oh, ok, okay, old, on, once, one, ones, only, onto, or, other, others, otherwise, ought, our, ours, ourselves, out, outside, over, overall, own, p, particular, particularly, per, perhaps, placed, please, plus, possible, presumably, probably, provides, q, que, quite, qv, r, rather, rd, re, really, reasonably, regarding, regardless, regards, relatively, respectively, right, s, said, same, saw, say, saying, says, second, secondly, see, seeing, seem, seemed, seeming, seems, seen, self, selves, sensible, sent, serious, seriously, seven, several, shall, she, should, shouldn't, since, six, so, some, somebody, somehow, someone, something, sometime, sometimes, somewhat, somewhere, soon, sorry, specified, specify, specifying, still, sub, such, sup, sure, t, t's, take, taken, tell, tends, th, than, thank, thanks, thanx, that, that's, thats, the, their, theirs, them, themselves, then, thence, there, there's, thereafter, thereby, therefore, therein, theres, thereupon, these, they, they'd, they'll, they're, they've, think, third, this, thorough, thoroughly, those, though, three, through, throughout, thru, thus, to, together, too, took, toward, towards, tried, tries, truly, try, trying, twice, two, u, un, under, unfortunately, unless, unlikely, until, unto, up, upon, us, use, used, useful, uses, using, usually, uucp, v, value, various, very, via, viz, vs, w, want, wants, was, wasn't, way, we, we'd, we'll, we're, we've, welcome, well, went, were, weren't, what, what's, whatever, when, whence, whenever, where, where's, whereafter, whereas, whereby, wherein, whereupon, wherever, whether, which, while, whither, who, who's, whoever, whole, whom, whose, why, will, willing, wish, with, within, without, won't, wonder, would, would, wouldn't, x, y, yes, yet, you, you'd, you'll, you're, you've, your, yours, yourself, yourselves, z, zero"
PUNC_IDS = [4, 6, 12, 22, 35, 36, 43, 60, 68, 72, 73, 108, 111, 113, 116, 128, 131, 207, 238, 322, 328, 359, 479, 480, 482, 640, 646, 734, 742, 787, 838, 845, 849, 947, 955, 1009, 1039, 1215, 1297, 1358, 1437, 1589, 1592, 1598, 1629, 1640, 1666, 1721, 1917, 2055, 2153, 2156, 2165, 2652, 2744, 2901, 3226, 3256, 3358, 3934, 4234, 4332, 4397, 4805, 4832, 4839, 5214, 5457, 5579, 6600, 6697, 7479, 7586, 7606, 7862, 8061, 8070, 8174, 8488, 8871, 9376, 9957, 10068, 10076, 10116, 10431, 10559, 10975, 11227, 11665, 12345, 12606, 12651, 12801, 12846, 12905, 13198, 13278, 13373, 13540, 13864, 13909, 14025, 14220, 14434, 15057, 15483, 15611, 15698, 16276, 16506, 16844, 16998, 17220, 17487, 17495, 17516, 17523, 17809, 18134, 18456, 18653, 19207, 19246, 19281, 19651, 20186, 20551, 21154, 21277, 21394, 21509, 21594, 21704, 21838, 22209, 22560, 22896, 23500, 23528, 23985, 24095, 24303, 24464, 24521, 24524, 24681, 24965, 24992, 25522, 25606, 25718, 26487, 26610, 26638, 27079, 27144, 27148, 27203, 27223, 27282, 27645, 27779, 27785, 27868, 28114, 28578, 28696, 28749, 28784, 29064, 29462, 29482, 29483, 29942, 30115, 30171, 30529, 30550, 30697, 30787, 30831, 31051, 31095, 31175, 31274, 31311, 31509, 31558, 31897, 32269, 32376, 32801, 32965, 33031, 33137, 33525, 33647, 34133, 34199, 34437, 35122, 35227, 35290, 35347, 35524, 35661, 35965, 36098, 36137, 36185, 36380, 36418, 36440, 36467, 36538, 36592, 36738, 36856, 36917, 37249, 37398, 37421, 37457, 37637, 37640, 38203, 38304, 38502, 38581, 38713, 38844, 38917, 38947, 39058, 39365, 39550, 39574, 39732, 39747, 40021, 40255, 40321, 40389, 40398, 40635, 40862, 41006, 41039, 41066, 41110, 41137, 41478, 41500, 41552, 41667, 41734, 41833, 41945, 42053, 42078, 42199, 42202, 42248, 42254, 42255, 42296, 42326, 42514, 42593, 42604, 42645, 42648, 42654, 42760, 42777, 42964, 43002, 43003, 43012, 43048, 43074, 43080, 43101, 43292, 43303, 43305, 43344, 43353, 43401, 43521, 43564, 43613, 43636, 43754, 43775, 43796, 43809, 43839, 43912, 43988, 44065, 44082, 44116, 44128, 44162, 44226, 44259, 44294, 44371, 44374, 44403, 44408, 44418, 44431, 44447, 44460, 44516, 44612, 44626, 44629, 44660, 44688, 44690, 44706, 44717, 44757, 44793, 44832, 44926, 44942, 45056, 45072, 45152, 45177, 45333, 45364, 45376, 45381, 45390, 45393, 45405, 45406, 45518, 45587, 45592, 45627, 45693, 45737, 45751, 45793, 45803, 45863, 45894, 45912, 45946, 45973, 45994, 46077, 46082, 46117, 46142, 46150, 46156, 46161, 46225, 46250, 46253, 46303, 46343, 46386, 46469, 46479, 46481, 46495, 46564, 46580, 46613, 46650, 46671, 46679, 46686, 46844, 46904, 46934, 46939, 46945, 46961, 46992, 47006, 47033, 47038, 47052, 47075, 47096, 47110, 47148, 47155, 47161, 47162, 47259, 47365, 47385, 47408, 47426, 47429, 47457, 47460, 47463, 47517, 47529, 47539, 47567, 47570, 47579, 47584, 47619, 47620, 47639, 47655, 47659, 47720, 47770, 47771, 47789, 47793, 47813, 47826, 47919, 47965, 48004, 48029, 48030, 48037, 48077, 48081, 48082, 48086, 48110, 48119, 48124, 48134, 48149, 48182, 48188, 48200, 48203, 48209, 48229, 48232, 48256, 48268, 48273, 48289, 48292, 48298, 48306, 48329, 48332, 48336, 48342, 48347, 48364, 48371, 48377, 48392, 48395, 48404, 48433, 48443, 48457, 48461, 48462, 48474, 48505, 48512, 48513, 48520, 48546, 48554, 48562, 48565, 48601, 48610, 48613, 48614, 48615, 48630, 48634, 48640, 48651, 48654, 48660, 48669, 48677, 48691, 48694, 48709, 48712, 48729, 48742, 48749, 48755, 48759, 48771, 48784, 48789, 48793, 48794, 48803, 48805, 48817, 48832, 48833, 48835, 48844, 48845, 48855, 48872, 48880, 48893, 48898, 48900, 48902, 48906, 48919, 48935, 48936, 48937, 48948, 48950, 48982, 48989, 48992, 48999, 49000, 49007, 49024, 49038, 49051, 49069, 49070, 49071, 49085, 49087, 49091, 49092, 49095, 49097, 49104, 49123, 49128, 49130, 49138, 49143, 49145, 49151, 49153, 49154, 49170, 49177, 49183, 49189, 49193, 49196, 49197, 49198, 49201, 49215, 49216, 49230, 49242, 49248, 49255, 49258, 49275, 49279, 49281, 49283, 49291, 49293, 49296, 49308, 49309, 49310, 49314, 49316, 49318, 49329, 49333, 49338, 49346, 49358, 49364, 49366, 49371, 49374, 49380, 49384, 49389, 49394, 49410, 49419, 49420, 49423, 49424, 49434, 49436, 49440, 49445, 49452, 49453, 49455, 49463, 49487, 49489, 49509, 49515, 49518, 49519, 49521, 49525, 49526, 49536, 49557, 49563, 49570, 49599, 49604, 49608, 49609, 49612, 49614, 49625, 49629, 49632, 49639, 49643, 49655, 49659, 49666, 49667, 49670, 49674, 49675, 49681, 49688, 49690, 49698, 49701, 49703, 49710, 49712, 49713, 49721, 49727, 49731, 49738, 49739, 49747, 49750, 49755, 49761, 49763, 49774, 49778, 49783, 49784, 49789, 49790, 49795, 49798, 49799, 49800, 49803, 49806, 49812, 49814, 49817, 49826, 49828, 49830, 49836, 49849, 49852, 49853, 49858, 49859, 49868, 49871, 49882, 49888, 49890, 49893, 49895, 49900, 49903, 49905, 49908, 49909, 49910, 49915, 49918, 49921, 49923, 49925, 49936, 49938, 49940, 49953, 49954, 49959, 49962, 49964, 49969, 49972, 49979, 49982, 49987, 49991, 49995, 50000, 50003, 50004, 50007, 50012, 50014, 50015, 50016, 50017, 50018, 50019, 50020, 50022, 50024, 50025, 50028, 50031, 50037, 50053, 50061, 50065, 50068, 50072, 50078, 50084, 50154, 50155, 50161, 50179, 50184, 50185, 50189, 50193, 50206, 50236, 50254, 50255]

def agg_demon(sa_policy, simulator, orgs, nsteps, verbose=False, use_idf=True,
              init_state=None, preserve_order=True, sub_collects=None):

    agg_orgs, agg_simps, agg_scores, agg_probs, agg_steps = [], [], [], [], []
    if not isinstance(orgs, list):
        orgs = list(orgs.values)
    orgs = [org for org in orgs if len(org.split()) > 1]
    simps_old = init_state if init_state else orgs.copy()

    r = Rake()
    kws_all = [] # store tokenized keywords for each sent in a batch
    for org in orgs:
        r.extract_keywords_from_text(org)
        kws = r.get_ranked_phrases()
        if kws:
            kws = [x for w in kws for x in w.split()]
            kws_all.append(kws)
        else:
            kws_all.append([])

    if sub_collects:
        sub_dicts=[{}] * len(orgs)
        for i, org in enumerate(orgs):
            org = ' ' + org + ' '
            for phra_k, phra_v in sub_collects.items():
                phra_k_ = ' ' + phra_k + ' '
                if phra_k_ in org:
                    sub_dicts[i][phra_k] = phra_v
    else: sub_dicts=None

    # simulator.init(orgs)

    for t in range(nsteps):

        kws_out = []
        for kws, simps in zip(kws_all, simps_old):
            kws_out.append([w for w in kws if not w in simps])
        # for single case
        sampled_simps, sampled_probs, sampled_scores, sampled_3d, sampled_sim_tuple = \
            sa_policy.run(case_idx, simps_old, orgs, keywords=kws_out, sub_dicts=sub_dicts)

        accept, accept_probs = simulator.anneal(t, sampled_simps, sampled_scores, verbose=verbose)

        for b, (x, y, ac, score, p) in enumerate(zip(orgs, sampled_simps, accept.cpu(), sampled_scores, sampled_probs)):
            if ac == 1.:
                agg_orgs.append(x)
                agg_simps.append(y)
                agg_scores.append(score.item())
                agg_probs.append(p.item())
                agg_steps.append(t)
                simps_old[b] = y
            else:
                pass

    if preserve_order:
        agg_orgs_ordered, agg_simps_ordered, agg_scores_ordered, agg_steps_ordered \
                        = [None]*len(orgs), [None]*len(orgs), [-1.]*len(orgs), [1]*len(orgs)
        assert len(orgs) == len(list(set(orgs)))
        for b, org in enumerate(agg_orgs):
            idx = orgs.index(org)
            agg_orgs_ordered[idx] = org
            if agg_scores[b] > agg_scores_ordered[idx]:
                agg_scores_ordered[idx] = agg_scores[b]
                agg_simps_ordered[idx] = agg_simps[b]
                agg_steps_ordered[idx] = agg_steps[b]
        # assert None not in agg_orgs_ordered
        return agg_orgs_ordered, agg_simps_ordered, agg_scores_ordered, agg_steps_ordered
    else:
        return agg_orgs, agg_simps, agg_scores, agg_steps

def train_mc(args, sa_policy, simulator, print_every=1, check_every=1):
    to_file = to_file = codecs.open(args.agg_to_file, 'w+', encoding='utf-8')
    train_dataset = torch.load(args.data_path+'train.pt')
    fields = _load_fields(train_dataset, 'text', args.data_path, None)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        smapler=torch.utils.data.SequentialSampler(train_dataset),
        collate_fn=lambda batch: process_batch(batch, fields),
        shuffle=False,
        num_workers=0)

    # with open(args.rule_path, 'rb') as f:
    #     sub_collects = pickle.load(f)
    sub_collects = None

    for i, batch_df in enumerate(train_dataloader):
        indices = batch_df.indices
        org_tokens = [' '.join(train_dataset.textset[i].src) for i in indices.tolist()]

        init_state = simp_tokens if post_edit else None
        agg_orgs, agg_simps, agg_scores, agg_steps = agg_demon(
                sa_policy, simulator, org_tokens, args.mc_steps,
                init_state=init_state, sub_collects=sub_collects)
        
        for x, y, p in zip(agg_orgs, agg_simps, agg_scores):
            to_file.write(x+'\t'+y+'\t'+str(p)+'\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to train data')

    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)

    parser.add_argument('--rbt_model', type=str, default='bert-base-uncased')
    parser.add_argument('--got_model', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')

    parser.add_argument("--mc_steps", type=int, default=100, help="mc sample steps")

    parser.add_argument("--agg_to_file", type=str, default=None, help='path to aggregated file')

    parser.add_argument('--tinit', type=float, default=3e-2, help='initial temperature')

    parser.add_argument('--C', type=float, default=3e-4, help='scale of temp')

    parser.add_argument("--K", type=int, default=100, help="sample subset size")

    parser.add_argument("--tk_layer", type=int, default=8, help="nth layer for sent bertscore")
    parser.add_argument("--sent_layer", type=int, default=1, help="nth layer for keyword bertscore")

    parser.add_argument("--do_copy", action='store_true', help="")

    parser.add_argument('--alpha', type=int, default=3, help='power for bert score (sentence level semantic sim)')
    parser.add_argument("--beta", type=int, default=8, help="power for bert_kw in LM for sentence candidate selection")

    parser.add_argument('--lm_model_path', type=str, default=None)
    parser.add_argument('--lm_vocab_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=40, help='maybe for lm')

    parser.add_argument('--fm_wght', type=float, default=0.125, help='weight of formal')
    parser.add_argument('--lm_wght', type=float, default=0.125, help='weight of lm')
    parser.add_argument('--temp', type=float, default=8., help='')

    parser.add_argument('--rule_path', type=str, default='')

    parser.add_argument('--score_mode', type=str, default='min',
                        choices=['sasm', 'sakm', 'smkm'], help='')
    parser.add_argument('--sa_policy', type=str, default='one', choices=['one', 'bs'], help='')
    parser.add_argument('--use_lm_diff', action='store_true')
    parser.add_argument('--use_cls_diff', action='store_true')
    parser.add_argument('--rand_pos', dest='rand_pos', action='store_true')
    parser.add_argument('--no_rand_pos', dest='rand_pos', action='store_false')
    parser.set_defaults(rand_pos=True)
    parser.add_argument('--nplh', type=int, default=2)

    args = parser.parse_args()

    gpt_config = GPT2Config.from_pretrianed(args.gpt_model)
    gpt_tknzr = GPT2Tokenizer.from_pretrianed(args.gpt_model, do_lower_case=args.do_lower_case)
    gpt_model = GPT2LMHeadModel.from_pretrianed(args.gpt_model, config=gpt_config, cache_dir=None)
    gpt_model.eval()

    rbt_config = RobertaConfig.from_pretrianed(args.rbt_model)
    rbt_tknzr = RobertaTokenizer.from_pretrianed(args.rbt_config, do_lower_case=args.do_lower_case)
    rbt_model = RobertaForMTL.from_pretrianed(args.rbt_model, config=rbt_config)
    rbt_model.eval()

    scorer = sa_score_funcs.SAScorer(args.batch_size, rbt_model, rbt_tknzr, punc_ids=PUNC_IDS,
                                       gpt_model=gpt_model, gpt_tknzr=gpt_tknzr, stopwords=STOPWORDS,
                                       verbose=False, tinit=args.tinit, C=args.C,
                                       sent_layer = args.tk_layer, kw_layer=args.sent_layer,
                                       alpha=args.alpha, beta=args.beta,
                                       fm_wght=args.fm_wght, lm_wght=args.lm_wght,
                                       score_mode=args.score_mode, use_lm_diff=args.use_lm_diff,
                                       use_cls_diff=args.use_cls_diff)
    
    seacher = sa_search.SASearcher(scorer, rbt_model=rbt_model, rbt_tokenizer=rbt_tknzr,
                          k=args.K, do_copy=args.do_copy,
                          device=device, temp=args.temp, rand_pos=args.rand_pos,
                          if_fm_voc=None, nplh=args.nplh)
    train_mc(args, seacher, scorer)


if __name__ == '__main__':
    main()